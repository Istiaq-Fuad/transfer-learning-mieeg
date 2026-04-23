from __future__ import annotations

import argparse
import copy
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

try:
    from data.loader import create_dataloaders, load_moabb_motor_imagery_dataset
    from models.heads import SSLProjectionHead
    from models.model import EEGModel
    from training.utils import lambda_scheduler
    from utils.reproducibility import (
        create_experiment_metadata,
        save_experiment_metadata,
        set_seed_everywhere,
    )
except ModuleNotFoundError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from data.loader import create_dataloaders, load_moabb_motor_imagery_dataset
    from models.heads import SSLProjectionHead
    from models.model import EEGModel
    from training.utils import lambda_scheduler
    from utils.reproducibility import (
        create_experiment_metadata,
        save_experiment_metadata,
        set_seed_everywhere,
    )


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pretrain_cross_dataset")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def _state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


@torch.no_grad()
def evaluate_task(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    for x, y, _ in data_loader:
        x = x.to(device)
        y = y.to(device)

        outputs = model(x, lambda_=0.0)
        logits = outputs["task"]
        loss = criterion(logits, y)
        preds = logits.argmax(dim=1)

        batch_size = y.size(0)
        total += batch_size
        running_loss += loss.item() * batch_size
        correct += (preds == y).sum().item()
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    accuracy = correct / max(total, 1)
    kappa = cohen_kappa_score(y_true, y_pred) if total > 0 else 0.0
    return {
        "accuracy": float(accuracy),
        "kappa": float(kappa),
        "task_loss": float(running_loss / max(total, 1)),
    }


def _augment_batch_ssl(
    x: torch.Tensor,
    noise_std: float,
    time_mask_ratio: float,
) -> torch.Tensor:
    x_aug = x.clone()
    if noise_std > 0.0:
        x_aug = x_aug + noise_std * torch.randn_like(x_aug)

    if time_mask_ratio > 0.0:
        t = x_aug.size(-1)
        mask_len = max(1, int(round(t * time_mask_ratio)))
        mask_len = min(mask_len, t)
        max_start = max(t - mask_len, 0)
        starts = torch.randint(0, max_start + 1, (x_aug.size(0),), device=x_aug.device)
        apply_mask = torch.rand(x_aug.size(0), device=x_aug.device) < 0.5
        for idx in range(x_aug.size(0)):
            if apply_mask[idx]:
                start = int(starts[idx].item())
                x_aug[idx, :, start : start + mask_len] = 0.0
    return x_aug


def _nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    if z1.size(0) < 2:
        return torch.tensor(0.0, device=z1.device)

    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    z = torch.cat([z1, z2], dim=0)

    logits = torch.matmul(z, z.transpose(0, 1)) / max(temperature, 1e-6)
    logits = logits.masked_fill(torch.eye(logits.size(0), device=z.device, dtype=torch.bool), float("-inf"))

    b = z1.size(0)
    targets = torch.cat(
        [
            torch.arange(b, 2 * b, device=z.device),
            torch.arange(0, b, device=z.device),
        ]
    )
    return F.cross_entropy(logits, targets)


@torch.no_grad()
def evaluate_ssl(
    model: nn.Module,
    projection_head: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    noise_std: float,
    time_mask_ratio: float,
    temperature: float,
) -> dict[str, float]:
    model.eval()
    projection_head.eval()

    running_ssl = 0.0
    n_samples = 0

    for xb, _, _ in data_loader:
        if xb.size(0) < 2:
            continue
        xb = xb.to(device)
        x1 = _augment_batch_ssl(xb, noise_std=noise_std, time_mask_ratio=time_mask_ratio)
        x2 = _augment_batch_ssl(xb, noise_std=noise_std, time_mask_ratio=time_mask_ratio)

        out1 = model(x1, lambda_=0.0)
        out2 = model(x2, lambda_=0.0)
        z1 = projection_head(out1["features"])
        z2 = projection_head(out2["features"])
        ssl_loss = _nt_xent_loss(z1, z2, temperature=temperature)

        bs = xb.size(0)
        n_samples += bs
        running_ssl += ssl_loss.item() * bs

    return {
        "ssl_loss": float(running_ssl / max(n_samples, 1)),
    }


def _make_domain_labels(
    dataset_index: int,
    source_subject_id: np.ndarray,
    domain_mode: str,
    subject_domain_offset: int,
) -> tuple[np.ndarray, int]:
    if domain_mode == "dataset":
        domain_id = np.full_like(source_subject_id, fill_value=dataset_index, dtype=np.int64)
        return domain_id, subject_domain_offset

    unique_subjects = sorted(np.unique(source_subject_id).tolist())
    subject_map = {
        int(subject): idx + subject_domain_offset
        for idx, subject in enumerate(unique_subjects)
    }
    domain_id = np.asarray([subject_map[int(s)] for s in source_subject_id], dtype=np.int64)
    next_offset = subject_domain_offset + len(unique_subjects)
    return domain_id, next_offset


def load_source_mix(
    datasets: list[str],
    data_path: str | None,
    domain_mode: str,
    class_policy: str,
    require_two_classes: bool,
    max_subjects_per_dataset: int | None,
    skip_failed_subjects: bool,
    subject_load_retries: int,
    redownload_on_failure: bool,
    redownload_once_per_subject: bool,
    skip_known_failed_subjects: bool,
    logger: logging.Logger,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    list[dict[str, Any]],
    list[dict[str, str]],
]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    domain_parts: list[np.ndarray] = []
    stats: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    subject_domain_offset = 0
    included_dataset_index = 0

    for ds_name in datasets:
        try:
            x, y, subject_id, subjects = load_moabb_motor_imagery_dataset(
                dataset_name=ds_name,
                data_path=data_path,
                class_policy=class_policy,
                max_subjects=max_subjects_per_dataset,
                show_progress=True,
                skip_failed_subjects=skip_failed_subjects,
                subject_load_retries=subject_load_retries,
                redownload_on_failure=redownload_on_failure,
                redownload_once_per_subject=redownload_once_per_subject,
                skip_known_failed_subjects=skip_known_failed_subjects,
            )
        except Exception as exc:
            if not skip_failed_subjects:
                raise
            reason = str(exc)
            skipped.append({"dataset": ds_name, "reason": reason})
            logger.warning("Skipping %s: %s", ds_name, reason)
            continue

        present_labels = sorted(np.unique(y).tolist())
        if require_two_classes and len(present_labels) < 2:
            reason = (
                "fewer than 2 classes after filtering "
                f"(present encoded labels={present_labels})"
            )
            skipped.append({"dataset": ds_name, "reason": reason})
            logger.warning("Skipping %s: %s", ds_name, reason)
            continue

        domain_id, subject_domain_offset = _make_domain_labels(
            dataset_index=included_dataset_index,
            source_subject_id=subject_id,
            domain_mode=domain_mode,
            subject_domain_offset=subject_domain_offset,
        )
        included_dataset_index += 1

        x_parts.append(x.astype(np.float32))
        y_parts.append(y.astype(np.int64))
        domain_parts.append(domain_id.astype(np.int64))

        unique_domains = int(np.unique(domain_id).shape[0])
        row = {
            "dataset": ds_name,
            "n_trials": int(x.shape[0]),
            "n_channels": int(x.shape[1]),
            "n_times": int(x.shape[2]),
            "n_subjects": int(len(subjects)),
            "n_classes": int(np.unique(y).shape[0]),
            "n_domain_labels": unique_domains,
        }
        stats.append(row)
        logger.info(
            "Loaded %s: trials=%d, channels=%d, times=%d, subjects=%d, classes=%d, domain_labels=%d",
            ds_name,
            row["n_trials"],
            row["n_channels"],
            row["n_times"],
            row["n_subjects"],
            row["n_classes"],
            row["n_domain_labels"],
        )

    if not x_parts:
        skipped_text = "; ".join(
            f"{entry['dataset']}: {entry['reason']}" for entry in skipped
        )
        raise ValueError(
            "No usable source datasets after class filtering. "
            f"Skipped: {skipped_text}"
        )

    target_channels = min(arr.shape[1] for arr in x_parts)
    target_times = min(arr.shape[2] for arr in x_parts)
    if any(arr.shape[1] != target_channels or arr.shape[2] != target_times for arr in x_parts):
        logger.warning(
            "Harmonizing source tensors to channels=%d, times=%d before concatenation",
            target_channels,
            target_times,
        )
    x_parts = [arr[:, :target_channels, :target_times] for arr in x_parts]

    x_all = np.concatenate(x_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    domain_all = np.concatenate(domain_parts, axis=0)
    num_domains = int(np.unique(domain_all).shape[0])

    logger.info(
        "Source mix ready: trials=%d, channels=%d, times=%d, num_domains=%d",
        int(x_all.shape[0]),
        int(x_all.shape[1]),
        int(x_all.shape[2]),
        num_domains,
    )

    return x_all, y_all, domain_all, num_domains, stats, skipped


def run(args: argparse.Namespace) -> None:
    set_seed_everywhere(args.seed, deterministic=args.deterministic)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag if args.tag else "_".join(args.source_datasets)
    run_dir = Path(args.output_dir) / f"pretrain_{tag}_{timestamp}"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir)

    class_policy = "all_mi" if args.pretrain_mode == "ssl" else "left_right"
    require_two_classes = args.pretrain_mode == "supervised"
    effective_domain_loss_weight = (
        args.ssl_domain_loss_weight
        if args.pretrain_mode == "ssl"
        else args.domain_loss_weight
    )

    logger.info("Starting cross-dataset pretraining")
    logger.info(
        "pretrain_mode=%s | class_policy=%s | domain_mode=%s",
        args.pretrain_mode,
        class_policy,
        args.domain_mode,
    )

    x, y, domain_id, num_domains, source_stats, skipped_datasets = load_source_mix(
        datasets=args.source_datasets,
        data_path=args.data_path,
        domain_mode=args.domain_mode,
        class_policy=class_policy,
        require_two_classes=require_two_classes,
        max_subjects_per_dataset=args.max_subjects_per_dataset,
        skip_failed_subjects=args.skip_failed_subjects,
        subject_load_retries=args.subject_load_retries,
        redownload_on_failure=args.redownload_on_failure,
        redownload_once_per_subject=args.redownload_once_per_subject,
        skip_known_failed_subjects=args.skip_known_failed_subjects,
        logger=logger,
    )

    if skipped_datasets:
        logger.info("Skipped datasets after label filtering: %s", skipped_datasets)

    train_loader, val_loader = create_dataloaders(
        x=x,
        y=y,
        subject_id=domain_id,
        batch_size=args.batch_size,
        test_size=args.val_split,
        random_state=args.seed,
        loso_subject=None,
        apply_euclidean_align=args.loader_euclidean_align,
        num_workers=args.num_workers,
        seed=args.seed,
        deterministic=args.deterministic,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGModel(
        num_channels=int(x.shape[1]),
        num_classes=int(np.max(y)) + 1,
        num_subjects=num_domains,
        cnn_out_channels=args.cnn_out_channels,
        cnn_dropout=args.cnn_dropout,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        temporal_kernels=tuple(args.temporal_kernels) if args.temporal_kernels else None,
        multiscale_preserve_capacity=args.multiscale_preserve_capacity,
        use_attention_pool=args.use_attention_pool,
        attention_mix_init=args.attention_mix_init,
        learnable_attention_mix=args.learnable_attention_mix,
        domain_head_hidden_dim=args.domain_head_hidden_dim,
        domain_head_layers=args.domain_head_layers,
        domain_head_dropout=args.domain_head_dropout,
        use_cnn_domain_head=args.use_cnn_domain_head,
        cnn_domain_weight=args.cnn_domain_weight,
        apply_model_pre_align_only=args.model_pre_align_only,
        apply_model_euclidean_alignment=args.model_euclidean_alignment,
        apply_model_riemannian_reweight=args.model_riemannian_reweight,
    ).to(device)

    projection_head: SSLProjectionHead | None = None
    if args.pretrain_mode == "ssl":
        projection_head = SSLProjectionHead(
            in_dim=args.embedding_dim,
            hidden_dim=args.ssl_hidden_dim,
            out_dim=args.ssl_proj_dim,
        ).to(device)

    params = list(model.parameters())
    if projection_head is not None:
        params += list(projection_head.parameters())

    optimizer = Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    task_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    config = {
        "source_datasets": args.source_datasets,
        "pretrain_mode": args.pretrain_mode,
        "class_policy": class_policy,
        "domain_mode": args.domain_mode,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "val_split": args.val_split,
        "domain_loss_weight": args.domain_loss_weight,
        "ssl_domain_loss_weight": args.ssl_domain_loss_weight,
        "effective_domain_loss_weight": effective_domain_loss_weight,
        "da_lambda_gamma": args.da_lambda_gamma,
        "loader_euclidean_align": args.loader_euclidean_align,
        "model_pre_align_only": args.model_pre_align_only,
        "model_euclidean_alignment": args.model_euclidean_alignment,
        "model_riemannian_reweight": args.model_riemannian_reweight,
        "temporal_kernels": args.temporal_kernels,
        "multiscale_preserve_capacity": args.multiscale_preserve_capacity,
        "use_attention_pool": args.use_attention_pool,
        "attention_mix_init": args.attention_mix_init,
        "learnable_attention_mix": args.learnable_attention_mix,
        "domain_head_hidden_dim": args.domain_head_hidden_dim,
        "domain_head_layers": args.domain_head_layers,
        "domain_head_dropout": args.domain_head_dropout,
        "use_cnn_domain_head": args.use_cnn_domain_head,
        "cnn_domain_weight": args.cnn_domain_weight,
        "cnn_out_channels": args.cnn_out_channels,
        "cnn_dropout": args.cnn_dropout,
        "embedding_dim": args.embedding_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "num_workers": args.num_workers,
        "data_path": args.data_path,
        "max_subjects_per_dataset": args.max_subjects_per_dataset,
        "skip_failed_subjects": args.skip_failed_subjects,
        "subject_load_retries": args.subject_load_retries,
        "redownload_on_failure": args.redownload_on_failure,
        "redownload_once_per_subject": args.redownload_once_per_subject,
        "skip_known_failed_subjects": args.skip_known_failed_subjects,
        "ssl_temperature": args.ssl_temperature,
        "ssl_proj_dim": args.ssl_proj_dim,
        "ssl_hidden_dim": args.ssl_hidden_dim,
        "ssl_weight": args.ssl_weight,
        "ssl_noise_std": args.ssl_noise_std,
        "ssl_time_mask_ratio": args.ssl_time_mask_ratio,
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    metadata = create_experiment_metadata(
        protocol="pretrain_cross_dataset",
        dataset="+".join(args.source_datasets),
        config=config,
        seed=args.seed,
        deterministic=args.deterministic,
    )
    save_experiment_metadata(run_dir / "metadata.json", metadata)

    history: list[dict[str, float]] = []
    best = {
        "accuracy": 0.0,
        "kappa": 0.0,
        "task_loss": float("inf"),
        "ssl_loss": float("inf"),
        "epoch": 0.0,
    }
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_proj_state_dict: dict[str, torch.Tensor] | None = None

    logger.info("Training on %s", str(device))
    for epoch in range(args.epochs):
        model.train()
        if projection_head is not None:
            projection_head.train()
        lam = lambda_scheduler(epoch, args.epochs, gamma=args.da_lambda_gamma)

        running_total = 0.0
        running_task = 0.0
        running_ssl = 0.0
        running_domain = 0.0
        running_domain_cnn = 0.0
        n_samples = 0

        for xb, yb, db in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            db = db.to(device)

            outputs = model(xb, lambda_=lam)
            domain_logits = outputs["domain"]

            if args.pretrain_mode == "ssl":
                if projection_head is None:
                    raise RuntimeError("projection_head is required in ssl mode")
                x1 = _augment_batch_ssl(
                    xb,
                    noise_std=args.ssl_noise_std,
                    time_mask_ratio=args.ssl_time_mask_ratio,
                )
                x2 = _augment_batch_ssl(
                    xb,
                    noise_std=args.ssl_noise_std,
                    time_mask_ratio=args.ssl_time_mask_ratio,
                )
                out1 = model(x1, lambda_=0.0)
                out2 = model(x2, lambda_=0.0)
                z1 = projection_head(out1["features"])
                z2 = projection_head(out2["features"])
                ssl_loss = _nt_xent_loss(z1, z2, temperature=args.ssl_temperature)
                task_loss = torch.tensor(0.0, device=device)
            else:
                task_logits = outputs["task"]
                task_loss = task_criterion(task_logits, yb)
                ssl_loss = torch.tensor(0.0, device=device)

            domain_loss = domain_criterion(domain_logits, db)

            domain_cnn_loss = torch.tensor(0.0, device=device)
            if "domain_cnn" in outputs:
                domain_cnn_loss = domain_criterion(outputs["domain_cnn"], db)

            domain_total = domain_loss + args.cnn_domain_weight * domain_cnn_loss
            total_loss = (
                task_loss
                + args.ssl_weight * ssl_loss
                + effective_domain_loss_weight * lam * domain_total
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_size = xb.size(0)
            n_samples += batch_size
            running_total += total_loss.item() * batch_size
            running_task += task_loss.item() * batch_size
            running_ssl += ssl_loss.item() * batch_size
            running_domain += domain_loss.item() * batch_size
            running_domain_cnn += domain_cnn_loss.item() * batch_size

        if args.pretrain_mode == "ssl":
            if projection_head is None:
                raise RuntimeError("projection_head is required in ssl mode")
            val_ssl = evaluate_ssl(
                model,
                projection_head,
                val_loader,
                device,
                noise_std=args.ssl_noise_std,
                time_mask_ratio=args.ssl_time_mask_ratio,
                temperature=args.ssl_temperature,
            )
            val_metrics = {
                "task_loss": -1.0,
                "accuracy": -1.0,
                "kappa": -1.0,
                "ssl_loss": val_ssl["ssl_loss"],
            }
        else:
            val_task = evaluate_task(model, val_loader, device, task_criterion)
            val_metrics = {
                "task_loss": val_task["task_loss"],
                "accuracy": val_task["accuracy"],
                "kappa": val_task["kappa"],
                "ssl_loss": -1.0,
            }

        row = {
            "epoch": float(epoch + 1),
            "lambda": float(lam),
            "train_loss": float(running_total / max(n_samples, 1)),
            "train_task_loss": float(running_task / max(n_samples, 1)),
            "train_ssl_loss": float(running_ssl / max(n_samples, 1)),
            "train_domain_loss": float(running_domain / max(n_samples, 1)),
            "train_domain_cnn_loss": float(running_domain_cnn / max(n_samples, 1)),
            "val_task_loss": float(val_metrics["task_loss"]),
            "val_ssl_loss": float(val_metrics["ssl_loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_kappa": float(val_metrics["kappa"]),
        }
        history.append(row)

        if args.pretrain_mode == "ssl":
            if val_metrics["ssl_loss"] <= best["ssl_loss"]:
                best = {
                    "accuracy": -1.0,
                    "kappa": -1.0,
                    "task_loss": -1.0,
                    "ssl_loss": float(val_metrics["ssl_loss"]),
                    "epoch": float(epoch + 1),
                }
                best_state_dict = copy.deepcopy(_state_dict_cpu(model))
                if projection_head is not None:
                    best_proj_state_dict = copy.deepcopy(_state_dict_cpu(projection_head))

            logger.info(
                "epoch=%d | mode=ssl | lambda=%.4f | train_loss=%.4f | ssl=%.4f | domain=%.4f | domain_cnn=%.4f | val_ssl=%.4f",
                epoch + 1,
                lam,
                row["train_loss"],
                row["train_ssl_loss"],
                row["train_domain_loss"],
                row["train_domain_cnn_loss"],
                row["val_ssl_loss"],
            )
        else:
            if val_metrics["accuracy"] >= best["accuracy"]:
                best = {
                    "accuracy": float(val_metrics["accuracy"]),
                    "kappa": float(val_metrics["kappa"]),
                    "task_loss": float(val_metrics["task_loss"]),
                    "ssl_loss": -1.0,
                    "epoch": float(epoch + 1),
                }
                best_state_dict = copy.deepcopy(_state_dict_cpu(model))

            logger.info(
                "epoch=%d | mode=supervised | lambda=%.4f | train_loss=%.4f | task=%.4f | domain=%.4f | domain_cnn=%.4f | val_loss=%.4f | val_acc=%.4f | val_kappa=%.4f",
                epoch + 1,
                lam,
                row["train_loss"],
                row["train_task_loss"],
                row["train_domain_loss"],
                row["train_domain_cnn_loss"],
                row["val_task_loss"],
                row["val_accuracy"],
                row["val_kappa"],
            )

    torch.save(model.state_dict(), ckpt_dir / "pretrain_last.pt")
    if projection_head is not None:
        torch.save(projection_head.state_dict(), ckpt_dir / "ssl_proj_last.pt")
    if best_state_dict is not None:
        torch.save(best_state_dict, ckpt_dir / "pretrain_best.pt")
    if projection_head is not None and best_proj_state_dict is not None:
        torch.save(best_proj_state_dict, ckpt_dir / "ssl_proj_best.pt")

    with (run_dir / "pretrain_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    summary = {
        "protocol": "pretrain_cross_dataset",
        "source_datasets": args.source_datasets,
        "skipped_datasets": skipped_datasets,
        "pretrain_mode": args.pretrain_mode,
        "class_policy": class_policy,
        "domain_mode": args.domain_mode,
        "num_domains": num_domains,
        "n_train": int(len(train_loader.dataset)),
        "n_val": int(len(val_loader.dataset)),
        "source_stats": source_stats,
        "best": best,
        "checkpoints": {
            "last": str(ckpt_dir / "pretrain_last.pt"),
            "best": str(ckpt_dir / "pretrain_best.pt"),
        },
    }
    if projection_head is not None:
        summary["checkpoints"]["ssl_proj_last"] = str(ckpt_dir / "ssl_proj_last.pt")
        summary["checkpoints"]["ssl_proj_best"] = str(ckpt_dir / "ssl_proj_best.pt")
    with (run_dir / "pretrain_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.pretrain_mode == "ssl":
        logger.info(
            "Done. best_val_ssl=%.4f at epoch %.0f | summary=%s",
            best["ssl_loss"],
            best["epoch"],
            str(run_dir / "pretrain_summary.json"),
        )
    else:
        logger.info(
            "Done. best_val_acc=%.4f at epoch %.0f | summary=%s",
            best["accuracy"],
            best["epoch"],
            str(run_dir / "pretrain_summary.json"),
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-dataset supervised + adversarial pretraining on MOABB MI datasets"
    )
    parser.add_argument(
        "--source_datasets",
        nargs="+",
        type=str,
        default=["physionetmi", "cho2017", "lee2019_mi"],
        choices=["physionetmi", "cho2017", "lee2019_mi", "bnci2014_001"],
    )
    parser.add_argument("--domain_mode", type=str, default="dataset", choices=["dataset", "subject"])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/pretrain_cross_dataset")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--max_subjects_per_dataset", type=int, default=None)
    parser.add_argument("--skip_failed_subjects", action="store_true", default=True)
    parser.add_argument(
        "--no-skip-failed-subjects",
        dest="skip_failed_subjects",
        action="store_false",
    )
    parser.add_argument("--subject_load_retries", type=int, default=1)
    parser.add_argument("--redownload_on_failure", action="store_true", default=True)
    parser.add_argument(
        "--no-redownload-on-failure",
        dest="redownload_on_failure",
        action="store_false",
    )
    parser.add_argument("--redownload_once_per_subject", action="store_true", default=True)
    parser.add_argument(
        "--no-redownload-once-per-subject",
        dest="redownload_once_per_subject",
        action="store_false",
    )
    parser.add_argument("--skip_known_failed_subjects", action="store_true", default=True)
    parser.add_argument(
        "--no-skip-known-failed-subjects",
        dest="skip_known_failed_subjects",
        action="store_false",
    )
    parser.add_argument(
        "--pretrain_mode",
        type=str,
        default="ssl",
        choices=["supervised", "ssl"],
    )

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--domain_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--ssl_domain_loss_weight",
        type=float,
        default=0.2,
    )
    parser.add_argument("--da_lambda_gamma", type=float, default=10.0)
    parser.add_argument("--ssl_weight", type=float, default=1.0)
    parser.add_argument("--ssl_temperature", type=float, default=0.2)
    parser.add_argument("--ssl_proj_dim", type=int, default=128)
    parser.add_argument("--ssl_hidden_dim", type=int, default=256)
    parser.add_argument("--ssl_noise_std", type=float, default=0.02)
    parser.add_argument("--ssl_time_mask_ratio", type=float, default=0.1)
    parser.add_argument("--loader_euclidean_align", action="store_true", default=True)
    parser.add_argument(
        "--no-loader-euclidean-align",
        dest="loader_euclidean_align",
        action="store_false",
    )
    parser.add_argument("--model_pre_align_only", action="store_true", default=False)
    parser.add_argument(
        "--no-model-pre-align-only",
        dest="model_pre_align_only",
        action="store_false",
    )
    parser.add_argument("--model_euclidean_alignment", action="store_true", default=True)
    parser.add_argument(
        "--no-model-euclidean-alignment",
        dest="model_euclidean_alignment",
        action="store_false",
    )
    parser.add_argument("--model_riemannian_reweight", action="store_true", default=True)
    parser.add_argument(
        "--no-model-riemannian-reweight",
        dest="model_riemannian_reweight",
        action="store_false",
    )

    parser.add_argument("--cnn_out_channels", type=int, default=32)
    parser.add_argument("--cnn_dropout", type=float, default=0.5)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temporal_kernels", nargs="+", type=int, default=None)
    parser.add_argument("--multiscale_preserve_capacity", action="store_true", default=False)
    parser.add_argument("--use_attention_pool", action="store_true", default=False)
    parser.add_argument("--attention_mix_init", type=float, default=0.5)
    parser.add_argument("--learnable_attention_mix", action="store_true", default=False)
    parser.add_argument("--domain_head_hidden_dim", type=int, default=0)
    parser.add_argument("--domain_head_layers", type=int, default=1)
    parser.add_argument("--domain_head_dropout", type=float, default=0.0)
    parser.add_argument("--use_cnn_domain_head", action="store_true", default=False)
    parser.add_argument("--cnn_domain_weight", type=float, default=0.0)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument(
        "--no-deterministic",
        dest="deterministic",
        action="store_false",
    )
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
