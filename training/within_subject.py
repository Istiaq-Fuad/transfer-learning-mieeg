from __future__ import annotations

import argparse
import copy
import json
import logging
from itertools import cycle
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from data.loader import (
        create_within_subject_domain_adaptation_dataloaders,
        create_within_subject_dataloaders,
        load_moabb_motor_imagery_dataset,
    )
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
    from data.loader import (
        create_within_subject_domain_adaptation_dataloaders,
        create_within_subject_dataloaders,
        load_moabb_motor_imagery_dataset,
    )
    from models.model import EEGModel
    from training.utils import lambda_scheduler
    from utils.reproducibility import (
        create_experiment_metadata,
        save_experiment_metadata,
        set_seed_everywhere,
    )


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("within_subject")
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


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []
    correct = 0
    total = 0

    for x, y, _ in data_loader:
        x = x.to(device)
        y = y.to(device)

        outputs = model(x, lambda_=0.0)
        logits = outputs["task"]
        preds = logits.argmax(dim=1)

        total += y.size(0)
        correct += (preds == y).sum().item()

        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(y.cpu().tolist())

    acc = correct / max(total, 1)
    kappa = cohen_kappa_score(all_targets, all_preds) if total > 0 else 0.0
    return {"accuracy": float(acc), "kappa": float(kappa)}


def _state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _augment_batch(
    x: torch.Tensor,
    augment_noise_std: float,
    augment_time_mask_ratio: float,
) -> torch.Tensor:
    if augment_noise_std > 0.0:
        x = x + augment_noise_std * torch.randn_like(x)

    if augment_time_mask_ratio > 0.0:
        t = x.size(-1)
        mask_len = max(1, int(round(t * augment_time_mask_ratio)))
        mask_len = min(mask_len, t)
        max_start = max(t - mask_len, 0)
        mask_starts = torch.randint(
            low=0,
            high=max_start + 1,
            size=(x.size(0),),
            device=x.device,
        )
        apply_mask = torch.rand(x.size(0), device=x.device) < 0.5
        for idx in range(x.size(0)):
            if apply_mask[idx]:
                start = int(mask_starts[idx].item())
                x[idx, :, start : start + mask_len] = 0.0
    return x


def train_one_subject(
    model: EEGModel,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    min_lr: float,
    weight_decay: float,
    label_smoothing: float,
    grad_clip_norm: float,
    early_stopping_patience: int,
    selection_metric: str,
    augment_noise_std: float,
    augment_time_mask_ratio: float,
    logger: logging.Logger,
) -> tuple[dict[str, float], list[dict[str, float]], dict[str, torch.Tensor] | None]:
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=min_lr)

    model.to(device)
    best = {"accuracy": 0.0, "kappa": 0.0, "epoch": 0}
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_score = float("-inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_samples = 0

        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)

            x = _augment_batch(x, augment_noise_std, augment_time_mask_ratio)

            outputs = model(x, lambda_=0.0)
            logits = outputs["task"]
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm > 0.0:
                clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            batch_size = x.size(0)
            n_samples += batch_size
            running_loss += loss.item() * batch_size

        train_loss = running_loss / max(n_samples, 1)
        test_metrics = evaluate(model, test_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        epoch_row = {
            "epoch": float(epoch + 1),
            "train_loss": float(train_loss),
            "test_accuracy": test_metrics["accuracy"],
            "test_kappa": test_metrics["kappa"],
            "lr": float(current_lr),
        }
        history.append(epoch_row)

        current_score = test_metrics[selection_metric]
        if current_score > best_score:
            best_score = current_score
            best = {
                "accuracy": test_metrics["accuracy"],
                "kappa": test_metrics["kappa"],
                "epoch": float(epoch + 1),
            }
            best_state_dict = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        logger.info(
            "epoch=%d | lr=%.6f | train_loss=%.4f | test_acc=%.4f | test_kappa=%.4f",
            epoch + 1,
            current_lr,
            train_loss,
            test_metrics["accuracy"],
            test_metrics["kappa"],
        )

        if early_stopping_patience > 0 and stale_epochs >= early_stopping_patience:
            logger.info(
                "Early stopping at epoch=%d (no %s improvement for %d epochs)",
                epoch + 1,
                selection_metric,
                early_stopping_patience,
            )
            break

    return best, history, best_state_dict


def train_one_subject_da(
    model: EEGModel,
    source_loader: torch.utils.data.DataLoader,
    target_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    min_lr: float,
    weight_decay: float,
    label_smoothing: float,
    grad_clip_norm: float,
    early_stopping_patience: int,
    selection_metric: str,
    augment_noise_std: float,
    augment_time_mask_ratio: float,
    domain_loss_weight: float,
    da_lambda_gamma: float,
    cnn_domain_weight: float,
    logger: logging.Logger,
) -> tuple[dict[str, float], list[dict[str, float]], dict[str, torch.Tensor] | None]:
    task_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    domain_criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=min_lr)

    model.to(device)
    best = {"accuracy": 0.0, "kappa": 0.0, "epoch": 0}
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_score = float("-inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        lam = lambda_scheduler(epoch, epochs, gamma=da_lambda_gamma)
        running_total_loss = 0.0
        running_task_loss = 0.0
        running_domain_loss = 0.0
        running_domain_cnn_loss = 0.0
        n_samples = 0

        for (x_s, y_s, d_s), (x_t, _, d_t) in zip(source_loader, cycle(target_loader)):
            x_s = x_s.to(device)
            y_s = y_s.to(device)
            d_s = d_s.to(device)
            x_t = x_t.to(device)
            d_t = d_t.to(device)

            x_s = _augment_batch(x_s, augment_noise_std, augment_time_mask_ratio)
            x_t = _augment_batch(x_t, augment_noise_std, augment_time_mask_ratio)

            out_src = model(x_s, lambda_=lam)
            out_tgt = model(x_t, lambda_=lam)

            task_logits = out_src["task"]
            domain_src = out_src["domain"]
            domain_tgt = out_tgt["domain"]

            task_loss = task_criterion(task_logits, y_s)
            domain_loss = 0.5 * (
                domain_criterion(domain_src, d_s) + domain_criterion(domain_tgt, d_t)
            )

            domain_cnn_loss = torch.tensor(0.0, device=device)
            if "domain_cnn" in out_src and "domain_cnn" in out_tgt:
                domain_cnn_src = out_src["domain_cnn"]
                domain_cnn_tgt = out_tgt["domain_cnn"]
                domain_cnn_loss = 0.5 * (
                    domain_criterion(domain_cnn_src, d_s)
                    + domain_criterion(domain_cnn_tgt, d_t)
                )

            domain_total = domain_loss + cnn_domain_weight * domain_cnn_loss
            total_loss = task_loss + domain_loss_weight * lam * domain_total

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if grad_clip_norm > 0.0:
                clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            bs = x_s.size(0)
            n_samples += bs
            running_total_loss += total_loss.item() * bs
            running_task_loss += task_loss.item() * bs
            running_domain_loss += domain_loss.item() * bs
            running_domain_cnn_loss += domain_cnn_loss.item() * bs

        train_loss = running_total_loss / max(n_samples, 1)
        train_task_loss = running_task_loss / max(n_samples, 1)
        train_domain_loss = running_domain_loss / max(n_samples, 1)
        train_domain_cnn_loss = running_domain_cnn_loss / max(n_samples, 1)

        test_metrics = evaluate(model, test_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        epoch_row = {
            "epoch": float(epoch + 1),
            "lambda": float(lam),
            "train_loss": float(train_loss),
            "train_task_loss": float(train_task_loss),
            "train_domain_loss": float(train_domain_loss),
            "train_domain_cnn_loss": float(train_domain_cnn_loss),
            "test_accuracy": test_metrics["accuracy"],
            "test_kappa": test_metrics["kappa"],
            "lr": float(current_lr),
        }
        history.append(epoch_row)

        current_score = test_metrics[selection_metric]
        if current_score > best_score:
            best_score = current_score
            best = {
                "accuracy": test_metrics["accuracy"],
                "kappa": test_metrics["kappa"],
                "epoch": float(epoch + 1),
            }
            best_state_dict = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        logger.info(
            "epoch=%d | lambda=%.4f | lr=%.6f | train_loss=%.4f | task=%.4f | domain=%.4f | domain_cnn=%.4f | test_acc=%.4f | test_kappa=%.4f",
            epoch + 1,
            lam,
            current_lr,
            train_loss,
            train_task_loss,
            train_domain_loss,
            train_domain_cnn_loss,
            test_metrics["accuracy"],
            test_metrics["kappa"],
        )

        if early_stopping_patience > 0 and stale_epochs >= early_stopping_patience:
            logger.info(
                "Early stopping at epoch=%d (no %s improvement for %d epochs)",
                epoch + 1,
                selection_metric,
                early_stopping_patience,
            )
            break

    return best, history, best_state_dict


def run(args: argparse.Namespace) -> None:
    set_seed_everywhere(args.seed, deterministic=args.deterministic)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"{args.dataset}_{timestamp}"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir)
    logger.info("Loading MOABB dataset: %s", args.dataset)

    if args.use_da:
        x, y, subject_ids, subjects, metadata = load_moabb_motor_imagery_dataset(
            dataset_name=args.dataset,
            data_path=args.data_path,
            include_metadata=True,
        )
    else:
        x, y, subject_ids, subjects = load_moabb_motor_imagery_dataset(
            dataset_name=args.dataset,
            data_path=args.data_path,
        )
        metadata = {}

    session_ids = metadata.get("session_id")
    if args.use_da and session_ids is None:
        raise ValueError(
            "Within-subject DA requires session metadata from MOABB; session_id not found"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_channels = x.shape[1]
    num_classes = int(np.max(y)) + 1

    logger.info(
        "Loaded data shape=%s, n_subjects=%d, n_classes=%d, device=%s",
        str(tuple(x.shape)),
        len(subjects),
        num_classes,
        device,
    )

    selected_subjects = subjects
    if args.subjects:
        requested = {int(s) for s in args.subjects}
        selected_subjects = [s for s in subjects if s in requested]

    if not selected_subjects:
        raise ValueError("No valid subjects selected")

    all_results: dict[str, dict[str, float]] = {}
    run_models_last: dict[str, dict[str, torch.Tensor]] = {}
    run_models_best: dict[str, dict[str, torch.Tensor]] = {}

    config = {
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "grad_clip_norm": args.grad_clip_norm,
        "early_stopping_patience": args.early_stopping_patience,
        "selection_metric": args.selection_metric,
        "augment_noise_std": args.augment_noise_std,
        "augment_time_mask_ratio": args.augment_time_mask_ratio,
        "use_da": args.use_da,
        "domain_loss_weight": args.domain_loss_weight,
        "da_lambda_gamma": args.da_lambda_gamma,
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
        "da_target_session": args.da_target_session,
        "test_size": args.test_size,
        "loader_euclidean_align": args.loader_euclidean_align,
        "model_pre_align_only": args.model_pre_align_only,
        "model_euclidean_alignment": args.model_euclidean_alignment,
        "model_riemannian_reweight": args.model_riemannian_reweight,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "checkpoint_mode": args.checkpoint_mode,
        "subjects": selected_subjects,
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    metadata = create_experiment_metadata(
        protocol="within_subject",
        dataset=args.dataset,
        config=config,
        seed=args.seed,
        deterministic=args.deterministic,
    )
    save_experiment_metadata(run_dir / "metadata.json", metadata)

    for subject in selected_subjects:
        logger.info("=== Subject %d ===", subject)

        if args.use_da:
            source_loader, target_loader, test_loader, selected_target_session = (
                create_within_subject_domain_adaptation_dataloaders(
                    x=x,
                    y=y,
                    subject_id=subject_ids,
                    session_id=session_ids,
                    target_subject=subject,
                    target_session=args.da_target_session,
                    batch_size=args.batch_size,
                    target_test_size=args.test_size,
                    random_state=args.seed,
                    apply_euclidean_align=args.loader_euclidean_align,
                    num_workers=args.num_workers,
                    seed=args.seed,
                    deterministic=args.deterministic,
                )
            )
            logger.info(
                "DA mode: subject=%d target_session=%d",
                subject,
                selected_target_session,
            )
        else:
            train_loader, test_loader = create_within_subject_dataloaders(
                x=x,
                y=y,
                subject_id=subject_ids,
                target_subject=subject,
                batch_size=args.batch_size,
                test_size=args.test_size,
                random_state=args.seed,
                apply_euclidean_align=args.loader_euclidean_align,
                num_workers=args.num_workers,
                seed=args.seed,
                deterministic=args.deterministic,
            )

        model = EEGModel(
            num_channels=num_channels,
            num_classes=num_classes,
            num_subjects=2 if args.use_da else 1,
            cnn_out_channels=args.cnn_out_channels,
            cnn_dropout=args.cnn_dropout,
            embedding_dim=args.embedding_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            temporal_kernels=(
                tuple(args.temporal_kernels) if args.temporal_kernels else None
            ),
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
        )

        if args.use_da:
            best, history, best_state_dict = train_one_subject_da(
                model=model,
                source_loader=source_loader,
                target_loader=target_loader,
                test_loader=test_loader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                min_lr=args.min_lr,
                weight_decay=args.weight_decay,
                label_smoothing=args.label_smoothing,
                grad_clip_norm=args.grad_clip_norm,
                early_stopping_patience=args.early_stopping_patience,
                selection_metric=args.selection_metric,
                augment_noise_std=args.augment_noise_std,
                augment_time_mask_ratio=args.augment_time_mask_ratio,
                domain_loss_weight=args.domain_loss_weight,
                da_lambda_gamma=args.da_lambda_gamma,
                cnn_domain_weight=args.cnn_domain_weight,
                logger=logger,
            )
        else:
            best, history, best_state_dict = train_one_subject(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                min_lr=args.min_lr,
                weight_decay=args.weight_decay,
                label_smoothing=args.label_smoothing,
                grad_clip_norm=args.grad_clip_norm,
                early_stopping_patience=args.early_stopping_patience,
                selection_metric=args.selection_metric,
                augment_noise_std=args.augment_noise_std,
                augment_time_mask_ratio=args.augment_time_mask_ratio,
                logger=logger,
            )

        subject_key = str(subject)
        all_results[subject_key] = best

        if args.checkpoint_mode == "per_subject":
            if best_state_dict is not None:
                torch.save(best_state_dict, ckpt_dir / f"subject_{subject}_best.pt")
            torch.save(model.state_dict(), ckpt_dir / f"subject_{subject}_last.pt")
        elif args.checkpoint_mode == "single_file":
            if best_state_dict is not None:
                run_models_best[subject_key] = {
                    k: v.detach().cpu() for k, v in best_state_dict.items()
                }
            run_models_last[subject_key] = _state_dict_cpu(model)

        with (run_dir / f"subject_{subject}_history.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(history, f, indent=2)

    accuracies = [v["accuracy"] for v in all_results.values()]
    kappas = [v["kappa"] for v in all_results.values()]

    summary = {
        "dataset": args.dataset,
        "n_subjects": len(all_results),
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "mean_kappa": float(np.mean(kappas)),
        "std_kappa": float(np.std(kappas)),
        "per_subject": all_results,
    }

    with (run_dir / "within_subject_results.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.checkpoint_mode == "single_file":
        torch.save(
            {
                "dataset": args.dataset,
                "protocol": "within_subject",
                "config": config,
                "per_subject_last_state_dict": run_models_last,
                "per_subject_best_state_dict": run_models_best,
            },
            ckpt_dir / "within_subject_run_models.pt",
        )

    logger.info(
        "Done. mean_acc=%.4f, std_acc=%.4f, results=%s",
        summary["mean_accuracy"],
        summary["std_accuracy"],
        str(run_dir / "within_subject_results.json"),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Within-subject training/evaluation on MOABB motor imagery datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="bnci2014_001",
        choices=["bnci2014_001", "physionetmi", "cho2017", "lee2019_mi"],
    )
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/within_subject")
    parser.add_argument(
        "--checkpoint_mode",
        type=str,
        default="per_subject",
        choices=["per_subject", "single_file", "none"],
    )
    parser.add_argument("--subjects", nargs="*", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--early_stopping_patience", type=int, default=15)
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="kappa",
        choices=["accuracy", "kappa"],
    )
    parser.add_argument("--augment_noise_std", type=float, default=0.01)
    parser.add_argument("--augment_time_mask_ratio", type=float, default=0.05)
    parser.add_argument("--use_da", action="store_true", default=False)
    parser.add_argument("--domain_loss_weight", type=float, default=1.0)
    parser.add_argument("--da_lambda_gamma", type=float, default=10.0)
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
    parser.add_argument("--da_target_session", type=int, default=None)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--loader_euclidean_align", action="store_true", default=True)
    parser.add_argument(
        "--no-loader-euclidean-align",
        dest="loader_euclidean_align",
        action="store_false",
    )
    parser.add_argument(
        "--model_euclidean_alignment", action="store_true", default=False
    )
    parser.add_argument(
        "--no-model-euclidean-alignment",
        dest="model_euclidean_alignment",
        action="store_false",
    )
    parser.add_argument("--model_pre_align_only", action="store_true", default=False)
    parser.add_argument(
        "--no-model-pre-align-only",
        dest="model_pre_align_only",
        action="store_false",
    )
    parser.add_argument(
        "--model_riemannian_reweight", action="store_true", default=True
    )
    parser.add_argument(
        "--no-model-riemannian-reweight",
        dest="model_riemannian_reweight",
        action="store_false",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument(
        "--no-deterministic", dest="deterministic", action="store_false"
    )

    parser.add_argument("--cnn_out_channels", type=int, default=32)
    parser.add_argument("--cnn_dropout", type=float, default=0.5)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())


# uv run training/within_subject.py \
#   --dataset bnci2014_001 \
#   --epochs 50 \
#   --lr 1e-3 \
#   --use_da \
#   --domain_loss_weight 1.5 \
#   --min_lr 1e-4 \
#   --weight_decay 1e-4 \
#   --label_smoothing 0.0 \
#   --augment_noise_std 0.0 \
#   --augment_time_mask_ratio 0.0 \
#   --early_stopping_patience 0 \
#   --selection_metric accuracy \
#   --output_dir results/within_subject_tuned_da_v3
