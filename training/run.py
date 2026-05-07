from __future__ import annotations

import argparse
import copy
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

try:
    from data.loader import (
        DataLoaderOptions,
        EEGDataset,
        MoabbLoadOptions,
        create_dataloaders,
        create_within_subject_dataloaders,
        load_moabb_motor_imagery_dataset,
    )
    from models.heads import CalibrationLayer
    from models.model import EEGModel
    from training.utils import apply_euclidean_alignment, fit_euclidean_alignment
    from utils.reproducibility import set_seed_everywhere
    from utils.reproducibility import build_torch_generator
except ModuleNotFoundError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from data.loader import (
        DataLoaderOptions,
        EEGDataset,
        MoabbLoadOptions,
        create_dataloaders,
        create_within_subject_dataloaders,
        load_moabb_motor_imagery_dataset,
    )
    from models.heads import CalibrationLayer
    from models.model import EEGModel
    from training.utils import apply_euclidean_alignment, fit_euclidean_alignment
    from utils.reproducibility import set_seed_everywhere
    from utils.reproducibility import build_torch_generator


DEFAULT_MODEL_CONFIG = {
    "cnn_out_channels": 32,
    "cnn_dropout": 0.25,
    "embedding_dim": 128,
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.1,
    "temporal_kernels": (32, 64, 128),
    "multiscale_preserve_capacity": True,
    "use_attention_pool": True,
    "attention_mix_init": 0.7,
    "learnable_attention_mix": True,
}


def _grl_lambda(epoch: int, total_epochs: int, lambda_max: float) -> float:
    if total_epochs <= 1:
        return float(lambda_max)
    p = epoch / max(1, total_epochs - 1)
    return float(lambda_max) * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)


def _apply_augmentations(
    x: torch.Tensor,
    crop_ratio: float,
    noise_sigma: float,
    channel_dropout: float,
) -> torch.Tensor:
    original_len = x.size(-1)

    if 0.0 < crop_ratio < 1.0:
        crop_len = max(1, int(original_len * crop_ratio))
        if crop_len < original_len:
            start = torch.randint(
                0,
                original_len - crop_len + 1,
                (1,),
                device=x.device,
            ).item()
            x = x[..., start : start + crop_len]
            x = F.interpolate(
                x,
                size=original_len,
                mode="linear",
                align_corners=False,
            )

    if noise_sigma > 0.0:
        scale = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        x = x + torch.randn_like(x) * (scale * noise_sigma)

    if channel_dropout > 0.0:
        keep_prob = 1.0 - channel_dropout
        mask = torch.bernoulli(
            torch.full(
                (x.size(0), x.size(1), 1),
                keep_prob,
                device=x.device,
            )
        )
        x = x * mask

    return x


def _feature_mixup(
    tokens: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0.0:
        return tokens, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(tokens.size(0), device=tokens.device)
    mixed = lam * tokens + (1.0 - lam) * tokens[index]
    return mixed, labels, labels[index], lam


def _mixup_loss(
    criterion: nn.Module,
    preds: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(preds, y_a) + (1.0 - lam) * criterion(preds, y_b)


def _set_cnn_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for param in model.cnn.parameters():
        param.requires_grad = requires_grad


def _split_calibration_indices(
    labels: np.ndarray,
    calibration_trials: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    total = labels.shape[0]
    if total <= 1:
        return np.arange(total), np.array([], dtype=int)

    calibration_trials = int(min(calibration_trials, total - 1))
    indices = np.arange(total)
    unique_y, counts = np.unique(labels, return_counts=True)
    stratify = labels if unique_y.shape[0] > 1 and np.min(counts) >= 2 else None
    calib_idx, eval_idx = train_test_split(
        indices,
        train_size=calibration_trials,
        random_state=seed,
        stratify=stratify,
    )
    return np.asarray(calib_idx), np.asarray(eval_idx)


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    calibration: nn.Module | None = None,
) -> dict[str, float]:
    model.eval()
    if calibration is not None:
        calibration.eval()
    total = 0
    correct = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for x, y, _ in data_loader:
        x = x.to(device)
        y = y.to(device)

        outputs = model(x, lambda_=0.0)
        logits = outputs["task"]
        if calibration is not None:
            logits = model.task_head(calibration(outputs["features"]))
        preds = logits.argmax(dim=1)

        total += y.size(0)
        correct += (preds == y).sum().item()
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    accuracy = correct / max(total, 1)
    kappa = cohen_kappa_score(y_true, y_pred) if total > 0 else 0.0
    if total > 0:
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
        cmatrix = confusion_matrix(y_true, y_pred).tolist()
    else:
        f1_macro = 0.0
        f1_per_class = []
        cmatrix = []
    return {
        "accuracy": float(accuracy),
        "kappa": float(kappa),
        "f1_macro": float(f1_macro),
        "f1_per_class": f1_per_class,
        "confusion_matrix": cmatrix,
    }


def calibrate_model(
    model: EEGModel,
    test_dataset: EEGDataset | Subset,
    device: torch.device,
    calibration_trials: int,
    calibration_steps: int,
    calibration_lr: float,
    calibration_weight_decay: float,
    calibration_label_smoothing: float,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[CalibrationLayer | None, dict[str, float] | None]:
    if calibration_trials <= 0 or calibration_steps <= 0:
        return None, None

    if isinstance(test_dataset, Subset):
        base = test_dataset.dataset
        indices = np.asarray(test_dataset.indices)
        x = base.x[indices]
        y = base.y[indices]
        subject_id = base.subject_id[indices]
        full_dataset = EEGDataset(x, y, subject_id)
    else:
        full_dataset = test_dataset

    labels = full_dataset.y.detach().cpu().numpy()
    calib_idx, eval_idx = _split_calibration_indices(labels, calibration_trials, seed)
    if calib_idx.size == 0 or eval_idx.size == 0:
        return None, None

    calib_dataset = Subset(full_dataset, calib_idx)
    eval_dataset = Subset(full_dataset, eval_idx)

    calib_loader = DataLoader(
        calib_dataset,
        batch_size=min(batch_size, len(calib_dataset)),
        shuffle=True,
        num_workers=num_workers,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    calibration = CalibrationLayer(model.embedding_dim).to(device)
    optimizer = AdamW(
        calibration.parameters(),
        lr=calibration_lr,
        weight_decay=calibration_weight_decay,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=calibration_label_smoothing)

    model.eval()
    calibration.train()
    for _ in range(calibration_steps):
        for x, y, _ in calib_loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                feats = model(x, lambda_=0.0)["features"]
            logits = model.task_head(calibration(feats))
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    calibrated_metrics = evaluate(
        model,
        eval_loader,
        device,
        calibration=calibration,
    )
    return calibration, calibrated_metrics


def train_one_subject(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader | None,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    label_smoothing: float,
    use_class_weights: bool,
    selection_metric: str,
    patience: int,
    min_delta: float,
    lr_schedule: str,
    warmup_epochs: int,
    eta_min: float,
    use_domain_loss: bool,
    domain_lambda_max: float,
    domain_loss_weight: float,
    mixup_alpha: float,
    augment: bool,
    augment_crop_ratio: float,
    augment_noise_sigma: float,
    augment_channel_dropout: float,
    freeze_cnn_epochs: int,
    logger: logging.Logger,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    model.to(device)

    weights_tensor = None
    if use_class_weights:
        labels = _extract_labels_from_loader(train_loader)
        if labels is not None:
            classes = np.unique(labels)
            if classes.shape[0] > 1:
                weights = compute_class_weight(
                    class_weight="balanced",
                    classes=classes,
                    y=labels,
                )
                weights_tensor = torch.tensor(
                    weights,
                    dtype=torch.float32,
                    device=device,
                )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=label_smoothing,
        weight=weights_tensor,
    )
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = _build_scheduler(
        optimizer=optimizer,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        eta_min=eta_min,
        schedule=lr_schedule,
    )
    history: list[dict[str, float]] = []
    best_score = float("-inf")
    best_row: dict[str, float] | None = None
    epochs_without_improvement = 0

    if freeze_cnn_epochs > 0:
        _set_cnn_requires_grad(model, False)

    for epoch in range(epochs):
        if freeze_cnn_epochs > 0 and epoch == freeze_cnn_epochs:
            _set_cnn_requires_grad(model, True)
        model.train()
        running_loss = 0.0
        n_samples = 0
        grl_lambda = (
            _grl_lambda(epoch, epochs, domain_lambda_max) if use_domain_loss else 0.0
        )
        use_mixup = mixup_alpha > 0.0

        for x, y, subject_id in train_loader:
            x = x.to(device)
            y = y.to(device)
            subject_id = subject_id.to(device)

            if augment:
                x = _apply_augmentations(
                    x,
                    crop_ratio=augment_crop_ratio,
                    noise_sigma=augment_noise_sigma,
                    channel_dropout=augment_channel_dropout,
                )

            if use_mixup:
                tokens, cnn_domain_output = model.encode_tokens(x, lambda_=grl_lambda)
                mixed_tokens, y_a, y_b, mix_lam = _feature_mixup(tokens, y, mixup_alpha)
                outputs = model.forward_from_tokens(
                    mixed_tokens,
                    lambda_=grl_lambda,
                    cnn_domain_output=cnn_domain_output,
                )
                task_loss = _mixup_loss(criterion, outputs["task"], y_a, y_b, mix_lam)
            else:
                outputs = model(x, lambda_=grl_lambda)
                task_loss = criterion(outputs["task"], y)

            loss = task_loss
            if use_domain_loss:
                domain_loss = F.cross_entropy(outputs["domain"], subject_id)
                loss = loss + domain_loss_weight * domain_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            running_loss += loss.item() * bs
            n_samples += bs

        train_loss = running_loss / max(n_samples, 1)
        val_metrics = None
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device)
        test_metrics = evaluate(model, test_loader, device)

        score_source = val_metrics if val_metrics is not None else test_metrics
        selection_score = score_source[selection_metric]

        if selection_score > (best_score + min_delta):
            best_score = selection_score
            best_row = {
                "epoch": float(epoch + 1),
                "train_loss": float(train_loss),
                "test_accuracy": test_metrics["accuracy"],
                "test_kappa": test_metrics["kappa"],
                "test_f1_macro": test_metrics["f1_macro"],
                "test_f1_per_class": test_metrics["f1_per_class"],
                "test_confusion_matrix": test_metrics["confusion_matrix"],
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            if val_metrics is not None:
                best_row["val_accuracy"] = val_metrics["accuracy"]
                best_row["val_kappa"] = val_metrics["kappa"]
                best_row["val_f1_macro"] = val_metrics["f1_macro"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": float(epoch + 1),
            "train_loss": float(train_loss),
            "test_accuracy": test_metrics["accuracy"],
            "test_kappa": test_metrics["kappa"],
            "test_f1_macro": test_metrics["f1_macro"],
            "test_f1_per_class": test_metrics["f1_per_class"],
            "test_confusion_matrix": test_metrics["confusion_matrix"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        if val_metrics is not None:
            row["val_accuracy"] = val_metrics["accuracy"]
            row["val_kappa"] = val_metrics["kappa"]
            row["val_f1_macro"] = val_metrics["f1_macro"]
        history.append(row)

        if val_metrics is None:
            logger.info(
                "epoch=%d | train_loss=%.4f | test_acc=%.4f | test_kappa=%.4f",
                epoch + 1,
                train_loss,
                test_metrics["accuracy"],
                test_metrics["kappa"],
            )
        else:
            logger.info(
                "epoch=%d | train_loss=%.4f | val_acc=%.4f | val_kappa=%.4f | test_acc=%.4f | test_kappa=%.4f",
                epoch + 1,
                train_loss,
                val_metrics["accuracy"],
                val_metrics["kappa"],
                test_metrics["accuracy"],
                test_metrics["kappa"],
            )

        if scheduler is not None:
            scheduler.step()

        if val_metrics is not None and patience > 0:
            if epochs_without_improvement >= patience:
                logger.info(
                    "Early stopping triggered after %d epochs without validation improvement",
                    patience,
                )
                break

    if best_row is None:
        best_row = history[-1]
    return best_row, history


def _split_train_val_loaders(
    dataset: EEGDataset,
    val_size: float,
    seed: int,
    batch_size: int,
    subject_balanced_sampling: bool,
    drop_last_train: bool,
    num_workers: int,
    deterministic: bool,
) -> tuple[DataLoader, DataLoader | None]:
    if val_size <= 0.0:
        generator = build_torch_generator(seed) if deterministic else None
        return (
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=drop_last_train,
                num_workers=num_workers,
                generator=generator,
            ),
            None,
        )

    indices = np.arange(len(dataset))
    y_np = dataset.y.detach().cpu().numpy()
    unique_y, counts = np.unique(y_np, return_counts=True)
    stratify = y_np if unique_y.shape[0] > 1 and np.min(counts) >= 2 else None
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        random_state=seed,
        stratify=stratify,
    )

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    generator = build_torch_generator(seed) if deterministic else None
    train_sampler = None
    train_shuffle = True
    if subject_balanced_sampling:
        sid = dataset.subject_id[train_idx]
        unique_sid, counts = sid.unique(return_counts=True)
        if unique_sid.numel() > 1:
            count_map = {
                int(subject.item()): float(count.item())
                for subject, count in zip(unique_sid, counts)
            }
            weights = torch.tensor(
                [1.0 / count_map[int(s.item())] for s in sid],
                dtype=torch.double,
            )
            train_sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(train_idx),
                replacement=True,
                generator=generator,
            )
            train_shuffle = False

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        drop_last=drop_last_train,
        num_workers=num_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def _aggregate_fold_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    accs = [row["test_accuracy"] for row in rows]
    kappas = [row["test_kappa"] for row in rows]
    return {
        "test_accuracy": float(np.mean(accs)) if accs else 0.0,
        "test_kappa": float(np.mean(kappas)) if kappas else 0.0,
        "folds": float(len(rows)),
    }


def _extract_labels_from_loader(
    loader: DataLoader,
) -> np.ndarray | None:
    dataset = loader.dataset
    if isinstance(dataset, Subset):
        base = dataset.dataset
        indices = np.asarray(dataset.indices)
        if hasattr(base, "y"):
            labels = base.y.detach().cpu().numpy()
            return labels[indices]
        return None
    if hasattr(dataset, "y"):
        return dataset.y.detach().cpu().numpy()
    return None


def _build_scheduler(
    optimizer: AdamW,
    epochs: int,
    warmup_epochs: int,
    eta_min: float,
    schedule: str,
) -> LambdaLR | CosineAnnealingLR | None:
    if schedule == "none":
        return None
    if schedule == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=eta_min)

    total_epochs = max(1, epochs)
    if total_epochs <= 1:
        return None

    warmup_steps = max(1, min(int(warmup_epochs), total_epochs - 1))
    start_factor = 0.1
    base_lr = optimizer.param_groups[0]["lr"]

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_steps:
            progress = (epoch + 1) / warmup_steps
            return start_factor + (1.0 - start_factor) * progress

        remaining = max(1, total_epochs - warmup_steps)
        t = epoch - warmup_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * t / remaining))
        lr = eta_min + (base_lr - eta_min) * cosine
        return lr / base_lr

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _parse_subjects(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _model_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "cnn_out_channels": args.cnn_out_channels,
        "cnn_dropout": args.cnn_dropout,
        "embedding_dim": args.embedding_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "temporal_kernels": DEFAULT_MODEL_CONFIG["temporal_kernels"],
        "multiscale_preserve_capacity": DEFAULT_MODEL_CONFIG[
            "multiscale_preserve_capacity"
        ],
        "use_attention_pool": DEFAULT_MODEL_CONFIG["use_attention_pool"],
        "attention_mix_init": DEFAULT_MODEL_CONFIG["attention_mix_init"],
        "learnable_attention_mix": DEFAULT_MODEL_CONFIG["learnable_attention_mix"],
    }


def _build_model(
    num_channels: int,
    num_classes: int,
    num_subjects: int,
    use_rope: bool,
    model_config: dict[str, Any],
) -> EEGModel:
    return EEGModel(
        num_channels=num_channels,
        num_classes=num_classes,
        num_subjects=num_subjects,
        cnn_out_channels=model_config["cnn_out_channels"],
        cnn_dropout=model_config["cnn_dropout"],
        embedding_dim=model_config["embedding_dim"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
        temporal_kernels=model_config["temporal_kernels"],
        multiscale_preserve_capacity=model_config["multiscale_preserve_capacity"],
        use_attention_pool=model_config["use_attention_pool"],
        attention_mix_init=model_config["attention_mix_init"],
        learnable_attention_mix=model_config["learnable_attention_mix"],
        use_rope=use_rope,
    )


def run(args: argparse.Namespace) -> None:
    set_seed_everywhere(args.seed, deterministic=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"{args.protocol}_{args.dataset}_{timestamp}"
    logger = setup_logger(run_dir)

    subjects = _parse_subjects(args.subjects)
    load_options = MoabbLoadOptions(
        subjects=subjects,
        class_policy=args.class_policy,
        use_common_channels=args.use_common_channels,
        show_progress=True,
    )

    logger.info("Loading MOABB dataset: %s", args.dataset)
    x, y, subject_ids, available_subjects = load_moabb_motor_imagery_dataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        options=load_options,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_channels = x.shape[1]
    num_classes = int(np.max(y)) + 1
    num_subjects = int(np.max(subject_ids)) + 1

    selected_subjects = available_subjects
    if subjects:
        wanted = {int(s) for s in subjects}
        selected_subjects = [s for s in available_subjects if s in wanted]

    if not selected_subjects:
        raise ValueError("No valid subjects selected")

    loader_options = DataLoaderOptions(
        batch_size=args.batch_size,
        test_size=args.test_size,
        random_state=args.seed,
        apply_euclidean_align=True,
        subject_balanced_sampling=args.subject_balanced_sampling,
        drop_last_train=args.drop_last_train,
        num_workers=args.num_workers,
        seed=args.seed,
        deterministic=True,
    )

    logger.info(
        "Loaded data shape=%s, n_subjects=%d, n_classes=%d, device=%s",
        str(tuple(x.shape)),
        len(selected_subjects),
        num_classes,
        device,
    )

    per_subject: dict[str, Any] = {}
    per_subject_raw: dict[str, Any] = {}
    per_subject_calibrated: dict[str, Any] = {}
    histories: dict[str, list[dict[str, float]]] = {}
    per_subject_folds: dict[str, list[dict[str, float]]] = {}

    base_lr = args.lr
    base_weight_decay = args.weight_decay
    base_label_smoothing = args.label_smoothing
    use_domain_loss = args.protocol == "loso" and args.domain_loss_weight > 0.0
    model_config = _model_config_from_args(args)
    if args.protocol == "within":
        base_lr = args.within_lr
        base_weight_decay = args.within_weight_decay
        base_label_smoothing = args.within_label_smoothing

    pretrained_state: dict[str, torch.Tensor] | None = None
    if args.protocol == "loso" and args.two_phase_loso and args.pretrain_epochs > 0:
        logger.info("Starting cross-subject pretraining")
        pretrain_dataset = EEGDataset(x, y, subject_ids)
        whitening = fit_euclidean_alignment(
            pretrain_dataset.x,
            eps=loader_options.align_eps,
        )
        pretrain_dataset.x = apply_euclidean_alignment(pretrain_dataset.x, whitening)
        pretrain_loader, pretrain_val = _split_train_val_loaders(
            dataset=pretrain_dataset,
            val_size=args.pretrain_val_size,
            seed=args.seed,
            batch_size=args.batch_size,
            subject_balanced_sampling=args.subject_balanced_sampling,
            drop_last_train=args.drop_last_train,
            num_workers=args.num_workers,
            deterministic=True,
        )
        pretrain_test = pretrain_val if pretrain_val is not None else pretrain_loader
        pretrain_model = _build_model(
            num_channels,
            num_classes,
            num_subjects,
            use_rope=args.use_rope,
            model_config=model_config,
        )
        pretrain_best, pretrain_history = train_one_subject(
            model=pretrain_model,
            train_loader=pretrain_loader,
            val_loader=pretrain_val,
            test_loader=pretrain_test,
            device=device,
            epochs=args.pretrain_epochs,
            lr=args.pretrain_lr,
            weight_decay=args.pretrain_weight_decay,
            label_smoothing=base_label_smoothing,
            use_class_weights=args.use_class_weights,
            selection_metric=args.selection_metric,
            patience=args.patience,
            min_delta=args.min_delta,
            lr_schedule=args.lr_schedule,
            warmup_epochs=args.warmup_epochs,
            eta_min=args.eta_min,
            use_domain_loss=False,
            domain_lambda_max=0.0,
            domain_loss_weight=0.0,
            mixup_alpha=args.mixup_alpha,
            augment=args.augment,
            augment_crop_ratio=args.augment_crop_ratio,
            augment_noise_sigma=args.augment_noise_sigma,
            augment_channel_dropout=args.augment_channel_dropout,
            freeze_cnn_epochs=0,
            logger=logger,
        )
        pretrained_state = copy.deepcopy(pretrain_model.state_dict())
        (run_dir / "pretrain_summary.json").write_text(
            json.dumps(pretrain_best, indent=2),
            encoding="utf-8",
        )
        (run_dir / "pretrain_history.json").write_text(
            json.dumps(pretrain_history, indent=2),
            encoding="utf-8",
        )

    for idx, subject in enumerate(selected_subjects, start=1):
        logger.info(
            "=== Subject %d/%d (id=%d) ===",
            idx,
            len(selected_subjects),
            subject,
        )
        if args.protocol == "within" and args.within_cv_folds > 1:
            mask = subject_ids == int(subject)
            x_sub = x[mask]
            y_sub = y[mask]
            if np.unique(y_sub).shape[0] < 2:
                raise ValueError(
                    f"Subject {subject} has fewer than 2 classes after filtering"
                )

            skf = StratifiedKFold(
                n_splits=args.within_cv_folds,
                shuffle=True,
                random_state=args.seed,
            )
            fold_rows: list[dict[str, float]] = []
            fold_histories: list[list[dict[str, float]]] = []

            for fold_idx, (train_idx, test_idx) in enumerate(
                skf.split(x_sub, y_sub), start=1
            ):
                train_idx = np.asarray(train_idx)
                test_idx = np.asarray(test_idx)

                train_labels = y_sub[train_idx]
                unique_y, counts = np.unique(train_labels, return_counts=True)
                stratify = (
                    train_labels
                    if unique_y.shape[0] > 1 and np.min(counts) >= 2
                    else None
                )
                train_inner_idx, val_idx = train_test_split(
                    train_idx,
                    test_size=args.val_size,
                    random_state=args.seed + fold_idx,
                    stratify=stratify,
                )

                s_train = np.full(
                    train_inner_idx.shape[0], int(subject), dtype=np.int64
                )
                s_val = np.full(val_idx.shape[0], int(subject), dtype=np.int64)
                s_test = np.full(test_idx.shape[0], int(subject), dtype=np.int64)

                train_dataset = EEGDataset(
                    x_sub[train_inner_idx],
                    y_sub[train_inner_idx],
                    s_train,
                )
                val_dataset = EEGDataset(
                    x_sub[val_idx],
                    y_sub[val_idx],
                    s_val,
                )
                test_dataset = EEGDataset(
                    x_sub[test_idx],
                    y_sub[test_idx],
                    s_test,
                )

                whitening = fit_euclidean_alignment(
                    train_dataset.x,
                    eps=loader_options.align_eps,
                )
                train_dataset.x = apply_euclidean_alignment(train_dataset.x, whitening)
                val_dataset.x = apply_euclidean_alignment(val_dataset.x, whitening)
                test_dataset.x = apply_euclidean_alignment(test_dataset.x, whitening)

                train_loader, val_loader = _split_train_val_loaders(
                    dataset=train_dataset,
                    val_size=0.0,
                    seed=args.seed + fold_idx,
                    batch_size=args.batch_size,
                    subject_balanced_sampling=args.subject_balanced_sampling,
                    drop_last_train=args.drop_last_train,
                    num_workers=args.num_workers,
                    deterministic=True,
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                )

                model = _build_model(
                    num_channels,
                    num_classes,
                    num_subjects,
                    use_rope=args.use_rope,
                    model_config=model_config,
                )
                if pretrained_state is not None:
                    model.load_state_dict(pretrained_state)
                best, history = train_one_subject(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    device=device,
                    epochs=args.finetune_epochs if args.two_phase_loso else args.epochs,
                    lr=args.finetune_lr if args.two_phase_loso else base_lr,
                    weight_decay=(
                        args.finetune_weight_decay
                        if args.two_phase_loso
                        else base_weight_decay
                    ),
                    label_smoothing=base_label_smoothing,
                    use_class_weights=args.use_class_weights,
                    selection_metric=args.selection_metric,
                    patience=args.patience,
                    min_delta=args.min_delta,
                    lr_schedule=args.lr_schedule,
                    warmup_epochs=args.warmup_epochs,
                    eta_min=args.eta_min,
                    use_domain_loss=use_domain_loss,
                    domain_lambda_max=args.domain_lambda_max,
                    domain_loss_weight=args.domain_loss_weight,
                    mixup_alpha=args.mixup_alpha,
                    augment=args.augment,
                    augment_crop_ratio=args.augment_crop_ratio,
                    augment_noise_sigma=args.augment_noise_sigma,
                    augment_channel_dropout=args.augment_channel_dropout,
                    freeze_cnn_epochs=(
                        args.finetune_freeze_cnn_epochs if args.two_phase_loso else 0
                    ),
                    logger=logger,
                )
                fold_rows.append(best)
                fold_histories.append(history)

            per_subject[str(subject)] = _aggregate_fold_metrics(fold_rows)
            per_subject_folds[str(subject)] = fold_rows
            histories[str(subject)] = fold_histories
            continue

        if args.protocol == "loso":
            train_loader, test_loader = create_dataloaders(
                x=x,
                y=y,
                subject_id=subject_ids,
                loso_subject=int(subject),
                options=loader_options,
            )
        else:
            train_loader, test_loader = create_within_subject_dataloaders(
                x=x,
                y=y,
                subject_id=subject_ids,
                target_subject=int(subject),
                options=loader_options,
            )

        train_loader, val_loader = _split_train_val_loaders(
            dataset=train_loader.dataset,
            val_size=args.val_size,
            seed=args.seed,
            batch_size=args.batch_size,
            subject_balanced_sampling=args.subject_balanced_sampling,
            drop_last_train=args.drop_last_train,
            num_workers=args.num_workers,
            deterministic=True,
        )

        model = _build_model(
            num_channels,
            num_classes,
            num_subjects,
            use_rope=args.use_rope,
            model_config=model_config,
        )
        if pretrained_state is not None:
            model.load_state_dict(pretrained_state)
        best, history = train_one_subject(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.finetune_epochs if args.two_phase_loso else args.epochs,
            lr=args.finetune_lr if args.two_phase_loso else base_lr,
            weight_decay=(
                args.finetune_weight_decay if args.two_phase_loso else base_weight_decay
            ),
            label_smoothing=base_label_smoothing,
            use_class_weights=args.use_class_weights,
            selection_metric=args.selection_metric,
            patience=args.patience,
            min_delta=args.min_delta,
            lr_schedule=args.lr_schedule,
            warmup_epochs=args.warmup_epochs,
            eta_min=args.eta_min,
            use_domain_loss=use_domain_loss,
            domain_lambda_max=args.domain_lambda_max,
            domain_loss_weight=args.domain_loss_weight,
            mixup_alpha=args.mixup_alpha,
            augment=args.augment,
            augment_crop_ratio=args.augment_crop_ratio,
            augment_noise_sigma=args.augment_noise_sigma,
            augment_channel_dropout=args.augment_channel_dropout,
            freeze_cnn_epochs=(
                args.finetune_freeze_cnn_epochs if args.two_phase_loso else 0
            ),
            logger=logger,
        )

        per_subject_raw[str(subject)] = best
        histories[str(subject)] = history

        if args.protocol == "loso" and args.calibration_trials > 0:
            _, calibrated_metrics = calibrate_model(
                model=model,
                test_dataset=test_loader.dataset,
                device=device,
                calibration_trials=args.calibration_trials,
                calibration_steps=args.calibration_steps,
                calibration_lr=args.calibration_lr,
                calibration_weight_decay=args.calibration_weight_decay,
                calibration_label_smoothing=args.calibration_label_smoothing,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=args.seed,
            )
            if calibrated_metrics is not None:
                per_subject_calibrated[str(subject)] = calibrated_metrics
                per_subject[str(subject)] = calibrated_metrics
            else:
                per_subject[str(subject)] = best
        else:
            per_subject[str(subject)] = best

    def _metric(row: dict[str, float], key: str) -> float:
        if key in row:
            return float(row[key])
        if key == "test_accuracy":
            return float(row.get("accuracy", 0.0))
        if key == "test_kappa":
            return float(row.get("kappa", 0.0))
        return 0.0

    accs = [_metric(row, "test_accuracy") for row in per_subject.values()]
    kappas = [_metric(row, "test_kappa") for row in per_subject.values()]
    summary = {
        "protocol": args.protocol,
        "dataset": args.dataset,
        "subjects": selected_subjects,
        "within_cv_folds": args.within_cv_folds,
        "val_size": args.val_size,
        "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "std_accuracy": float(np.std(accs)) if accs else 0.0,
        "mean_kappa": float(np.mean(kappas)) if kappas else 0.0,
        "std_kappa": float(np.std(kappas)) if kappas else 0.0,
        "per_subject": per_subject,
    }
    if per_subject_folds:
        summary["per_subject_folds"] = per_subject_folds
    if per_subject_raw:
        summary["per_subject_raw"] = per_subject_raw
    if per_subject_calibrated:
        summary["per_subject_calibrated"] = per_subject_calibrated

    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (run_dir / "history.json").write_text(
        json.dumps(histories, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "Done. mean_acc=%.4f, std_acc=%.4f, results=%s",
        summary["mean_accuracy"],
        summary["std_accuracy"],
        str(run_dir / "summary.json"),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LOSO or within-subject evaluation on MOABB motor imagery datasets.",
    )
    parser.add_argument("--protocol", choices=["loso", "within"], default="loso")
    parser.add_argument("--dataset", type=str, default="bnci2014_001")
    parser.add_argument("--subjects", type=str, default="")
    parser.add_argument(
        "--class_policy",
        type=str,
        default="all",
        choices=["all", "left_right"],
    )
    parser.add_argument("--use_common_channels", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--cnn_out_channels",
        type=int,
        default=DEFAULT_MODEL_CONFIG["cnn_out_channels"],
    )
    parser.add_argument(
        "--cnn_dropout",
        type=float,
        default=DEFAULT_MODEL_CONFIG["cnn_dropout"],
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=DEFAULT_MODEL_CONFIG["embedding_dim"],
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=DEFAULT_MODEL_CONFIG["num_heads"],
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=DEFAULT_MODEL_CONFIG["num_layers"],
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_MODEL_CONFIG["dropout"],
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--within_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--within_weight_decay", type=float, default=2e-4)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--within_cv_folds", type=int, default=1)
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "kappa"],
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="warmup_cosine",
        choices=["none", "cosine", "warmup_cosine"],
    )
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--within_label_smoothing", type=float, default=0.1)
    parser.add_argument(
        "--no_rope", dest="use_rope", action="store_false", default=True
    )
    parser.add_argument("--domain_loss_weight", type=float, default=0.1)
    parser.add_argument("--domain_lambda_max", type=float, default=0.3)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument(
        "--no_mixup",
        dest="mixup_alpha",
        action="store_const",
        const=0.0,
    )
    parser.add_argument(
        "--no_augment", dest="augment", action="store_false", default=True
    )
    parser.add_argument("--augment_crop_ratio", type=float, default=0.85)
    parser.add_argument("--augment_noise_sigma", type=float, default=0.01)
    parser.add_argument("--augment_channel_dropout", type=float, default=0.05)
    parser.add_argument("--two_phase_loso", action="store_true", default=False)
    parser.add_argument("--pretrain_epochs", type=int, default=50)
    parser.add_argument("--pretrain_lr", type=float, default=1e-3)
    parser.add_argument("--pretrain_weight_decay", type=float, default=1e-4)
    parser.add_argument("--pretrain_val_size", type=float, default=0.1)
    parser.add_argument("--finetune_epochs", type=int, default=20)
    parser.add_argument("--finetune_lr", type=float, default=1e-4)
    parser.add_argument("--finetune_weight_decay", type=float, default=1e-4)
    parser.add_argument("--finetune_freeze_cnn_epochs", type=int, default=5)
    parser.add_argument("--calibration_trials", type=int, default=0)
    parser.add_argument("--calibration_steps", type=int, default=50)
    parser.add_argument("--calibration_lr", type=float, default=1e-3)
    parser.add_argument("--calibration_weight_decay", type=float, default=0.0)
    parser.add_argument("--calibration_label_smoothing", type=float, default=0.0)
    parser.add_argument(
        "--no_class_weights",
        dest="use_class_weights",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--subject_balanced_sampling", action="store_true", default=False
    )
    parser.add_argument("--drop_last_train", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_euclidean_align", action="store_true", default=False)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
