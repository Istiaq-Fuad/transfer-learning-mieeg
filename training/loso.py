from __future__ import annotations

import argparse
import json
import logging
from itertools import cycle
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW

try:
    from data.loader import (
        EEGDataset,
        DataLoaderOptions,
        MoabbLoadOptions,
        create_dataloaders,
        create_loso_domain_adaptation_dataloaders,
        load_moabb_motor_imagery_dataset,
        subsample_train_trials_per_subject_class,
    )
    from models.model import EEGModel
    from training.utils import lambda_scheduler
    from utils.reproducibility import (
        build_torch_generator,
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
        EEGDataset,
        DataLoaderOptions,
        MoabbLoadOptions,
        create_dataloaders,
        create_loso_domain_adaptation_dataloaders,
        load_moabb_motor_imagery_dataset,
        subsample_train_trials_per_subject_class,
    )
    from models.model import EEGModel
    from training.utils import lambda_scheduler
    from utils.reproducibility import (
        build_torch_generator,
        create_experiment_metadata,
        save_experiment_metadata,
        set_seed_everywhere,
    )


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("loso")
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
) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for x, y, _ in data_loader:
        x = x.to(device)
        y = y.to(device)

        outputs = model(x, lambda_=0.0)
        logits = outputs["task"]
        preds = logits.argmax(dim=1)

        total += y.size(0)
        correct += (preds == y).sum().item()
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    accuracy = correct / max(total, 1)
    kappa = cohen_kappa_score(y_true, y_pred) if total > 0 else 0.0
    return {"accuracy": float(accuracy), "kappa": float(kappa)}


def _state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _infer_checkpoint_pretrain_mode(checkpoint_path: str) -> str | None:
    """Best-effort lookup of the pretraining mode next to a run checkpoint."""
    path = Path(checkpoint_path)
    candidates = [
        path.parent.parent / "config.json",
        path.parent / "config.json",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            with candidate.open("r", encoding="utf-8") as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        pretrain_mode = config.get("pretrain_mode")
        if isinstance(pretrain_mode, str):
            return pretrain_mode
    return None


def _should_load_task_head(task_head_policy: str, pretrain_mode: str | None) -> bool:
    if task_head_policy == "load":
        return True
    if task_head_policy == "skip":
        return False
    return pretrain_mode != "ssl"


def _load_init_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    logger: logging.Logger,
    task_head_policy: str = "auto",
    load_domain_heads: bool = False,
) -> None:
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    pretrain_mode = _infer_checkpoint_pretrain_mode(checkpoint_path)
    load_task_head = _should_load_task_head(task_head_policy, pretrain_mode)
    model_state = model.state_dict()
    filtered_state: dict[str, torch.Tensor] = {}
    skipped_shape: list[str] = []
    skipped_policy: list[str] = []
    for key, value in state.items():
        if key not in model_state:
            continue
        if key.startswith("task_head.") and not load_task_head:
            skipped_policy.append(key)
            continue
        if not load_domain_heads and (
            key.startswith("domain_head.") or key.startswith("cnn_domain_head.")
        ):
            skipped_policy.append(key)
            continue
        if model_state[key].shape != value.shape:
            skipped_shape.append(key)
            continue
        filtered_state[key] = value

    incompatible = model.load_state_dict(filtered_state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    logger.info(
        "Warm-started model from %s | pretrain_mode=%s | task_head_policy=%s | load_domain_heads=%s | loaded=%d | policy_skipped=%d | shape_skipped=%d | missing_keys=%d | unexpected_keys=%d",
        checkpoint_path,
        pretrain_mode or "unknown",
        task_head_policy,
        load_domain_heads,
        len(filtered_state),
        len(skipped_policy),
        len(skipped_shape),
        len(missing),
        len(unexpected),
    )
    if skipped_policy:
        logger.info("Policy-skipped key sample: %s", skipped_policy[:5])
    if skipped_shape:
        logger.info("Shape-mismatch key sample: %s", skipped_shape[:5])
    if missing:
        logger.info("Missing key sample: %s", missing[:5])
    if unexpected:
        logger.info("Unexpected key sample: %s", unexpected[:5])


def _configure_head_only_finetune(model: nn.Module, logger: logging.Logger) -> None:
    """Freeze backbone and domain heads; train only the task head."""
    for param in model.parameters():
        param.requires_grad = False

    for param in model.task_head.parameters():
        param.requires_grad = True

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    logger.info(
        "Head-only fine-tune enabled: trainable_params=%d / total_params=%d",
        trainable_params,
        total_params,
    )


def _backbone_parameters(model: nn.Module) -> list[nn.Parameter]:
    backbone_modules = [model.cnn, model.tokenizer, model.vit]
    attn_pool = getattr(model, "attn_pool", None)
    if attn_pool is not None:
        backbone_modules.append(attn_pool)

    params: list[nn.Parameter] = []
    for module in backbone_modules:
        params.extend(list(module.parameters()))
    return params


def _rebuild_loader_with_fraction(
    loader: torch.utils.data.DataLoader,
    fraction: float,
    seed: int,
    batch_size: int,
    subject_balanced_sampling: bool,
    drop_last_train: bool,
    num_workers: int,
    deterministic: bool,
) -> torch.utils.data.DataLoader:
    if fraction >= 1.0:
        return loader

    dataset = loader.dataset
    if not isinstance(dataset, EEGDataset):
        raise TypeError("Reduced-data fine-tuning expects an EEGDataset loader")

    x_np = dataset.x.detach().cpu().numpy()
    y_np = dataset.y.detach().cpu().numpy()
    s_np = dataset.subject_id.detach().cpu().numpy()
    x_sub, y_sub, s_sub = subsample_train_trials_per_subject_class(
        x=x_np,
        y=y_np,
        subject_id=s_np,
        fraction=fraction,
        random_state=seed,
    )
    subset_dataset = EEGDataset(x_sub, y_sub, s_sub)

    generator = build_torch_generator(seed) if deterministic else None
    train_sampler = None
    train_shuffle = True
    if subject_balanced_sampling:
        sid = subset_dataset.subject_id
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
            train_sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights,
                num_samples=len(subset_dataset),
                replacement=True,
                generator=generator,
            )
            train_shuffle = False

    return torch.utils.data.DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        drop_last=drop_last_train,
        num_workers=num_workers,
        generator=generator,
    )


def _dataset_arrays(
    dataset: EEGDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        dataset.x.detach().cpu().numpy(),
        dataset.y.detach().cpu().numpy(),
        dataset.subject_id.detach().cpu().numpy(),
    )


def _subset_eeg_dataset(dataset: EEGDataset, indices: np.ndarray) -> EEGDataset:
    x_np, y_np, s_np = _dataset_arrays(dataset)
    return EEGDataset(x_np[indices], y_np[indices], s_np[indices])


def _make_eval_loader(
    dataset: EEGDataset,
    batch_size: int,
    num_workers: int,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def _make_train_loader(
    dataset: EEGDataset,
    seed: int,
    batch_size: int,
    subject_balanced_sampling: bool,
    drop_last_train: bool,
    num_workers: int,
    deterministic: bool,
) -> torch.utils.data.DataLoader:
    generator = build_torch_generator(seed) if deterministic else None
    train_sampler = None
    train_shuffle = True
    if subject_balanced_sampling:
        sid = dataset.subject_id
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
            train_sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights,
                num_samples=len(dataset),
                replacement=True,
                generator=generator,
            )
            train_shuffle = False

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        drop_last=drop_last_train,
        num_workers=num_workers,
        generator=generator,
    )


def _split_source_train_validation_loaders(
    loader: torch.utils.data.DataLoader,
    validation_size: float,
    train_fraction: float,
    min_trials_per_class: int,
    seed: int,
    batch_size: int,
    subject_balanced_sampling: bool,
    drop_last_train: bool,
    num_workers: int,
    deterministic: bool,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if not (0.0 < validation_size < 1.0):
        raise ValueError("--source_val_size must be in (0, 1)")
    dataset = loader.dataset
    if not isinstance(dataset, EEGDataset):
        raise TypeError("Source validation split expects an EEGDataset loader")

    _, y_np, _ = _dataset_arrays(dataset)
    indices = np.arange(len(dataset))
    unique_y, counts = np.unique(y_np, return_counts=True)
    stratify = y_np if unique_y.shape[0] > 1 and np.min(counts) >= 2 else None
    train_idx, val_idx = train_test_split(
        indices,
        test_size=validation_size,
        random_state=seed,
        stratify=stratify,
    )
    train_dataset = _subset_eeg_dataset(dataset, train_idx)
    val_dataset = _subset_eeg_dataset(dataset, val_idx)

    if train_fraction < 1.0:
        x_train, y_train, s_train = _dataset_arrays(train_dataset)
        x_sub, y_sub, s_sub = subsample_train_trials_per_subject_class(
            x=x_train,
            y=y_train,
            subject_id=s_train,
            fraction=train_fraction,
            random_state=seed,
            min_trials_per_class=min_trials_per_class,
        )
        train_dataset = EEGDataset(x_sub, y_sub, s_sub)

    train_loader = _make_train_loader(
        dataset=train_dataset,
        seed=seed,
        batch_size=batch_size,
        subject_balanced_sampling=subject_balanced_sampling,
        drop_last_train=drop_last_train,
        num_workers=num_workers,
        deterministic=deterministic,
    )
    val_loader = _make_eval_loader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def _build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    strategy: str,
    backbone_lr_multiplier: float,
    head_lr_multiplier: float,
) -> AdamW:
    if strategy == "head_only":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.task_head.parameters():
            param.requires_grad = True
        return AdamW(
            [
                {
                    "params": list(model.task_head.parameters()),
                    "lr": lr * head_lr_multiplier,
                }
            ],
            weight_decay=weight_decay,
        )

    for param in model.parameters():
        param.requires_grad = True

    param_groups = [
        {
            "params": _backbone_parameters(model),
            "lr": lr * backbone_lr_multiplier,
            "weight_decay": weight_decay,
            "name": "backbone",
        },
        {
            "params": list(model.task_head.parameters()),
            "lr": lr * head_lr_multiplier,
            "weight_decay": weight_decay,
            "name": "head",
        },
    ]
    return AdamW(param_groups, weight_decay=weight_decay)


def _update_optimizer_lrs(
    optimizer: AdamW,
    lr: float,
    strategy: str,
    epoch: int,
    warmup_epochs: int,
    backbone_lr_multiplier: float,
    head_lr_multiplier: float,
) -> tuple[float, float]:
    backbone_lr = 0.0
    if strategy != "head_only":
        if strategy == "progressive" and epoch < warmup_epochs:
            backbone_lr = 0.0
        else:
            backbone_lr = lr * backbone_lr_multiplier

    head_lr = lr * head_lr_multiplier

    for group in optimizer.param_groups:
        if group.get("name") == "backbone":
            group["lr"] = backbone_lr
        elif group.get("name") == "head":
            group["lr"] = head_lr

    return backbone_lr, head_lr


def train_one_fold(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    finetune_strategy: str,
    warmup_epochs: int,
    backbone_lr_multiplier: float,
    head_lr_multiplier: float,
    max_grad_norm: float,
    patience: int,
    min_delta: float,
    selection_metric: str,
    label_smoothing: float,
    logger: logging.Logger,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = _build_optimizer(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        strategy=finetune_strategy,
        backbone_lr_multiplier=backbone_lr_multiplier,
        head_lr_multiplier=head_lr_multiplier,
    )

    model.to(device)
    best = {
        "accuracy": 0.0,
        "kappa": 0.0,
        "epoch": 0.0,
        "selection_accuracy": 0.0,
        "selection_kappa": 0.0,
        "selection_score": float("-inf"),
        "legacy_best_test_accuracy": 0.0,
        "legacy_best_test_kappa": 0.0,
        "legacy_best_test_epoch": 0.0,
    }
    best_state = _state_dict_cpu(model)
    best_score = float("-inf")
    legacy_best_test = {"accuracy": 0.0, "kappa": 0.0, "epoch": 0.0}
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        backbone_lr, head_lr = _update_optimizer_lrs(
            optimizer=optimizer,
            lr=lr,
            strategy=finetune_strategy,
            epoch=epoch,
            warmup_epochs=warmup_epochs,
            backbone_lr_multiplier=backbone_lr_multiplier,
            head_lr_multiplier=head_lr_multiplier,
        )
        running_loss = 0.0
        n_samples = 0

        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x, lambda_=0.0)
            logits = outputs["task"]
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_grad_norm,
                )
            optimizer.step()

            bs = x.size(0)
            n_samples += bs
            running_loss += loss.item() * bs

        train_loss = running_loss / max(n_samples, 1)
        val_metrics = evaluate(model, val_loader, device)
        test_metrics = evaluate(model, test_loader, device)
        current_score = val_metrics[selection_metric]

        row = {
            "epoch": float(epoch + 1),
            "train_loss": float(train_loss),
            "val_accuracy": val_metrics["accuracy"],
            "val_kappa": val_metrics["kappa"],
            "test_accuracy": test_metrics["accuracy"],
            "test_kappa": test_metrics["kappa"],
            "backbone_lr": float(backbone_lr),
            "head_lr": float(head_lr),
        }
        history.append(row)

        if test_metrics["accuracy"] > legacy_best_test["accuracy"]:
            legacy_best_test = {
                "accuracy": test_metrics["accuracy"],
                "kappa": test_metrics["kappa"],
                "epoch": float(epoch + 1),
            }

        if current_score > (best_score + min_delta):
            best_score = current_score
            best = {
                "accuracy": test_metrics["accuracy"],
                "kappa": test_metrics["kappa"],
                "epoch": float(epoch + 1),
                "selection_accuracy": val_metrics["accuracy"],
                "selection_kappa": val_metrics["kappa"],
                "selection_score": float(current_score),
                "legacy_best_test_accuracy": legacy_best_test["accuracy"],
                "legacy_best_test_kappa": legacy_best_test["kappa"],
                "legacy_best_test_epoch": legacy_best_test["epoch"],
            }
            best_state = _state_dict_cpu(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        logger.info(
            "epoch=%d | train_loss=%.4f | val_acc=%.4f | val_kappa=%.4f | test_acc=%.4f | test_kappa=%.4f | backbone_lr=%.2e | head_lr=%.2e",
            epoch + 1,
            train_loss,
            val_metrics["accuracy"],
            val_metrics["kappa"],
            test_metrics["accuracy"],
            test_metrics["kappa"],
            backbone_lr,
            head_lr,
        )

        if patience > 0 and epochs_without_improvement >= patience:
            logger.info(
                "Early stopping triggered after %d epochs without validation improvement",
                patience,
            )
            break

    model.load_state_dict(best_state)
    best["legacy_best_test_accuracy"] = legacy_best_test["accuracy"]
    best["legacy_best_test_kappa"] = legacy_best_test["kappa"]
    best["legacy_best_test_epoch"] = legacy_best_test["epoch"]

    return best, history


def train_one_fold_da(
    model: nn.Module,
    source_loader: torch.utils.data.DataLoader,
    target_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    label_smoothing: float,
    domain_loss_weight: float,
    da_lambda_gamma: float,
    cnn_domain_weight: float,
    selection_metric: str,
    min_delta: float,
    logger: logging.Logger,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    task_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    domain_criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found for optimizer")
    optimizer = AdamW(trainable_params, lr=lr)

    model.to(device)
    best = {
        "accuracy": 0.0,
        "kappa": 0.0,
        "epoch": 0.0,
        "selection_accuracy": 0.0,
        "selection_kappa": 0.0,
        "selection_score": float("-inf"),
        "legacy_best_test_accuracy": 0.0,
        "legacy_best_test_kappa": 0.0,
        "legacy_best_test_epoch": 0.0,
    }
    best_state = _state_dict_cpu(model)
    best_score = float("-inf")
    legacy_best_test = {"accuracy": 0.0, "kappa": 0.0, "epoch": 0.0}
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        lam = lambda_scheduler(epoch, epochs, gamma=da_lambda_gamma)
        running_loss = 0.0
        running_task = 0.0
        running_domain = 0.0
        running_domain_cnn = 0.0
        n_samples = 0

        for (x_s, y_s, d_s), (x_t, _, d_t) in zip(source_loader, cycle(target_loader)):
            x_s = x_s.to(device)
            y_s = y_s.to(device)
            d_s = d_s.to(device)
            x_t = x_t.to(device)
            d_t = d_t.to(device)

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
            loss = task_loss + domain_loss_weight * lam * domain_total

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = x_s.size(0)
            n_samples += bs
            running_loss += loss.item() * bs
            running_task += task_loss.item() * bs
            running_domain += domain_loss.item() * bs
            running_domain_cnn += domain_cnn_loss.item() * bs

        train_loss = running_loss / max(n_samples, 1)
        train_task_loss = running_task / max(n_samples, 1)
        train_domain_loss = running_domain / max(n_samples, 1)
        train_domain_cnn_loss = running_domain_cnn / max(n_samples, 1)
        val_metrics = evaluate(model, val_loader, device)
        test_metrics = evaluate(model, test_loader, device)
        current_score = val_metrics[selection_metric]

        row = {
            "epoch": float(epoch + 1),
            "lambda": float(lam),
            "train_loss": float(train_loss),
            "train_task_loss": float(train_task_loss),
            "train_domain_loss": float(train_domain_loss),
            "train_domain_cnn_loss": float(train_domain_cnn_loss),
            "val_accuracy": val_metrics["accuracy"],
            "val_kappa": val_metrics["kappa"],
            "test_accuracy": test_metrics["accuracy"],
            "test_kappa": test_metrics["kappa"],
        }
        history.append(row)

        if test_metrics["accuracy"] > legacy_best_test["accuracy"]:
            legacy_best_test = {
                "accuracy": test_metrics["accuracy"],
                "kappa": test_metrics["kappa"],
                "epoch": float(epoch + 1),
            }

        if current_score > (best_score + min_delta):
            best_score = current_score
            best = {
                "accuracy": test_metrics["accuracy"],
                "kappa": test_metrics["kappa"],
                "epoch": float(epoch + 1),
                "selection_accuracy": val_metrics["accuracy"],
                "selection_kappa": val_metrics["kappa"],
                "selection_score": float(current_score),
                "legacy_best_test_accuracy": legacy_best_test["accuracy"],
                "legacy_best_test_kappa": legacy_best_test["kappa"],
                "legacy_best_test_epoch": legacy_best_test["epoch"],
            }
            best_state = _state_dict_cpu(model)

        logger.info(
            "epoch=%d | lambda=%.4f | train_loss=%.4f | task=%.4f | domain=%.4f | domain_cnn=%.4f | val_acc=%.4f | val_kappa=%.4f | test_acc=%.4f | test_kappa=%.4f",
            epoch + 1,
            lam,
            train_loss,
            train_task_loss,
            train_domain_loss,
            train_domain_cnn_loss,
            val_metrics["accuracy"],
            val_metrics["kappa"],
            test_metrics["accuracy"],
            test_metrics["kappa"],
        )

    model.load_state_dict(best_state)
    best["legacy_best_test_accuracy"] = legacy_best_test["accuracy"]
    best["legacy_best_test_kappa"] = legacy_best_test["kappa"]
    best["legacy_best_test_epoch"] = legacy_best_test["epoch"]

    return best, history


def run(args: argparse.Namespace) -> None:
    set_seed_everywhere(args.seed, deterministic=args.deterministic)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"{args.dataset}_loso_{timestamp}"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir)

    if args.head_only_finetune and args.use_da:
        raise ValueError(
            "--head_only_finetune cannot be combined with --use_da in LOSO. "
            "Disable DA for head-only transfer or disable head-only to use DA."
        )

    finetune_strategy = (
        "head_only" if args.head_only_finetune else args.finetune_strategy
    )

    logger.info("Loading MOABB dataset: %s", args.dataset)

    x, y, subject_ids, subjects = load_moabb_motor_imagery_dataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        options=MoabbLoadOptions(
            subjects=args.subjects if args.subjects else None,
            show_progress=True,
        ),
    )

    loader_options = DataLoaderOptions(
        batch_size=args.batch_size,
        target_test_size=args.target_test_size,
        random_state=args.seed,
        apply_euclidean_align=args.loader_euclidean_align,
        subject_balanced_sampling=args.subject_balanced_sampling,
        drop_last_train=args.drop_last_train,
        num_workers=args.num_workers,
        seed=args.seed,
        deterministic=args.deterministic,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_channels = x.shape[1]
    num_classes = int(np.max(y)) + 1
    num_subjects = int(np.max(subject_ids)) + 1

    logger.info(
        "Loaded data shape=%s, n_subjects=%d, n_classes=%d, device=%s",
        str(tuple(x.shape)),
        len(subjects),
        num_classes,
        device,
    )
    if args.train_fraction < 1.0:
        logger.info(
            "Reduced-data fine-tuning enabled: train_fraction=%.2f",
            args.train_fraction,
        )

    selected_subjects = subjects
    if args.subjects:
        wanted = {int(s) for s in args.subjects}
        selected_subjects = [s for s in subjects if s in wanted]

    if not selected_subjects:
        raise ValueError("No valid LOSO subjects selected")

    config = {
        "dataset": args.dataset,
        "init_checkpoint": args.init_checkpoint,
        "head_only_finetune": args.head_only_finetune,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "source_val_size": args.source_val_size,
        "target_test_size": args.target_test_size,
        "selection_metric": args.selection_metric,
        "checkpoint_task_head_policy": args.checkpoint_task_head_policy,
        "load_domain_heads": args.load_domain_heads,
        "use_da": args.use_da,
        "domain_loss_weight": args.domain_loss_weight,
        "da_lambda_gamma": args.da_lambda_gamma,
        "loader_euclidean_align": args.loader_euclidean_align,
        "subject_balanced_sampling": args.subject_balanced_sampling,
        "drop_last_train": args.drop_last_train,
        "train_fraction": args.train_fraction,
        "finetune_strategy": finetune_strategy,
        "warmup_epochs": args.warmup_epochs,
        "backbone_lr_multiplier": args.backbone_lr_multiplier,
        "head_lr_multiplier": args.head_lr_multiplier,
        "max_grad_norm": args.max_grad_norm,
        "patience": args.patience,
        "min_delta": args.min_delta,
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
        "seed": args.seed,
        "deterministic": args.deterministic,
        "checkpoint_mode": args.checkpoint_mode,
        "subjects": selected_subjects,
        "protocol": "LOSO",
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    metadata = create_experiment_metadata(
        protocol="loso",
        dataset=args.dataset,
        config=config,
        seed=args.seed,
        deterministic=args.deterministic,
    )
    save_experiment_metadata(run_dir / "metadata.json", metadata)

    per_subject: dict[str, dict[str, float]] = {}
    run_models: dict[str, dict[str, torch.Tensor]] = {}

    for held_out in selected_subjects:
        logger.info("=== LOSO held-out subject %d ===", held_out)

        if args.use_da:
            source_loader, target_loader, test_loader = (
                create_loso_domain_adaptation_dataloaders(
                    x=x,
                    y=y,
                    subject_id=subject_ids,
                    target_subject=held_out,
                    options=loader_options,
                )
            )
            source_loader, val_loader = _split_source_train_validation_loaders(
                loader=source_loader,
                validation_size=args.source_val_size,
                train_fraction=args.train_fraction,
                min_trials_per_class=args.train_fraction_min_trials_per_class,
                seed=args.seed,
                batch_size=args.batch_size,
                subject_balanced_sampling=args.subject_balanced_sampling,
                drop_last_train=args.drop_last_train,
                num_workers=args.num_workers,
                deterministic=args.deterministic,
            )
        else:
            train_loader, test_loader = create_dataloaders(
                x=x,
                y=y,
                subject_id=subject_ids,
                loso_subject=held_out,
                options=loader_options,
            )
            train_loader, val_loader = _split_source_train_validation_loaders(
                loader=train_loader,
                validation_size=args.source_val_size,
                train_fraction=args.train_fraction,
                min_trials_per_class=args.train_fraction_min_trials_per_class,
                seed=args.seed,
                batch_size=args.batch_size,
                subject_balanced_sampling=args.subject_balanced_sampling,
                drop_last_train=args.drop_last_train,
                num_workers=args.num_workers,
                deterministic=args.deterministic,
            )

        model = EEGModel(
            num_channels=num_channels,
            num_classes=num_classes,
            num_subjects=2 if args.use_da else num_subjects,
            cnn_out_channels=args.cnn_out_channels,
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

        if args.init_checkpoint:
            _load_init_checkpoint(
                model,
                args.init_checkpoint,
                device,
                logger,
                task_head_policy=args.checkpoint_task_head_policy,
                load_domain_heads=args.load_domain_heads,
            )

        if args.head_only_finetune:
            _configure_head_only_finetune(model, logger)

        if args.use_da:
            best, history = train_one_fold_da(
                model=model,
                source_loader=source_loader,
                target_loader=target_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                label_smoothing=args.label_smoothing,
                domain_loss_weight=args.domain_loss_weight,
                da_lambda_gamma=args.da_lambda_gamma,
                cnn_domain_weight=args.cnn_domain_weight,
                selection_metric=args.selection_metric,
                min_delta=args.min_delta,
                logger=logger,
            )
        else:
            best, history = train_one_fold(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                finetune_strategy=finetune_strategy,
                warmup_epochs=args.warmup_epochs,
                backbone_lr_multiplier=args.backbone_lr_multiplier,
                head_lr_multiplier=args.head_lr_multiplier,
                max_grad_norm=args.max_grad_norm,
                patience=args.patience,
                min_delta=args.min_delta,
                selection_metric=args.selection_metric,
                label_smoothing=args.label_smoothing,
                logger=logger,
            )

        key = str(held_out)
        per_subject[key] = best

        if args.checkpoint_mode == "per_subject":
            torch.save(
                model.state_dict(), ckpt_dir / f"loso_subject_{held_out}_last.pt"
            )
        elif args.checkpoint_mode == "single_file":
            run_models[key] = _state_dict_cpu(model)

        with (run_dir / f"loso_subject_{held_out}_history.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(history, f, indent=2)

    accuracies = [v["accuracy"] for v in per_subject.values()]
    kappas = [v["kappa"] for v in per_subject.values()]
    legacy_accuracies = [v["legacy_best_test_accuracy"] for v in per_subject.values()]
    legacy_kappas = [v["legacy_best_test_kappa"] for v in per_subject.values()]

    summary = {
        "dataset": args.dataset,
        "protocol": "LOSO",
        "n_subjects": len(per_subject),
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "mean_kappa": float(np.mean(kappas)),
        "std_kappa": float(np.std(kappas)),
        "selection_metric": args.selection_metric,
        "mean_legacy_best_test_accuracy": float(np.mean(legacy_accuracies)),
        "std_legacy_best_test_accuracy": float(np.std(legacy_accuracies)),
        "mean_legacy_best_test_kappa": float(np.mean(legacy_kappas)),
        "std_legacy_best_test_kappa": float(np.std(legacy_kappas)),
        "per_subject": per_subject,
    }

    with (run_dir / "loso_results.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.checkpoint_mode == "single_file":
        torch.save(
            {
                "dataset": args.dataset,
                "protocol": "LOSO",
                "config": config,
                "per_subject_state_dict": run_models,
            },
            ckpt_dir / "loso_run_models.pt",
        )

    logger.info(
        "Done. mean_acc=%.4f, std_acc=%.4f, results=%s",
        summary["mean_accuracy"],
        summary["std_accuracy"],
        str(run_dir / "loso_results.json"),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LOSO training/evaluation with MOABB")
    parser.add_argument(
        "--dataset",
        type=str,
        default="bnci2014_001",
        choices=["bnci2014_001", "physionetmi", "cho2017", "lee2019_mi"],
    )
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument("--head_only_finetune", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="results/loso")
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
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument("--train_fraction_min_trials_per_class", type=int, default=2)
    parser.add_argument("--source_val_size", type=float, default=0.2)
    parser.add_argument("--target_test_size", type=float, default=0.2)
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "kappa"],
    )
    parser.add_argument(
        "--checkpoint_task_head_policy",
        type=str,
        default="auto",
        choices=["auto", "load", "skip"],
        help=(
            "auto loads task head from supervised checkpoints and skips it for SSL "
            "checkpoints when a neighboring config.json is available"
        ),
    )
    parser.add_argument("--load_domain_heads", action="store_true", default=False)
    parser.add_argument(
        "--finetune_strategy",
        type=str,
        default="progressive",
        choices=["progressive", "full", "head_only"],
    )
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--backbone_lr_multiplier", type=float, default=0.1)
    parser.add_argument("--head_lr_multiplier", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        dest="max_grad_norm",
        help="Alias for --max_grad_norm",
    )
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        dest="patience",
        help="Alias for --patience",
    )
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--use_da", action="store_true", default=False)
    parser.add_argument("--domain_loss_weight", type=float, default=1.0)
    parser.add_argument("--da_lambda_gamma", type=float, default=10.0)
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
    parser.add_argument(
        "--model_euclidean_alignment", action="store_true", default=True
    )
    parser.add_argument(
        "--no-model-euclidean-alignment",
        dest="model_euclidean_alignment",
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
    parser.add_argument("--temporal_kernels", nargs="+", type=int, default=None)
    parser.add_argument(
        "--multiscale_preserve_capacity", action="store_true", default=False
    )
    parser.add_argument("--use_attention_pool", action="store_true", default=False)
    parser.add_argument("--attention_mix_init", type=float, default=0.5)
    parser.add_argument("--learnable_attention_mix", action="store_true", default=False)
    parser.add_argument("--domain_head_hidden_dim", type=int, default=0)
    parser.add_argument("--domain_head_layers", type=int, default=1)
    parser.add_argument("--domain_head_dropout", type=float, default=0.0)
    parser.add_argument("--use_cnn_domain_head", action="store_true", default=False)
    parser.add_argument("--cnn_domain_weight", type=float, default=0.0)
    parser.add_argument(
        "--subject_balanced_sampling", action="store_true", default=False
    )
    parser.add_argument("--drop_last_train", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument(
        "--no-deterministic", dest="deterministic", action="store_false"
    )

    parser.add_argument("--cnn_out_channels", type=int, default=32)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
