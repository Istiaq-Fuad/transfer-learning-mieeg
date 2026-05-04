from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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
    selection_metric: str,
    patience: int,
    min_delta: float,
    lr_schedule: str,
    logger: logging.Logger,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if lr_schedule == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    model.to(device)
    history: list[dict[str, float]] = []
    best_score = float("-inf")
    best_row: dict[str, float] | None = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
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
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            if val_metrics is not None:
                best_row["val_accuracy"] = val_metrics["accuracy"]
                best_row["val_kappa"] = val_metrics["kappa"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": float(epoch + 1),
            "train_loss": float(train_loss),
            "test_accuracy": test_metrics["accuracy"],
            "test_kappa": test_metrics["kappa"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        if val_metrics is not None:
            row["val_accuracy"] = val_metrics["accuracy"]
            row["val_kappa"] = val_metrics["kappa"]
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


def _parse_subjects(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _build_model(num_channels: int, num_classes: int, num_subjects: int) -> EEGModel:
    return EEGModel(
        num_channels=num_channels,
        num_classes=num_classes,
        num_subjects=num_subjects,
        cnn_out_channels=DEFAULT_MODEL_CONFIG["cnn_out_channels"],
        cnn_dropout=DEFAULT_MODEL_CONFIG["cnn_dropout"],
        embedding_dim=DEFAULT_MODEL_CONFIG["embedding_dim"],
        num_heads=DEFAULT_MODEL_CONFIG["num_heads"],
        num_layers=DEFAULT_MODEL_CONFIG["num_layers"],
        dropout=DEFAULT_MODEL_CONFIG["dropout"],
        temporal_kernels=DEFAULT_MODEL_CONFIG["temporal_kernels"],
        multiscale_preserve_capacity=DEFAULT_MODEL_CONFIG[
            "multiscale_preserve_capacity"
        ],
        use_attention_pool=DEFAULT_MODEL_CONFIG["use_attention_pool"],
        attention_mix_init=DEFAULT_MODEL_CONFIG["attention_mix_init"],
        learnable_attention_mix=DEFAULT_MODEL_CONFIG["learnable_attention_mix"],
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
        apply_euclidean_align=not args.no_euclidean_align,
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
    histories: dict[str, list[dict[str, float]]] = {}
    per_subject_folds: dict[str, list[dict[str, float]]] = {}

    base_lr = args.lr
    base_weight_decay = args.weight_decay
    base_label_smoothing = args.label_smoothing
    if args.protocol == "within":
        base_lr = args.within_lr
        base_weight_decay = args.within_weight_decay
        base_label_smoothing = args.within_label_smoothing

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

                if loader_options.apply_euclidean_align:
                    whitening = fit_euclidean_alignment(
                        train_dataset.x,
                        eps=loader_options.align_eps,
                    )
                    train_dataset.x = apply_euclidean_alignment(
                        train_dataset.x, whitening
                    )
                    val_dataset.x = apply_euclidean_alignment(val_dataset.x, whitening)
                    test_dataset.x = apply_euclidean_alignment(
                        test_dataset.x, whitening
                    )

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

                model = _build_model(num_channels, num_classes, num_subjects)
                best, history = train_one_subject(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    device=device,
                    epochs=args.epochs,
                    lr=base_lr,
                    weight_decay=base_weight_decay,
                    label_smoothing=base_label_smoothing,
                    selection_metric=args.selection_metric,
                    patience=args.patience,
                    min_delta=args.min_delta,
                    lr_schedule=args.lr_schedule,
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

        model = _build_model(num_channels, num_classes, num_subjects)
        best, history = train_one_subject(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=base_lr,
            weight_decay=base_weight_decay,
            label_smoothing=base_label_smoothing,
            selection_metric=args.selection_metric,
            patience=args.patience,
            min_delta=args.min_delta,
            lr_schedule=args.lr_schedule,
            logger=logger,
        )

        per_subject[str(subject)] = best
        histories[str(subject)] = history

    accs = [row["test_accuracy"] for row in per_subject.values()]
    kappas = [row["test_kappa"] for row in per_subject.values()]
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
        default="cosine",
        choices=["none", "cosine"],
    )
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--within_label_smoothing", type=float, default=0.1)
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
