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
from torch import nn
from torch.optim import Adam

try:
    from data.loader import (
        create_dataloaders,
        create_loso_domain_adaptation_dataloaders,
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
        create_dataloaders,
        create_loso_domain_adaptation_dataloaders,
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


def train_one_fold(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    logger: logging.Logger,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    model.to(device)
    best = {"accuracy": 0.0, "kappa": 0.0, "epoch": 0.0}
    history: list[dict[str, float]] = []

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
            n_samples += bs
            running_loss += loss.item() * bs

        train_loss = running_loss / max(n_samples, 1)
        test_metrics = evaluate(model, test_loader, device)

        row = {
            "epoch": float(epoch + 1),
            "train_loss": float(train_loss),
            "test_accuracy": test_metrics["accuracy"],
            "test_kappa": test_metrics["kappa"],
        }
        history.append(row)

        if test_metrics["accuracy"] >= best["accuracy"]:
            best = {
                "accuracy": test_metrics["accuracy"],
                "kappa": test_metrics["kappa"],
                "epoch": float(epoch + 1),
            }

        logger.info(
            "epoch=%d | train_loss=%.4f | test_acc=%.4f | test_kappa=%.4f",
            epoch + 1,
            train_loss,
            test_metrics["accuracy"],
            test_metrics["kappa"],
        )

    return best, history


def train_one_fold_da(
    model: nn.Module,
    source_loader: torch.utils.data.DataLoader,
    target_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    domain_loss_weight: float,
    da_lambda_gamma: float,
    cnn_domain_weight: float,
    logger: logging.Logger,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    task_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    model.to(device)
    best = {"accuracy": 0.0, "kappa": 0.0, "epoch": 0.0}
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
        test_metrics = evaluate(model, test_loader, device)

        row = {
            "epoch": float(epoch + 1),
            "lambda": float(lam),
            "train_loss": float(train_loss),
            "train_task_loss": float(train_task_loss),
            "train_domain_loss": float(train_domain_loss),
            "train_domain_cnn_loss": float(train_domain_cnn_loss),
            "test_accuracy": test_metrics["accuracy"],
            "test_kappa": test_metrics["kappa"],
        }
        history.append(row)

        if test_metrics["accuracy"] >= best["accuracy"]:
            best = {
                "accuracy": test_metrics["accuracy"],
                "kappa": test_metrics["kappa"],
                "epoch": float(epoch + 1),
            }

        logger.info(
            "epoch=%d | lambda=%.4f | train_loss=%.4f | task=%.4f | domain=%.4f | domain_cnn=%.4f | test_acc=%.4f | test_kappa=%.4f",
            epoch + 1,
            lam,
            train_loss,
            train_task_loss,
            train_domain_loss,
            train_domain_cnn_loss,
            test_metrics["accuracy"],
            test_metrics["kappa"],
        )

    return best, history


def run(args: argparse.Namespace) -> None:
    set_seed_everywhere(args.seed, deterministic=args.deterministic)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"{args.dataset}_loso_{timestamp}"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir)
    logger.info("Loading MOABB dataset: %s", args.dataset)

    x, y, subject_ids, subjects = load_moabb_motor_imagery_dataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
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

    selected_subjects = subjects
    if args.subjects:
        wanted = {int(s) for s in args.subjects}
        selected_subjects = [s for s in subjects if s in wanted]

    if not selected_subjects:
        raise ValueError("No valid LOSO subjects selected")

    config = {
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "use_da": args.use_da,
        "domain_loss_weight": args.domain_loss_weight,
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
                    batch_size=args.batch_size,
                    apply_euclidean_align=args.loader_euclidean_align,
                    num_workers=args.num_workers,
                    seed=args.seed,
                    deterministic=args.deterministic,
                )
            )
        else:
            train_loader, test_loader = create_dataloaders(
                x=x,
                y=y,
                subject_id=subject_ids,
                batch_size=args.batch_size,
                loso_subject=held_out,
                apply_euclidean_align=args.loader_euclidean_align,
                num_workers=args.num_workers,
                seed=args.seed,
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

        if args.use_da:
            best, history = train_one_fold_da(
                model=model,
                source_loader=source_loader,
                target_loader=target_loader,
                test_loader=test_loader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                domain_loss_weight=args.domain_loss_weight,
                da_lambda_gamma=args.da_lambda_gamma,
                cnn_domain_weight=args.cnn_domain_weight,
                logger=logger,
            )
        else:
            best, history = train_one_fold(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                logger=logger,
            )

        key = str(held_out)
        per_subject[key] = best

        if args.checkpoint_mode == "per_subject":
            torch.save(model.state_dict(), ckpt_dir / f"loso_subject_{held_out}_last.pt")
        elif args.checkpoint_mode == "single_file":
            run_models[key] = _state_dict_cpu(model)

        with (run_dir / f"loso_subject_{held_out}_history.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(history, f, indent=2)

    accuracies = [v["accuracy"] for v in per_subject.values()]
    kappas = [v["kappa"] for v in per_subject.values()]

    summary = {
        "dataset": args.dataset,
        "protocol": "LOSO",
        "n_subjects": len(per_subject),
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "mean_kappa": float(np.mean(kappas)),
        "std_kappa": float(np.std(kappas)),
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
