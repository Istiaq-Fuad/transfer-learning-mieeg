from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from data.loader import DataLoaderOptions, create_dataloaders
from models.model import EEGModel
from training.finetune import finetune
from training.pretrain import pretrain
from utils.reproducibility import (
    create_experiment_metadata,
    save_experiment_metadata,
    set_seed_everywhere,
)


@dataclass
class Config:
    x_path: str = "data/x.npy"
    y_path: str = "data/y.npy"
    subject_path: str = "data/subject_id.npy"
    batch_size: int = 32
    pretrain_epochs: int = 20
    finetune_epochs: int = 25
    lr_pretrain: float = 1e-3
    lr_finetune: float = 1e-4
    loso_subject: int | None = None
    embedding_dim: int = 128
    cnn_out_channels: int = 32
    num_heads: int = 4
    num_layers: int = 2
    save_path: str = "checkpoints/pretrained.pt"
    run_finetune: bool = False
    seed: int = 42
    deterministic: bool = True


def load_config(path: str = "config.yaml") -> Config:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return Config()

    with cfg_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    return Config(**raw)


def evaluate(
    model: EEGModel, data_loader: torch.utils.data.DataLoader, device: torch.device
) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y, _ in data_loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x, lambda_=0.0)
            logits = outputs["task"]
            preds = logits.argmax(dim=1)

            total += y.size(0)
            correct += (preds == y).sum().item()

    return correct / max(total, 1)


def main() -> None:
    cfg = load_config()
    set_seed_everywhere(cfg.seed, deterministic=cfg.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_path = Path(cfg.x_path)
    y_path = Path(cfg.y_path)
    s_path = Path(cfg.subject_path)
    if not (x_path.exists() and y_path.exists() and s_path.exists()):
        raise FileNotFoundError(
            "Missing EEG .npy files. Expected x, y, and subject_id paths in config or defaults."
        )

    x = np.load(x_path)
    y = np.load(y_path)
    subject_id = np.load(s_path)

    train_loader, val_loader = create_dataloaders(
        x=x,
        y=y,
        subject_id=subject_id,
        loso_subject=cfg.loso_subject,
        options=DataLoaderOptions(
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            deterministic=cfg.deterministic,
        ),
    )

    num_channels = x.shape[1]
    num_classes = int(np.max(y)) + 1
    num_subjects = int(np.max(subject_id)) + 1

    model = EEGModel(
        num_channels=num_channels,
        num_classes=num_classes,
        num_subjects=num_subjects,
        cnn_out_channels=cfg.cnn_out_channels,
        embedding_dim=cfg.embedding_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
    )

    pretrain_history = pretrain(
        model=model,
        train_loader=train_loader,
        device=device,
        max_epoch=cfg.pretrain_epochs,
        lr=cfg.lr_pretrain,
    )

    save_path = Path(cfg.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)

    config_dict = {
        "x_path": cfg.x_path,
        "y_path": cfg.y_path,
        "subject_path": cfg.subject_path,
        "batch_size": cfg.batch_size,
        "pretrain_epochs": cfg.pretrain_epochs,
        "finetune_epochs": cfg.finetune_epochs,
        "lr_pretrain": cfg.lr_pretrain,
        "lr_finetune": cfg.lr_finetune,
        "loso_subject": cfg.loso_subject,
        "embedding_dim": cfg.embedding_dim,
        "cnn_out_channels": cfg.cnn_out_channels,
        "num_heads": cfg.num_heads,
        "num_layers": cfg.num_layers,
        "save_path": cfg.save_path,
        "run_finetune": cfg.run_finetune,
        "seed": cfg.seed,
        "deterministic": cfg.deterministic,
    }
    metadata = create_experiment_metadata(
        protocol="main",
        dataset="local_npy",
        config=config_dict,
        seed=cfg.seed,
        deterministic=cfg.deterministic,
    )
    save_experiment_metadata(save_path.parent / "metadata.json", metadata)

    val_acc = evaluate(model, val_loader, device)
    print(f"Final pretrain loss: {pretrain_history[-1]['loss']:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")

    if cfg.run_finetune:
        finetune_history = finetune(
            model=model,
            train_loader=train_loader,
            device=device,
            pretrained_path=str(save_path),
            num_epochs=cfg.finetune_epochs,
            lr=cfg.lr_finetune,
        )

        val_acc = evaluate(model, val_loader, device)
        print(f"Final finetune loss: {finetune_history[-1]['loss']:.4f}")
        print(f"Validation accuracy after finetune: {val_acc:.4f}")


if __name__ == "__main__":
    main()
