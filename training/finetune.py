from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.optim import AdamW


def _load_partial_checkpoint(
    model: nn.Module,
    pretrained_path: str,
    device: torch.device,
) -> None:
    state = torch.load(pretrained_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {pretrained_path}")

    model_state = model.state_dict()
    filtered_state: dict[str, torch.Tensor] = {}
    skipped_shape: list[str] = []

    for key, value in state.items():
        if key not in model_state:
            continue
        if model_state[key].shape != value.shape:
            skipped_shape.append(key)
            continue
        filtered_state[key] = value

    incompatible = model.load_state_dict(filtered_state, strict=False)
    print(
        f"[Finetune] Warm-started from {pretrained_path} | loaded={len(filtered_state)} | shape_skipped={len(skipped_shape)} | missing={len(incompatible.missing_keys)} | unexpected={len(incompatible.unexpected_keys)}"
    )


def _freeze_for_finetune(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, "tokenizer"):
        for param in model.tokenizer.parameters():
            param.requires_grad = True

    if len(model.vit.blocks) > 0:
        for param in model.vit.blocks[-1].parameters():
            param.requires_grad = True

    if hasattr(model.vit, "norm"):
        for param in model.vit.norm.parameters():
            param.requires_grad = True

    if hasattr(model, "attn_pool") and model.attn_pool is not None:
        for param in model.attn_pool.parameters():
            param.requires_grad = True

    for param in model.task_head.parameters():
        param.requires_grad = True


def finetune(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    pretrained_path: str,
    num_epochs: int = 25,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
) -> list[dict[str, Any]]:
    """Fine-tune on target data without domain adaptation."""
    _load_partial_checkpoint(model, pretrained_path, device)
    model.to(device)
    _freeze_for_finetune(model)

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    history: list[dict[str, Any]] = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        n_samples = 0

        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x, lambda_=0.0)
            task_out = outputs["task"]
            task_loss = criterion(task_out, y)

            optimizer.zero_grad()
            task_loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            n_samples += batch_size
            running_loss += task_loss.item() * batch_size

        epoch_loss = running_loss / max(n_samples, 1)
        history.append({"epoch": epoch + 1, "loss": epoch_loss})
        print(
            f"[Finetune] Epoch {epoch + 1:03d}/{num_epochs:03d} | loss={epoch_loss:.4f}"
        )

    return history
