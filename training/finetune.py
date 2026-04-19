from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.optim import Adam


def _freeze_for_finetune(model: nn.Module) -> None:
    # Freeze all by default, then unfreeze selected modules.
    for param in model.parameters():
        param.requires_grad = False

    for param in model.cnn.parameters():
        param.requires_grad = False

    if len(model.vit.blocks) > 0:
        for param in model.vit.blocks[0].parameters():
            param.requires_grad = False

    if len(model.vit.blocks) > 0:
        for param in model.vit.blocks[-1].parameters():
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
) -> list[dict[str, Any]]:
    """Fine-tune on target data without domain adaptation."""
    state = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(state)

    model.to(device)
    _freeze_for_finetune(model)

    optimizer = Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    criterion = nn.CrossEntropyLoss()

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
