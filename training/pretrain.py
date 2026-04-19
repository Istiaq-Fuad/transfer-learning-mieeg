from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.optim import Adam

from training.utils import lambda_scheduler


def pretrain(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_epoch: int = 30,
    lr: float = 1e-3,
) -> list[dict[str, Any]]:
    """Pretrain with adversarial domain adaptation."""
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    task_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    history: list[dict[str, Any]] = []

    for epoch in range(max_epoch):
        model.train()
        lam = lambda_scheduler(epoch, max_epoch)

        running_loss = 0.0
        running_task = 0.0
        running_domain = 0.0
        n_samples = 0

        for x, y, subject_id in train_loader:
            x = x.to(device)
            y = y.to(device)
            subject_id = subject_id.to(device)

            outputs = model(x, lam)
            task_out = outputs["task"]
            domain_out = outputs["domain"]

            task_loss = task_criterion(task_out, y)
            domain_loss = domain_criterion(domain_out, subject_id)
            total_loss = task_loss + lam * domain_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            n_samples += batch_size
            running_loss += total_loss.item() * batch_size
            running_task += task_loss.item() * batch_size
            running_domain += domain_loss.item() * batch_size

        epoch_stats = {
            "epoch": epoch + 1,
            "lambda": lam,
            "loss": running_loss / max(n_samples, 1),
            "task_loss": running_task / max(n_samples, 1),
            "domain_loss": running_domain / max(n_samples, 1),
        }
        history.append(epoch_stats)
        print(
            f"[Pretrain] Epoch {epoch + 1:03d}/{max_epoch:03d} | "
            f"lambda={lam:.4f} | loss={epoch_stats['loss']:.4f}"
        )

    return history
