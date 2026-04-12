from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function


class _GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx: Function, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Function, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


class GRL(nn.Module):
    """Gradient Reversal Layer."""

    def forward(self, x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        return _GradientReversalFunction.apply(x, lambda_)


class TaskHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class DomainHead(nn.Module):
    def __init__(self, embedding_dim: int, num_subjects: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, num_subjects)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
