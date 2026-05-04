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
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.dropout(x)
        return self.classifier(x)


class AttentionPool(nn.Module):
    """Learned attention pooling over sequence tokens."""

    def __init__(self, embedding_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.score = nn.Linear(embedding_dim, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, N, D)
        attn_logits = self.score(tokens).squeeze(-1)  # (B, N)
        attn = torch.softmax(attn_logits, dim=1)
        pooled = torch.bmm(attn.unsqueeze(1), tokens).squeeze(1)  # (B, D)
        return self.dropout(pooled)


class DomainHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_subjects: int,
        hidden_dim: int = 0,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers <= 1 or hidden_dim <= 0:
            self.classifier = nn.Linear(embedding_dim, num_subjects)
        else:
            layers: list[nn.Module] = []
            in_dim = embedding_dim
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, num_subjects))
            self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
