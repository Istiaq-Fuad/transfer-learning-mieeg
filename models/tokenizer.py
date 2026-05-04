from __future__ import annotations

import torch
from torch import nn


class EEGTokenizer(nn.Module):
    """Converts CNN EEG features into transformer tokens with per-trial normalization."""

    def __init__(
        self, in_features: int, embedding_dim: int = 128, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(in_features)
        self.proj = nn.Linear(in_features, embedding_dim)
        self.out_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, F, 1, T')

        Returns:
            (B, N, D), where N=T'
        """
        x = x.squeeze(2)  # (B, F, T')
        x = self.norm(x)  # normalize per feature channel
        x = x.transpose(1, 2)  # (B, T', F)
        x = self.proj(x)  # (B, T', D)
        x = self.out_norm(x)
        x = self.dropout(x)
        return x
