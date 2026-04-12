from __future__ import annotations

import torch
from torch import nn


class TransformerBlock(nn.Module):
    """Pre-norm transformer block for lightweight EEG encoding."""

    def __init__(
        self, embedding_dim: int, num_heads: int = 4, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        hidden_dim = 2 * embedding_dim
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + self.dropout1(attn_out)

        ffn_in = self.norm2(x)
        ffn_out = self.ffn(ffn_in)
        x = x + self.dropout2(ffn_out)
        return x


class ViTEncoder(nn.Module):
    """Lightweight transformer encoder with CLS token for EEG sequences."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, max_seq_len + 1, embedding_dim)
            )
        else:
            self.register_parameter("pos_embed", None)

        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embedding_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, D)

        Returns:
            sequence: (B, N+1, D)
            cls_token: (B, D)
        """
        batch_size, seq_len, _ = x.shape
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        if self.use_positional_encoding:
            if x.size(1) > self.pos_embed.size(1):
                raise ValueError(
                    f"Sequence length {x.size(1)} exceeds max positional length {self.pos_embed.size(1)}"
                )
            x = x + self.pos_embed[:, : x.size(1), :]

        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_token = x[:, 0, :]
        return x, cls_token
