from __future__ import annotations

import math

import torch
from torch import nn


def _build_rope_cache(
    seq_len: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head_dim")
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def _apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    out = torch.stack(
        [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
        dim=-1,
    )
    return out.flatten(-2)


class SelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_rope: bool = False,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_rope = use_rope

        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_rope:
            cos, sin = _build_rope_cache(seq_len, self.head_dim, x.device, x.dtype)
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        out = self.proj(out)
        return self.proj_drop(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block for lightweight EEG encoding."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_rope: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attn = SelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=use_rope,
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
        attn_out = self.attn(attn_in)
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
        use_rope: bool = True,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        self.use_rope = use_rope
        self.use_positional_encoding = use_positional_encoding and not use_rope
        if self.use_positional_encoding:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, max_seq_len + 1, embedding_dim)
            )
        else:
            self.register_parameter("pos_embed", None)

        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim,
                    num_heads,
                    dropout,
                    use_rope=self.use_rope,
                )
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
