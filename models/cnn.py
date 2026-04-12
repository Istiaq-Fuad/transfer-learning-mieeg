from __future__ import annotations

import torch
from torch import nn


class CNNBlock(nn.Module):
    """EEG CNN block with temporal + depthwise spatial convolution and residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 32,
        temporal_kernel: int = 64,
        pool_kernel: int = 4,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=(1, temporal_kernel),
            padding=(0, temporal_kernel // 2),
            bias=False,
        )
        self.temporal_bn = nn.BatchNorm2d(out_channels)

        self.spatial_conv = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(in_channels, 1),
            groups=out_channels,
            bias=False,
        )
        self.spatial_bn = nn.BatchNorm2d(out_channels)

        self.residual_proj = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )

        self.activation = nn.ELU()
        self.pool = nn.AvgPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_kernel))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, C, T)

        Returns:
            (B, F, 1, T')
        """
        main = self.temporal_bn(self.temporal_conv(x))
        main = self.spatial_bn(self.spatial_conv(main))

        # 1x1 projection matches channel count; average over spatial axis to match height=1.
        residual = self.residual_proj(x).mean(dim=2, keepdim=True)

        # With even temporal kernel (64) and padding (32), temporal conv can produce T+1.
        # Align both branches before residual addition.
        if main.size(-1) != residual.size(-1):
            t = min(main.size(-1), residual.size(-1))
            main = main[..., :t]
            residual = residual[..., :t]

        out = self.activation(main + residual)
        out = self.pool(out)
        out = self.dropout(out)
        return out
