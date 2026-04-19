from __future__ import annotations

import torch
from torch import nn


class CNNBlock(nn.Module):
    """EEG CNN block with multi-scale temporal conv + depthwise spatial conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 32,
        temporal_kernel: int = 64,
        temporal_kernels: tuple[int, ...] | None = None,
        multiscale_preserve_capacity: bool = False,
        pool_kernel: int = 4,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        kernels = (
            temporal_kernels if temporal_kernels is not None else (temporal_kernel,)
        )
        if len(kernels) == 0:
            raise ValueError("temporal_kernels must contain at least one kernel size")

        n_branches = len(kernels)
        if multiscale_preserve_capacity and n_branches > 1:
            branch_channels = [out_channels for _ in range(n_branches)]
            merged_channels = out_channels * n_branches
            self.temporal_merge = nn.Sequential(
                nn.Conv2d(
                    in_channels=merged_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            base = out_channels // n_branches
            remainder = out_channels % n_branches
            branch_channels = [
                base + (1 if i < remainder else 0) for i in range(n_branches)
            ]
            if min(branch_channels) <= 0:
                raise ValueError(
                    "out_channels must be at least the number of temporal branches"
                )
            self.temporal_merge = nn.Identity()

        self.temporal_convs = nn.ModuleList()
        self.temporal_bns = nn.ModuleList()
        for k, ch in zip(kernels, branch_channels):
            self.temporal_convs.append(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=ch,
                    kernel_size=(1, k),
                    padding=(0, k // 2),
                    bias=False,
                )
            )
            self.temporal_bns.append(nn.BatchNorm2d(ch))

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
        temporal_feats = []
        for conv, bn in zip(self.temporal_convs, self.temporal_bns):
            temporal_feats.append(bn(conv(x)))

        if len(temporal_feats) > 1:
            min_t = min(t_feat.size(-1) for t_feat in temporal_feats)
            temporal_feats = [t_feat[..., :min_t] for t_feat in temporal_feats]

        main = torch.cat(temporal_feats, dim=1)
        main = self.temporal_merge(main)
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
