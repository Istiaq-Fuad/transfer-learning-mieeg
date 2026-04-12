from __future__ import annotations

import torch
from torch import nn

from models.cnn import CNNBlock
from models.heads import DomainHead, GRL, TaskHead
from models.tokenizer import EEGTokenizer
from models.vit import ViTEncoder
from training.utils import euclidean_alignment, riemannian_reweight


class EEGModel(nn.Module):
    """Unified EEG model with CNN + tokenizer + lightweight ViT + adversarial domain head."""

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        num_subjects: int,
        cnn_out_channels: int = 32,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.cnn = CNNBlock(
            in_channels=num_channels, out_channels=cnn_out_channels, dropout=0.5
        )
        self.tokenizer = EEGTokenizer(
            in_features=cnn_out_channels,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        self.vit = ViTEncoder(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_positional_encoding=True,
        )

        self.task_head = TaskHead(embedding_dim, num_classes)
        self.grl = GRL()
        self.domain_head = DomainHead(embedding_dim, num_subjects)

    def forward(
        self, x: torch.Tensor, lambda_: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, T)
            lambda_: GRL scale

        Returns:
            task_output: (B, num_classes)
            domain_output: (B, num_subjects)
        """
        x = euclidean_alignment(x)
        x = riemannian_reweight(x)

        x = x.unsqueeze(1)  # (B, 1, C, T)
        x = self.cnn(x)  # (B, F, 1, T')
        x = self.tokenizer(x)  # (B, N, D)

        _, cls_token = self.vit(x)

        task_output = self.task_head(cls_token)
        domain_input = self.grl(cls_token, lambda_)
        domain_output = self.domain_head(domain_input)
        return task_output, domain_output
