from __future__ import annotations

import torch
from torch import nn

from models.cnn import CNNBlock
from models.heads import AttentionPool, DomainHead, GRL, TaskHead
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
        cnn_dropout: float = 0.5,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        temporal_kernels: tuple[int, ...] | None = None,
        multiscale_preserve_capacity: bool = False,
        use_attention_pool: bool = False,
        attention_mix_init: float = 0.5,
        learnable_attention_mix: bool = False,
        domain_head_hidden_dim: int = 0,
        domain_head_layers: int = 1,
        domain_head_dropout: float = 0.0,
        use_cnn_domain_head: bool = False,
        cnn_domain_weight: float = 0.0,
        apply_model_pre_align_only: bool = False,
        apply_model_euclidean_alignment: bool = True,
        apply_model_riemannian_reweight: bool = True,
    ) -> None:
        super().__init__()

        self.apply_model_euclidean_alignment = apply_model_euclidean_alignment
        self.apply_model_riemannian_reweight = apply_model_riemannian_reweight
        self.use_attention_pool = use_attention_pool
        self.use_cnn_domain_head = use_cnn_domain_head
        self.cnn_domain_weight = float(cnn_domain_weight)
        self.apply_model_pre_align_only = apply_model_pre_align_only

        self.cnn = CNNBlock(
            in_channels=num_channels,
            out_channels=cnn_out_channels,
            dropout=cnn_dropout,
            temporal_kernels=temporal_kernels,
            multiscale_preserve_capacity=multiscale_preserve_capacity,
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

        if use_attention_pool:
            self.attn_pool = AttentionPool(embedding_dim=embedding_dim, dropout=dropout)
            if learnable_attention_mix:
                mix_init = float(max(0.0, min(1.0, attention_mix_init)))
                self.attention_mix = nn.Parameter(torch.tensor([mix_init]))
            else:
                self.register_parameter("attention_mix", None)
                self.attention_mix_fixed = float(max(0.0, min(1.0, attention_mix_init)))
        else:
            self.attn_pool = None
            self.register_parameter("attention_mix", None)

        self.task_head = TaskHead(embedding_dim, num_classes)
        self.grl = GRL()
        self.domain_head = DomainHead(
            embedding_dim,
            num_subjects,
            hidden_dim=domain_head_hidden_dim,
            num_layers=domain_head_layers,
            dropout=domain_head_dropout,
        )
        if use_cnn_domain_head:
            self.cnn_domain_head = DomainHead(
                cnn_out_channels,
                num_subjects,
                hidden_dim=max(cnn_out_channels, domain_head_hidden_dim),
                num_layers=max(1, domain_head_layers),
                dropout=domain_head_dropout,
            )
        else:
            self.cnn_domain_head = None

    def forward(self, x: torch.Tensor, lambda_: float = 0.0) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, T)
            lambda_: GRL scale

        Returns:
            dict with task/domain logits.
        """
        if self.apply_model_euclidean_alignment and not self.apply_model_pre_align_only:
            x = euclidean_alignment(x)
        if self.apply_model_riemannian_reweight and not self.apply_model_pre_align_only:
            x = riemannian_reweight(x)

        x = x.unsqueeze(1)  # (B, 1, C, T)
        x = self.cnn(x)  # (B, F, 1, T')

        cnn_domain_output = None
        if self.cnn_domain_head is not None:
            cnn_feat = x.squeeze(2).mean(dim=-1)  # (B, F)
            cnn_domain_input = self.grl(cnn_feat, lambda_)
            cnn_domain_output = self.cnn_domain_head(cnn_domain_input)

        x = self.tokenizer(x)  # (B, N, D)

        sequence, cls_token = self.vit(x)

        if self.attn_pool is not None:
            pooled = self.attn_pool(sequence[:, 1:, :])
            if self.attention_mix is not None:
                alpha = torch.sigmoid(self.attention_mix).view(1, 1)
                task_feature = alpha * cls_token + (1.0 - alpha) * pooled
            else:
                alpha = self.attention_mix_fixed
                task_feature = alpha * cls_token + (1.0 - alpha) * pooled
        else:
            task_feature = cls_token

        task_output = self.task_head(task_feature)
        domain_input = self.grl(task_feature, lambda_)
        domain_output = self.domain_head(domain_input)

        output: dict[str, torch.Tensor] = {
            "task": task_output,
            "domain": domain_output,
        }
        if cnn_domain_output is not None:
            output["domain_cnn"] = cnn_domain_output
        return output
