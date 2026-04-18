from __future__ import annotations

import math

import torch


def fit_euclidean_alignment(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Estimate a global whitening matrix from calibration data.

    Args:
        x: (N, C, T) calibration signals.
        eps: Numerical stability term for tiny eigenvalues.

    Returns:
        Whitening matrix of shape (C, C).
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x to have shape (N, C, T), got {tuple(x.shape)}")

    _, _, t = x.shape
    cov = torch.matmul(x, x.transpose(-1, -2)) / max(t, 1)  # (N, C, C)
    mean_cov = cov.mean(dim=0)  # (C, C)

    eigvals, eigvecs = torch.linalg.eigh(mean_cov)
    eigvals = torch.clamp(eigvals, min=eps)
    inv_sqrt = torch.diag(torch.rsqrt(eigvals))
    whitening = eigvecs @ inv_sqrt @ eigvecs.transpose(-1, -2)
    return whitening


def apply_euclidean_alignment(x: torch.Tensor, whitening: torch.Tensor) -> torch.Tensor:
    """
    Apply a precomputed Euclidean-alignment whitening matrix.

    Args:
        x: (N, C, T) signals to transform.
        whitening: (C, C) whitening matrix from fit_euclidean_alignment.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x to have shape (N, C, T), got {tuple(x.shape)}")
    if whitening.ndim != 2:
        raise ValueError(
            f"Expected whitening to have shape (C, C), got {tuple(whitening.shape)}"
        )

    return torch.matmul(whitening.unsqueeze(0), x)


def euclidean_alignment(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Batch-wise Euclidean Alignment (whitening with mean covariance).

    Args:
        x: (B, C, T)
    """
    whitening = fit_euclidean_alignment(x, eps=eps)
    return apply_euclidean_alignment(x, whitening)


def riemannian_reweight(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Riemannian-inspired covariance reweighting.

    Args:
        x: (B, C, T)
    """
    cov = torch.matmul(x, x.transpose(-1, -2))  # (B, C, C)
    fro = torch.linalg.norm(cov, ord="fro", dim=(-2, -1), keepdim=True)
    cov_norm = cov / (fro + eps)
    return torch.matmul(cov_norm, x)


def lambda_scheduler(
    epoch: int,
    max_epoch: int,
    gamma: float = 10.0,
) -> float:
    """DANN-style lambda schedule."""
    denom = max(max_epoch, 1)
    p = float(epoch) / float(denom)
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0
