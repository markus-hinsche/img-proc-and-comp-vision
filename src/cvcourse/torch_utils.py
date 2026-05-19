"""Small PyTorch utility helpers used across the course notebooks."""

from __future__ import annotations

import torch
import torch.nn as nn


def get_device() -> torch.device:
    """Return the best available device. CPU is always a valid fallback."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_params(module: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters in a module."""
    params = module.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
