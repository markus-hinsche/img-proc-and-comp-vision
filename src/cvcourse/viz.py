"""Visualization helpers used throughout the course.

Everything here assumes PyTorch tensors in CHW (or NCHW) layout, values typically
in [0, 1]. We keep matplotlib usage minimal and explicit so students can see
exactly how images get from tensors to pixels on screen.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_displayable(img: torch.Tensor) -> np.ndarray:
    """Convert a CHW (or HW) tensor to an HWC numpy array suitable for imshow."""
    if img.ndim == 2:
        return img.detach().cpu().numpy()
    if img.ndim == 3:
        c = img.shape[0]
        arr = img.detach().cpu().numpy()
        if c == 1:
            return arr[0]
        return np.transpose(arr, (1, 2, 0))
    raise ValueError(f"Expected a 2D or 3D tensor, got shape {tuple(img.shape)}")


def show_image(img: torch.Tensor, title: str | None = None, ax=None, cmap: str = "gray"):
    """Show a single image tensor (CHW or HW)."""
    arr = _to_displayable(img)
    own_axis = ax is None
    if own_axis:
        _, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(arr, cmap=cmap if arr.ndim == 2 else None)
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title)
    if own_axis:
        plt.show()


def show_grid(
    images: Sequence[torch.Tensor] | torch.Tensor,
    titles: Iterable[str] | None = None,
    cols: int = 8,
    figsize_per_cell: float = 1.5,
    cmap: str = "gray",
):
    """Show a list of images (or an NCHW batch) on a grid."""
    if isinstance(images, torch.Tensor) and images.ndim == 4:
        images = [images[i] for i in range(images.shape[0])]
    images = list(images)
    n = len(images)
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * figsize_per_cell, rows * figsize_per_cell)
    )
    axes = np.atleast_2d(axes).reshape(rows, cols)
    titles = list(titles) if titles is not None else [None] * n
    for i, ax in enumerate(axes.flat):
        if i < n:
            show_image(images[i], title=titles[i], ax=ax, cmap=cmap)
        else:
            ax.set_axis_off()
    fig.tight_layout()
    plt.show()


def show_kernels(weight: torch.Tensor, cols: int = 8, title: str | None = None):
    """Show conv-layer kernels.

    weight shape is (out_channels, in_channels, kH, kW). For visualization we
    average over the input-channel dimension so each output channel becomes
    one image.
    """
    if weight.ndim != 4:
        raise ValueError(f"Expected 4D conv weight, got shape {tuple(weight.shape)}")
    w = weight.detach().cpu()
    w = w.mean(dim=1)  # (out, kH, kW)
    # Normalize each kernel independently for display contrast.
    w_min = w.amin(dim=(-1, -2), keepdim=True)
    w_max = w.amax(dim=(-1, -2), keepdim=True)
    w_norm = (w - w_min) / (w_max - w_min + 1e-8)
    show_grid([w_norm[i] for i in range(w_norm.shape[0])], cols=cols, cmap="viridis")
    if title:
        plt.suptitle(title)


def show_feature_maps(maps: torch.Tensor, cols: int = 8, max_maps: int = 16):
    """Show feature maps from a conv layer.

    maps shape: either (C, H, W) for a single image, or (N, C, H, W) — in the
    second case we take the first image of the batch.
    """
    if maps.ndim == 4:
        maps = maps[0]
    if maps.ndim != 3:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(maps.shape)}")
    maps = maps.detach().cpu()
    n = min(maps.shape[0], max_maps)
    show_grid([maps[i] for i in range(n)], cols=cols, cmap="viridis")
