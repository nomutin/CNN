"""Data visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from einops import pack, unpack
from torch import Tensor

from wandb import Image


def to_wandb_images(images_tensor: Tensor) -> list[Image]:
    """Convert images tensor to wandb.Image."""
    return [Image(image) for image in images_tensor.detach().cpu()]


def to_wandb_scatter(
    data: Tensor,
    x_label: str,
    y_label: str,
) -> Image:
    """Visualize 2D data."""
    fig, axe = plt.subplots(figsize=(8, 8), tight_layout=True)
    axe.scatter(data[:, 0], data[:, 1], alpha=0.5)
    axe.set_xlabel(xlabel=x_label)
    axe.set_ylabel(ylabel=y_label)
    wandb_image = Image(fig)
    plt.close(fig)
    return wandb_image


def pca(data: Tensor, n_components: int = 2) -> tuple[Tensor, Tensor]:
    """
    Apply PCA on 2D+ Tensor.

    References
    ----------
    * https://pytorch.org/docs/stable/generated/torch.pca_lowrank.html

    Returns
    -------
    Tensor
        PCA-transformed data. Tensor shaped [batch*, n_components].
    Tensor
        Explained variance ratio. Tensor shaped [n_components].
    """
    data, ps = pack([data], "* d")
    _, s, v = torch.pca_lowrank(data, q=n_components)
    [data_pca] = unpack(torch.matmul(data, v), ps, "* d")
    ratio = (s**2) / (data.shape[0] - 1) / data.var(dim=0).sum()
    return data_pca, ratio
