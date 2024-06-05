"""Utility functions."""

from __future__ import annotations

import math
from itertools import tee

import torch
from einops import repeat
from torch import Tensor, arange, nn


def pairwise(iterable: tuple[int, ...]) -> list[tuple[int, int]]:
    """S -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return list(zip(a, b))


def calc_conv_out_size(
    in_size: int,
    padding: int,
    kernel_size: int,
    stride: int,
) -> int:
    """
    Calculate the output size of a convolutional layer.

    References
    ----------
    * https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    """
    return math.floor((in_size + 2 * padding - kernel_size) / stride + 1)


def calc_convt_out_size(
    in_size: int,
    padding: int,
    kernel_size: int,
    stride: int,
    output_padding: int,
) -> int:
    """
    Calculate the output size of a convolutional layer.

    References
    ----------
    * https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html

    """
    size = (in_size + 2 * padding - kernel_size - output_padding) / stride
    return math.floor(size) + 1


def coord_conv(x: Tensor) -> Tensor:
    """
    Add coordinate channels to the input tensor.

    References
    ----------
    * https://github.com/Wizaron/coord-conv-pytorch

    Parameters
    ----------
    x : Tensor
        Batched image tensor. shape=[B, C, H, W]

    Returns
    -------
    Tensor
        Batched image tensor with coordinate channels. shape=[B, C + 2, H, W]
    """
    b, _, h, w = x.size()
    y_coords = repeat(arange(h), "h -> h w", w=w).mul(2).div(h - 1).sub(1)
    x_coords = repeat(arange(w), "w -> h w", h=h).mul(2).div(w - 1).sub(1)
    coords = torch.stack((y_coords, x_coords), dim=0).to(x.device)
    coords = repeat(coords, "C H W -> B C H W", B=b)
    return torch.cat((coords, x), dim=1)


def get_activation(activation_name: str) -> type[nn.Module]:
    """Get activation function from its name."""
    if issubclass(m := getattr(nn, activation_name), nn.Module):
        return m
    msg = f"Activation function not found: {activation_name}"
    raise AttributeError(msg)
