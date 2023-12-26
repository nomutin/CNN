"""Utility functions."""

from __future__ import annotations

import math
from itertools import tee
from typing import Iterable


def pairwise(iterable: Iterable) -> list:
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
