"""Encoder/Decoder."""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING

from einops import pack, unpack
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from torchgeometry.contrib import SpatialSoftArgmax2d

from cnn.utils import (
    CoordConv2d,
    calc_conv_out_size,
    calc_convt_out_size,
    pairwise,
)

if TYPE_CHECKING:
    from cnn.config import DecoderConfig, EncoderConfig


class Encoder(nn.Module):
    """Image Encoder."""

    def __init__(self, config: EncoderConfig) -> None:
        """Set Hyperparameters."""
        super().__init__()
        self.config = config
        self.model = self.build()

    def build(self) -> nn.Sequential:
        """Build the model as `nn.Sequential`."""
        seq: list[nn.Module] = []

        if self.config.coord_conv:
            seq += [CoordConv2d()]

        channels = (self.config.observation_shape[0], *self.config.channels)
        for channel_io, kernel_size, stride, padding in zip(
            pairwise(channels),
            self.config.kernel_sizes,
            self.config.strides,
            self.config.paddings,
        ):
            conv = nn.Conv2d(
                in_channels=channel_io[0],
                out_channels=channel_io[1],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            norm = nn.BatchNorm2d(channel_io[1])
            seq += [conv, norm, self.config.activation()]

        if self.config.spatial_softmax:
            seq += [SpatialSoftArgmax2d()]

        seq += [nn.Flatten()]
        linear_sizes = (prod(self.conv_out_shape), *self.config.linear_sizes)
        for linear_size_pair in pairwise(linear_sizes):
            seq += [nn.Linear(*linear_size_pair), self.config.activation()]

        seq[-1] = self.config.out_activation()

        return nn.Sequential(*seq)

    @property
    def conv_out_shape(self) -> tuple[int, int, int]:
        """Return the output shape of the convolutional layers."""
        _, h, w = self.config.observation_shape
        for kernel_size, stride, padding in zip(
            self.config.kernel_sizes,
            self.config.strides,
            self.config.paddings,
        ):
            h = calc_conv_out_size(h, padding, kernel_size, stride)
            w = calc_conv_out_size(w, padding, kernel_size, stride)

        if self.config.spatial_softmax:
            return self.config.channels[-1], 2, 1

        return self.config.channels[-1], h, w

    def forward(self, observations: Tensor) -> Tensor:
        """Encode observation(s) into features."""
        observations, ps = pack([observations], "* c h w")
        feature: Tensor = self.model.forward(observations)
        return unpack(feature, ps, "* d")[0]


class Decoder(nn.Module):
    """Image Decoder."""

    def __init__(self, config: DecoderConfig) -> None:
        """Set hyperparameters."""
        super().__init__()
        self.config = config
        self.model = self.build()

    def build(self) -> nn.Sequential:
        """Build the model as `nn.Sequential`."""
        seq: list[nn.Module] = []

        linear_sizes = (*self.config.linear_sizes, prod(self.convt_out_shape))
        for linear_size_pair in pairwise(linear_sizes):
            seq += [nn.Linear(*linear_size_pair), self.config.activation()]

        rearrange = Rearrange(
            "b (c h w) -> b c h w",
            c=self.convt_out_shape[0],
            h=self.convt_out_shape[1],
            w=self.convt_out_shape[2],
        )
        seq += [rearrange]

        channels = (*self.config.channels, self.config.observation_shape[0])
        for channel_io, kernel_size, stride, padding, output_padding in zip(
            pairwise(channels),
            self.config.kernel_sizes,
            self.config.strides,
            self.config.paddings,
            self.config.output_paddings,
        ):
            conv = nn.ConvTranspose2d(
                in_channels=channel_io[0],
                out_channels=channel_io[1],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            )
            norm = nn.BatchNorm2d(channel_io[1])
            seq += [conv, norm, self.config.activation()]

        seq[-2] = nn.Identity()
        seq[-1] = self.config.out_activation()
        return nn.Sequential(*seq)

    @property
    def convt_out_shape(self) -> tuple[int, int, int]:
        """Return the output shape of the convolutional layers."""
        _, h, w = self.config.observation_shape
        for kernel_size, stride, padding, out_pad in zip(
            reversed(self.config.kernel_sizes),
            reversed(self.config.strides),
            reversed(self.config.paddings),
            reversed(self.config.output_paddings),
        ):
            h = calc_convt_out_size(h, padding, kernel_size, stride, out_pad)
            w = calc_convt_out_size(w, padding, kernel_size, stride, out_pad)
        return self.config.channels[0], h, w

    def forward(self, features: Tensor) -> Tensor:
        """Reconstruct observation(s) from features."""
        features, ps = pack([features], "* d")
        reconstruction = self.model(features)
        return unpack(reconstruction, ps, "* c h w")[0]
