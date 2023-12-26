"""Encoder/Decoder."""


from __future__ import annotations

from math import prod

from einops import pack, unpack
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from cnn.utils import calc_conv_out_size, calc_convt_out_size, pairwise


class Encoder(nn.Module):
    """Image Encoder."""

    def __init__(  # noqa: PLR0913
        self,
        linear_sizes: tuple[int, ...],
        activation_name: str,
        out_activation_name: str,
        channels: tuple[int, ...],
        kernel_sizes: tuple[int, ...],
        strides: tuple[int, ...],
        paddings: tuple[int, ...],
        observation_shape: tuple[int, int, int],
    ) -> None:
        """Set Hyperparameters."""
        super().__init__()
        self.linear_sizes = linear_sizes
        self.activation_name = activation_name
        self.out_activation_name = out_activation_name
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.observation_shape = observation_shape

        self.model = self.build()

    def build(self) -> nn.Sequential:
        """Build the model as `nn.Sequential`."""
        seq: list[nn.Module] = []

        channels = (self.observation_shape[0], *self.channels)
        for channel_io, kernel_size, stride, padding in zip(
            pairwise(channels),
            self.kernel_sizes,
            self.strides,
            self.paddings,
        ):
            conv = nn.Conv2d(
                in_channels=channel_io[0],
                out_channels=channel_io[1],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            norm = nn.BatchNorm2d(channel_io[1])
            seq += [conv, norm, self.activation()]

        seq += [nn.Flatten()]
        linear_sizes = (prod(self.conv_out_shape), *self.linear_sizes)
        for linear_size_pair in pairwise(linear_sizes):
            seq += [nn.Linear(*linear_size_pair), self.activation()]

        seq[-1] = self.out_activation()

        return nn.Sequential(*seq)

    @property
    def conv_out_shape(self) -> tuple[int, int, int]:
        """Return the output shape of the convolutional layers."""
        _, h, w = self.observation_shape
        for kernel_size, stride, padding in zip(
            self.kernel_sizes,
            self.strides,
            self.paddings,
        ):
            h = calc_conv_out_size(h, padding, kernel_size, stride)
            w = calc_conv_out_size(w, padding, kernel_size, stride)
        return self.channels[-1], h, w

    @property
    def activation(self) -> nn.Module:
        """Return the activation function."""
        return getattr(nn, self.activation_name)

    @property
    def out_activation(self) -> nn.Module:
        """Return the activation function."""
        return getattr(nn, self.out_activation_name)

    def forward(self, observations: Tensor) -> Tensor:
        """Encode observation(s) into features."""
        observations, ps = pack([observations], "* c h w")
        feature = self.model.forward(observations)
        return unpack(feature, ps, "* d")[0]


class Decoder(nn.Module):
    """Image Decoder."""

    def __init__(  # noqa: PLR0913
        self,
        linear_sizes: tuple[int, ...],
        activation_name: str,
        out_activation_name: str,
        channels: tuple[int, ...],
        kernel_sizes: tuple[int, ...],
        strides: tuple[int, ...],
        paddings: tuple[int, ...],
        output_paddings: tuple[int, ...],
        observation_shape: tuple[int, int, int],
    ) -> None:
        """Set hyperparameters."""
        super().__init__()
        self.linear_sizes = linear_sizes
        self.activation_name = activation_name
        self.out_activation_name = out_activation_name
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.output_paddings = output_paddings
        self.observation_shape = observation_shape

        self.model = self.build()

    def build(self) -> nn.Sequential:
        """Build the model as `nn.Sequential`."""
        seq: list[nn.Module] = []

        linear_sizes = (*self.linear_sizes, prod(self.convt_out_shape))
        for linear_size_pair in pairwise(linear_sizes):
            seq += [nn.Linear(*linear_size_pair), self.activation()]

        rearrange = Rearrange(
            "b (c h w) -> b c h w",
            c=self.convt_out_shape[0],
            h=self.convt_out_shape[1],
            w=self.convt_out_shape[2],
        )
        seq += [rearrange]

        channels = (*self.channels, self.observation_shape[0])
        for channel_io, kernel_size, stride, padding, output_padding in zip(
            pairwise(channels),
            self.kernel_sizes,
            self.strides,
            self.paddings,
            self.output_paddings,
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
            seq += [conv, norm, self.activation()]

        seq[-2] = nn.Identity()
        seq[-1] = self.out_activation()
        return nn.Sequential(*seq)

    @property
    def convt_out_shape(self) -> tuple[int, int, int]:
        """Return the output shape of the convolutional layers."""
        _, h, w = self.observation_shape
        for kernel_size, stride, padding, out_pad in zip(
            reversed(self.kernel_sizes),
            reversed(self.strides),
            reversed(self.paddings),
            reversed(self.output_paddings),
        ):
            h = calc_convt_out_size(h, padding, kernel_size, stride, out_pad)
            w = calc_convt_out_size(w, padding, kernel_size, stride, out_pad)
        return self.channels[0], h, w

    @property
    def activation(self) -> nn.Module:
        """Return the activation function."""
        return getattr(nn, self.activation_name)

    @property
    def out_activation(self) -> nn.Module:
        """Return the activation function."""
        return getattr(nn, self.out_activation_name)

    def forward(self, features: Tensor) -> Tensor:
        """Reconstruct observation(s) from features."""
        features, ps = pack([features], "* d")
        reconstruction = self.model.forward(features)
        return unpack(reconstruction, ps, "* c h w")[0]
