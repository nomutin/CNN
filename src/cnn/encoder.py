"""Networks."""

from torch import Tensor, nn
from torchgeometry.contrib import SpatialSoftArgmax2d

from cnn.config import EncoderConfig
from cnn.utils import CoordConv2d, ResidualBlock, packdim


class Encoder(nn.Module):
    """
    Observation Encoder.

    Parameters
    ----------
    config : EncoderConfig
        Encoder configuration.

    Example
    -------
    >>> import torch
    >>> encoder = Encoder(EncoderConfig())
    >>> features = encoder.forward(torch.randn(1, 3, 64, 64))
    >>> features.shape
    torch.Size([1, 128])
    >>> features = encoder.training_step(torch.randn(1, 3, 64, 64))
    >>> features.shape
    torch.Size([1, 128])
    """

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.pre_conv = self.build_pre_conv()
        self.conv = self.build_conv()
        self.res_block = self.build_res_block()
        self.post_conv = self.build_post_conv()
        self.flatten = nn.Flatten()
        self.linear = self.build_linear()

    def build_pre_conv(self) -> nn.Module:
        """
        Build the pre-convolutional layers.

        Returns
        -------
        nn.Module
            First layers of the encoder.
            Coordinate Convolution is added if `config.coord_conv` is True.
        """
        pre_conv_list: list[nn.Module] = []
        if self.config.coord_conv:
            pre_conv_list += [CoordConv2d()]
        return nn.Sequential(*pre_conv_list)

    def build_conv(self) -> nn.Sequential:
        """
        Build the convolutional layers.

        Returns
        -------
        nn.Sequential
            The convolutional layers.
            Input shape : Any
        """
        conv_list: list[nn.Module] = []
        for out_channels, kernel_size, stride, padding in zip(
            self.config.channels,
            self.config.kernel_sizes,
            self.config.strides,
            self.config.paddings,
            strict=False,
        ):
            conv = nn.LazyConv2d(
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            norm = nn.BatchNorm2d(out_channels)
            conv_list += [conv, norm, self.config.activation()]
        return nn.Sequential(*conv_list)

    def build_res_block(self) -> nn.Sequential:
        """
        Build the residual blocks.

        Returns
        -------
        nn.Sequential
            The residual blocks.
            Input : [config.channels[-1], H, W]
            Output : [config.residual_output_size, H, W]
        """
        res_block_list: list[nn.Module] = []
        if self.config.num_residual_blocks:
            for _ in range(self.config.num_residual_blocks):
                res_block = ResidualBlock(
                    io=self.config.channels[-1],
                    intermediate=self.config.residual_intermediate_size,
                    activation_name=self.config.activation_name,
                )
                res_block_list += [res_block]
            out_conv = nn.Conv2d(
                self.config.channels[-1],
                self.config.residual_output_size,
                kernel_size=3,
                padding=1,
            )
            res_block_list += [out_conv]
        return nn.Sequential(*res_block_list)

    def build_post_conv(self) -> nn.Module:
        """
        Build the post-convolutional layers.

        Returns
        -------
        nn.Module
            Last layers of the encoder.
            Spatial Softmax is added if `config.spatial_softmax` is True.
        """
        post_conv_list: list[nn.Module] = []
        if self.config.spatial_softmax:
            post_conv_list += [SpatialSoftArgmax2d()]
        return nn.Sequential(*post_conv_list)

    def build_linear(self) -> nn.Sequential:
        """
        Build the linear layers.

        Returns
        -------
        nn.Sequential
            The linear layers.
            Input : [Any]
            Output : [config.linear_sizes[-1]]
        """
        linear_list: list[nn.Module] = []
        if 0 not in self.config.linear_sizes:
            for linear_size in self.config.linear_sizes:
                linear_list += [nn.LazyLinear(linear_size)]
                linear_list += [self.config.activation()]
            linear_list[-1] = self.config.out_activation()
        return nn.Sequential(*linear_list)

    def training_step(self, observations: Tensor) -> Tensor:
        """
        Encode observation(s) into features.

        Same as `forward` method.

        Parameters
        ----------
        observations : Tensor
            Observation(s) to encode. [*B, C, H, W]

        Returns
        -------
        Tensor
            Encoded features. [*B, D]
        """
        return self.forward(observations)

    @packdim(in_pattern="* c h w", out_pattern="* d")
    def forward(self, observations: Tensor) -> Tensor:
        """
        Encode observation(s) into features.

        Parameters
        ----------
        observations : Tensor
            Observation(s) to encode. [*B, C, H, W]

        Returns
        -------
        Tensor
            Encoded features. [*B, D]
        """
        feature_map = self.pre_conv(observations)
        feature_map = self.conv(feature_map)
        feature_map = self.res_block(feature_map)
        feature_map = self.post_conv(feature_map)
        feature = self.flatten(feature_map)
        return self.linear(feature)  # type: ignore[no-any-return]
