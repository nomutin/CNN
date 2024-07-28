
"""Networks."""

from einops import pack, unpack
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from cnn.config import DecoderConfig
from cnn.utils import ResidualBlock


class Decoder(nn.Module):
    """Image Decoder."""

    def __init__(self, config: DecoderConfig) -> None:
        """Set hyperparameters."""
        super().__init__()
        self.config = config
        self.linear = self.build_linear()
        self.rearrange = Rearrange(
            pattern="b (c h w) -> b c h w",
            c=self.config.conv_in_shape[0],
            h=self.config.conv_in_shape[1],
            w=self.config.conv_in_shape[2],
        )
        self.res_block = self.build_res_block()
        self.conv = self.build_conv()

    def build_linear(self) -> nn.Sequential:
        """Build the linear layers."""
        linear_list: list[nn.Module] = []
        if 0 not in self.config.linear_sizes:
            for linear_size in self.config.linear_sizes:
                linear_list += [nn.LazyLinear(linear_size)]
                linear_list += [self.config.activation()]
        return nn.Sequential(*linear_list)

    def build_res_block(self) -> nn.Sequential:
        """Build the residual blocks."""
        res_block_list: list[nn.Module] = []
        if self.config.num_residual_blocks:
            conv = nn.Conv2d(
                self.config.conv_in_shape[0],
                self.config.residual_input_size,
                kernel_size=3,
                padding=1,
            )
            res_block_list += [conv]
            for _ in range(self.config.num_residual_blocks):
                res_block = ResidualBlock(
                    io=self.config.residual_input_size,
                    intermediate=self.config.residual_intermediate_size,
                    activation_name=self.config.activation_name,
                )
                res_block_list += [res_block]
        return nn.Sequential(*res_block_list)

    def build_conv(self) -> nn.Sequential:
        """Build the convolutional layers."""
        conv_list: list[nn.Module] = []
        for out_channels, kernel_size, stride, padding, output_padding in zip(
            self.config.channels,
            self.config.kernel_sizes,
            self.config.strides,
            self.config.paddings,
            self.config.output_paddings,
            strict=False,
        ):
            conv = nn.LazyConvTranspose2d(
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            )
            norm = nn.BatchNorm2d(out_channels)
            conv_list += [conv, norm, self.config.activation()]
        conv_list[-2] = nn.Identity()
        conv_list[-1] = self.config.out_activation()
        return nn.Sequential(*conv_list)

    def forward(self, features: Tensor) -> Tensor:
        """Reconstruct observation(s) from features."""
        features, ps = pack([features], "* d")
        feature = self.linear(features)
        feature_map = self.rearrange(feature)
        feature_map = self.res_block(feature_map)
        reconstructions = self.conv(feature_map)
        return unpack(reconstructions, ps, "* c h w")[0]
