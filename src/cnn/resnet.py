"""
Resnet-based encoder and decoders.

Modified from: https://github.com/hirahirashun
"""

import timm
import torch
from einops import pack, unpack
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from cnn.config import DecoderConfig, EncoderConfig
from cnn.utils import get_activation


class PretrainerEncoder(nn.Module):
    """
    事前学習済みモデル(`efficientnet_b0`)を使うエンコーダ.

    Parameters
    ----------
    config : EncoderConfig
        エンコーダの設定.
        .linear_sizes[-1] のみが使用される.
    """

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            model_name="efficientnet_b0",
            pretrained=True,
            num_classes=0,
        )
        in_features = self.encoder.num_features

        self.mlp = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Mish(),
            nn.Linear(128, config.linear_sizes[-1]),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        順伝播.

        Parameters
        ----------
        x : Tensor
            画像データ. Shape: (*, C, H, W).

        Returns
        -------
        Tensor
            エンコードされた画像データ. Shape: (*, D).

        """
        x, ps = pack([x], "* C H W")
        x = self.encoder(x)
        x = self.mlp(x)
        return unpack(x, ps, "* D")[0]


class ResidualBlock(nn.Module):
    """Residual block."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        順伝播.

        Parameters
        ----------
        x : Tensor
            入力データ. Shape: (B, C, H, W).

        Returns
        -------
        Tensor
            出力データ. Shape: (B, C, H, W).
        """
        return x + self.conv_block.forward(x)


class ResNetDecoder(nn.Module):
    """
    ResNet-based decoder.

    Parameters
    ----------
    config : DecoderConfig
        Decoder の設定.
    """

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        channels, height, width = config.observation_shape

        self.mlp = nn.Sequential(
            nn.Linear(config.linear_sizes[0], 256),
            nn.ELU(),
            nn.Linear(256, height // 4 * width // 4),
            Rearrange("B (H W) -> B 1 H W", H=height // 4, W=width // 4),
        )

        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU(),
        )

        # Residual blocks
        res_blocks = [ResidualBlock(64) for _ in range(config.depth)]
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
        )

        # Upsampling layers
        upsampling = []
        for _ in range(2):
            upsampling += [
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, channels, kernel_size=9, stride=1, padding=4),
            get_activation(config.out_activation_name)(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        順伝播.

        Parameters
        ----------
        x : Tensor
            入力データ. Shape: (*, D).

        Returns
        -------
        Tensor
            出力データ. Shape: (*, C, H, W).
        """
        x, ps = pack([x], "* D")

        x = self.mlp(x)

        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)

        out = self.upsampling(out)
        out = self.conv3(out)

        return unpack(out, ps, "* C H W")[0]  # type: ignore[no-any-return]
