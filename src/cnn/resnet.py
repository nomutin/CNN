"""
Resnet-based encoder and decoders.

Modified from: https://github.com/hirahirashun
"""

import timm
import torch
from einops import pack, unpack
from einops.layers.torch import Rearrange
from torch import Tensor, nn


class PretrainerEncoder(nn.Module):
    """
    事前学習済みモデルを使うエンコーダ.

    Parameters
    ----------
    backbone_name : str
        事前学習済みモデル名.
    obs_embed_size : int
        画像をエンコードするサイズ.
    """

    def __init__(self, backbone_name: str, obs_embed_size: int) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
        )
        in_features = self.encoder.num_features

        self.mlp = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Mish(),
            nn.Linear(128, obs_embed_size),
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
        return x + self.conv_block(x)


class ResNetDecoder(nn.Module):
    """
    ResNet-based decoder.

    Parameters
    ----------
    n_residual_blocks : int
        Residual blockの数.
    embed_size : int
        入力の次元.
    height : int
        再構成する画像の高さ.
    width : int
        再構成する画像の幅.
    """

    def __init__(
        self,
        n_residual_blocks: int = 16,
        embed_size: int = 16,
        height: int = 128,
        width: int = 128,
    ) -> None:
        super().__init__()

        out_channels = 3

        self.mlp = nn.Sequential(
            nn.Linear(embed_size, 256),
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
        res_blocks = [ResidualBlock(64) for _ in range(n_residual_blocks)]
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
            nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4),
            nn.Sigmoid(),
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

        return unpack(out, ps, "* C H W")[0]
