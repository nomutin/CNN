"""Encoder/Decoder."""

from __future__ import annotations

from einops import pack, unpack
from torch import Tensor, nn
from torchrl.modules import ObsDecoder, ObsEncoder


class Encoder(ObsEncoder):
    def __init__(
        self,
        obs_embed_size: int,
        channels: int = 32,
        num_layers: int = 4,
    ) -> None:
        super().__init__(channels, num_layers)
        self.encoder[0] = nn.Conv2d(
            in_channels=3,
            out_channels=channels,
            kernel_size=4,
            stride=2,
        )
        self.embed_to_latent = nn.Linear(channels * 32, obs_embed_size)


class Decoder(ObsDecoder):
    def __init__(
        self,
        feature_size: int,
        channels: int = 32,
        num_layers: int = 4,
    ) -> None:
        super().__init__(channels, num_layers)
        self.state_to_latent = nn.Linear(feature_size, channels * 32)
        self.decoder[0] = nn.ConvTranspose2d(
            in_channels=feature_size,
            out_channels=channels * num_layers,
            kernel_size=5,
            stride=2,
        )

    def forward(self, latent: Tensor) -> Tensor:
        features, ps = pack([latent], "* d")
        reconstruction = self.model.forward(features)
        return unpack(reconstruction, ps, "* c h w")[0]
