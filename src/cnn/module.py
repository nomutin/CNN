"""
Define convolutional encoder-decoder models.

Impletemtations
---------------
- Variational Autoencoder (VAE)
"""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass


import lightning
import torch.distributions as td
import torch.nn.functional as tf
from torch import Tensor, optim

from cnn.loss import kl_between_standart_normal, likelihood
from cnn.network import Decoder, Encoder


@dataclass
class TestParams:
    v1: int
    v2: int


class VAE(lightning.LightningModule):
    """Variational Auto Encoder."""

    def __init__(  # noqa: PLR0913
        self,
        testparams: TestParams,
        encoder_linear_sizes: tuple[int, ...] = (128, 64),
        encoder_activation_name: str = "ELU",
        encoder_out_activation_name: str = "Identity",
        encoder_channels: tuple[int, ...] = (8, 16, 32),
        encoder_kernel_sizes: tuple[int, ...] = (3, 3, 3),
        encoder_strides: tuple[int, ...] = (2, 2, 2),
        encoder_paddings: tuple[int, ...] = (1, 1, 1),
        decoder_linear_sizes: tuple[int, ...] = (32, 128),
        decoder_activation_name: str = "ELU",
        decoder_out_activation_name: str = "Sigmoid",
        decoder_channels: tuple[int, ...] = (32, 16, 8),
        decoder_kernel_sizes: tuple[int, ...] = (4, 4, 4),
        decoder_strides: tuple[int, ...] = (2, 2, 2),
        decoder_paddings: tuple[int, ...] = (1, 1, 1),
        decoder_output_paddings: tuple[int, ...] = (0, 0, 0),
        observation_shape: tuple[int, int, int] = (3, 64, 64),
    ) -> None:
        """Set networks."""
        super().__init__()
        self.save_hyperparameters()
        print(testparams)
        self.encoder = Encoder(
            linear_sizes=encoder_linear_sizes,
            activation_name=encoder_activation_name,
            out_activation_name=encoder_out_activation_name,
            channels=encoder_channels,
            kernel_sizes=encoder_kernel_sizes,
            strides=encoder_strides,
            paddings=encoder_paddings,
            observation_shape=observation_shape,
        )
        self.decoder = Decoder(
            linear_sizes=decoder_linear_sizes,
            activation_name=decoder_activation_name,
            out_activation_name=decoder_out_activation_name,
            channels=decoder_channels,
            kernel_sizes=decoder_kernel_sizes,
            strides=decoder_strides,
            paddings=decoder_paddings,
            output_paddings=decoder_output_paddings,
            observation_shape=observation_shape,
        )

    def configure_optimizers(self) -> optim.Optimizer:
        """Choose what optimizers to use."""
        return optim.AdamW(self.parameters(), lr=1e-3)

    def encode(self, observations: Tensor) -> dict[str, Any]:
        """
        Encode images into latent.

        Returns
        -------
        dict:
            Includes k:v below.
            - mean: Tensor
            - obs_embed: Tensor
            - distribution: torch.distributions.Independent
        """
        obs_embed = self.encoder(observations)
        mean, std = obs_embed.chunk(2, dim=1)
        std = tf.softplus(std)
        distribution = td.Independent(td.Normal(mean, std), 1)
        sample = distribution.rsample()
        return {
            "mean": mean,
            "obs_embed": sample,
            "distribution": distribution,
        }

    def decode(self, obs_embed: Tensor) -> dict[str, Any]:
        """
        Reconstruct images from latent.

        Returns
        -------
        dict:
            Includes k:v below.
            - reconstruction: Tensor
        """
        return {"reconstruction": self.decoder(obs_embed)}

    def _shared_step(self, batch: list[Tensor]) -> dict[str, Tensor]:
        """Return the loss of the batch."""
        inputs, targets = batch
        encoder_output = self.encode(inputs)
        distribution = encoder_output["distribution"]
        obs_embed = encoder_output["obs_embed"]
        reconstruction = self.decode(obs_embed)["reconstruction"]

        recon_loss = likelihood(
            prediction=reconstruction,
            target=targets,
            event_ndims=3,
        )
        kl_loss = kl_between_standart_normal(distribution)
        return {
            "loss": recon_loss + kl_loss,
            "reconstruction": recon_loss,
            "kl divergence": kl_loss,
        }

    def training_step(self, batch: list, **_: dict) -> dict[str, Tensor]:
        """Rollout training step."""
        loss_dict = self._shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict

    def validation_step(self, batch: list, _: int) -> dict[str, Tensor]:
        """Rollout validation step."""
        loss_dict = self._shared_step(batch)
        loss_dict = {"val_" + k: v for k, v in loss_dict.items()}
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict
