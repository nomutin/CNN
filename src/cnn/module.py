"""
Define convolutional encoder-decoder models.

Impletemtations
---------------
- Variational Autoencoder (VAE)
- Categorical Autoencoder (CategoricalAE)
"""

from __future__ import annotations

import tempfile
from typing import Any

import lightning
import torch
import wandb
from distribution_extention import (
    MultiDimentionalOneHotCategoricalFactory,
    NormalFactory,
)
from torch import Tensor, optim

from cnn.loss import likelihood
from cnn.network import Decoder, DecoderConfig, Encoder, EncoderConfig


class VAE(lightning.LightningModule):
    """Variational Auto Encoder."""

    def __init__(
        self,
        encoder_config: EncoderConfig,
        decoder_config: DecoderConfig,
    ) -> None:
        """Set networks."""
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(encoder_config)
        self.decoder = Decoder(decoder_config)
        self.distribution_factory = NormalFactory()

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
            - obs_embed: Tensor
            - distribution: torch.distributions.Independent
        """
        obs_embed = self.encoder(observations)
        distribution = self.distribution_factory(obs_embed)
        sample = distribution.rsample()
        return {
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
        kl_loss = distribution.kl_divergence_starndard_normal()
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

    @classmethod
    def load_from_wandb(cls, reference: str) -> VAE:
        """Load the model from wandb checkpoint."""
        run = wandb.Api().run(reference)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_name, cpu = "best_model.ckpt", torch.device("cpu")
            ckpt = run.file(ckpt_name).download(replace=True, root=tmpdir)
            return cls.load_from_checkpoint(ckpt.name, map_location=cpu)


class CategoricalAE(VAE):
    """Categorical Auto Encoder."""

    def __init__(
        self,
        encoder_config: EncoderConfig,
        decoder_config: DecoderConfig,
        class_size: int,
        category_size: int,
    ) -> None:
        """Set networks."""
        super().__init__(encoder_config, decoder_config)
        self.save_hyperparameters(class_size, category_size)
        self.class_size = class_size
        self.category_size = category_size
        self.distribution_factory = MultiDimentionalOneHotCategoricalFactory(
            category_size=category_size,
            class_size=class_size,
        )

    def encode(self, observation: Tensor) -> dict[str, Any]:
        """
        Encode images into latent.

        Returns
        -------
        dict:
            Includes k:v below.
            - logit: Tensor
            - obs_embed: Tensor
            - distribution: torch.distributions.Independent
        """
        obs_embed = self.encoder.forward(observation)
        distribution = self.distribution_factory(obs_embed)
        sample = distribution.rsample()
        return {
            "logit": obs_embed,
            "obs_embed": sample,
            "distribution": distribution,
        }

    def _shared_step(self, batch: list[Tensor]) -> dict[str, Tensor]:
        """Return the loss of the batch."""
        inputs, targets = batch
        encoder_output = self.encode(inputs)
        obs_embed = encoder_output["obs_embed"]
        reconstruction = self.decode(obs_embed)["reconstruction"]
        recon_loss = torch.nn.functional.mse_loss(
            input=reconstruction,
            target=targets,
            reduction="mean",
        )
        return {"loss": recon_loss}
