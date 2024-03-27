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
from distribution_extension import (
    MultiDimentionalOneHotCategoricalFactory,
    NormalFactory,
)
from torch import Tensor
from torch import distributions as td

from cnn.network import Decoder, Encoder


class VAE(lightning.LightningModule):
    """Variational Auto Encoder."""

    def __init__(
        self,
        encoder_output_size: int,
        decoder_input_size: int,
        num_layers: int,
        channels: int,
    ) -> None:
        """Set networks."""
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(
            obs_embed_size=encoder_output_size,
            num_layers=num_layers,
            channels=channels,
        )
        self.decoder = Decoder(
            feature_size=decoder_input_size,
            num_layers=num_layers,
            channels=channels,
        )
        self.distribution_factory = NormalFactory()

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

        dist = td.Independent(td.Normal(reconstruction, 1.0), 3)
        recon_loss = -dist.log_prob(targets).mean()
        kl_loss = distribution.kl_divergence_starndard_normal()
        return {
            "loss": recon_loss + kl_loss,
            "reconstruction": recon_loss,
            "kl divergence": kl_loss,
        }

    def training_step(
        self,
        batch: list[Tensor],
        **_: dict,
    ) -> dict[str, Tensor]:
        """Rollout training step."""
        loss_dict = self._shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict

    def validation_step(
        self,
        batch: list[Tensor],
        _: int,
    ) -> dict[str, Tensor]:
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
        encoder_output_size: int,
        num_layers: int,
        channels: int,
        class_size: int,
        category_size: int,
    ) -> None:
        """Set networks."""
        super().__init__(
            encoder_output_size=encoder_output_size,
            decoder_input_size=class_size * category_size,
            num_layers=num_layers,
            channels=channels,
        )
        self.save_hyperparameters(class_size, category_size)
        self.distribution_factory = MultiDimentionalOneHotCategoricalFactory(
            category_size=category_size,
            class_size=class_size,
        )

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
