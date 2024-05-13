"""Define convolutional encoder-decoder models.

Impletemtations
---------------
- Variational Autoencoder (VAE)
- Categorical Autoencoder (CategoricalAE)
"""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING

import torch
import wandb
from distribution_extension import (
    MultiOneHot,
    MultiOneHotFactory,
    Normal,
    NormalFactory,
)
from lightning import LightningModule
from torch import Tensor

from cnn.loss import likelihood
from cnn.network import Decoder, DecoderConfig, Encoder, EncoderConfig

if TYPE_CHECKING:
    from cnn.custom_types import DataGroup, LossDict


class AE(LightningModule):
    """Base class for autoencoders."""

    def shared_step(self, batch: DataGroup) -> LossDict:
        """Return the loss of the batch."""
        raise NotImplementedError

    def training_step(self, batch: DataGroup, **_: int) -> LossDict:
        """Rollout training step."""
        loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict

    def validation_step(self, batch: DataGroup, _: int) -> LossDict:
        """Rollout validation step."""
        loss_dict = self.shared_step(batch)
        loss_dict = {"val_" + k: v for k, v in loss_dict.items()}
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict

    @classmethod
    def load_from_wandb(cls, reference: str) -> VAE:
        """Load the model from wandb checkpoint."""
        run = wandb.Api().run(reference)  # type: ignore[no-untyped-call]
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_name, cpu = "best_model.ckpt", torch.device("cpu")
            ckpt = run.file(ckpt_name).download(replace=True, root=tmpdir)
            model = cls.load_from_checkpoint(ckpt.name, map_location=cpu)
        if isinstance(model, VAE):
            return model
        msg = f"Model type not supported: {type(model)}"
        raise ValueError(msg)


class VAE(AE):
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

    def encode(self, observations: Tensor) -> Normal:
        """Encode the observations and return the latent distribution."""
        obs_embed = self.encoder(observations)
        return self.distribution_factory.forward(obs_embed)

    def decode(self, obs_embed: Tensor) -> Tensor:
        """Decode the latent representation."""
        return self.decoder.forward(obs_embed)

    def shared_step(self, batch: DataGroup) -> LossDict:
        """Return the loss of the batch."""
        inputs, targets = batch
        distribution = self.encode(inputs)
        obs_embed = distribution.rsample()
        reconstruction = self.decode(obs_embed)
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


class CategoricalAE(AE):
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
        self.encoder = Encoder(encoder_config)
        self.decoder = Decoder(decoder_config)
        self.class_size = class_size
        self.category_size = category_size
        self.distribution_factory = MultiOneHotFactory(
            category_size=category_size,
            class_size=class_size,
        )

    def encode(self, observation: Tensor) -> MultiOneHot:
        """Encode the observation and return the latent distribution."""
        obs_embed = self.encoder.forward(observation)
        return self.distribution_factory.forward(obs_embed)

    def decode(self, obs_embed: Tensor) -> Tensor:
        """Decode the latent representation."""
        return self.decoder.forward(obs_embed)

    def shared_step(self, batch: DataGroup) -> LossDict:
        """Return the loss of the batch."""
        inputs, targets = batch
        distribution = self.encode(inputs)
        obs_embed = distribution.rsample()
        reconstruction = self.decode(obs_embed)
        recon_loss = torch.nn.functional.mse_loss(
            input=reconstruction,
            target=targets,
            reduction="mean",
        )
        return {"loss": recon_loss}
