"""LightningModule for autoencoder models."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import wandb
from lightning import LightningModule
from torch import Tensor
from typing_extensions import Self

from cnn.config import DecoderConfig, EncoderConfig
from cnn.network import Decoder, Encoder


class ObservationModule(LightningModule):
    """Base class for autoencoder models."""

    def __init__(
        self,
        encoder_config: EncoderConfig,
        decoder_config: DecoderConfig,
    ) -> None:
        """Set Hyperparameters."""
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(encoder_config)
        self.decoder = Decoder(decoder_config)

    def shared_step(self, batch: tuple[Tensor, ...]) -> dict[str, Tensor]:
        """Shared training/validation step."""
        inputs, targets = batch
        z = self.encoder(inputs)
        reconstructions = self.decoder(z)
        loss = (reconstructions - targets).abs().mean()
        return {"loss": loss}

    def training_step(self, batch: tuple[Tensor, ...], **_: str) -> dict[str, Tensor]:
        """Rollout training step."""
        loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict

    def validation_step(self, batch: tuple[Tensor, ...], _: int) -> dict[str, Tensor]:
        """Rollout validation step."""
        loss_dict = self.shared_step(batch)
        loss_dict = {"val_" + k: v for k, v in loss_dict.items()}
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict

    @classmethod
    def load_from_wandb(cls, reference: str) -> Self:
        """
        Load the model from wandb checkpoint.

        Parameters
        ----------
        reference : str
            The reference to the wandb artifact.

        Returns
        -------
        Self
            The model loaded from the wandb artifact.

        Raises
        ------
        TypeError
            If the loaded model is not an instance of the class.
        """
        run = wandb.Api().artifact(reference)
        with TemporaryDirectory() as tmpdir:
            ckpt = Path(run.download(root=tmpdir))
            model = cls.load_from_checkpoint(
                checkpoint_path=ckpt / "model.ckpt",
                map_location=torch.device("cpu"),
            )
        if not isinstance(model, cls):
            msg = f"Model is not an instance of {cls}"
            raise TypeError(msg)
        return model
