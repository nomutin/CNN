"""Self-contained programs that works on the Hook."""

from __future__ import annotations

from typing import TYPE_CHECKING

import lightning
from lightning.pytorch.loggers import WandbLogger

from cnn.module import VAE
from cnn.visualize import pca, to_wandb_images, to_wandb_scatter

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import STEP_OUTPUT
    from torch import Tensor


class LogCNNOutput(lightning.Callback):
    """Callback to log results in wandb."""

    def __init__(self, every_n_epochs: int, indices: int) -> None:
        """Set parameters."""
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.indices = indices

    def on_validation_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: STEP_OUTPUT,  # noqa: ARG002
        batch: list[Tensor],
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:
        """Log observation/reconstruction/latent to wandb."""
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        if not isinstance(trainer.logger, WandbLogger):
            return
        if not isinstance(pl_module, VAE):
            return

        inputs, targets = batch
        distribution = pl_module.encode(inputs)
        obs_embed = distribution.rsample()
        recon = pl_module.decode(obs_embed)
        obs_embed_pca, variance_ratio = pca(obs_embed)
        latent_fig = to_wandb_scatter(
            data=obs_embed_pca.detach().cpu(),
            x_label=f"PC1({variance_ratio[0]:.2f})",
            y_label=f"PC2({variance_ratio[1]:.2f})",
        )
        trainer.logger.experiment.log(
            {
                "image inputs": to_wandb_images(targets[self.indices]),
                "reconstruction images": to_wandb_images(recon[self.indices]),
                "latent pca": latent_fig,
            },
        )
