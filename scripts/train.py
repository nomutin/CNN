"""Script to train a model."""

from __future__ import annotations

from pathlib import Path

import click
import lightning
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf


@click.command()
@click.argument("train_config", type=str)
@click.option("--dev", is_flag=True, default=False)
def train(train_config: str, dev: bool) -> None:  # noqa: FBT001
    """Train a model."""
    lightning.seed_everything(42)
    path = Path(train_config)
    config = OmegaConf.load(path)
    OmegaConf.resolve(config)
    model = instantiate(config.model)
    logger = WandbLogger(project=path.stem, save_dir="./wandb/")
    log_model = ModelCheckpoint(
        dirpath=logger.experiment.dir,
        filename="best_model",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    callbacks = [instantiate(config=cb) for cb in config.callbacks]
    callbacks.append(log_model)
    trainer = instantiate(
        config=config.trainer,
        logger=logger,
        fast_dev_run=dev,
        callbacks=callbacks,
    )
    datamodule = instantiate(config=config.datamodule)
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    train()
