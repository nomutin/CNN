"""Datamodule that reads local `.pt` files."""

from __future__ import annotations

from pathlib import Path

import lightning
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose


class CNNDataset(Dataset):
    """Observation pytorch dataset."""

    def __init__(self, path_to_data: Path, transforms: Compose) -> None:
        """Initialize `PlayDataset` ."""
        super().__init__()
        self.path_to_data = path_to_data
        self.transforms = transforms
        self.data_size = len(list(self.path_to_data.glob("observation_*.pt")))

    def __len__(self) -> int:
        """Return the number of data."""
        return self.data_size

    def load_data(self, idx: int) -> Tensor:
        """Load observation data (sequence)."""
        observation_path = self.path_to_data / f"observation_{idx}.pt"
        return torch.load(observation_path)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Apply transforms to and return input & target Tensors."""
        data = self.load_data(idx)
        return self.transforms(data), data


class CNNDataModule(lightning.LightningDataModule):
    """Observation lightning datamodule."""

    def __init__(
        self,
        data_name: str,
        batch_size: int,
        num_workers: int,
        transforms: Compose,
    ) -> None:
        """Initialize variables."""
        super().__init__()
        self.data_name = data_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms

        self.path_to_train = Path("data") / data_name / "train"
        self.path_to_val = Path("data") / data_name / "validation"

    def setup(self, stage: str = "train") -> None:  # noqa: ARG002
        """Set up train/val/test dataset."""
        self.train_dataset = CNNDataset(
            path_to_data=self.path_to_train,
            transforms=self.transforms,
        )
        self.val_dataset = CNNDataset(
            path_to_data=self.path_to_val,
            transforms=Compose([]),
        )

    def train_dataloader(self) -> DataLoader:
        """Define training dataloader."""
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Define validation dataloader."""
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.val_dataset.data_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
