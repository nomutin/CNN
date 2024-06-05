"""テストで共通で使うメソッドの定義."""

import pytest
import torch
from torch import Tensor

batch_size = 4
seq_len = 8
channels, height, width = 3, 32, 32


@pytest.fixture()
def sample_4d_data() -> Tensor:
    """Create a sample 4D(batched image) tensor."""
    return torch.rand(batch_size, channels, height, width)


@pytest.fixture()
def sample_5d_data() -> Tensor:
    """Create a sample 5D(batched sequence of images) tensor."""
    return torch.rand(batch_size, seq_len, channels, height, width)
