"""Tests for `utils.py`."""

from torch import Tensor

from cnn.utils import (
    CoordConv2d,
    calc_conv_out_size,
    calc_convt_out_size,
    get_activation,
    pairwise,
)
from tests.conftest import batch_size, channels, height, width


def test__pairwise() -> None:
    """Test `pairwise`."""
    inputs = (1, 2, 3, 4)
    targets = [(1, 2), (2, 3), (3, 4)]
    assert pairwise(inputs) == targets


def test__calc_conv_out_size() -> None:
    """Test `calc_conv_out_size`."""
    in_size = 64
    expected = 32
    out_size = calc_conv_out_size(
        in_size=in_size,
        padding=1,
        kernel_size=3,
        stride=2,
    )
    assert out_size == expected


def test__calc_convt_out_size() -> None:
    """Test `calc_convt_out_size`."""
    in_size = 32
    expected = 16
    out_size = calc_convt_out_size(
        in_size=in_size,
        padding=1,
        kernel_size=4,
        stride=2,
        output_padding=0,
    )
    assert out_size == expected


def test__coord_conv_2d(sample_4d_data: Tensor) -> None:
    """Test `coord_conv`."""
    out = CoordConv2d()(sample_4d_data)
    assert out.shape == (batch_size, channels + 2, height, width)


def test__get_activation() -> None:
    """Test `get_activation`."""
    activation_name = "ReLU"
    activation = get_activation(activation_name)
    assert isinstance(activation, type)
    assert activation.__name__ == activation_name
    assert activation.__module__ == "torch.nn.modules.activation"
