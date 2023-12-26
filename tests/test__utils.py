"""Tests for `utils.py`."""

from cnn.utils import (
    ConvNetworkParameter,
    calc_conv_out_size,
    calc_convt_out_size,
    pairwise,
)


def test__pairwise() -> None:
    """Test `pairwise`."""
    inputs = [1, 2, 3, 4]
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


class TestConvNetWorkParameter:
    """Test for `ConvNetWorkParameter`."""

    def test__conv_out_shape(self) -> None:
        """Test `conv_out_shape`."""
        params = ConvNetworkParameter(
            linear_sizes=(0, 0),
            activation_name="ReLU",
            out_activation_name="Sigmoid",
            channels=(32, 64, 128),
            kernel_sizes=(3, 3, 3),
            strides=(2, 2, 2),
            paddings=(1, 1, 1),
            output_paddings=(0, 0, 0),
            observation_shape=(3, 64, 64),
        )
        assert params.conv_out_shape == (128, 8, 8)

    def test__convt_out_shape(self) -> None:
        """Test `convt_out_shape`."""
        params = ConvNetworkParameter(
            linear_sizes=(0, 0),
            activation_name="ReLU",
            out_activation_name="Sigmoid",
            channels=(128, 64, 32),
            kernel_sizes=(4, 4, 4),
            strides=(2, 2, 2),
            paddings=(1, 1, 1),
            output_paddings=(0, 0, 0),
            observation_shape=(3, 64, 64),
        )
        assert params.convt_out_shape == (128, 8, 8)
