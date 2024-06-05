"""Tests for `utils.py`."""

from cnn.utils import calc_conv_out_size, calc_convt_out_size, pairwise


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
