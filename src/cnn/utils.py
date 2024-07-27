"""Utility functions."""

import torch
from einops import repeat
from torch import Tensor, arange, nn


class ResidualBlock(nn.Module):
    """Residual block."""

    def __init__(
        self,
        io: int,
        intermediate: int,
        activation_name: str,
    ) -> None:
        super().__init__()
        self.activation = get_activation(activation_name)()
        self.skip = nn.Sequential(
            nn.Conv2d(io, intermediate, kernel_size=1),
            nn.BatchNorm2d(intermediate),
            self.activation,
            nn.Conv2d(intermediate, intermediate, kernel_size=3, padding=1),
            nn.BatchNorm2d(intermediate),
            self.activation,
            nn.Conv2d(intermediate, io, kernel_size=1),
            nn.BatchNorm2d(io),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        順伝播.

        Parameters
        ----------
        x : Tensor
            入力データ. Shape: (B, C, H, W).

        Returns
        -------
        Tensor
            出力データ. Shape: (B, C, H, W).
        """
        identity = x.clone()
        x = self.skip(x)
        x += identity
        return self.activation(x)


class CoordConv2d(nn.Module):
    """
    Add coordinate channels to the input tensor.

    References
    ----------
    * https://github.com/Wizaron/coord-conv-pytorch
    """

    def forward(self, x: Tensor) -> Tensor:  # noqa: PLR6301
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Batched image tensor. shape=[B, C, H, W]

        Returns
        -------
        Tensor
            Batched image tensor with coordinate channels.
            shape=[B, C + 2, H, W]
        """
        b, _, h, w = x.size()
        y_coords = repeat(arange(h), "h -> h w", w=w).mul(2).div(h - 1).sub(1)
        x_coords = repeat(arange(w), "w -> h w", h=h).mul(2).div(w - 1).sub(1)
        coords = torch.stack((y_coords, x_coords), dim=0).to(x.device)
        coords = repeat(coords, "C H W -> B C H W", B=b)
        return torch.cat((coords, x), dim=1)


def get_activation(activation_name: str) -> type[nn.Module]:
    """
    Get activation function from its name.

    Parameters
    ----------
    activation_name : str
        Activation function name.

    Returns
    -------
    type[nn.Module]
        Activation function.

    Raises
    ------
    AttributeError
        If the activation function is not found in `torch.nn`.
    """
    if issubclass(m := getattr(nn, activation_name), nn.Module):
        return m
    msg = f"Activation function not found: {activation_name}"
    raise AttributeError(msg)
