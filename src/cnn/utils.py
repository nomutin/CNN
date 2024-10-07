"""Utility functions."""

from collections.abc import Callable
from typing import Any

import torch
from einops import pack, repeat, unpack
from torch import Tensor, arange, nn


class ResidualBlock(nn.Module):
    """
    Residual block.

    Parameters
    ----------
    io : int
        Input/output channels.
    intermediate : int
        Intermediate channels.
    activation_name : str
        Activation function name.

    Examples
    --------
    >>> import torch
    >>> block = ResidualBlock(16, 32, "ReLU")
    >>> output = block.forward(torch.randn(1, 16, 8, 8))
    >>> output.shape
    torch.Size([1, 16, 8, 8])
    """

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
        return self.activation(x)  # type: ignore[no-any-return]


class CoordConv2d(nn.Module):
    """
    Add coordinate channels to the input tensor.

    References
    ----------
    * https://github.com/Wizaron/coord-conv-pytorch

    Examples
    --------
    >>> import torch
    >>> coord_conv = CoordConv2d()
    >>> input_tensor = torch.randn(1, 3, 32, 32)
    >>> output_tensor = coord_conv.forward(input_tensor)
    >>> output_tensor.shape
    torch.Size([1, 5, 32, 32])
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
        return m  # type: ignore[no-any-return]
    msg = f"Activation function not found: {activation_name}"
    raise AttributeError(msg)


ForwardLike = Callable[[Any, Tensor], Tensor]


def packdim(in_pattern: str, out_pattern: str) -> Callable[[ForwardLike], ForwardLike]:
    """
    Perform packing/unpacking for functions with one argument and one return value as Tensor.

    Parameters
    ----------
    in_pattern : str
        einops packing pattern of input.
    out_pattern : str
        einops packing pattern of output.

    Returns
    -------
    Callable[[ForwardLike], ForwardLike]
        Decorator.

    Examples
    --------
    >>> from torch import nn, randn, Tensor
    >>> class Model(nn.Flatten):
    ...     @packdim(in_pattern="* c h w", out_pattern="* d")
    ...     def forward(self, x: Tensor) -> Tensor:
    ...         return super().forward(x)
    >>> Model().forward(randn(4, 10, 3, 8, 8)).shape
    torch.Size([4, 10, 192])
    """

    def decorator(forward_func: ForwardLike) -> ForwardLike:
        def decorated_forward(self_: nn.Module, input_tensor: Tensor) -> Tensor:
            input_tensor, ps = pack([input_tensor], in_pattern)
            output_tensor = forward_func(self_, input_tensor)
            return unpack(output_tensor, ps, out_pattern)[0]

        return decorated_forward

    return decorator
