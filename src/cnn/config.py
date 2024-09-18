"""全ての Encoder/Decoder 共通で使用する設定."""

from dataclasses import dataclass

from cnn.utils import get_activation


@dataclass
class EncoderConfig:
    """Encoder configuration."""

    linear_sizes: tuple[int, ...] = (256, 128)
    activation_name: str = "ReLU"
    out_activation_name: str = "Identity"
    channels: tuple[int, ...] = (16, 32, 64)
    kernel_sizes: tuple[int, ...] = (3, 3, 3)
    strides: tuple[int, ...] = (2, 2, 2)
    paddings: tuple[int, ...] = (1, 1, 1)
    num_residual_blocks: int = 3
    residual_intermediate_size: int = 128
    residual_output_size: int = 64
    coord_conv: bool = False
    spatial_softmax: bool = False

    def __post_init__(self) -> None:
        """Make a non-tuple Iterable attributes into tuples."""
        self.linear_sizes = tuple(self.linear_sizes)
        self.channels = tuple(self.channels)
        self.kernel_sizes = tuple(self.kernel_sizes)
        self.strides = tuple(self.strides)
        self.paddings = tuple(self.paddings)
        self.activation = get_activation(self.activation_name)
        self.out_activation = get_activation(self.out_activation_name)


@dataclass
class DecoderConfig:
    """Decoder configuration."""

    linear_sizes: tuple[int, ...] = (128, 512)
    activation_name: str = "ReLU"
    out_activation_name: str = "Sigmoid"
    channels: tuple[int, ...] = (32, 16, 3)
    kernel_sizes: tuple[int, ...] = (4, 4, 4)
    strides: tuple[int, ...] = (2, 2, 2)
    paddings: tuple[int, ...] = (1, 1, 1)
    output_paddings: tuple[int, ...] = (0, 0, 0)
    conv_in_shape: tuple[int, ...] = (8, 8, 8)
    num_residual_blocks: int = 3
    residual_intermediate_size: int = 128
    residual_input_size: int = 64

    def __post_init__(self) -> None:
        """Make a non-tuple Iterable attributes into tuples."""
        self.linear_sizes = tuple(self.linear_sizes)
        self.channels = tuple(self.channels)
        self.kernel_sizes = tuple(self.kernel_sizes)
        self.strides = tuple(self.strides)
        self.paddings = tuple(self.paddings)
        self.output_paddings = tuple(self.output_paddings)
        self.conv_in_shape = tuple(self.conv_in_shape)
        self.activation = get_activation(self.activation_name)
        self.out_activation = get_activation(self.out_activation_name)
