"""全ての Encoder/Decoder 共通で使用する設定."""

from dataclasses import dataclass

from cnn.utils import get_activation


@dataclass
class EncoderConfig:
    """Encoder configuration."""

    linear_sizes: tuple[int, ...]
    activation_name: str
    out_activation_name: str
    channels: tuple[int, ...]
    kernel_sizes: tuple[int, ...]
    strides: tuple[int, ...]
    paddings: tuple[int, ...]
    observation_shape: tuple[int, ...]
    num_residual_blocks: int
    residual_intermediate_size: int
    residual_output_size: int
    coord_conv: bool = False
    vector_quantize: bool = False
    spatial_softmax: bool = False

    def __post_init__(self) -> None:
        """Make a non-tuple Iterable attributes into tuples."""
        self.linear_sizes = tuple(self.linear_sizes)
        self.channels = tuple(self.channels)
        self.kernel_sizes = tuple(self.kernel_sizes)
        self.strides = tuple(self.strides)
        self.paddings = tuple(self.paddings)
        self.observation_shape = tuple(self.observation_shape)
        self.activation = get_activation(self.activation_name)
        self.out_activation = get_activation(self.out_activation_name)


@dataclass
class DecoderConfig:
    """Decoder configuration."""

    linear_sizes: tuple[int, ...]
    activation_name: str
    out_activation_name: str
    channels: tuple[int, ...]
    kernel_sizes: tuple[int, ...]
    strides: tuple[int, ...]
    paddings: tuple[int, ...]
    output_paddings: tuple[int, ...]
    observation_shape: tuple[int, ...]
    conv_in_shape: tuple[int, ...]
    num_residual_blocks: int
    residual_intermediate_size: int
    residual_input_size: int

    def __post_init__(self) -> None:
        """Make a non-tuple Iterable attributes into tuples."""
        self.linear_sizes = tuple(self.linear_sizes)
        self.channels = tuple(self.channels)
        self.kernel_sizes = tuple(self.kernel_sizes)
        self.strides = tuple(self.strides)
        self.paddings = tuple(self.paddings)
        self.output_paddings = tuple(self.output_paddings)
        self.observation_shape = tuple(self.observation_shape)
        self.conv_in_shape = tuple(self.conv_in_shape)
        self.activation = get_activation(self.activation_name)
        self.out_activation = get_activation(self.out_activation_name)
