"""A from-scratch residual CNN encoder with a separate CUB-200 head.

No pretrained parameters exist anywhere in this module: construction always
starts from the caller's torch RNG (the Task 1.2 baseline required by the
project specification). Seed torch before construction for reproducible
initialization. The wrapper mirrors the pretrained adapter contract —
``forward_features`` before the head and a Grad-CAM target layer — so one
training loop can serve both model families. Input is the project's prepared
224 fit transform; normalization is a caller decision applied by the data
pipeline, not the model.
"""

from dataclasses import dataclass
import math

from torch import Tensor, nn


_ARCHITECTURE = "scratch_resnet"
_STEM_WIDTH = 32
_STAGE_WIDTHS = (64, 128, 256, 256)
_STAGE_BLOCKS = (2, 2, 2, 2)
_LEAKY_SLOPE = 0.1
_ACTIVATIONS = {"relu", "leaky_relu"}


def _make_activation(name: str) -> nn.Module:
    """Return a fresh activation layer; both options are plain torch layers."""
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=_LEAKY_SLOPE, inplace=True)
    return nn.ReLU(inplace=True)


@dataclass(frozen=True)
class ScratchConfig:
    """Validated immutable construction settings for the from-scratch baseline."""

    num_classes: int = 200
    head_dropout: float = 0.0
    activation: str = "relu"  # "relu" or "leaky_relu"

    def __post_init__(self) -> None:
        if type(self.num_classes) is not int or self.num_classes != 200:
            raise ValueError("num_classes must be the integer 200")
        if (type(self.head_dropout) not in (int, float)
                or not math.isfinite(self.head_dropout)
                or not 0 <= self.head_dropout < 1):
            raise ValueError("head_dropout must be a finite number in [0, 1)")
        if self.activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}")


@dataclass(frozen=True)
class ScratchSpec:
    """Serializable architecture metadata, independent of random parameters.

    ``encoder_channels`` is the spatial/pooled channel count; ``output_stride``
    is 32, matching the pretrained encoders' geometry. There is deliberately no
    weight id or native preprocessing: no pretrained checkpoint exists for this
    architecture.
    """

    architecture: str
    encoder_channels: int
    output_stride: int
    gradcam_target_path: str


def get_scratch_spec() -> ScratchSpec:
    """Resolve the fixed architecture metadata locally, without model construction."""
    return ScratchSpec(
        architecture=_ARCHITECTURE,
        encoder_channels=_STAGE_WIDTHS[-1],
        output_stride=32,
        gradcam_target_path="encoder.stages.3",
    )


def _init_conv(conv: nn.Conv2d, activation: str = "relu") -> None:
    # He initialization matched to the activation that follows the convolution.
    nn.init.kaiming_normal_(conv.weight, a=_LEAKY_SLOPE if activation == "leaky_relu" else 0,
                            mode="fan_out", nonlinearity=activation)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


class _BasicBlock(nn.Module):
    """Two 3x3 convolutions with a residual add; identity or 1x1 projection skip."""

    def __init__(self, in_channels: int, channels: int, stride: int, activation: str = "relu") -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = _make_activation(activation)
        self.downsample = None
        if stride != 1 or in_channels != channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        _init_conv(self.conv1, activation)
        _init_conv(self.conv2, activation)
        if self.downsample is not None:
            _init_conv(self.downsample[0], activation)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.act(out + identity)


class ScratchEncoder(nn.Module):
    """A stem, four residual stages, and the final spatial feature map."""

    def __init__(self, stem_width: int, stage_widths: tuple[int, ...], stage_blocks: tuple[int, ...],
                 activation: str = "relu") -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_width),
            _make_activation(activation),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        _init_conv(self.stem[0], activation)
        stages = []
        in_channels = stem_width
        for stage_index, (width, blocks) in enumerate(zip(stage_widths, stage_blocks)):
            stride = 1 if stage_index == 0 else 2
            modules = [_BasicBlock(in_channels, width, stride=stride, activation=activation)]
            modules.extend(_BasicBlock(width, width, stride=1, activation=activation)
                           for _ in range(blocks - 1))
            stages.append(nn.Sequential(*modules))
            in_channels = width
        self.stages = nn.Sequential(*stages)

    def forward(self, images: Tensor) -> Tensor:
        return self.stages(self.stem(images))


class ScratchClassifier(nn.Module):
    """A fully trainable spatial encoder, global average pooling, and a 200-logit head.

    Every parameter is trained from random initialization, so there is no
    fine-tuning policy to apply. CAM callers may still set
    ``images.requires_grad_(True)`` and read ``gradcam_target_layer`` output.
    """

    def __init__(self, config: ScratchConfig) -> None:
        super().__init__()
        self.config = config
        self.spec = get_scratch_spec()
        self.encoder = ScratchEncoder(_STEM_WIDTH, _STAGE_WIDTHS, _STAGE_BLOCKS, config.activation)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(config.head_dropout),
            nn.Linear(_STAGE_WIDTHS[-1], config.num_classes),
        )
        nn.init.normal_(self.head[1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.head[1].bias)

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """The final spatial stage, resolved without duplicate module registration."""
        return self.get_submodule(self.spec.gradcam_target_path)

    def forward_features(self, images: Tensor) -> dict[str, Tensor]:
        """Return spatial NCHW features and pooled embeddings before head dropout."""
        spatial = self.encoder(images)
        pooled = self.pool(spatial).flatten(1)
        return {"spatial": spatial, "pooled": pooled}

    def forward(self, images: Tensor) -> Tensor:
        return self.head(self.forward_features(images)["pooled"])


def create_scratch_model(config: ScratchConfig) -> ScratchClassifier:
    """Construct the from-scratch baseline; initialization uses the caller's torch RNG."""
    if not isinstance(config, ScratchConfig):
        raise TypeError("config must be a ScratchConfig")
    return ScratchClassifier(config)
