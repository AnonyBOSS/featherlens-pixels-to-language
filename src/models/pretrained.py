"""Torchvision CNN encoders with explicit weights and a separate CUB-200 head.

Construction loads the pinned ImageNet weights by default. Pass
``load_weights=False`` explicitly for offline construction with random parameters.
Native preprocessing is metadata; it does not change the project's data transforms.
"""

from collections import OrderedDict
from dataclasses import dataclass, replace
import math

from torch import Tensor, nn
from torchvision import models

from .finetuning import (
    apply_parameter_policy,
    apply_training_policy,
    validate_fine_tuning,
)


_WEIGHT_IDS = {
    "convnext_tiny": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
    "efficientnet_b3": "EfficientNet_B3_Weights.IMAGENET1K_V1",
    "regnet_y_3_2gf": "RegNet_Y_3_2GF_Weights.IMAGENET1K_V2",
}
_ENCODER_CHANNELS = {"convnext_tiny": 768, "efficientnet_b3": 1536, "regnet_y_3_2gf": 1512}
_PARTIAL_PATHS = {
    "convnext_tiny": ("6", "7"),
    "efficientnet_b3": ("7", "8"),
    "regnet_y_3_2gf": ("trunk_output.block4",),
}


def _weight_id(architecture: str) -> str:
    if not isinstance(architecture, str) or architecture not in _WEIGHT_IDS:
        raise ValueError(f"architecture must be one of {tuple(_WEIGHT_IDS)}")
    return _WEIGHT_IDS[architecture]


def _weight_enum(architecture: str):
    enum_name, member = _weight_id(architecture).split(".")
    return getattr(getattr(models, enum_name), member)


@dataclass(frozen=True)
class ModelConfig:
    """Validated immutable construction settings; ``None`` selects the pinned weight ID."""

    architecture: str
    weights: str | None = None
    num_classes: int = 200
    fine_tuning: str = "frozen"
    head_dropout: float = 0.0
    freeze_bn_stats: bool = True

    def __post_init__(self) -> None:
        pinned = _weight_id(self.architecture)
        if self.weights is not None and (not isinstance(self.weights, str) or self.weights != pinned):
            raise ValueError(f"weights for {self.architecture} must be {pinned!r} or None")
        object.__setattr__(self, "weights", pinned)
        if type(self.num_classes) is not int or self.num_classes != 200:
            raise ValueError("num_classes must be the integer 200")
        validate_fine_tuning(self.fine_tuning, self.freeze_bn_stats)
        if (type(self.head_dropout) not in (int, float)
                or not math.isfinite(self.head_dropout)
                or not 0 <= self.head_dropout < 1):
            raise ValueError("head_dropout must be a finite number in [0, 1)")


@dataclass(frozen=True)
class ModelSpec:
    """Serializable architecture metadata, independent of any loaded parameters.

    ``encoder_channels`` is the spatial/pooled channel count; ``output_stride``
    is 32 for these encoders. ``gradcam_target_path`` resolves on the wrapper.
    The native recipe describes Torchvision's weight preset, not baseline input.
    """

    architecture: str
    weight_id: str
    native_preprocessing: dict
    encoder_channels: int
    output_stride: int
    gradcam_target_path: str


def get_model_spec(architecture: str) -> ModelSpec:
    """Resolve the pinned weight recipe locally, without model construction or I/O."""
    weight_id = _weight_id(architecture)
    transforms = _weight_enum(architecture).transforms()
    return ModelSpec(
        architecture=architecture,
        weight_id=weight_id,
        native_preprocessing={
            "resize_size": list(transforms.resize_size),
            "crop_size": list(transforms.crop_size),
            "interpolation": transforms.interpolation.value,
            "mean": list(transforms.mean),
            "std": list(transforms.std),
            "antialias": transforms.antialias,
        },
        encoder_channels=_ENCODER_CHANNELS[architecture],
        output_stride=32,
        gradcam_target_path="encoder." + _PARTIAL_PATHS[architecture][-1],
    )


class PretrainedClassifier(nn.Module):
    """A real spatial encoder, native pooling/normalization, and a fresh 200-logit head.

    Fine-tuning transitions update ``config`` to reflect the current policy. Frozen
    encoders retain autograd when inputs require gradients, so CAM callers may use
    ``images.requires_grad_(True)`` even in the fully frozen policy.
    """

    def __init__(self, config: ModelConfig, native: nn.Module, *, weights_loaded: bool) -> None:
        super().__init__()
        self.config = config
        self.spec = get_model_spec(config.architecture)
        self.configured_weight_id = self.spec.weight_id
        self.weights_loaded = weights_loaded
        self.loaded_weight_id = self.spec.weight_id if weights_loaded else None
        self.pool = native.avgpool
        if config.architecture == "convnext_tiny":
            self.encoder = native.features
            self.encoder_norm = native.classifier[0]
            channels = native.classifier[2].in_features
        elif config.architecture == "efficientnet_b3":
            self.encoder = native.features
            self.encoder_norm = nn.Identity()
            channels = native.classifier[1].in_features
        else:
            self.encoder = nn.Sequential(OrderedDict(stem=native.stem, trunk_output=native.trunk_output))
            self.encoder_norm = nn.Identity()
            channels = native.fc.in_features
        if channels != self.spec.encoder_channels:
            raise ValueError("Torchvision encoder dimensions differ from the supported model specification")
        self.head = nn.Sequential(nn.Dropout(config.head_dropout), nn.Linear(channels, config.num_classes))
        self.set_fine_tuning(config.fine_tuning)

    def _partial_modules(self) -> tuple[nn.Module, ...]:
        return tuple(self.encoder.get_submodule(path) for path in _PARTIAL_PATHS[self.config.architecture])

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """The final spatial stage, resolved without duplicate module registration."""
        return self.get_submodule(self.spec.gradcam_target_path)

    def forward_features(self, images: Tensor) -> dict[str, Tensor]:
        """Return spatial NCHW features and pooled embeddings before head dropout."""
        spatial = self.encoder(images)
        pooled = self.encoder_norm(self.pool(spatial)).flatten(1)
        return {"spatial": spatial, "pooled": pooled}

    def forward(self, images: Tensor) -> Tensor:
        return self.head(self.forward_features(images)["pooled"])

    def set_fine_tuning(self, mode: str, *, freeze_bn_stats: bool | None = None) -> None:
        """Set frozen/partial/full masks and preserve frozen-stage evaluation state."""
        freeze_stats = self.config.freeze_bn_stats if freeze_bn_stats is None else freeze_bn_stats
        validate_fine_tuning(mode, freeze_stats)
        self.config = replace(self.config, fine_tuning=mode, freeze_bn_stats=freeze_stats)
        apply_parameter_policy(self.encoder, self.encoder_norm, self.head, self._partial_modules(), mode)
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        apply_training_policy(
            self.encoder, self.encoder_norm, self._partial_modules(), self.config.fine_tuning,
            training=mode, freeze_bn_stats=self.config.freeze_bn_stats,
        )
        return self


def create_pretrained_model(
    config: ModelConfig, *, load_weights: bool = True, progress: bool = True,
) -> PretrainedClassifier:
    """Construct the selected CNN; propagate weight-loading failures unchanged.

    With ``load_weights=False``, the Torchvision constructor receives
    ``weights=None`` and no pretrained parameters are loaded.
    """
    if not isinstance(config, ModelConfig):
        raise TypeError("config must be a ModelConfig")
    if type(load_weights) is not bool or type(progress) is not bool:
        raise TypeError("load_weights and progress must be bools")
    native = getattr(models, config.architecture)(
        weights=_weight_enum(config.architecture) if load_weights else None,
        progress=progress,
    )
    return PretrainedClassifier(config, native, weights_loaded=load_weights)
