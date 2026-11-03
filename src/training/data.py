"""Pretrained baselines over the existing, validated CUB classification pipeline."""
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader
from torchvision.transforms import ColorJitter, Compose

from src.data.classification import CubClassificationDataset, make_classification_loaders
from src.data.preparation import load_classification_manifest
from src.data.source import CubSource, load_cub_split
from src.data.transforms import IMAGENET_NORMALIZATION, classification_transform
from src.models.pretrained import ModelConfig, get_model_spec


def _model_spec(config: ModelConfig):
    if not isinstance(config, ModelConfig):
        raise ValueError("model_config must be a validated ModelConfig")
    return get_model_spec(config.architecture, config.weights)


@dataclass(frozen=True)
class PreprocessingConfig:
    """Full-image fit and white padding, followed by one optional normalization.

    This deliberately uses the Task 1.1 native image decoding and transforms.
    The architecture's native weight crop recipe is descriptive metadata only.
    """

    image_size: int = 224
    normalization: tuple[Sequence[float], Sequence[float]] | None = IMAGENET_NORMALIZATION
    color_jitter: float = 0.0

    def __post_init__(self) -> None:
        if (type(self.color_jitter) not in (int, float) or not math.isfinite(self.color_jitter)
                or not 0 <= self.color_jitter <= 0.5):
            raise ValueError('color_jitter must be finite and between 0 and 0.5')
        object.__setattr__(self, 'color_jitter', float(self.color_jitter) if self.color_jitter else 0.0)
        # Keep one validation authority for dimensions and mean/std semantics.
        classification_transform(training=False, image_size=self.image_size,
                                 geometry="fit", normalization=self.normalization)
        if self.normalization is not None:
            object.__setattr__(self, "normalization", tuple(tuple(values) for values in self.normalization))

    def build_transform(self, *, training: bool) -> Compose:
        """Keep full-image geometry; optional exposure jitter applies only to training.

        Jitter changes brightness/contrast, never hue or saturation. Apply it to
        the original RGB image before fitting so added padding stays white.
        Zero keeps the legacy transform and its RNG consumption unchanged.
        """
        transform = classification_transform(training=training, image_size=self.image_size,
                                             geometry="fit", normalization=self.normalization)
        if training and self.color_jitter:
            transform.transforms.insert(0, ColorJitter(
                brightness=self.color_jitter, contrast=self.color_jitter))
        return transform

    def describe(self, model_config: ModelConfig) -> dict[str, Any]:
        """Describe the applied baseline separately from unapplied native weights."""
        spec = _model_spec(model_config)
        normalization = None if self.normalization is None else {
            "mean": list(self.normalization[0]), "std": list(self.normalization[1]),
        }
        return {
            "baseline": {
                "image_size": self.image_size, "geometry": "fit",
                "padding_rgb": [255, 255, 255], "normalization": normalization,
                "train_horizontal_flip_probability": 0.5, "validation_augmentation": False,
                "train_color_jitter": {"brightness": self.color_jitter, "contrast": self.color_jitter,
                                       "saturation": 0, "hue": 0},
            },
            "native_weights": {
                "weight_id": spec.weight_id,
                "preprocessing": deepcopy(spec.native_preprocessing), "applied": False,
            },
        }


def make_pretrained_loaders(
    prepared_dir: str | Path,
    model_config: ModelConfig,
    *,
    preprocessing: PreprocessingConfig | None = None,
    source: CubSource | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool = False,
) -> dict[str, DataLoader]:
    """Validate complete preparation, then open only official train for train/val.

    Model/configuration checks never construct a model or fetch weights. A
    supplied source avoids source fetching too. The existing manifest validator
    runs before source loading or dataset construction; it owns stored partition
    validation. ``seed`` controls loader sampling, never prepared membership.
    With zero workers, callers seed global RNGs for training augmentation, as in
    ``make_classification_loaders``. Validation preprocessing is deterministic.
    """
    _model_spec(model_config)
    if preprocessing is None:
        preprocessing = PreprocessingConfig()
    if not isinstance(preprocessing, PreprocessingConfig):
        raise ValueError("preprocessing must be a PreprocessingConfig or None")
    # Validate before potential source network effects, using the existing
    # loader's public parameter contract (actual construction remains below).
    for name, value, minimum in (("batch_size", batch_size, 1), ("num_workers", num_workers, 0), ("seed", seed, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if seed >= 2**32 or not isinstance(pin_memory, bool):
        raise ValueError("seed must be below 2**32 and pin_memory must be bool")

    manifest = load_classification_manifest(prepared_dir)
    if source is None:
        source = load_cub_split("train")
    datasets = {
        partition: CubClassificationDataset(
            source, manifest, partition=partition,
            transform=preprocessing.build_transform(training=partition == "train"),
        )
        for partition in ("train", "val")
    }
    return make_classification_loaders(datasets, batch_size=batch_size, num_workers=num_workers,
                                       seed=seed, pin_memory=pin_memory)
