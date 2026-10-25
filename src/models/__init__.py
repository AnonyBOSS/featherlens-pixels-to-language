"""Supported pretrained CNN adapters, the from-scratch baseline, and fine-tuning policies."""

from .pretrained import (
    ModelConfig,
    ModelSpec,
    PretrainedClassifier,
    create_pretrained_model,
    get_model_spec,
)
from .scratch import (
    ScratchClassifier,
    ScratchConfig,
    ScratchSpec,
    create_scratch_model,
    get_scratch_spec,
)

__all__ = [
    "ModelConfig",
    "ModelSpec",
    "PretrainedClassifier",
    "ScratchClassifier",
    "ScratchConfig",
    "ScratchSpec",
    "create_pretrained_model",
    "create_scratch_model",
    "get_model_spec",
    "get_scratch_spec",
]
