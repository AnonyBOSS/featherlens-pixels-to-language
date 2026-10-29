"""Training workflows for both scratch ResNet and pretrained models."""

# Scratch model training
from .scratch_trainer import (
    OPTIMIZERS,
    EpochReport,
    TrainingConfig,
    TrainingResult,
    build_optimizer,
    evaluate_classification,
    train_classification,
)

# Pretrained model training
from .pretrained_trainer import (
    Phase,
    TrainConfig,
    evaluate as evaluate_pretrained,
    fit,
    train_epoch as train_pretrained_epoch,
)

__all__ = [
    "OPTIMIZERS",
    "EpochReport",
    "TrainingConfig",
    "TrainingResult",
    "build_optimizer",
    "evaluate_classification",
    "train_classification",
    "Phase",
    "TrainConfig",
    "evaluate_pretrained",
    "fit",
    "train_pretrained_epoch",
]
