"""Compatibility imports for the canonical pretrained training implementation.

The implementation and serialized configuration live in ``trainer``. Keeping
these names as aliases preserves callers without a second config or fit loop.
"""
from .trainer import Phase, TrainConfig, evaluate, fit, resolve_precision, train_epoch

__all__ = ["Phase", "TrainConfig", "evaluate", "fit", "resolve_precision", "train_epoch"]
