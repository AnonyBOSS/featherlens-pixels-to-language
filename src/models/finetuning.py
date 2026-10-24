"""Small, explicit fine-tuning policies for the supported CNN adapters."""

from collections.abc import Sequence

from torch import nn


FINE_TUNING_MODES = ("frozen", "partial", "full")


def validate_fine_tuning(mode: str, freeze_bn_stats: bool) -> None:
    """Validate before changing any parameter or module state."""
    if not isinstance(mode, str) or mode not in FINE_TUNING_MODES:
        raise ValueError(f"fine_tuning must be one of {FINE_TUNING_MODES}")
    if type(freeze_bn_stats) is not bool:
        raise TypeError("freeze_bn_stats must be a bool")


def apply_parameter_policy(
    encoder: nn.Module,
    encoder_norm: nn.Module,
    head: nn.Module,
    partial_modules: Sequence[nn.Module],
    mode: str,
) -> None:
    """Set the parameter mask and discard gradients on newly frozen tensors."""
    for module in (encoder, encoder_norm):
        for parameter in module.parameters():
            parameter.requires_grad_(mode == "full")
    if mode == "partial":
        for module in (*partial_modules, encoder_norm):
            module.requires_grad_(True)
    head.requires_grad_(True)
    for module in (encoder, encoder_norm):
        for parameter in module.parameters():
            if not parameter.requires_grad:
                parameter.grad = None


def apply_training_policy(
    encoder: nn.Module,
    encoder_norm: nn.Module,
    partial_modules: Sequence[nn.Module],
    mode: str,
    *,
    training: bool,
    freeze_bn_stats: bool,
) -> None:
    """Keep frozen stages in eval, including their stochastic layers and buffers."""
    encoder.train(training and mode == "full")
    encoder_norm.train(training and mode != "frozen")
    if training and mode == "partial":
        for module in partial_modules:
            module.train(True)
    if freeze_bn_stats:
        for module in encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
