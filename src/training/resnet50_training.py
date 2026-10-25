"""Building blocks for ResNet-50 classification training."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from src.models.resnet50 import ResNet50Classifier


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float
    samples: int


def build_optimizer(
    model: ResNet50Classifier,
    *,
    classifier_learning_rate: float,
    weight_decay: float,
    encoder_learning_rate: float | None = None,
) -> AdamW:
    """Build AdamW for the model's current training stage.

    Move the model to its device and select its stage before calling.
    Rebuild the optimizer whenever the stage changes.
    """
    if (
        not math.isfinite(classifier_learning_rate)
        or classifier_learning_rate <= 0
    ):
        raise ValueError("classifier_learning_rate must be finite and positive.")

    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative.")

    parameter_groups = [
        {
            "params": model.classifier.parameters(),
            "lr": classifier_learning_rate,
        }
    ]

    if model.stage == "head":
        if encoder_learning_rate is not None:
            raise ValueError("Head training does not use encoder_learning_rate.")
    elif model.stage == "finetune":
        if (
            encoder_learning_rate is None
            or not math.isfinite(encoder_learning_rate)
            or encoder_learning_rate <= 0
        ):
            raise ValueError("encoder_learning_rate must be finite and positive.")

        parameter_groups.append(
            {
                "params": model.encoder.layer4.parameters(),
                "lr": encoder_learning_rate,
            }
        )
    else:
        raise ValueError(f"Unsupported training stage: {model.stage!r}")

    return AdamW(
        parameter_groups,
        lr=classifier_learning_rate,
        weight_decay=weight_decay,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
) -> EpochMetrics:
    """Train for one pass over the supplied loader.

    The caller selects the training stage, places the model on device,
    and constructs its optimizer. Loss is unweighted mean cross-entropy.
    """
    model.train()
    loss_fn = nn.CrossEntropyLoss()

    loss_sum = 0.0
    correct = 0
    samples = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = loss_fn(logits, labels)

        if not torch.isfinite(loss).item():
            raise FloatingPointError("Training loss is not finite.")

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        samples += batch_size

    if samples == 0:
        raise ValueError("The training loader produced no samples.")

    return EpochMetrics(
        loss=loss_sum / samples,
        accuracy=correct / samples,
        samples=samples,
    )


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> EpochMetrics:
    """Measure validation performance without updating the model."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    loss_sum = 0.0
    correct = 0
    samples = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        if not torch.isfinite(loss).item():
            raise FloatingPointError("Validation loss is not finite.")

        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        samples += batch_size

    if samples == 0:
        raise ValueError("The validation loader produced no samples.")

    return EpochMetrics(
        loss=loss_sum / samples,
        accuracy=correct / samples,
        samples=samples,
    )
