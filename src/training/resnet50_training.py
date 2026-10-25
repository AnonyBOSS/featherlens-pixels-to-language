"""Building blocks for ResNet-50 classification training."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float
    samples: int


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
