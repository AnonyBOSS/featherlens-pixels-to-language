"""An explicit classification training loop over the prepared CUB loaders.

The loop is pure computation: it opens no files, writes no checkpoints, and
talks to no tracking service. Callers supply any dict-batch loaders whose
batches contain ``image`` and ``label`` tensors and receive a full per-epoch
history plus the best-validation state dict; persistence and experiment
tracking stay with the CLI layer. The model family is irrelevant — pretrained
adapters and the scratch baseline share this loop.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
import random

import numpy as np
from torch import Tensor, nn
from torch.utils.data import DataLoader
import torch

from src.evaluation.metrics import confusion_matrix, summarize_classification


@dataclass(frozen=True)
class TrainingConfig:
    """Validated immutable optimization settings for one training run."""

    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    seed: int = 42
    amp: bool = True

    def __post_init__(self) -> None:
        if type(self.epochs) is not int or isinstance(self.epochs, bool) or self.epochs < 1:
            raise ValueError("epochs must be an integer >= 1")
        if (isinstance(self.learning_rate, bool) or type(self.learning_rate) not in (int, float)
                or not math.isfinite(self.learning_rate) or self.learning_rate <= 0):
            raise ValueError("learning_rate must be a finite number > 0")
        if (isinstance(self.weight_decay, bool) or type(self.weight_decay) not in (int, float)
                or not math.isfinite(self.weight_decay) or self.weight_decay < 0):
            raise ValueError("weight_decay must be a finite number >= 0")
        if (isinstance(self.label_smoothing, bool) or type(self.label_smoothing) not in (int, float)
                or not math.isfinite(self.label_smoothing) or not 0 <= self.label_smoothing < 1):
            raise ValueError("label_smoothing must be a finite number in [0, 1)")
        if type(self.seed) is not int or isinstance(self.seed, bool) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if type(self.amp) is not bool:
            raise ValueError("amp must be a bool")


@dataclass(frozen=True)
class EpochReport:
    """One epoch's averaged metrics; ``epoch`` counts from 1."""

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    learning_rate: float


@dataclass(frozen=True)
class TrainingResult:
    """Full run history plus the state dict of the best validation epoch.

    ``best_state`` tensors are detached CPU copies, so the result stays
    portable regardless of the training device. The live model keeps the
    final epoch's parameters.
    """

    reports: tuple[EpochReport, ...]
    best_epoch: int
    best_val_accuracy: float
    best_state: dict[str, Tensor]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        return torch.device(device)
    if isinstance(device, torch.device):
        return device
    raise TypeError("device must be a str, torch.device, or None")


def _snapshot(model: nn.Module) -> dict[str, Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def evaluate_classification(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: str | torch.device,
    criterion: nn.Module | None = None,
    amp: bool = False,
) -> tuple[float, Tensor]:
    """Average loss and one CPU confusion matrix over a full loader pass.

    The model must already live on ``device``. Batches are dict batches with
    ``image`` and ``label``; the returned matrix follows
    :func:`src.evaluation.metrics.confusion_matrix` orientation. Evaluation
    runs in full precision unless ``amp`` is requested.
    """
    resolved = _resolve_device(device)
    loss_fn = criterion if criterion is not None else nn.CrossEntropyLoss()
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    seen = 0
    predictions: list[Tensor] = []
    labels: list[Tensor] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(resolved, non_blocking=True)
            targets = batch["label"].to(resolved, non_blocking=True)
            with torch.autocast(device_type=resolved.type, dtype=torch.float16, enabled=amp):
                logits = model(images)
                loss = loss_fn(logits, targets)
            loss_sum += float(loss.detach()) * images.size(0)
            seen += images.size(0)
            predictions.append(logits.argmax(dim=1).cpu())
            labels.append(targets.cpu())
    if was_training:
        model.train()
    if seen == 0:
        raise ValueError("loader yielded no batches")
    matrix = confusion_matrix(torch.cat(predictions), torch.cat(labels), num_classes=int(logits.shape[-1]))
    return loss_sum / seen, matrix


def train_classification(
    model: nn.Module,
    loaders: Mapping[str, DataLoader],
    config: TrainingConfig,
    *,
    device: str | torch.device | None = None,
    on_epoch: Callable[[EpochReport], None] | None = None,
) -> TrainingResult:
    """Optimize ``model`` over prepared loaders and return the run history.

    AdamW with a per-step cosine schedule trains every parameter; validation
    runs after each epoch in evaluation mode. The best validation accuracy
    snapshot is retained, and ``on_epoch`` receives every
    :class:`EpochReport` in order for live experiment tracking. Training seeds
    global Python/NumPy/torch RNGs with ``config.seed`` first, matching the
    loaders' own generator seeding. Evaluation restores the prior mode, so the
    returned model is left in training mode with final-epoch parameters.
    """
    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig")
    if not isinstance(loaders, Mapping) or "train" not in loaders or "val" not in loaders:
        raise ValueError("loaders must be a mapping containing 'train' and 'val'")
    for name, loader in loaders.items():
        if not isinstance(loader, DataLoader):
            raise TypeError(f"loader {name!r} must be a DataLoader")
    if on_epoch is not None and not callable(on_epoch):
        raise TypeError("on_epoch must be callable or None")
    _seed_everything(config.seed)
    resolved = _resolve_device(device)
    model.to(resolved)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    total_steps = config.epochs * len(loaders["train"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    use_amp = config.amp and resolved.type == "cuda"
    scaler = torch.amp.GradScaler(resolved.type, enabled=use_amp)

    reports: list[EpochReport] = []
    best_accuracy = -1.0
    best_epoch = 0
    best_state: dict[str, Tensor] = {}
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        epoch_predictions: list[Tensor] = []
        epoch_labels: list[Tensor] = []
        for batch in loaders["train"]:
            images = batch["image"].to(resolved, non_blocking=True)
            targets = batch["label"].to(resolved, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=resolved.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += float(loss.detach()) * images.size(0)
            seen += images.size(0)
            epoch_predictions.append(logits.argmax(dim=1).cpu())
            epoch_labels.append(targets.cpu())
        if seen == 0:
            raise ValueError("train loader yielded no batches")
        train_summary = summarize_classification(
            confusion_matrix(torch.cat(epoch_predictions), torch.cat(epoch_labels),
                             num_classes=int(logits.shape[-1])),
        )
        val_loss, val_matrix = evaluate_classification(
            model, loaders["val"], device=resolved, criterion=criterion, amp=use_amp,
        )
        val_summary = summarize_classification(val_matrix)
        report = EpochReport(
            epoch=epoch,
            train_loss=loss_sum / seen,
            train_accuracy=train_summary["accuracy"],
            val_loss=val_loss,
            val_accuracy=val_summary["accuracy"],
            learning_rate=scheduler.get_last_lr()[0],
        )
        reports.append(report)
        if report.val_accuracy > best_accuracy:
            best_accuracy = report.val_accuracy
            best_epoch = epoch
            best_state = _snapshot(model)
        if on_epoch is not None:
            on_epoch(report)
    return TrainingResult(
        reports=tuple(reports),
        best_epoch=best_epoch,
        best_val_accuracy=best_accuracy,
        best_state=best_state,
    )
