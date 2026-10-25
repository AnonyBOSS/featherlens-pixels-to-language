# Classification training progress

**Scope:** Task 1.2 — the from-scratch CNN baseline and the training
foundation it needs. This document records what is implemented on
`feat/scratch-cnn-baseline` and `feat/classification-training`, how to use it,
and what remains before the Milestone 1 submission.

**Date:** 2026-10-25

## Done

### From-scratch CNN encoder (`src/models/scratch.py`)

A residual CNN written from random initialization, with no pretrained
parameters anywhere (the required Task 1.2 baseline):

- 3x3 stem (stride 2) + maxpool, four residual stages at 64/128/256/256 with
  two basic blocks each (BatchNorm, ReLU, 1x1 projection skips), global
  average pooling, dropout, and a 200-logit head.
- 5,236,712 parameters; about 1.35 GMACs per 224x224 image; output stride 32
  with a 7x7 final spatial map, matching the pretrained adapters' geometry.
- Same wrapper contract as `PretrainedClassifier`: `forward_features`
  returning `{"spatial", "pooled"}`, `gradcam_target_layer`, serializable
  `ScratchSpec`, and a validated frozen `ScratchConfig`.
- Construction initializes from the caller's torch RNG; seed before
  construction for reproducible initialization.

### Classification metrics (`src/evaluation/metrics.py`)

Pure torch metrics with explicit validation and no new dependencies:

- `confusion_matrix(predictions, labels, num_classes=...)` — row-true,
  column-predicted int64 counts; matrices from separate batches sum exactly.
- `summarize_classification(matrix)` — overall accuracy plus macro precision,
  recall, and F1; classes without predictions or support contribute 0.0.

### Training loop (`src/training/trainer.py`)

A pure, file-free optimization loop shared by every model family:

- `TrainingConfig` — validated frozen settings: epochs, learning rate,
  weight decay, label smoothing, seed, AMP.
- `train_classification(model, loaders, config)` — AdamW with a per-step
  cosine schedule, label-smoothing cross-entropy, per-epoch train and
  validation evaluation, best-validation snapshot retained as detached CPU
  tensors, and an `on_epoch` hook for live experiment tracking.
- `evaluate_classification(model, loader, device=...)` — averaged loss and one
  full confusion matrix; the standalone path for validation and test splits.
- Full per-epoch history returned as `TrainingResult.reports`; no files are
  written and no tracking service is contacted from this module.

### Verification

- `tests/models/test_scratch.py` — 18 offline tests for the baseline model.
- `tests/evaluation/test_metrics.py` — orientation, hand-computed macro
  scores, zero-division semantics, and input rejection.
- `tests/training/test_trainer.py` — determinism, snapshot portability,
  ordering, input rejection, empty-loader errors, and a real one-epoch run on
  the scratch model.
- `pytest tests/models tests/evaluation tests/training/test_trainer.py -q`
  passes with 124 tests; the broader offline suites remain green.

## How to run a training pass

```python
from pathlib import Path
import torch

from src.data.preparation import load_classification_manifest
from src.data.source import load_cub_split
from src.data.classification import CubClassificationDataset, make_classification_loaders
from src.data.transforms import classification_transform
from src.models.scratch import ScratchConfig, create_scratch_model
from src.training.trainer import TrainingConfig, train_classification

rows = load_classification_manifest(Path("outputs/preparation"))
source = load_cub_split("train")
datasets = {part: CubClassificationDataset(source, rows, partition=part,
            transform=classification_transform(training=(part == "train")))
            for part in ("train", "val")}
loaders = make_classification_loaders(datasets, batch_size=64, num_workers=4, pin_memory=True)

model = create_scratch_model(ScratchConfig(head_dropout=0.1))
result = train_classification(model, loaders, TrainingConfig(epochs=30))
print(result.best_epoch, result.best_val_accuracy)
```

## Remaining

In dependency order:

1. **CLI and configuration** — `src/cli.py` with
   `train --task classification --config configs/classification.yaml`,
   checkpoint persistence, and the first real config file (`configs/` is
   still empty). The trainer intentionally performs no I/O, so persistence
   lands here.
2. **Experiment tracking** — MLflow around `on_epoch`: hyperparameters,
   per-epoch curves, and checkpoint artifacts for every run.
3. **First real training run** — the scratch baseline on the prepared
   loaders (local GPU or Colab); record the achieved validation metrics.
4. **Training visualizations** — loss/accuracy curves and per-class views
   from saved histories, in the established report-rendering style.
5. **Hyperparameter tuning** — grid and random search over a subset
   (learning rate, weight decay, head dropout), with the required
   method-vs-method comparison (best score, trials, time).
6. **Grad-CAM** — CAM computation and rendering over `gradcam_target_layer`
   for correct and wrong predictions; the model-side hooks already exist.
7. **Transfer-learning comparison** — frozen, partial, and full fine-tuning
   of the pretrained adapters over `make_pretrained_loaders`, evaluated with
   the same metrics.
8. **Benchmarking** — comparison against published CUB-200 results with a
   fair split and metric statement.
9. **Milestone 2** — colorization (grayscale-pair builder plus encoder
   decoder) and captioning (frozen encoder plus autoregressive decoder).

## Limits

- The trainer assumes dict batches with `image` and `label`; loaders from
  `make_classification_loaders` and `make_pretrained_loaders` satisfy this.
- Training seeds global RNGs at loop start; loader-level shuffling keeps its
  own generator seeds from `make_classification_loaders`.
- The official test split is never opened by the trainer; it stays reserved
  for final evaluation and benchmarking.
