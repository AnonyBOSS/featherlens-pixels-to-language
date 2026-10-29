# ResNet-50 overfitting mitigation: measured trade-offs

This records earlier ImageNet experiments. For the current selected iNaturalist model, use the [selected workflow](resnet50_training.md).

This document records validation tuning before official test evaluation. The
subsequent fixed-checkpoint results are in [official test results](resnet50_test_results.md).

All experiments use seed 42 and the same prepared 4794 training / 1200 validation
images across 200 classes. Clean training and validation accuracy below both use
evaluation mode and identical deterministic fit-to-224 preprocessing with ImageNet
normalization. Training-time augmentation, dropout and Mixup target agreement are
not used to calculate these gaps. Official test images were not used for tuning.

| Model | Clean train accuracy | Validation accuracy | Gap (percentage points) | Validation macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Original baseline | 97.98% | 60.33% | 37.64 | 0.6008 |
| Previous regularized best | 94.20% | 63.00% | 31.20 | 0.6187 |
| Fine-tune last block only | 79.95% | 53.92% | 26.04 | 0.5263 |
| Full layer4 with Mixup | 89.97% | 61.58% | 28.38 | 0.6071 |
| Mixup encoder + stronger L2 classifier | **92.30%** | **64.17%** | **28.14** | **0.6317** |
| Previous best encoder + weaker L2 classifier | 100.00% | **67.50%** | 32.50 | 0.6695 |

The stronger L2 classifier is the balanced candidate for the current request: it
reduces the previous best's gap by 3.06 points and improves validation accuracy by
1.17 points. Overfitting is mitigated, not eliminated. A smaller gap alone is not a
selection objective: restricting fine-tuning to the last block reduced the gap
but substantially hurt validation performance, so it was not selected.

The weaker L2 classifier has the highest observed validation accuracy, yet fits
all training samples. Keep it as an accuracy-oriented alternative rather than
describe it as resolving overfitting. A training/validation gap does not by itself
invalidate a model with better independent generalization; final comparison must
follow a declared selection criterion and a single untouched-test evaluation.

## What changed

The CNN trials retain dropout 0.3, mild augmentation, weight decay 0.01, and
validation-loss early stopping. The last-block trial limits trainable encoder
capacity. The Mixup trial instead trains all of layer4 with alpha 0.2, and disables
additional fine-tuning label smoothing. Neither trial beat the previous 63% best
with its original classifier.

The additional classifier refit freezes an already trained encoder and learns a
new 2048-to-200 linear classifier using train-only standardized embeddings.
It optimizes cross entropy plus an explicit L2 penalty with LBFGS. The balanced
candidate uses the Mixup encoder and L2 strength 0.35; the accuracy alternative
uses the previous regularized encoder and strength 0.003. Normalization is folded
into the saved linear weights. No encoder parameters change during this step.

A positive confidence scale is fitted on validation NLL and folded into the
classifier. It preserves class ordering and therefore cannot improve accuracy.
The resulting validation NLLs, about 1.3460 and 1.3143, are calibration tuning
measurements, not independent estimates. Validation labels never fit feature
statistics or the classifier's cross-entropy/L2 objective.

Exploratory comparisons also included ridge classifiers, compressed PCA/ridge
classifiers and horizontal-flip prediction averaging. They did not give a better
balanced result. Analyses are retained under
`outputs/resnet50/head_refit_analysis/`; these are adaptive exploratory trials,
not the project's complete required grid-versus-random-search comparison.
An internal exploratory filter asked for a gap at most 28 points; no candidate
met all its criteria. The balanced result's actual gap is 28.14 points.

## Local artifacts and limitations

| Purpose | Local run ID |
| --- | --- |
| Previous regularized best CNN | `487597dac3854f30a101214619872d38` |
| Last-block CNN | `396c958a120a4bb0a16ae3fbf1aa1797` |
| Mixup CNN / balanced encoder source | `f842795088f84057867000e7378ae907` |
| Balanced classifier export | `029780ab815f4c8092c402153444767f` |
| Accuracy-oriented classifier export | `3897e4b1675c49a7a6c4402fa23ebfa5` |

CNN checkpoints are under `outputs/resnet50/training/<run_id>/best.pt`.
Complete classifier-refit exports are under
`outputs/resnet50/head_refit/<run_id>/model.pt`, with config and metrics JSON.
They are inference-only schema-2 checkpoints, with no optimizer/resume state.
They retain the same model state-dictionary names and can load strictly through
`ResNet50Classifier`. See [loading and commands](resnet50_training.md).

Runs, metrics and exports are recorded in local MLflow. Exploratory probe JSONs
are attached after the analysis and marked as such. Model files remain ignored
by Git and require separate Google Drive delivery after review.

Only one seed was trained, and the same validation set was consulted repeatedly
to select hyperparameters and calibration. Improvements may be optimistic and
are not test results. Freeze the chosen configuration before final test
evaluation. Formal grid/random search, final metrics/Grad-CAM, scratch-CNN
comparison and final integration remain separate project work.

The transfer-learning workflow follows the
[PyTorch tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).
Mixup blends training examples and their targets as described in the
[original paper](https://arxiv.org/abs/1710.09412).

## Archived last-block configuration

This historical experiment is not part of the selected iNaturalist workflow.
Its former standalone configuration, `configs/resnet50_last_block.yaml`, is preserved below
so the measured trial remains documented without an unused active YAML file.

```yaml
experiment:
  name: resnet50_capacity_regularization
  seed: 42
model:
  num_classes: 200
  weights: IMAGENET1K_V2
  dropout: 0.3
  finetune_scope: last_block
data:
  prepared_dir: preparation
  image_size: 224
  geometry: fit
  normalization: imagenet
  augmentation: mild
  batch_size: 32
  num_workers: 0
training:
  device: cuda
  optimizer: adamw
  weight_decay: 0.01
  label_smoothing: 0.1
  head:
    epochs: 3
    learning_rate: 0.001
  finetune:
    epochs: 10
    encoder_learning_rate: 0.00001
    classifier_learning_rate: 0.0001
    early_stopping:
      patience: 3
      min_delta: 0.001
checkpoints:
  local_root: outputs/resnet50
  selection_metric: val_accuracy
tracking:
  backend: mlflow
  local_dir: outputs/resnet50/mlruns
```

## Archived Mixup configuration

This historical experiment is not part of the selected iNaturalist workflow.
Its former standalone configuration, `configs/resnet50_mixup.yaml`, is preserved below
so the measured trial remains documented without an unused active YAML file.

```yaml
experiment:
  name: resnet50_capacity_regularization
  seed: 42
model:
  num_classes: 200
  weights: IMAGENET1K_V2
  dropout: 0.3
  finetune_scope: layer4
data:
  prepared_dir: preparation
  image_size: 224
  geometry: fit
  normalization: imagenet
  augmentation: mild
  batch_size: 32
  num_workers: 0
training:
  device: cuda
  optimizer: adamw
  weight_decay: 0.01
  label_smoothing: 0.1
  head:
    epochs: 3
    learning_rate: 0.001
  finetune:
    epochs: 10
    encoder_learning_rate: 0.00001
    classifier_learning_rate: 0.0001
    mixup_alpha: 0.2
    label_smoothing: 0.0
    early_stopping:
      patience: 3
      min_delta: 0.001
checkpoints:
  local_root: outputs/resnet50
  selection_metric: val_accuracy
tracking:
  backend: mlflow
  local_dir: outputs/resnet50/mlruns
```
