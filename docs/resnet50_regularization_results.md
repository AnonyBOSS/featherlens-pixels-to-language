# ResNet-50 regularization results

These are earlier ImageNet regularization experiments. The [selected workflow](resnet50_training.md) documents the current iNaturalist model.

These are exploratory validation results from one seed (42), using the original
4,794 training and 1,200 validation images. No official test images were used.

## Settings

| Setting | Baseline | Regularized | Regularized, lower LR |
| --- | ---: | ---: | ---: |
| Classifier dropout | 0 | 0.3 | 0.3 |
| Training label smoothing | 0 | 0.1 | 0.1 |
| AdamW weight decay | 0.0001 | 0.01 | 0.01 |
| Training augmentation | Flip | Flip, mild brightness/contrast, rotation | Same |
| Encoder fine-tuning LR | 0.00001 | 0.00001 | 0.000003 |
| Classifier fine-tuning LR | 0.0001 | 0.0001 | 0.00003 |
| Head epochs | 3 | 3 | 3 |
| Maximum fine-tuning epochs | 15 | 10 | 10 |
| Fine-tuning early stopping | Disabled | Validation loss, patience 3, delta 0.001 | Same |

Validation preprocessing is identical across all runs. The regularized trials
combine several changes; this comparison does not isolate each change's effect.
The lower-LR trial changes only the fine-tuning learning rates relative to the
regularized trial. Both repeated head stages produced identical metrics.

## Selected checkpoint comparison

All metrics below were recomputed on the same GPU. Clean training evaluation
disables augmentation and dropout and uses the same preprocessing as validation.
Validation loss is unsmoothed cross-entropy for all models. Macro F1 averages
the 200 per-class F1 scores equally.

| Run | Selected fine-tuning epoch | Clean train accuracy | Validation accuracy | Validation macro F1 | Validation loss | Accuracy gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 4 | 97.98% | 60.33% | 60.08% | 1.6953 | 37.64 pp |
| Regularized | 10 | 94.20% | 63.00% | 61.87% | 1.5445 | 31.20 pp |
| Regularized, lower LR | 10 | 84.40% | 58.00% | 56.85% | 1.6925 | 26.40 pp |

The regularized configuration achieved the highest validation accuracy:
63.00%, an improvement of 2.67 percentage points over the baseline.
Its clean train/validation accuracy gap decreased from 37.64 to 31.20 percentage points.
Overfitting remains substantial; these results show mitigation, not elimination.
The lower learning rates learned more slowly and did not beat the baseline
within the same ten-epoch fine-tuning cap.

Checkpoint selection used validation accuracy. Early stopping monitored
validation loss. In the regularized trial, the patience criterion was reached
at epoch 10, also the configured cap; it did not shorten that particular run.
The lower-LR trial reached its cap without triggering early stopping.

## Configurations and artifacts

- Baseline: [resnet50.yaml](../configs/resnet50.yaml).
- Selected: [resnet50_regularized.yaml](../configs/resnet50_regularized.yaml).
- Lower LR: [archived configuration below](#archived-lower-learning-rate-configuration).

- Baseline run ID: acf1269e5a9a40d09ef80915fc34e7a9.
- Regularized run ID: 487597dac3854f30a101214619872d38.
- Regularized, lower LR run ID: 57c43480e7d94f1f9f6334b2a3df6698.

Each run folder under outputs/resnet50/training contains its best.pt,
last.pt, stage-best checkpoints, effective configuration, and epoch summary.
Post-training validation_report_gpu.json contains per-class precision/recall/F1,
the confusion matrix, and example identities. clean_train_report.json contains
the deterministic training evaluation. These reports and metrics were also
added to the existing local MLflow runs.

The combined comparison is outputs/resnet50/regularization_comparison/comparison.json;
its plot is validation_comparison.png in the same directory. Generated models
and plots remain ignored by Git; upload required trained models to Drive manually.

## Remaining work

These two manually chosen trials do not satisfy the project requirement to
compare grid and random search. Broader tuning, repeated seeds, Grad-CAM,
scratch-CNN/benchmark comparisons, and final held-out test evaluation remain.

## Archived lower-learning-rate configuration

This historical experiment is not part of the selected iNaturalist workflow.
Its former standalone configuration, `configs/resnet50_regularized_slow.yaml`, is preserved below
so the measured trial remains documented without an unused active YAML file.

```yaml
experiment:
  name: resnet50_regularization
  seed: 42
model:
  num_classes: 200
  weights: IMAGENET1K_V2
  dropout: 0.3
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
    encoder_learning_rate: 0.000003
    classifier_learning_rate: 0.00003
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
