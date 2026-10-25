# ResNet-50 regularization results

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
- Lower LR: [resnet50_regularized_slow.yaml](../configs/resnet50_regularized_slow.yaml).

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
