# Official CUB test evaluation

**Pretraining limitation:** the [official CUB dataset page](https://www.vision.caltech.edu/datasets/cub_200_2011/)
warns about image overlap with ImageNet. The following checkpoints use ImageNet
pretraining. Our audit established isolation of the provided CUB partitions, not
disjointness from ImageNet. These scores are held out from CUB fine-tuning; they
are not certified independent of pretraining. The overlap count and its score
effect have not been measured. Original evaluation metrics remain unchanged.

On 2026-10-25, two fixed classifier-refit checkpoints were evaluated on every one
of the 5794 official test images across 200 classes. They were declared before
evaluation: the highest-validation-accuracy candidate as primary, and the
stronger-regularization candidate as a comparison. Neither checkpoint was changed
after seeing test labels. No augmentation or test-fitted calibration was used.

| Candidate | Clean training accuracy | Validation accuracy | Official test accuracy | Test macro precision | Test macro recall | Test macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Primary: L2 0.003, previous regularized encoder | 100.00% | 67.50% | **65.98%** | 66.77% | 66.28% | **65.99%** |
| Comparison: L2 0.35, Mixup encoder | 92.30% | 64.17% | 62.69% | 63.01% | 63.06% | 62.06% |

The primary model classified 3823/5794 images correctly, with cross entropy
1.319263. The comparison classified 3632/5794 correctly, with cross entropy
1.362653. Both retain confidence scales previously fitted on validation labels;
these scales were not adjusted using the test set.

The primary model has a 34.02-point clean-training/test gap. This remains evidence
of overfitting, but its test accuracy is only 1.52 points below validation
accuracy. The comparison has a smaller 29.62-point training/test gap and lower
test accuracy. A smaller gap by itself is therefore not the right reason to
prefer it. The primary model was preferred by validation accuracy before testing
and also performed better on this CUB test-split evaluation, subject to the
pretraining limitation above.

Whether 65.98% meets the project's desired performance needs a fair baseline and
the eventual scratch-CNN comparison; these scores alone do not establish that
the full project is complete. Do not keep optimizing hyperparameters against
this test set and continue calling it an untouched final assessment.

## Reproduce and inspect

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.resnet50_evaluation --checkpoint outputs/resnet50/head_refit/3897e4b1675c49a7a6c4402fa23ebfa5/model.pt --checkpoint outputs/resnet50/head_refit/029780ab815f4c8092c402153444767f/model.pt --device cuda
```

| Candidate | Checkpoint run ID | Local evaluation run ID |
| --- | --- | --- |
| Primary | `3897e4b1675c49a7a6c4402fa23ebfa5` | `1bd8b7c972424ca7baf03d8d5920e57d` |
| Comparison | `029780ab815f4c8092c402153444767f` | `7a1c1ce2a524417eb4360fdbeedbc95c` |

Checkpoints are under `outputs/resnet50/head_refit/<checkpoint_run_id>/model.pt`.
Reports are under `outputs/resnet50/evaluation/<evaluation_run_id>/`:

- `report.json`: checkpoint SHA-256, pinned data provenance, preprocessing,
  total counts, aggregate and per-class metrics.
- `predictions.jsonl`: one stable image ID, true class, prediction and confidence
  for each of the 5794 images.
- `confusion_matrix.csv` and `.png`: rows are true classes; columns are predicted
  classes. Class IDs map to the report's ordered `class_names` array.

The same artifacts and aggregate metrics are recorded in the local MLflow
experiment `resnet50_official_test`. Nothing is uploaded automatically.

The evaluator requires the complete pinned test population, rejects duplicate
image IDs or incomplete coverage, strictly loads model weights, and runs in eval
and inference modes. Four unit tests check metric definitions, the final partial
batch, duplicate/missing-image rejection, nonfinite outputs and unchanged weights.
