# Official CUB test evaluation

## Balanced iNaturalist option: smaller validation gap

The later gap-focused comparison kept validation accuracy within one percentage
point of the 86.67% accuracy-oriented candidate, then minimized the clean
train/validation gap. Twelve additional stronger-regularization settings were
measured on cached training/validation features; previously measured candidates
remained eligible. No official test images were loaded for this comparison or
for the balanced model's verification.

The selected candidate uses the original, frozen iNaturalist encoder and a
discriminant head with shrinkage **0.9**. It was already present among the
earlier trials; none of the additional settings gave a smaller gap while
meeting the accuracy floor. Training-only features determine class centers and
covariance. Positive confidence scaling uses validation labels, so validation
NLL remains a tuning measurement.

| Measurement | Accuracy-oriented model | Balanced model |
| --- | ---: | ---: |
| Clean training accuracy | 97.54% | 91.01% |
| Validation accuracy | 86.67% | 85.75% |
| Train/validation gap | 10.87 points | **5.26 points** |
| Official test accuracy | 85.23% | **Not evaluated** |

This reduces the measured validation gap by 5.61 points at a cost of 0.92 points
of validation accuracy. It does not demonstrate higher test accuracy or the
elimination of overfitting. The previous tested model remains available as the
accuracy-oriented option. Both are retained for an explicit tradeoff, not
selected against test scores.

Balanced export run: `7e70b477f56041ceaac8d3e30961e559`.
Complete inference-only model:
`outputs/resnet50_inaturalist/head_refit/7e70b477f56041ceaac8d3e30961e559/model.pt`.
Keep its adjacent `config.json` and `metrics.json` for handoff. Verification
strictly reloaded its weights, checked the encoder against its source, matched
real-image and cached-feature logits at batch size 32, and reproduced 85.75%
accuracy on all 1,200 validation images. The selection plan, all measured trials
and verification remain under `outputs/resnet50_inaturalist/validation_search/`
and in local MLflow.

## iNaturalist2021 classifier improvement: 2026-10-26

A bounded validation-only comparison tested 12 heads: two fixed encoders
(the best fine-tuned checkpoint and the original iNaturalist encoder), each
with discriminant shrinkage 0.1/0.5/0.9 and logistic L2 strength
0.003/0.03/0.3. Selection used validation accuracy, then macro F1, then the
smaller clean train/validation gap. No official test data was loaded during
this search. Only the selected winner was subsequently tested.

The winner retains the fine-tuned encoder from the baseline below and replaces
its final classifier with an equal-prior linear discriminant head using
shrinkage 0.1. Class centers and shared covariance use training features only;
the parameters are folded into the existing 200-output linear layer. A positive
confidence scale is fitted on validation labels and does not change the class
ranking. Its validation NLL is therefore a tuning measurement.

| Model | Clean training accuracy | Validation accuracy | Official test accuracy | Train/validation gap |
| --- | ---: | ---: | ---: | ---: |
| Initial iNaturalist CNN | 96.77% | 84.08% | 84.05% | 12.68 points |
| Same encoder, selected regularized head | 97.54% | **86.67%** | **85.23%** | **10.87 points** |

The new model classified **4,938/5,794** test images correctly, 68 more than
the initial model. Test macro precision is 85.96%, macro recall 85.34%, macro
F1 85.31%, and cross entropy 0.570746. Validation correct count increased from
1,009 to 1,040 out of 1,200. This is a modest measured improvement, not the
elimination of overfitting. The clean training/test gap is still 12.31 points.

### Extended metrics for the recommended accuracy-oriented model

All 5,794 test predictions were reproduced exactly from the same frozen
checkpoint to collect all 200 class scores. Model buffers and checkpoint bytes
were unchanged; no confidence scaling was refitted on test labels. The
previous report remains intact. Additional results are in `extended_metrics.json`
and `full_class_scores.npz` beside that evaluation's `report.json`, with copies
in the same MLflow evaluation run.

| Test metric | Value |
| --- | ---: |
| Top-1 accuracy | 85.23% |
| Error rate | 14.77% (856 images) |
| Macro precision | 85.96% |
| Macro recall / balanced accuracy | 85.34% |
| Macro F1 | 85.31% |
| Weighted precision | 86.01% |
| Weighted recall | 85.23% |
| Weighted F1 | 85.27% |
| Micro precision / recall / F1 | 85.23% |
| Top-5 accuracy | 96.98% |
| ROC-AUC, one-vs-rest macro | 0.996991 |
| Mean average precision, macro | 0.903568 |
| Cohen's kappa | 0.851512 |
| Matthews correlation coefficient | 0.851551 |
| Cross entropy / NLL | 0.570746 |
| Multiclass Brier score, summed over classes | 0.212134 |
| Expected calibration error, 15 equal-width bins | 1.97 percentage points |

Macro metrics give each species equal weight; weighted metrics use its test
support. Top-5 asks whether the correct species is among five suggestions.
One-vs-rest ROC-AUC measures probability ranking for each species against all
other species. Its high value is not 99.70% classification accuracy; the actual
top-1 accuracy remains 85.23%. Mean average precision is not trapezoidal PR AUC.
Brier score is the mean sum of squared probability errors across 200 classes,
without division by 200. These definitions matter when comparing other models.

Mean top-1 confidence was 84.11%, versus 85.23% accuracy. Mean confidence on
incorrect predictions was 52.79%. Nevertheless, **77 wrong predictions had
confidence at least 90%**, and 51 had confidence at least 95%. A low aggregate
calibration error does not prevent individual confident mistakes.

The weakest species by recall include:

| Species | Correct / test images | Recall |
| --- | ---: | ---: |
| California Gull | 10 / 30 | 33.33% |
| Herring Gull | 13 / 30 | 43.33% |
| American Crow | 14 / 30 | 46.67% |
| Pelagic Cormorant | 15 / 30 | 50.00% |
| Yellow bellied Flycatcher | 15 / 29 | 51.72% |

The largest directed confusions were California Gull to Western Gull (11),
American Crow to Fish Crow (10), and Anna Hummingbird to Ruby throated
Hummingbird (9). Per-species supports are small, so these recall estimates are
noisy. The overall score should not be interpreted as equal reliability across
all 200 species. The JSON report contains every species' precision, recall,
F1 and support, plus calibration bins and leading confusion pairs.

An image-level binomial Wilson approximation gives a 95% accuracy interval of
84.29%-86.12%. This interval excludes uncertainty from training seeds, repeated
model selection, possible pretraining overlap and dependence between images.

The exported encoder tensors were checked against their source and are
unchanged. Strict model loading passed; real-image logits exactly matched cached
feature logits at the configured batch size of 32. The exported model also
reproduced 86.67% accuracy on all 1,200 real validation images before test
evaluation. Small floating-point differences were observed when comparing
batch size 2 with cached features computed at batch size 32; use the saved batch
size when reproducing these measurements.

- Search parent run: `54602834697c41f6805bb3aa92bad34b`.
- Winning trial: `6db84c6f0d6049f592656f8623a44a8f`.
- Export run: `375ca28ba526439090f6000aedfc6ee1`.
- Evaluation run: `604ff15d9c734932b911490840fc7a72`.
- Selected complete model:
  `outputs/resnet50_inaturalist/head_refit/375ca28ba526439090f6000aedfc6ee1/model.pt`.
- Model SHA-256:
  `7fe7c0a1adde882e3830e6fddd67fe97c56fd71482813add553784fd5938e32e`.
- Test report and predictions:
  `outputs/resnet50_inaturalist/evaluation/604ff15d9c734932b911490840fc7a72/`.
- Search plan, all trials and export verification:
  `outputs/resnet50_inaturalist/validation_search/`.

All trials and metadata are recorded in local MLflow. The model is a schema-2,
inference-only export; the original training checkpoint and optimizer state are
preserved separately. Upload this `model.pt` with its adjacent `config.json` and
`metrics.json` for handoff. The original iNaturalist pretraining file is not
needed to load the complete exported model.

The validation split has been used repeatedly, and earlier project test scores
were already observed. This is not a newly blinded benchmark. An image-level
CUB/iNaturalist overlap audit has not been performed. No further changes were
selected using this test result.

## Initial iNaturalist2021 CNN: 2026-10-26

ResNet-50 was initialized from the authors'
`inat2021_supervised_large_from_scratch.pth.tar` checkpoint, whose
[catalogue entry](https://github.com/visipedia/newt/blob/main/benchmark/configs.py)
specifies iNaturalist2021 pretraining without ImageNet initialization. A new
200-class CUB head was trained using `configs/resnet50_inaturalist.yaml`. The
fixed EDA memberships, fit/pad geometry, class order and regularized training
recipe were retained. This is transfer learning for CUB, not our scratch-CNN
baseline.

| Partition | Images | Correct | Accuracy | Macro F1 | Unsmoothed cross entropy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean training | 4,794 | 4,639 | 96.77% | 96.74% | 0.276041 |
| Validation | 1,200 | 1,009 | 84.08% | 84.02% | 0.762655 |
| Official test | 5,794 | 4,870 | **84.05%** | **83.93%** | 0.751154 |

Test macro precision is 84.81% and macro recall is 84.14%. Clean training and
validation measurements both disable augmentation and dropout. The
training/validation gap is **12.68 percentage points**, versus approximately
31.20 points for the previous ImageNet-initialized regularized CNN (94.20%
training, 63.00% validation). The matched recipe's validation accuracy improved
by 21.08 points in this one-seed comparison. Some overfitting remains; the
improvement does not establish that pretraining-data overlap was its cause.

Three head epochs and five fine-tuning epochs completed. Validation-loss early
stopping ended training; the highest-validation-accuracy checkpoint came from
fine-tuning epoch 2 and was fixed before its single test evaluation. No
classifier refit, test augmentation or confidence-scale fitting was used for
this model. Earlier project test results had already been observed, so this is
not a newly blinded benchmark. An image-level CUB/iNaturalist overlap audit has
not been performed. Do not use this test result to tune the same model further.

Training run: `9c6e5c07be24414b95fc5717e3d01eeb`.
Evaluation run: `da4d2a33feb543748f06178c2638ddb7`.

- Selected complete CUB model:
  `outputs/resnet50_inaturalist/training/9c6e5c07be24414b95fc5717e3d01eeb/best.pt`.
- Clean train/validation audit: `clean_train_val_report.json` beside the model.
- Test report, 5,794 predictions and confusion matrix:
  `outputs/resnet50_inaturalist/evaluation/da4d2a33feb543748f06178c2638ddb7/`.
- External pretraining file identity:
  `e845bd2512b7728c390523afc8696dafedf2675c93e9d7508a48b416cda8e8d9` (SHA-256).
- Selected CUB checkpoint identity:
  `a6ed5697a0a7555292e84ab2219848428cda7a18eb43fe9e02dff43c88eeb221` (SHA-256).

The model and saved `config.json` can be uploaded to Google Drive for handoff.
The complete CUB checkpoint restores without the original iNaturalist file.
Implementation checks passed 591 tests, with the opt-in cached-data check
passing separately (592 passing checks overall).

## Earlier ImageNet-initialized models: 2026-10-25

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

### Reproduce and inspect the earlier models

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
