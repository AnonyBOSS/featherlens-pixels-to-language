# Classification pipeline review after the first official test evaluation

This review uses the 30-page Task 1.1 EDA report, the nine-page project
specification, the shared source/preparation/image/Dataset/transform code, the
ResNet model, all training modules/configurations, the classifier refit, the
evaluation implementation and their relevant tests. Plotting and caption-analysis
modules are outside the model's execution path.

## Pretraining overlap is a separate limitation

The [official Caltech CUB page](https://www.vision.caltech.edu/datasets/cub_200_2011/)
warns that CUB images overlap with ImageNet and that a pretrained network may
have seen CUB test photos. Our internal train/validation/test ID/hash isolation
checks do not audit ImageNet. The supplied EDA audit covered the two pinned CUB
sources, not the pretraining corpus. Therefore, none of these checkpoints has
verified CUB/ImageNet disjointness. Scores are held out from CUB fine-tuning, not
certified independent of pretraining. No exact overlap count is established here.

ImageNet pretraining itself is the intended transfer-learning starting point.
Possible reuse of the exact same photographs is the separate contamination
concern. Switching to another ImageNet-pretrained architecture would retain that
concern. Its impact on these particular scores has not been measured; it could
bias evaluation upward and is not evidence explaining the large training gap.
Do not silently change the official split or label an unverified subset clean.

For a strict pretraining-disjoint benchmark, verify image-level overlap against
the relevant pretraining corpus and report an audited protocol, or use weights
with verifiably disjoint pretraining data. The existing scratch-CNN baseline is
also essential for comparison. Those are separate from regularization tuning.

## Confirmed findings

- The fixed memberships contain 4794 training, 1200 validation and 5794 official
  test records. Training has 23 or 24 images per species; validation has six per
  species. Class weighting or oversampling is not supported by a material count
  imbalance in these data.
- Independently checking canonical IDs, encoded SHA-256 and normalized-pixel
  SHA-256 found zero intersections between each pair of partitions. This does not
  certify absence of perceptual near-duplicates or semantic annotation mistakes.
  The known exact pair remains together in validation, as the EDA policy specifies.
- Class labels follow the source's ordered names, not alphabetical filenames.
  Dataset access checks labels, filenames and source IDs. Manifest fingerprints,
  source revisions and eligibility are checked before loading samples.
- Source images are fully decoded, EXIF-oriented and converted to RGB; transparency
  is composited on white. Four original grayscale training images are supported.
- The pipeline uses external ImageNet normalization. The report's raw 5994-image
  mean/SD summaries include validation and are descriptive, not fitted tensor
  normalization parameters. They are not substituted into the model pipeline.
- The EDA report records that earlier crops removed bird anatomy in source rows
  1429 and 305. Full-image fit/padding is deliberately used. It preserves content
  but reduces effective bird resolution, especially for unusual aspect ratios.
- Current mild augmentation is flip plus modest brightness/contrast and expanded
  rotation after fitting/padding. The shared EDA transforms are unchanged. Eval
  preprocessing is deterministic and disables all augmentation.
- Encoder freezing works in the head stage. Fine-tuning unfreezes layer4 or its
  final bottleneck as configured. Frozen layers remain in eval mode. BatchNorm
  running buffers remain fixed; selected affine parameters can still train.
- The selected encoder parameters and classifier have separate AdamW learning
  rates. Optimizers are rebuilt on stage transitions and stale gradients cleared.
  Training calls backward/step; validation and official testing do neither.
- CNN training uses dropout 0.3, AdamW weight decay 0.01 and label smoothing 0.1
  in the regularized trial. The Mixup trial replaces fine-tuning smoothing with
  mixed targets. Its training accuracy is weighted target agreement, which must
  not be compared directly with clean validation accuracy.
- The three-epoch head warm-up is not a complete frozen-backbone benchmark.
  The runner currently has no learning-rate scheduler. Neither fact proves the
  cause of the observed gap.
- Best checkpoints use highest validation accuracy, keeping earlier ties. Early
  stopping separately monitors validation loss. That is the implemented policy;
  no test score selects an epoch. Full training resumption is not implemented.
- The extra L2 classifier can fit all training examples even when its CNN encoder
  is never fine-tuned. A weakly penalized high-dimensional classifier is therefore
  one contributor to perfect training accuracy; it is not evidence that freezing
  failed. Posthoc confidence scaling changes NLL, not argmax accuracy.

Configuration clarification: 0.35 was not an AdamW weight decay; it was the
explicit frozen-feature classifier L2 strength in
`CE + 0.5 * strength * sum(W**2)`. The 60.33% run was the original baseline,
not the regularized comparison; the regularized CNN reached 63.00%, while the slower
learning-rate trial reached 58.00%.

## Verification and diagnostics

The relevant data, model, training and evaluation regression suite passed 176
tests. One optional cached-source integration test was initially skipped because
its opt-in environment variable was unset; it subsequently passed with the real
cached source after opting in. Fresh diagnostic loading independently
verified the real manifest and train/validation mapping. No source algorithms
were changed before these checks.

Validation errors were regenerated from the selected classifier and cached pooled
features, after confirming cached labels against the sorted manifest. Accuracy
reproduced exactly at 67.50%. Examples include Acadian to Yellow-bellied
Flycatcher (four of six), Rufous to Ruby-throated Hummingbird (three), and Great
Grey to Loggerhead Shrike (three). Tree Sparrow has zero of six correct. Six
examples with native/224-pixel views are retained. These are selected examples,
not estimates of object size, blur or annotation quality across the dataset.

Existing official-test per-class statistics were read descriptively. Further
development diagnostics open only the official training source and its derived
validation members. The official test is not reopened for tuning.

### Original ImageNet encoder baseline

Diagnostic MLflow run: `d52e1354ab704529a57c4f1c14b8bfa5`.
Every encoder parameter and buffer was compared with the cached original
ImageNet weights; all matched. Deterministic 224-pixel fit features were collected
on train/validation. Four L2 strengths were declared before fitting; feature mean
and SD came exclusively from training. Confidence calibration was disabled.

| L2 strength | Clean training accuracy | Validation accuracy | Validation macro F1 |
| --- | ---: | ---: | ---: |
| 0.003 | 100.00% | 57.83% | 0.5671 |
| 0.01 | 100.00% | 56.67% | 0.5573 |
| 0.03 | 99.73% | 56.75% | 0.5573 |
| 0.1 | 95.45% | 54.08% | 0.5279 |

In this small predefined comparison, removing encoder adaptation did not improve
on the selected fine-tuned model's 67.50% validation accuracy. The diagnostic is
not an exhaustive hyperparameter search and does not prove every frozen approach
is worse.

### Resolution hypothesis

The next diagnostic changes only deterministic evaluation resolution from 224
to 320 while retaining full-image fit, normalization and exact saved weights.
Both raw CNN heads and their previously fitted classifier alternatives are
measured on clean training and validation. Higher effective detail is motivated
by the EDA's explicit padding/resolution trade-off, not by test-class errors.
It is a validation-selected preprocessing experiment, not a new CNN training run.

The shortcut failed: the regularized encoder's old classifier fell from 63.00%
to 56.67% validation accuracy, and its refitted classifier fell from 67.50% to
62.67%. The Mixup/refitted candidate fell from 64.17% to 59.92%. The corresponding
MLflow diagnostic is `bcc3c82f50464668a3e4fa50189bd5c4`. The existing weights/configs
are therefore retained at their original 224-pixel evaluation resolution. This
does not rule out separately training and validating a model at higher resolution.

### Class-centroid diagnostic

A lightweight train-only class-centroid classifier was also checked using the
existing 224-pixel embeddings, with and without per-image normalization before
averaging each class. Its best regularized-encoder result was 87.94% clean training
and 65.00% validation accuracy, a 22.94-point gap. This is a possible lower-fit
trade-off, but does not beat the primary model's validation accuracy. It is kept
as a diagnostic, not automatically substituted into the project's architecture.

### Controlled training augmentation comparison

The implemented `spatial` profile changes only augmentation relative to the
regularized CNN configuration. It rotates native pixels before fitting, then
randomly varies full-image size (85-100% of the longest canvas side) and placement.
It retains the complete image and avoids the EDA's demonstrated cropping failure.
Brightness/contrast and flip settings remain mild; validation remains identical.
No scheduler, architecture change, test-set retuning or added erasing is combined
with this comparison. Its config is `configs/resnet50_spatial.yaml`.

Ten new geometry/validation checks plus the existing regularization/transform
checks passed (32 total). Colored corner markers verify full-content retention,
and seeded replay and identical validation tensors are checked. Actual EDA cases
were visually inspected in the saved preview before training.

After the augmentation implementation, the complete repository suite passed
**571 tests in 113.66 seconds, with no failures or skips**, including the opt-in
real cached-source integration. The EDA/shared transform implementation and
manifest memberships remain unchanged.

The full controlled run (`0f42a834b6124cb4a86ee9845b7c773f`) finished its three
head and ten fine-tuning epochs. Highest validation accuracy was 63.17%, selected
at fine-tuning epoch eight because earlier ties are retained. Clean training
accuracy was 92.41%, giving a 29.24-point gap. Compared with the earlier CNN's
94.20% / 63.00% / 31.20-point gap, this is a 1.96-point gap reduction and only
two additional correct validation images. It is a modest result, not a complete
solution or a replacement for the best classifier-refit candidate.

### More constrained discriminant classifier

Using the same frozen encoders, a linear discriminant classifier estimates class
means and a pooled within-class covariance from training features. Shrinkage
regularizes the covariance toward `trace(S)/D * I`; it is not AdamW decay or the
logistic objective's L2 coefficient. The fitted coefficients remain compatible
with the existing PyTorch linear classifier. This is the standard shared-covariance
linear discriminant form, following the
[official mathematical reference](https://scikit-learn.org/stable/modules/lda_qda.html).

The diagnostic first tested shrinkage 0.5, then declared 0.8/0.9 before checking
those values, across the existing regularized and Mixup encoders. This was a small
adaptive validation exploration, not an independent confirmation or a completed
grid-versus-random-search comparison. No new backbone was trained for these fits.

| Candidate | Clean training accuracy | Validation accuracy | Gap (points) | Validation macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Previous primary logistic fit | 100.00% | 67.50% | 32.50 | 0.6695 |
| New balanced discriminant fit (Mixup, shrinkage 0.9) | **91.91%** | **67.92%** | **23.99** | **0.6758** |
| New accuracy-oriented discriminant fit (regularized, shrinkage 0.5) | 98.64% | **70.08%** | 28.56 | **0.6987** |

The balanced candidate reduces the previous primary's gap by 8.51 points while
gaining five correct validation images. The accuracy-oriented candidate gains
31 correct validation images and reduces its gap by 3.94 points. Some overfitting
remains in both. The smaller validation change is particularly sensitive to
sampling/seed variation; repeated use of this same validation set may also cause
optimism. The new candidates have not been evaluated on the official test set.

Both exports include a positive validation-fitted confidence scale. The unscaled
Gaussian-derived logits were overconfident (e.g., NLL about 10.75 for the 0.5
regularized diagnostic), despite useful accuracy. Scaling preserves class
ordering and lowers the selected exports' validation NLL to about 1.17. This is
calibration tuning, not an independent probability-quality estimate.

| Export | Local head-refit run ID |
| --- | --- |
| Balanced | `d55f4961d5c64e8e94a937acdef75d29` |
| Accuracy-oriented | `6a390726b83c41969abe8a9d1d7a9f6f` |

Complete models are in `outputs/resnet50/head_refit/<run_id>/model.pt`, with
metrics/config JSON, local MLflow tracking and the usual inference-only schema-2
marker. The original encoder weights/buffers are retained. See
[training/refit commands](resnet50_training.md) for reproducible fitting and loading.

Ten discriminant checks test nearest-mean equivalence at full shrinkage,
training-only fitting, rank-deficient features, invalid shrinkage and degenerate
covariance. Existing logistic tests continue to pass. The final complete suite
passed **581 tests in 113.53 seconds, with no failures or skips**.

## Whether to change the backbone

The review found no implementation bug that requires replacing ResNet-50. With
the 200-class head, it has 23,917,832 total parameters and 15,374,536 trainable
parameters during layer4 fine-tuning. Frozen-head training uses 409,800 trainable
parameters. The frozen ImageNet diagnostic already showed classifier overfitting,
so reducing backbone capacity alone is not an established fix.

The follow-up comparison keeps ResNet-50 and changes its pretraining source to the
authors' `inat2021_supervised_from_scratch` checkpoint. Their
[model catalogue](https://github.com/visipedia/newt/blob/main/benchmark/configs.py)
documents iNaturalist2021 pretraining without ImageNet initialization. This
changes the learned representation, not the architecture or parameter count.
It does not establish that iNaturalist and CUB contain no duplicate photographs;
that requires a separate audit. Validation accuracy and overfitting must still
be measured, rather than inferred from the pretraining dataset's name.
The completed follow-up reached 84.08% validation and 84.05% official test
accuracy, with a 12.68-point clean training/validation gap; see the
[baseline results and limitations](resnet50_test_results.md#initial-inaturalist2021-cnn-2026-10-26).
A later validation-only comparison of 12 classifier fits, with fixed encoders,
selected discriminant shrinkage 0.1 on the fine-tuned encoder. This improved
validation accuracy to 86.67% and official test accuracy to 85.23%, reducing the
clean train/validation gap to 10.87 points. The encoder is unchanged; see the
[classifier comparison](resnet50_test_results.md#inaturalist2021-classifier-improvement-2026-10-26).
A subsequent gap-focused comparison selected the original frozen iNaturalist
encoder with discriminant shrinkage 0.9: 91.01% clean training, 85.75%
validation, and a 5.26-point gap. This balanced option has not been tested on
the official test set; the earlier model's 85.23% test score remains attached
to that earlier checkpoint. Both options are preserved.

## Boundaries for further work

Do not change architecture, class mappings, fixed memberships or official test
assignments as a reaction to these scores. Do not report a smaller gap obtained
by worse validation performance as a success. Compare accuracy, macro F1, clean
train/validation gap and unsmoothed validation NLL together. Calibration-fitted
NLL must remain labelled as a tuning measurement.

If the resolution diagnostic is insufficient, the smallest next controlled
sequence is a content-preserving augmentation comparison, then a scheduler
comparison with the augmentation held fixed. Add one main factor per comparison.
Any new geometry must be visually checked against the EDA's crop-failure cases.
Full-backbone unfreezing and additional architectures are later options, not
automatic fixes. Multiple seeds remain needed to judge stability.

The project specification requires both grid and random search and a comparison
between those methods. These adaptive diagnostics do not satisfy that complete
deliverable. Grad-CAM, scratch-CNN comparison and final integration also remain.
Previously reported official-test scores belong to the fixed candidates tested
then; they must not be relabelled as untouched results for later tuned candidates.
