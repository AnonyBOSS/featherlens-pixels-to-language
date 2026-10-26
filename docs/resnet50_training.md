# ResNet-50 training

Run these commands from the repository root using the project's virtual environment.
Install a matching CUDA-enabled PyTorch build first for GPU training, then install
the data and tracking dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-resnet50.txt
```

The configuration is `configs/resnet50.yaml`. Preparation paths, checkpoint paths,
and tracking paths are relative to the repository root. The runner reuses the
validated EDA manifest and pinned official training source for training/validation;
it does not load official test images.

## Check the data/model connection

```powershell
.\.venv\Scripts\python.exe -m src.training.resnet50_runner --config configs/resnet50.yaml --check-batch
```

This checks one validation batch without updating weights. The first model
construction downloads the ImageNet weights into PyTorch's normal cache.

## Check the complete training pipeline

```powershell
.\.venv\Scripts\python.exe -m src.training.resnet50_runner --config configs/resnet50.yaml --smoke-test
```

Smoke mode runs one epoch per stage with batch size at most four and at most two
batches of images per partition. Effective settings are saved with the run. Runs
are tagged `smoke` and use a separate experiment name ending in `_smoke`. Their
scores check the pipeline and must not be reported as model evaluation results.

## Run a full experiment

```powershell
.\.venv\Scripts\python.exe -m src.training.resnet50_runner --config configs/resnet50.yaml --train
```

The current defaults train the classifier alone for three epochs, then train the
classifier and encoder `layer4` for fifteen epochs. Each stage gets a fresh AdamW
optimizer. Fine-tuning continues from the final head-stage weights. Encoder
BatchNorm running statistics stay fixed. The best overall checkpoint is selected
by validation accuracy across both stages, retaining the earlier one on ties.

Each run has a unique folder under `outputs/resnet50/training/<run-id>/` (or
`outputs/resnet50/smoke/<run-id>/`) containing:

- `last.pt`: latest completed epoch, including optimizer state.
- `best.pt`: highest validation accuracy across both stages.
- `head_best.pt` and `finetune_best.pt`: best checkpoint from each stage.
- `config.json`: effective settings, ordered class names, preparation fingerprint,
  and source revisions.
- `summary.json`: epoch metrics and the selected best epoch after a successful run.

Checkpoints contain model/optimizer states, stage, stage-local epoch, validation
metrics, best accuracy, and configuration. A fresh model can load its weights with
`torch.load(path, map_location=device, weights_only=True)` followed by
`model.load_state_dict(checkpoint['model_state_dict'])`. Set the saved stage before
continuing training and reconstruct the matching optimizer before loading its
state. The runner currently starts fresh runs; it has no resume command and does
not restore RNG/loader state for exact replay.

## View local MLflow results

MLflow records parameters, per-epoch loss/accuracy, stage, duration, run status,
configuration, summary, and a copy of the best checkpoint. The local SQLite
database is `outputs/resnet50/mlruns/mlflow.db`; artifacts are stored alongside it.
No tracking server is needed during training. To view the results afterward:

```powershell
$resnetTrackingDb = (Resolve-Path 'outputs/resnet50/mlruns/mlflow.db').Path.Replace('\', '/')
.\.venv\Scripts\python.exe -m mlflow server --backend-store-uri "sqlite:///$resnetTrackingDb" --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000` in your browser. Stop the viewer with Ctrl+C.

Generated models and logs are ignored by Git. Upload trained model files from
completed reported experiments to Google Drive manually and include the required
Drive link in the project submission. Smoke-test checkpoints are test artifacts.
Final test-set evaluation, broader metrics, tuning searches, and Grad-CAM remain
separate steps.

## iNaturalist2021 pretraining without ImageNet initialization

`configs/resnet50_inaturalist.yaml` keeps the same ResNet-50 architecture and
regularized training recipe, but selects `INAT2021_FROM_SCRATCH`. The authors'
[catalogue](https://github.com/visipedia/newt/blob/main/benchmark/configs.py)
identifies `inat2021_supervised_from_scratch` as supervised iNaturalist2021
pretraining with no earlier ImageNet weights. The similarly named
`inat2021_supervised` checkpoint does use ImageNet initialization and is not the
checkpoint selected here. "From scratch" describes the authors' pretraining;
our CUB model still uses transfer learning.

The authors distribute the checkpoint inside their
[model archive](https://cornell.app.box.com/s/bnyhq5lwobu6fgjrub44zle0pyjijbmw).
The required member is
`cvpr21_newt_pretrained_models/pt/inat2021_supervised_large_from_scratch.pth.tar`.
Place that file in `outputs/pretrained/`. The configuration pins its SHA-256;
the loader rejects a different or incomplete file. This is a locally computed
file identity, not an independently published author checksum. Checkpoint files
and download provenance stay outside Git under the ignored `outputs/` tree.

The runner constructs the model with `weights=None`, then loads the external
encoder strictly and discards its 10,000-class classifier. The new CUB head has
200 outputs. It does not fetch ImageNet weights. The `normalization: imagenet`
setting refers only to fixed RGB mean/std constants, also used by the authors'
[preprocessing](https://github.com/visipedia/newt/blob/main/benchmark/pt_resnet_feature_extractor.py);
normalization does not load or train on ImageNet images. Geometry remains the
EDA full-image fit/pad transform so bird anatomy is preserved.

```powershell
.\.venv\Scripts\python.exe -m src.training.resnet50_runner --config configs/resnet50_inaturalist.yaml --check-batch
.\.venv\Scripts\python.exe -m src.training.resnet50_runner --config configs/resnet50_inaturalist.yaml --train
```

Runs use the separate `resnet50_inaturalist` experiment and
`outputs/resnet50_inaturalist/training/<run-id>/` directory. The MLflow database
is shared with the existing ResNet experiments. Saved CUB checkpoints contain
the complete encoder and classifier, so evaluation does not require the
original external pretraining file. Keep the saved configuration and class
mapping with models uploaded to Google Drive.

Nature pretraining is a reasonable hypothesis for bird recognition, not an
accuracy guarantee. Compare clean training and validation accuracy, macro F1
and validation loss on the same fixed split. No image-level CUB/iNaturalist
overlap audit has been performed. Pretraining provenance alone does not prove
that the two datasets contain no duplicate photographs.

The first completed run selected fine-tuning epoch 2, reaching **84.08%
validation** and **84.05% official test accuracy**. Clean training accuracy was
96.77%, leaving a 12.68-point training/validation gap. See the
[recorded baseline results and model paths](resnet50_test_results.md#initial-inaturalist2021-cnn-2026-10-26).
The fixed test result is a report, not a target for additional tuning.

A subsequent 12-trial train/validation-only head comparison selected a
regularized discriminant classifier on this same fine-tuned encoder. It reached
**86.67% validation** and **85.23% official test accuracy**. The clean
train/validation gap fell to 10.87 points. No Python source changes were needed;
the existing head-refit command reproduces the selected fit:

```powershell
.\.venv\Scripts\python.exe -m src.training.resnet50_head_refit --source-checkpoint outputs/resnet50_inaturalist/training/9c6e5c07be24414b95fc5717e3d01eeb/best.pt --method discriminant --shrinkage 0.1 --device cuda
```

The selected inference-only export is
`outputs/resnet50_inaturalist/head_refit/375ca28ba526439090f6000aedfc6ee1/model.pt`.
Keep its adjacent `config.json` and `metrics.json` with it for Google Drive
handoff. The original CNN training checkpoints remain available. See the
[comparison, verification and limitations](resnet50_test_results.md#inaturalist2021-classifier-improvement-2026-10-26).

For a smaller train/validation gap, the balanced alternative keeps the original
iNaturalist encoder frozen and uses discriminant shrinkage 0.9:

```powershell
.\.venv\Scripts\python.exe -m src.training.resnet50_head_refit --source-checkpoint outputs/resnet50_inaturalist/training/9c6e5c07be24414b95fc5717e3d01eeb/head_best.pt --method discriminant --shrinkage 0.9 --device cuda
```

Its saved export is
`outputs/resnet50_inaturalist/head_refit/7e70b477f56041ceaac8d3e30961e559/model.pt`.
It scores 91.01% on clean training images and 85.75% on validation, leaving a
5.26-point gap. It has not been evaluated on official test images; the previous
model's 85.23% test accuracy must not be assigned to this model. See the
[balanced-model tradeoff](resnet50_test_results.md#balanced-inaturalist-option-smaller-validation-gap).

### Search script and generated-file cleanup

The temporary `outputs/resnet50_inaturalist/validation_search/run_search.py`
collected deterministic encoder features, compared 12 classifier fits, selected
by validation, and exported the winner. It is unnecessary for normal training,
head refitting, evaluation, prediction or feature extraction. Its working copy
was deleted after verifying that a byte-identical copy is retained in the
MLflow search run's `code` artifacts. Restore that copy to its original working
location if the complete initial search needs to be replayed.

The selected fits can be reproduced directly with the commands above. Temporary
test/smoke outputs and regenerable feature caches were removed. Source, tests,
preparation files, real training checkpoints, final exports, reports and MLflow
history were preserved. The retained search JSON and MLflow records still show
what was tried. New calls to the head-refit command collect their own features;
they do not depend on the removed cache files. Cleanup is recorded in
`outputs/resnet50_inaturalist/cleanup_report.json`.

## Regularization trials

The original `configs/resnet50.yaml` retains its baseline behavior. Optional
settings default to zero dropout, zero label smoothing, baseline augmentation,
and no early stopping, so existing checkpoints remain compatible.

`configs/resnet50_regularized.yaml` tests classifier dropout 0.3, label smoothing
0.1 for training, weight decay 0.01, and mild training-only augmentation. The mild
profile adds brightness/contrast jitter and rotations up to ten degrees on the
already resized image. Expanded rotation followed by resizing avoids clipping;
validation preprocessing is unchanged. The shared EDA transform file is unchanged.

`configs/resnet50_regularized_slow.yaml` uses the same settings but lowers the
fine-tuning encoder/classifier learning rates to 0.000003/0.00003. Both trials cap
fine-tuning at ten epochs and stop after three epochs without validation-loss
improvement of at least 0.001. Checkpoint selection still uses validation accuracy;
stopping and checkpoint selection therefore monitor different quantities.

```powershell
.\.venv\Scripts\python.exe -m src.training.resnet50_runner --config configs/resnet50_regularized.yaml --train
.\.venv\Scripts\python.exe -m src.training.resnet50_runner --config configs/resnet50_regularized_slow.yaml --train
```

Training loss includes label smoothing in these trials and cannot be compared
directly to the baseline training loss. Training accuracy also uses augmentation
and dropout, while validation disables both. Compare identical validation metrics
and, when measuring the train/validation gap, evaluate both partitions with the
same deterministic preprocessing and the model in evaluation mode. These two
trials are controlled experiments, not the full required grid/random search.

## Conservative fine-tuning and Mixup

`model.finetune_scope` defaults to `layer4`. Setting it to `last_block` unfreezes
only the last residual bottleneck during fine-tuning: 4,462,592 encoder parameters
plus 409,800 classifier parameters, instead of all 14,964,736 layer4 parameters.
Earlier blocks remain frozen in both parameter updates and training mode. Encoder
BatchNorm running statistics remain fixed, including inside the selected block.
Module names and state-dictionary keys are unchanged; when reconstructing training,
pass the saved scope into the constructor before building/loading the optimizer.

`training.finetune.mixup_alpha` optionally enables Mixup during fine-tuning. The
default is zero (disabled). Positive values draw a Beta-distributed mixing weight,
blend pairs of images, and blend their two classification losses. Training
accuracy then measures weighted target agreement, not ordinary classification
accuracy; this definition is tagged in MLflow. Validation never uses Mixup.
Stage-local `label_smoothing` can override the global setting; the Mixup trial
uses zero fine-tuning label smoothing to avoid stacking the two target-softening
methods. Its head warm-up remains identical to the other regularized runs.

- `configs/resnet50_last_block.yaml`: smaller fine-tuning scope with the existing
  regularization settings.
- `configs/resnet50_mixup.yaml`: the successful full-layer4 architecture with
  Mixup alpha 0.2 and no additional label smoothing during fine-tuning. The
  queued experiment was changed from the last-block scope after that scope
  showed slower validation adaptation in its early epochs.

Both trials use the same seed, data split, batch size, ten-epoch fine-tuning cap,
and validation-loss stopping rule. Rank checkpoints by identical validation
metrics, and measure any accuracy gap using deterministic clean evaluation.

## Refit a regularized classifier on a frozen encoder

`src/training/resnet50_head_refit.py` fits a new linear classifier on deterministic
2048-dimensional embeddings from a saved encoder. It uses the existing 4794/1200
train/validation split and never opens the official test split. It verifies the
checkpoint's class mapping, manifest hash and pinned source revisions.

The objective is mean cross entropy plus `0.5 * l2_strength * sum(weight**2)`.
This explicit L2 penalty is different from the AdamW weight-decay setting in the
CNN training configuration. The encoder stays fixed. LBFGS optimizes the small
classifier using all training embeddings; it replaces AdamW only for this refit.
Feature means and standard deviations come exclusively from training embeddings.

For the balanced candidate, run this after its source CNN experiment exists:

```powershell
.\.venv\Scripts\python.exe -m src.training.resnet50_head_refit --source-checkpoint outputs/resnet50/training/f842795088f84057867000e7378ae907/best.pt --l2-strength 0.35 --device cuda
```

For the accuracy-oriented alternative, use the source checkpoint from
`487597dac3854f30a101214619872d38` and `--l2-strength 0.003`.
These parameters were selected through exploratory validation tuning, not an
independent test or the complete required grid/random-search comparison.

A positive logit scale is fitted on validation NLL. It changes confidence and
NLL, while preserving class ordering and accuracy. Its reported validation NLL
is therefore a tuning result. Feature normalization and this scale are folded
into the ordinary linear layer, requiring no new inference preprocessing.

Each command creates a local MLflow run and an inference-only schema-2
`outputs/resnet50/head_refit/<run_id>/model.pt`, plus metrics/config JSON. It
contains no optimizer state or invented training epoch. Load it as follows:

```python
import torch
from src.models.resnet50 import ResNet50Classifier

checkpoint = torch.load("path/to/model.pt", map_location="cpu", weights_only=True)
settings = checkpoint["config"]["model"]
model = ResNet50Classifier(
    num_classes=checkpoint["num_classes"], weights=None,
    dropout=settings.get("dropout", 0.0),
    finetune_scope=settings.get("finetune_scope", "layer4"),
)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()
```

Use the saved image size, fit geometry and ImageNet normalization for images.
These exports support inference, not resuming the original AdamW run. Their raw
encoder embeddings are unchanged; the fitted normalization affects only the
classifier. Upload selected model files and their configuration to Google Drive
for delivery after review. The command does not upload files automatically.

## Official test evaluation

After selecting a fixed checkpoint using validation data, evaluate it with:

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.resnet50_evaluation --checkpoint outputs/resnet50/head_refit/3897e4b1675c49a7a6c4402fa23ebfa5/model.pt --device cuda
```

Repeat `--checkpoint` to evaluate a predeclared comparison model in the same
command. This is evaluation only: strict checkpoint loading, `model.eval()`,
inference mode, deterministic saved preprocessing, no augmentation and no
optimizer. All 5794 official test images must be counted exactly once.
The evaluator checks preparation fingerprints, pinned revisions and class order.

Each evaluation creates a local MLflow run in `resnet50_official_test` and saves
`outputs/resnet50/evaluation/<run_id>/report.json`, predictions JSONL, a confusion
matrix CSV and PNG. The report includes checkpoint SHA-256, accuracy, unsmoothed
cross entropy, macro precision/recall/F1 and per-class metrics. Confidence scaling
already present in a checkpoint is retained; it is never refitted on test labels.

Once these test scores are inspected, do not repeatedly tune models using them
and continue presenting the same test set as an untouched final assessment.

## Content-preserving spatial augmentation

`configs/resnet50_spatial.yaml` changes only the augmentation profile relative
to `resnet50_regularized.yaml` (plus the experiment name for tracking). It keeps
the same seed, 224-pixel fit geometry, batch size, split, model, epoch budgets,
learning rates, dropout, smoothing, weight decay and stopping/selection rules.

`data.augmentation: spatial` flips and mildly adjusts brightness/contrast,
rotates the native image with an expanded canvas, then fits its complete content
at 85-100% of the output's longest side. `RandomFitAndPad` randomly chooses an
available placement inside the square. It uses Torch RNG for seeded replay and
preserves aspect ratio within pixel rounding. It never crops or wraps content.
No hue/saturation jitter or random erasing is added. The profile requires `fit`
geometry. Validation/test still use the original deterministic shared transform.

```powershell
.\.venv\Scripts\python.exe -m src.training.resnet50_runner --config configs/resnet50_spatial.yaml --train
```

The EDA report's crop-sensitive rows 1429/305 and extreme-aspect rows 3228/5590
were previewed before training. Preview and diagnostic artifacts are local under
`outputs/resnet50/pipeline_review/`. See [pipeline review](resnet50_pipeline_review.md)
for confirmed findings, negative diagnostics and experiment limitations. The
presence of this configuration does not itself establish improved accuracy or
elimination of overfitting; compare the measured checkpoints.

## Shrinkage discriminant classifier

The same frozen-feature refit module also supports a more constrained linear
classifier (`--method discriminant`). It computes one mean feature vector per
species and a shared within-class covariance from training features. It shrinks
that covariance toward a uniform-variance diagonal matrix before solving for
linear weights. All fitted means/covariances use training records only. Class
priors are equal; the pooled covariance divisor is `N - K` (images minus classes).

```powershell
# Balanced candidate: Mixup encoder, stronger covariance shrinkage.
.\.venv\Scripts\python.exe -m src.training.resnet50_head_refit --source-checkpoint outputs/resnet50/training/f842795088f84057867000e7378ae907/best.pt --method discriminant --shrinkage 0.9 --device cuda

# Accuracy-oriented candidate: regularized encoder, weaker covariance shrinkage.
.\.venv\Scripts\python.exe -m src.training.resnet50_head_refit --source-checkpoint outputs/resnet50/training/487597dac3854f30a101214619872d38/best.pt --method discriminant --shrinkage 0.5 --device cuda
```

For shrinkage `a`, the fitted covariance is
`(1-a) * S + a * trace(S)/D * I`. At 0.9, 90% of the estimate comes from that
simpler target, reducing dependence on noisy correlations between the 2048
features. This is a different parameter from both AdamW weight decay and the
logistic head's explicit L2 penalty. It changes the fitting rule, not the number
of stored classifier weights.

The fit uses a closed-form `torch.linalg.solve`, not gradient optimization.
Global feature centering and a positive validation-fitted confidence scale are
folded into the ordinary linear layer. Raw Gaussian-derived logits can be very
overconfident, so calibrated exports retain the validation-fitted scale. It
preserves class ordering; validation NLL after calibration remains a tuning
measurement. The shared-covariance Gaussian assumption is an approximation,
not a verified distributional property of the encoder features.

Output paths, schema-2 inference-only format, strict model loading and Google
Drive delivery are the same as for logistic refits. Logistic remains the default
method and its existing commands are supported. The discriminant command requires
`--shrinkage` and rejects `--l2-strength` to avoid confusing the two penalties.
The mathematical classifier and covariance-regularization rationale follow the
[official discriminant-analysis documentation](https://scikit-learn.org/stable/modules/lda_qda.html).
The implementation here uses PyTorch and needs no new fitting dependency.
