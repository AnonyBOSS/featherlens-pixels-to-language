# Selected ResNet-50 transfer-learning workflow

Start here for the selected model. Older experiment results remain in
[the evaluation history](resnet50_test_results.md),
[regularization results](resnet50_regularization_results.md),
[capacity results](resnet50_capacity_results.md), and
[pipeline review](resnet50_pipeline_review.md).

## Selected model

- Architecture: ResNet-50, pretrained on iNaturalist2021 without ImageNet initialization.
- Encoder training: 224px, using `configs/resnet50_inaturalist.yaml`.
- Final classifier: shrinkage LDA, shrinkage 0.05, fitted on frozen 320px features
  from original and horizontally flipped training images.
- Prediction: deterministic full-image fit-and-pad at 320px; one view, no TTA.
- Checkpoint: `outputs/resnet50_inaturalist/augmented_pca_search/accuracy_challenger/model.pt`.
- SHA256: `fcb1e22360fb802da312b9075797d5ada8f8250f975769b1cba3b48211f185c8`.

| Split | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Train | 98.06% | 98.17% | 98.06% | 98.06% |
| Validation | 87.92% | 89.61% | 87.92% | 87.91% |
| Test | 87.54% | 88.29% | 87.65% | 87.62% |

These are measured results for this saved checkpoint, not a guarantee for a rerun.
Overfitting remains: train minus validation accuracy is 10.14 percentage points.
Validation was reused across experiments, one encoder seed was trained, and test
scores were previously inspected. The test is not an untouched final assessment.

## Read the code in this order

| File | Responsibility |
|---|---|
| `configs/resnet50_inaturalist.yaml` | Training settings: paths, seed, batch size, learning rates, regularization and epoch budgets. |
| `src/models/resnet50.py` | Encoder, pooling and classifier; head/fine-tuning freezing; strict iNaturalist weight loading. |
| `src/data/resnet50_transforms.py` | Training augmentation and deterministic evaluation preprocessing, reusing shared image transforms. |
| `src/training/resnet50_training.py` | Build AdamW, train one epoch, validate one epoch and save a checkpoint safely. |
| `src/training/resnet50_experiment.py` | Coordinate stages and epochs, early stopping, best checkpoints and MLflow. |
| `src/training/resnet50_runner.py` | Read command arguments/configuration, seed randomness, select device and build data/model before starting the experiment. |
| `src/training/resnet50_head_refit.py` | Optional post-training classifier fitting; the selected model uses its LDA path and training-only original/flip features. |
| `src/evaluation/resnet50_evaluation.py` | Evaluate a fixed checkpoint on the full official test set and save metrics/confusion matrix. |
| `src/resnet50_predict.py` | Load a checkpoint and predict species for one local image. |
| `src/visualization/resnet50_gradcam.py` | Explain a class logit with a heatmap without updating model weights. |

Shared `src/data/source.py`, `preparation.py`, `classification.py`, `image_io.py`
and `transforms.py` provide source revisions, verified split membership, image
loading and batching. They also serve other project work. Tests check freezing,
normalization, metrics, export/reload, label order and explanation correctness.

```text
YAML -> runner -> data/model -> experiment -> train/validate -> encoder checkpoint
                                                                  |
                                               optional classifier refit
                                                                  |
                                                   selected full checkpoint
                                                                  |
                                                evaluation / prediction / Grad-CAM
```

Head training freezes the encoder and learns the classifier through AdamW.
Fine-tuning learns the selected encoder block AND the classifier together. The
current YAML permits 3 head epochs and up to 10 fine-tuning epochs, with early
stopping. It is different from the older 3+15 ImageNet baseline.

Refitting is a separate operation after those stages. It freezes the adapted
encoder again and calculates replacement classifier weights. The selected LDA
path uses training class means and a shared regularized covariance estimate;
it does not use an epoch loop or `optimizer.step()`. Refitting is optional for
transfer learning, but it is already included in this selected checkpoint.
The module retains the earlier logistic fitting method for historical comparisons.

## iNaturalist2021 pretraining without ImageNet initialization

For encoder training, obtain the authors' pretrained file from their
[model archive](https://cornell.app.box.com/s/bnyhq5lwobu6fgjrub44zle0pyjijbmw).
Extract `cvpr21_newt_pretrained_models/pt/inat2021_supervised_large_from_scratch.pth.tar`
to `outputs/pretrained/`. The configuration pins its locally computed SHA256;
the loader rejects a different file. The
[authors' catalogue](https://github.com/visipedia/newt/blob/main/benchmark/configs.py)
distinguishes this checkpoint from their ImageNet-initialized variant.

The runner constructs the network with `weights=None` and loads the iNaturalist
encoder. `normalization: imagenet` names RGB mean/std constants; it does not load
ImageNet weights. The prepared canonical data must be available in `preparation/`.

Use the working CUDA-enabled project environment. Workflow dependencies are in
`requirements-resnet50.txt`; commands below run from the repository root.

## Training and checks

```powershell
.\.venv\Scripts\python.exe -m src.training.resnet50_runner --config configs/resnet50_inaturalist.yaml --check-batch
.\.venv\Scripts\python.exe -m src.training.resnet50_runner --config configs/resnet50_inaturalist.yaml --smoke-test
.\.venv\Scripts\python.exe -m src.training.resnet50_runner --config configs/resnet50_inaturalist.yaml --train
```

The runner defaults to the iNaturalist YAML when `--config` is omitted. Explicit
`configs/resnet50.yaml` remains available for the historical ImageNet baseline.
`--check-batch` only checks a batch; `--smoke-test` trains tiny bounded subsets
and does not produce reportable performance. `--train` starts a new full run.
The CLI modes are mutually exclusive. You do not need to train again to use the
selected checkpoint.

Training prints a unique run directory. It saves best/last/stage checkpoints,
effective configuration and an epoch summary; model tensors and optimizer state
are included for training checkpoints. The ResNet runner does not implement
training resumption. Smoke runs remain separate from genuine training results.
MLflow records parameters, epoch metrics and checkpoint references/artifacts.

## Reproduce the selected classifier refit

The archived encoder source is
`outputs/resnet50_inaturalist/training/9c6e5c07be24414b95fc5717e3d01eeb/best.pt`.
To fit the selected recipe on that frozen encoder:

```powershell
.\.venv\Scripts\python.exe -m src.training.resnet50_head_refit --source-checkpoint "outputs/resnet50_inaturalist/training/9c6e5c07be24414b95fc5717e3d01eeb/best.pt" --method discriminant --shrinkage 0.05 --image-size 320 --batch-size 16 --train-horizontal-flip --device cuda
```

This changes feature resolution without retraining the encoder. The 4,794 training
images provide 9,588 feature vectors: original plus a deterministic horizontal
flip AFTER preprocessing. Training metrics use only the 4,794 original images.
Validation is unflipped; the official test source is never loaded by fitting.

The selected recipe fits a float64 TRAIN mean, rounds centered features to
float32, and solves the shrinkage discriminant classifier with equal class priors
and covariance divisor N-K. Centering is folded into the final linear layer.
There is no feature standardization or PCA reduction. A positive confidence scale
is fitted on validation and folded into the weights; it preserves class ordering.
Validation NLL is therefore a calibration/tuning measurement.

The command prints a new directory containing `model.pt`, `config.json` and
`metrics.json`. Source files/checkpoints are preserved. Metadata records the
source hash, resolutions, batch size and view counts. Exports are inference-only:
they cannot resume encoder training. Use the printed new path if reproducing a
run; hardware/library/batch differences can affect borderline predictions.

## Prediction and official-test evaluation

Replace the image path with a real local image. Prediction requires only the
project dependencies and complete saved checkpoint, not the dataset or original
pretraining file. It reads class names and preprocessing from the checkpoint.

```powershell
.\.venv\Scripts\python.exe -m src.resnet50_predict --image "D:\birds\bird.jpg" --checkpoint "outputs/resnet50_inaturalist/augmented_pca_search/accuracy_challenger/model.pt" --device cpu --top-k 5
```

Use `--device cuda` for GPU prediction. Optional `--output prediction.json` writes
a new JSON file; its parent directory must exist and existing files are preserved.
Class IDs are zero-based. Confidence is the softmax score among 200 known classes,
not a correctness guarantee or an unknown-species detector. EXIF correction,
RGB conversion and white transparency compositing match the dataset pipeline.

For a fixed model, the evaluator loads all 5,794 official test images exactly
once and saves metrics, per-class results, predictions, a confusion matrix and
an MLflow run. Do not use test scores to select further configurations.

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.resnet50_evaluation --checkpoint "outputs/resnet50_inaturalist/augmented_pca_search/accuracy_challenger/model.pt" --device cuda --output-dir outputs/resnet50_inaturalist/evaluation
```

## Grad-CAM

```powershell
.\.venv\Scripts\python.exe -m src.visualization.resnet50_gradcam --image "D:\birds\bird.jpg" --checkpoint "outputs/resnet50_inaturalist/augmented_pca_search/accuracy_challenger/model.pt" --output "outputs/gradcam/bird.png" --device cpu
```

The command saves an input/map/overlay PNG and adjacent JSON. It refuses to
replace either file. By default it explains the predicted class logit using the
last ResNet block (`encoder.layer4.2`). `--class-id N` explains another class;
`--true-class-id N` annotates correctness when a true label is known. A supplied
label does not change the prediction or explanation target.

Grad-CAM averages spatial logit gradients, weights feature channels, applies
ReLU and resizes to the actual input canvas, following
[Selvaraju et al.](https://arxiv.org/abs/1610.02391). A private input copy requires
gradients; no optimizer or parameter update occurs. Maps are coarse positive
class support, not segmentation or proof of causality. They are independently
normalized, so intensities cannot be compared numerically across images. A zero
map means no positive spatial contrast. Include both correct and wrong examples
and discuss bird, background and padding regions in the report.

## Grid versus random search

Both methods try hyperparameters using training/validation data. Grid search
tries every combination in fixed lists. Random search samples a chosen number
of combinations from predefined distributions. The comparison should report
search spaces, trial budgets, best validation accuracy/macro F1 and elapsed time.
Compare like-for-like tasks/budgets, or explicitly explain any differences; do
not rank methods from test scores.

Existing local records include a 64-combination feature/refit grid and a
16-trial random shrinkage search. They differ in search spaces and budgets, so
they do not yet support an unqualified claim that one search is better. Their
reproducible comparison remains a separate deliverable. A frozen-feature head
comparison need not repeat full encoder training.

## Delivery and later encoder reuse

Keep the model, adjacent configuration, class mapping, metrics, source hashes and
loading instructions together. Weights, local MLflow and research outputs are
ignored by Git. Upload the required trained-model collection to Google Drive
and verify the download/share link; no automatic upload is performed here.

`forward_features()` returns spatial features (2048x10x10 at 320px) and `encode()`
returns 2048 pooled features per image. Document the selected preprocessing and
checkpoint for later colorization/captioning integration. Historical experiments
and failures belong in the report; their settings/results remain in the linked
history documents, effective run configurations and Git history.
