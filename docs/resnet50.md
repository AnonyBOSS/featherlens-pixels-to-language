# ResNet-50 workflow and retained results

This guide preserves the merged ResNet-50 workflow. Run commands from the repository root. For the shared data contract, see [classification data](classification_data.md). These results retain their original limitations; joint integration does not constitute a new training or test run.

### Selected iNaturalist ResNet-50 workflow

The same ResNet-50 implementation supports two distinct pretraining choices:
`configs/resnet50.yaml` selects the earlier ImageNet baseline, while
`configs/resnet50_inaturalist.yaml` selects the authors' iNaturalist2021 checkpoint
trained without ImageNet initialization. Their weights and experiment outputs
are separate. The default runner configuration is the iNaturalist YAML;
the historical ImageNet baseline requires its explicit configuration.

Run the PowerShell commands below from the repository root, in a Python 3.12
environment with a suitable CUDA PyTorch build. Install the ResNet dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-resnet50.txt
```

For encoder training, download the authors' [pretrained model archive](https://cornell.box.com/s/bnyhq5lwobu6fgjrub44zle0pyjijbmw).
Extract `cvpr21_newt_pretrained_models/pt/inat2021_supervised_large_from_scratch.pth.tar`
and place that file at `outputs/pretrained/inat2021_supervised_large_from_scratch.pth.tar`.
The [authors' catalogue](https://raw.githubusercontent.com/visipedia/newt/main/benchmark/configs.py)
identifies this variant as `inat2021_supervised_from_scratch`, with no ImageNet
initialization. The runner verifies the file against the SHA-256 pinned in
`configs/resnet50_inaturalist.yaml`:
`e845bd2512b7728c390523afc8696dafedf2675c93e9d7508a48b416cda8e8d9`.
This is the recorded local fingerprint, not an independently published author checksum.
`normalization: imagenet` names RGB mean/std constants; it does not load ImageNet weights.

The tracked `preparation/` snapshot supplies the canonical split; image data is
downloaded/cached from the pinned Hugging Face source when needed. Check a batch,
then start a new encoder-training run:

```powershell
.\.venv\Scripts\python.exe -m src.training.resnet50_runner --config configs/resnet50_inaturalist.yaml --check-batch
.\.venv\Scripts\python.exe -m src.training.resnet50_runner --config configs/resnet50_inaturalist.yaml --train
```

Head training freezes the encoder; fine-tuning updates `layer4` and the classifier
together. The YAML allows 3 head epochs and up to 10 fine-tuning epochs with
early stopping. Training prints a unique run directory containing `best.pt`,
`last.pt`, stage checkpoints, configuration and epoch history. MLflow records
settings, train/validation loss and accuracy histories, and checkpoint artifacts
or references under `outputs/resnet50/mlruns/`. `--train` starts a new run; this
ResNet runner does not implement checkpoint resumption. Optional `--smoke-test`
uses tiny subsets and produces no reportable performance result.

#### Refit the classifier at 320px

Encoder training uses 224px. The selected final model adds a separate classifier
refit at 320px; `--train` does not run this step automatically. Use the archived
encoder below to reproduce the recorded refit, or replace the path with the
`best.pt` printed by your own completed training run:

```powershell
$encoderCheckpoint = "outputs/resnet50_inaturalist/training/9c6e5c07be24414b95fc5717e3d01eeb/best.pt"
.\.venv\Scripts\python.exe -m src.training.resnet50_head_refit --source-checkpoint $encoderCheckpoint --method discriminant --shrinkage 0.05 --image-size 320 --batch-size 16 --train-horizontal-flip --device cuda
```

The archived encoder is an external model artifact, not included in Git. Its
SHA-256 is `a6ed5697a0a7555292e84ab2219848428cda7a18eb43fe9e02dff43c88eeb221`.
Obtain it from the team's model handoff when reproducing the recorded result.
Refitting freezes the encoder and fits an LDA linear classifier on original and
flipped TRAIN features. No test data is loaded. A positive confidence scale is
fitted using validation data and folded into the classifier; it preserves class
ordering. The command prints a new directory containing `model.pt`, `config.json`
and `metrics.json`. This final checkpoint is inference-only.

Measured results for the selected saved 320px checkpoint:

| Split | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Train (4,794 images) | 98.06% | 98.17% | 98.06% | 98.06% |
| Validation (1,200 images) | 87.92% | 89.61% | 87.92% | 87.91% |
| Official test (5,794 images) | 87.54% | 88.29% | 87.65% | 87.62% |

These are saved-checkpoint measurements, not guaranteed rerun scores. Overfitting
remains, one encoder seed was trained, validation was repeatedly reused, and
earlier test scores were inspected. The test is not an untouched final benchmark.

#### Prediction, evaluation and Grad-CAM

Prediction and explanation use the saved full checkpoint without retraining:

```powershell
$modelCheckpoint = "outputs/resnet50_inaturalist/augmented_pca_search/accuracy_challenger/model.pt"
.\.venv\Scripts\python.exe -m src.resnet50_predict --image "path/to/bird.jpg" --checkpoint $modelCheckpoint --device cpu --top-k 5
.\.venv\Scripts\python.exe -m src.visualization.resnet50_gradcam --image "path/to/bird.jpg" --checkpoint $modelCheckpoint --output "outputs/gradcam/bird.png" --device cpu
```

The selected checkpoint is an external artifact. Replace `$modelCheckpoint`
with the `model.pt` extracted from the model delivery ZIP or printed by your own
refit run, and replace the image path with your image. Prediction reads saved
preprocessing and class order; it needs neither the CUB dataset nor the original
iNaturalist pretraining file. Grad-CAM saves a new PNG and adjacent JSON; choose
an unused output filename and add `--true-class-id N` when the correct zero-based
label is known. Include correct and wrong examples with interpretation in the report.

To evaluate a fixed checkpoint on the full official test set:

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.resnet50_evaluation --checkpoint $modelCheckpoint --device cuda --output-dir outputs/resnet50_inaturalist/evaluation
```

Evaluation saves metrics, predictions and confusion matrices, and logs an MLflow
run. Choose settings/checkpoints using validation; do not use test scores for tuning.

#### Grid versus random search for ResNet-50

The controlled comparison tunes LDA shrinkage in [0.03, 0.95]: 8 log-spaced grid
points versus 8 log-uniform random samples, seed 42. Both methods use the same
frozen 320px TRAIN/VAL features and two CPU threads. This study uses original
training views only; the selected final model used original and flipped views.

| Method | Trials | Best shrinkage | Validation accuracy | Validation macro F1 | Fit + score time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Grid | 8 | 0.049147 | 87.9167% | 87.8934% | 12.80s |
| Random | 8 | 0.435032 | 87.9167% | 87.8052% | 11.87s |

Timing excludes shared feature extraction/loading and output/logging. These
measurements do not establish a universal winner. Trial coefficients, metrics,
timing, input hashes and the comparison plot are saved locally and tracked in MLflow.

To rerun, obtain the archived `finetuned_features.pt` cache and the matching
encoder from the team's experiment handoff. Neither input is included in Git.
The cache SHA-256 is `b01132aaeb6b902bc284d3dee924093b424de5a44296941509187106099263c3`;
the matching encoder hash is listed above. Use a new output directory for each run:

```powershell
.\.venv\Scripts\python.exe -m src.training.resnet50_search --features outputs/resnet50_inaturalist/resolution_review/finetuned_features.pt --source-checkpoint $encoderCheckpoint --output-dir outputs/resnet50_inaturalist/grid_random_comparison_rerun --trials 8 --seed 42
```

For this recorded comparison, `$encoderCheckpoint` must identify the archived
encoder matching that cache. The loader verifies encoder/cache identity and
canonical row membership. It loads no official test images and does not replace
the selected deployment model.

Model files, reports and local MLflow records are ignored by Git and delivered
separately. Keep model configuration and class mapping with Google Drive exports.
The ResNet encoder exposes spatial and pooled features for later project tasks.
This workflow uses its own runner; the shared JSON-based CLI below supports the
other pretrained adapters.
