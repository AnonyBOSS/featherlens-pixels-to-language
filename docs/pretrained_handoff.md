# Pretrained CNN classification handoff

This handoff covers our ConvNeXt-Tiny + ConvNeXt-V2 Base classification track.
ResNet50, scratch-CNN experiments and the joint Milestone 1 comparison remain team-owned work.
The delivered default is the original-view, equal-probability ensemble; it uses two CNNs at inference.

## Setup and supplied files

Use Python 3.12 and run commands from the repository/package root. Create and activate
an isolated environment, then install the inference dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-inference.txt
python -m pip check
python -m src.cli --help
```

The verified local runtime used torch 2.11.0+cu128 and torchvision 0.26.0+cu128.
For CUDA, first install matching accelerator builds for your system using the
[official PyTorch installer](https://pytorch.org/get-started/locally/), then install
the remaining pins. Do not apply the older CPU-development requirements to that environment.
The direct pins are not a complete transitive lockfile; a fresh installation on every
platform is not established by the local verification. CPU inference uses FP32 and is slower.

The separately supplied `classifier/` contains `bundle.json`, `tiny.pt`, `base.pt`
and SHA-256 sidecars. Keep these together; the bundle uses relative paths and the
loader checks component hashes, class order, preprocessing and manifest identity.
These are inference weights, without optimizer state, datasets or original training checkpoints.
Loading the supplied bundle requires no pretrained-weight download or CUB cache.
Only load artifacts from the trusted team handoff and verify its inventory first.

## Predict and explain a local image

```bash
python -m src.cli predict --bundle classifier/bundle.json --image examples/bird.jpg --device cpu
python -m src.cli gradcam --bundle classifier/bundle.json --image examples/bird.jpg --component tiny --output outputs/cam-tiny.png --device cpu
python -m src.cli gradcam --bundle classifier/bundle.json --image examples/bird.jpg --component base --output outputs/cam-base.png --device cpu
```

Replace `examples/bird.jpg` with your image; image files are supplied separately.
Prediction prints JSON with zero-based class ID, class name, top-five probabilities
and device. Probabilities are uncalibrated model confidence, not a correctness guarantee.
Grad-CAM saves a native-aspect overlay and JSON metadata. Its default target is the
ensemble prediction; `--target-class ID` explicitly selects another class. Maps explain
one component's raw FP32 class score, not causal evidence or its ensemble contribution.
A zero/degenerate map is flagged rather than presented as fabricated localization.
Existing export/evaluation/Grad-CAM destinations are protected against overwrite.

## Reproduce canonical validation

Obtain the approved Task 1.1 preparation directory and populate the revision-pinned
Hugging Face source cache using the [classification data handoff](classification_data.md).
Unlike local-image prediction, partition evaluation requires those source images and
validated preparation metadata. It does not generate a new split or silently filter records.

```bash
python -m src.cli evaluate --bundle classifier/bundle.json --prepared-dir outputs/classification-v1 --partition val --batch-size 8 --num-workers 0 --device cuda --output-dir outputs/validation-check
```

Use `--device cpu` if CUDA is unavailable. `num-workers 0` is the portable default,
including on Windows. The output contains `metrics.json`, `predictions.csv`,
`probabilities.pt` and `evaluation.png`; metrics include per-class results, macro/weighted
precision, recall, F1, top-1/top-5 and the confusion matrix (truth rows, prediction columns).
The evaluator accounts for every member of the selected partition and checks IDs and labels
across components. Failure interrupts evaluation rather than silently dropping images.

The fixed seed-42 split is **4,794 train / 1,200 validation / 5,794 official test**,
with six validation members per class. Both exact-duplicate Pigeon Guillemot images
are in validation. The required manifest SHA-256 is
`a8bd05e048febf32df6d84a6e1bc16eddf6416be723e12dfc89621044962736b`.
Use train for fitting and validation for selection. Test is a frozen final assessment;
`--partition test` requires the separate `--allow-official-test` opt-in. Do not use
test images, labels or results to select new settings. Delivery verification reuses
saved test findings and does not authorize a new test run.

## Preprocessing and measured results

The shared decoder applies EXIF orientation, converts to RGB and composites transparency
on white; it does not apply ICC profile correction. The whole image is fitted without
cropping into a centered white square, preserving aspect ratio within pixel rounding.
Tiny uses **448 x 448**, Base **384 x 384**. Each uses ImageNet mean
`[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]` once, after conversion to [0,1].
There is no inference augmentation. Full-image fitting preserves content but can reduce
bird resolution in wide/tall images. Configurable normalization remains available in the
shared preprocessing API; changing the supplied recipe changes the measured protocol.

The CLI sets deterministic algorithms, disables cuDNN benchmarking, and uses two CPU
threads. It permits cuDNN TF32 kernels, matching the measured Base encoder, while
disabling TF32 matrix multiplication. CUDA prediction uses Tiny FP16 microbatches of
eight and Base FP32 tensors in microbatches of four, followed by Base's CPU FP32 affine
head. FP32 tensor storage does not imply that cuDNN convolutions use full FP32 precision.
Base's training-derived score scale is already folded into the exported coefficients: **do not apply it again**.
The two softmax probability vectors are averaged 50/50. CPU FP32 execution is a separate
numerical profile; other devices, versions and batch sizes may introduce small differences.

| Frozen original-view ensemble | Top-1 | Macro precision | Macro recall | Macro F1 | Top-5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation, 1,200 images | 90.9167% | 91.9721% | 90.9167% | 90.6110% | 98.4167% |
| Official test, 5,794 images | 90.3003% | 90.6041% | 90.3735% | 90.1246% | 98.1878% |

The separately recorded fixed flip-TTA experiment reached 91.00% validation top-1;
it is not this default protocol and has no corresponding official-test result.
94% was not achieved. Original experiment failures and negative results remain evidence;
historical MLflow imports are explicitly labelled and do not imply new training.

## Reuse one encoder

```python
import torch
from src.data.image_io import load_rgb
from src.evaluation.inference import load_component
from src.training.data import PreprocessingConfig

model, metadata = load_component("classifier/tiny.pt", device="cpu")
norm = metadata["preprocessing"]["normalization"]
recipe = PreprocessingConfig(image_size=metadata["preprocessing"]["image_size"],
    normalization=None if norm is None else (norm["mean"], norm["std"]))
with load_rgb({"path": "examples/bird.jpg"}) as image:
    batch = recipe.build_transform(training=False)(image).unsqueeze(0)
with torch.no_grad():
    features = model.forward_features(batch)
print(features["spatial"].shape, features["pooled"].shape)
```

Tiny produces `[N,768,14,14]` spatial / `[N,768]` pooled features at 448 pixels.
Base (`base.pt`) produces `[N,1024,12,12]` / `[N,1024]` at 384. Features precede
classification-head dropout. Downstream colorization/captioning adaptation remains future work.
