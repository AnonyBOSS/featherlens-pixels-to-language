# Pretrained CNN model and training reference

Use the [pretrained handoff](pretrained_handoff.md) for the delivered ensemble,
inference setup, measured results, evaluation and component Grad-CAM. This guide
covers model construction, reusable features, data integration and training/recovery.
It retains the Task 1.1 source, preparation, image and sample contracts.

## CPU development setup

The historical Stage 1A CPU environment (October 25, 2026) used Windows Python
3.12.15, torch `2.14.1+cpu` and torchvision `0.29.1+cpu`. To reproduce that development
setup, use Python 3.12 and the pinned [training requirements](../requirements.txt):

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

These CPU pins differ from the verified inference runtime in the main handoff.
Preserve a working accelerator stack in Colab/Kaggle; do not apply these CPU
installation commands there. Direct pins are not a complete transitive lockfile.
All examples run from the repository root with the environment active. Keep weights,
source caches, prepared manifests and experiment outputs outside Git. Initial cache
population needs network access; `load_weights=False` examples avoid weight downloads.

## Model configuration and weight identity

The [model API](../src/models/pretrained.py) uses an immutable, validated `ModelConfig`:

```python
from src.models.pretrained import ModelConfig, get_model_spec

config = ModelConfig(architecture="convnext_tiny", weights=None, num_classes=200,
                     fine_tuning="frozen", head_dropout=0.0, freeze_bn_stats=True)
spec = get_model_spec(config.architecture, config.weights)
print(spec.weight_id, spec.native_preprocessing)
```

`weights=None` selects the architecture's pinned pretrained identity. It does not
request random initialization. Exact qualified IDs are also accepted. Unsupported
architectures, IDs, class counts, policies, types and invalid dropout are rejected.
The output has 200 classes in the original ordered source mapping.

| Architecture key | Exact configured weight ID | Native resize / crop | Interpolation | Encoder channels | Stride |
| --- | --- | --- | --- | ---: | ---: |
| `convnext_tiny` | `ConvNeXt_Tiny_Weights.IMAGENET1K_V1` | 236 / 224 | Bilinear | 768 | 32 |
| `efficientnet_b3` | `EfficientNet_B3_Weights.IMAGENET1K_V1` | 320 / 300 | Bicubic | 1,536 | 32 |
| `regnet_y_3_2gf` | `RegNet_Y_3_2GF_Weights.IMAGENET1K_V2` | 232 / 224 | Bilinear | 1,512 | 32 |
| `convnextv2_base` | `timm/convnextv2_base.fcmae_ft_in22k_in1k_384` | 384 / 384 (squash) | Bicubic | 1,024 | 32 |

ConvNeXt-Tiny optionally accepts `facebook/convnext_tiny.in22k_ft_in1k_384`
(`FB22K_WEIGHT_ID`), the paper-author ImageNet-22K → ImageNet-1K profile with a
fresh 200-class head. Revision `32c741ba106f9e062174ef829354a16719c3dcac` and the
full SHA-256 are checked before safe CPU loading. Conversion matched upstream timm
features, embeddings and 1,000-class logits exactly at four CPU resolutions and on
two real training images; production loading needs no timm. Native preprocessing
is bicubic 384-pixel squashing; the chosen project recipe still preserves the whole
image with white padding. Defaults stay unchanged. Use the same weight ID in
`TrainConfig`/training JSON; a changed profile needs a fresh experiment.

ConvNeXt-V2 Base uses pinned `timm==1.0.30` and `safetensors==0.8.0`. Its FCMAE →
ImageNet-22K → ImageNet-1K artifact is revision-pinned and full-SHA-256 verified;
native spatial/pooled outputs matched the adapter exactly. Install only those two
packages when adding it to an existing GPU environment, retaining the Torch stack.
The model card identifies CC-BY-NC-4.0 weights; preserve attribution. This adapter
has separate real GPU resource checks and is outside the original three-model preflight.

`create_pretrained_model(config, *, load_weights=True, progress=True)` loads the
resolved torchvision enum or pinned optional artifact. Load/download failures
propagate; no random-weight fallback occurs. A new task head starts randomly.
Explicit `load_weights=False` constructs a random encoder while retaining its
configured identity. The wrapper exposes `config`, `spec`, `configured_weight_id`,
`weights_loaded` and `loaded_weight_id`; the latter two are `False`/`None` for this
offline construction. Configured identities or mocked construction alone do not
verify downloaded-checkpoint compatibility.

The JSON-serializable native recipe records `resize_size`, `crop_size`,
`interpolation`, `mean`, `std` and `antialias` separately from chosen preprocessing.
The three torchvision recipes use antialiasing and ImageNet normalization. They
do not execute inside the model or apply to already prepared tensors; notably,
EfficientNet-B3's native 300-pixel crop differs from the 224 full-image baseline.

## Classification and reusable encoder features

```python
import torch
from src.models.pretrained import ModelConfig, create_pretrained_model

model = create_pretrained_model(ModelConfig(architecture="convnext_tiny"),
                                load_weights=False).eval()
images = torch.zeros(2, 3, 224, 224)
with torch.no_grad():
    logits = model(images)
    features = model.forward_features(images)
print(logits.shape)               # [2, 200]
print(features["spatial"].shape)  # [2, 768, 7, 7]
print(features["pooled"].shape)   # [2, 768]
print(model.gradcam_target_layer)
```

`forward` returns logits without softmax; `forward_features` returns exactly
`spatial` and `pooled`, independently of the head. At 224 pixels, the three
torchvision adapters produce NCHW `[B,D,7,7]` spatial features, `[B,D]` pooled
features and `[B,200]` logits; `D` appears above. Other sizes change spatial shape;
stride 32 does not guarantee exact integer division at every size.

ConvNeXt retains final encoder normalization; EfficientNet pooled features precede
head dropout. The separate `head` applies configured dropout and a linear projection.
Calling both interfaces performs two encoder passes. Feature access preserves
autograd: omit `no_grad()` for gradients and set the input's `requires_grad` when
backpropagating through a frozen encoder's spatial representation.

## Fine-tuning policies and mode changes

Using the model above:

```python
model.train()
model.set_fine_tuning("partial")
model.eval()
model.train()  # Restores selected policy and normalization constraints.
model.set_fine_tuning("full", freeze_bn_stats=False)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
model.set_fine_tuning("frozen", freeze_bn_stats=True)
```

The head is always trainable. `frozen` freezes the encoder; `full` enables all its
parameters. `partial` enables the final transition/stage and applicable final
projection or normalization:

| Architecture | Trainable encoder portion in `partial` | CAM target reference |
| --- | --- | --- |
| ConvNeXt-Tiny | `encoder[6]`, `encoder[7]`, `encoder_norm` | `encoder.7` |
| EfficientNet-B3 | `encoder[7]`, `encoder[8]` | `encoder.8` |
| RegNetY-3.2GF | `encoder.trunk_output.block4` | `encoder.trunk_output.block4` |

Frozen stages, including stochastic layers, stay in evaluation mode during training.
Default `freeze_bn_stats=True` fixes BatchNorm running statistics even in trainable
stages; affine parameters follow the parameter policy. With `False`, active stages
may update statistics while frozen stages remain in evaluation mode. `eval()` sets
all modules to evaluation; subsequent `train()` reapplies these constraints.
Policy changes clear newly frozen gradients and replace immutable `model.config`.
Custom loops must also update optimizer groups when parameter ownership changes.

`gradcam_target_layer` resolves an existing spatial module without registering a
second copy; `spec.gradcam_target_path` records its path for external hooks.
For the delivered component renderer, use the main handoff.

## Baseline preprocessing and normalization

The [classification transforms](classification_data.md#preprocessing-and-configurable-normalization)
lazily decode independent RGB8 pixels, apply EXIF orientation and composite
transparency on white. The default fits the complete image, without cropping, into
a centered white 224×224 canvas with bilinear interpolation and aspect ratio
preserved within rounding. Source assets stay unchanged. Training alone adds a
horizontal flip with probability 0.5; validation is deterministic.

```python
from src.models.pretrained import ModelConfig
from src.training.data import PreprocessingConfig

config = ModelConfig(architecture="efficientnet_b3")
default = PreprocessingConfig()
plain = PreprocessingConfig(normalization=None)
custom = PreprocessingConfig(normalization=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
train_transform = default.build_transform(training=True)
val_transform = default.build_transform(training=False)
print(default.describe(config))
```

Default ImageNet means are `(0.485, 0.456, 0.406)` and standard deviations
`(0.229, 0.224, 0.225)`, applied once after float RGB conversion to `[0,1]`.
Disabled normalization retains that range; custom pairs need three finite means
and three finite positive standard deviations. `image_size` changes the square
size explicitly. Match chosen geometry/constants across training and evaluation;
estimate future dataset-derived constants from the 4,794 training records only.

`describe(model_config)` records chosen geometry, padding, normalization, train
flips and deterministic validation, alongside native identity/recipe with
`applied=False`. Changing architecture never silently selects its native crop.
The model consumes prepared tensors and never normalizes again. The delivered
ensemble's selected image sizes and protocol are in the main handoff.

## Prepared Dataset and DataLoader integration

Use the complete Task 1.1 ZIP's `preparation/` directory from the
[EDA handoff](eda_handoff.md#reports-and-delivery-bundle), or reproduce it with
`prepare_classification(audit_dir, output_dir, val_fraction=0.20, seed=42)` as in
the [classification guide](classification_data.md#reproduce-the-preparation).
Packaged audit summaries alone cannot regenerate preparation. Metadata contains
no images; the pinned official-training source must also be accessible/cached.

Replace the placeholder with the directory containing `preparation.json` and
`classification_manifest.jsonl`. Save as a script; the main guard supports Windows workers:

```python
import random
from pathlib import Path
import numpy as np
import torch
from src.models.pretrained import ModelConfig, create_pretrained_model
from src.training.data import make_pretrained_loaders

def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    prepared_dir = Path("path/to/verified/preparation")  # Replace this placeholder.
    config = ModelConfig(architecture="convnext_tiny", fine_tuning="frozen")
    loaders = make_pretrained_loaders(prepared_dir, config, batch_size=2,
                                     num_workers=0, seed=seed, pin_memory=False)
    batch = next(iter(loaders["val"]))
    model = create_pretrained_model(config, load_weights=False).eval()
    with torch.no_grad():
        logits = model(batch["image"])
    print(batch["image"].shape, batch["label"].dtype, batch["image_id"][0], logits.shape)
    print(len(loaders["train"].dataset), len(loaders["val"].dataset))

if __name__ == "__main__":
    main()
```

`make_pretrained_loaders(prepared_dir, model_config, *, preprocessing=None,
source=None, batch_size=32, num_workers=0, seed=42, pin_memory=False)` returns
`{"train": DataLoader, "val": DataLoader}`. It validates the complete preparation
with `load_classification_manifest()` before opening sources/constructing datasets.
By default it opens only `load_cub_split("train")` for both derived partitions.
An injected pinned `CubSource` gets the same identity/mapping checks. No split is
created or changed, and no official-test source is fetched.

Samples contain exactly `image`, `label`, `image_id`: default batches have float32
`[B,3,224,224]` images, int64 `[B]` labels 0–199 and stable source-ID lists.
Class order comes from original `ClassLabel.names`, never alphabetic sorting.
IDs keep `hf:<repository>@<revision>:<split>:<row>`; indexed source/manifest
mismatches raise instead of skipping records.

Canonical seed-42 preparation has **4,794 train / 1,200 validation / 5,794 official
test rows**, no exclusions and the unchanged ordered 200 classes. Both exact-duplicate
Pigeon Guillemot rows stay together in validation. The helper consumes train/val
only, preserving test assignments, captions, assets and class IDs. See the
[preparation policy](classification_data.md#preparation-and-cleaning-policy).

Only train shuffles. Each partition uses its loader generator; workers seed Python
and NumPy from PyTorch's worker seed. With zero workers, caller RNG controls train
augmentation, so seed it before iteration as above. Recreate loaders/reset seeds to
reproduce a run; versions, worker count and iteration schedule matter. The trainer
below supplies checkpointed RNG restoration and the training loop.

## Classification training and recovery

The [training notebook](../notebooks/pretrained/02_train_convnext.ipynb) and CLI
share the API below. Cloud setup verifies exact clean published Git HEAD before
imports, validates canonical metadata before pixels, preserves CUDA/global caches
and installs only missing non-Torch dependencies. See the
[cloud workflow](gpu_preflight.md#first-run-online-colab-gpu) for bootstrap details.
Required manifest SHA-256:
`a8bd05e048febf32df6d84a6e1bc16eddf6416be723e12dfc89621044962736b`.
Only pinned official-training pixels supply train/validation data.

```bash
python -m src.cli train --task classification --config configs/pretrained_acceptance.json --prepared-dir outputs/classification-v1 --run-dir outputs/experiments/convnext-frozen-acceptance-v1 --device cuda --tracking mlflow
```

```python
import json
from pathlib import Path
from src.training.trainer import TrainConfig
from src.training.experiment import run_training

config = TrainConfig.from_dict(json.loads(Path("configs/pretrained_acceptance.json").read_text()))
summary = run_training(
    "outputs/classification-v1", "outputs/experiments/convnext-frozen-acceptance-v1",
    config, device="cuda", resume=False, tracking="mlflow",
)
```

This bounded acceptance config freezes Tiny for two epochs: physical batch 8,
accumulation 2, AdamW LR 0.001/decay 0.0001, seed 42, FP16, workers 0, frozen BN,
224 full-image fitting, ImageNet normalization and train-only flips. Successful
steps would yield 600 updates; actual history/skipped AMP updates are authoritative.
It has no tuned-accuracy or duration promise; selected results are in the main handoff.

Preserve the **entire run directory**. Colab's acceptance path is
`/content/drive/MyDrive/featherlens/experiments/convnext-frozen-acceptance-v1`.
Recover with the same published code, preparation, config, directory and tracking
mode, setting `resume=True` / `--resume`. Use one writer. Fresh runs refuse nonempty
directories; changed configs need a new name. `latest.pt` is authoritative at
completed epochs, with optimizer/scaler/scheduler/RNG state; partial epochs replay.
Before any completed checkpoint, retain failure evidence and choose a new name.
Completed-run resume regenerates reports without duplicate training. CPU tests do
not establish fresh-session GPU or cross-version restoration.

`run_training(..., initial_head="path/to/probe-head.pt")` starts fresh optimizer
state/history from a frozen-probe classifier. It verifies the SHA sidecar,
backbone profile/hash, manifest, class order and dimensions before pixels.
`folded_raw_head` must include train-only feature standardization. Resume restores
the full checkpoint and does not reread the probe artifact.

`run_summary.json` records best/latest validation aggregates, successful/skipped
updates, runtime, checkpoint hashes, reload proof and tracking identity/status.
`history.json`/`.csv`, validation JSON, image-ID prediction CSVs and `run.log` remain
independent of tracking. Genuine training completion reloads best and checks finite
200-class validation logits; terminal recovery may report `state_loaded_without_forward`.
Inspect the actual proof/failures. Return `run_summary.json` and `evidence.zip` (with
SHA inventory); keep latest/best checkpoints, sidecars and `mlruns` separately.
The ZIP excludes checkpoints, data, caches and `mlruns`.

`tracking="json"` / `--tracking json` needs no MLflow. MLflow uses local
`run_dir/mlruns` FileStore without server/autologging. Failures remain visible and
can be reconciled under the same run ID on resume; cross-machine artifact-location
relocation is not guaranteed. Kaggle uses Internet + GPU, `CLOUD="kaggle"`, metadata
under `/kaggle/input` and runs under `/kaggle/working/experiments`; **Save Version**,
then download/verify the entire saved run. No offline training package is verified.
The training notebook warns for seven reviewed unrelated Kaggle package relationships
(BigFrames, Colab, Dopamine, Gradio and MoviePy owners) without repairs; unknown
relationships, malformed/process errors and required-import failures remain fatal.
Colab keeps strict `pip check` behavior.

Optional `label_smoothing` and brightness/contrast `color_jitter` default to zero
and affect training only; jitter precedes fitting so padding stays white.
`mixup_alpha=0.2` opts into within-microbatch image/target mixing. Default `0.0`
preserves arithmetic and consumes no MixUp draws. Enabled train accuracy is
`mixup_weighted`; validation retains hard labels and unchanged cross-entropy/metrics.
Legacy checkpoints missing the field load disabled; changing it on resume is rejected.

## Provenance, verification, and limits

Retain source revisions/class order, manifest SHA-256, audit fingerprints,
policy/split versions, counts and seed from `preparation.json`, plus model config,
configured/loaded weight identities and `PreprocessingConfig.describe(...)`.
Recipes alone do not prove downloads, audits or executed training. Preparation
validation checks metadata/stored audit evidence, not a fresh hash of every image.

Historical Stage 1A verification on October 25, 2026 passed **614 tests, no failures
or skips, in 90.57 seconds** on Windows CPU, with offline cached real data via
`FEATHERLENS_PREPARED_DIR`. Real train/validation smoke checks used all three random
encoders (`weights_loaded=False`, `loaded_weight_id=None`), retained 4,794/1,200
members and reproduced validation pixels/logits with the feature shapes above.
Compatibility checks passed for 53 dependencies without requirements changes.
This establishes CPU integration, not pretrained/GPU quality. CPU model/data and
synthetic trainer/checkpoint tests run with `python -m pytest -q`; the cached
real-data opt-in otherwise skips without its populated source cache/prepared path.
Task 1.1 evidence remains in its original handoffs.

Returned Stage 1B1 T4 evidence verified genuine official weights and 36/36 FP32/FP16
checks; corrected automatic precision passed 12/12 plus storage readback before
PR 13 merged as `6ad1adfe47f339414693c2034658a32bc6df2ca6`.
The [GPU evidence guide](gpu_preflight.md) preserves BF16 failures, report hashes
and limits. Preflight made zero optimizer updates. Later canonical CUDA training
artifacts and the main handoff establish delivered results separately; scratch
CNNs, automated search and downstream adaptation remain separate work.
