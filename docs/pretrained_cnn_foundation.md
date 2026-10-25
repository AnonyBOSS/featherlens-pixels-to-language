# Pretrained CNN foundation — Task 1.2 Stage 1A

Stage 1A supplies three torchvision model adapters, separate 200-class heads,
reusable encoder features, fine-tuning policies, and a thin connection to the
existing classification data pipeline. It preserves the Task 1.1 source, audit,
preparation, image, and sample contracts. This is a foundation for experiments;
Stage 1B now supplies verified official weights, GPU preflight and classification
training/recovery. Genuine canonical training results are still pending.

## CPU development setup

Use Python 3.12 and the pinned [training requirements](../requirements.txt), which
include the classification requirements and `mlflow-skinny==3.17.2`.
From the repository root, create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell, activate it with `.\.venv\Scripts\Activate.ps1`; on
macOS/Linux, use `source .venv/bin/activate`. With that environment active:

```bash
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

The established development environment is Windows Python 3.12.15 with torch
`2.14.1+cpu` and torchvision `0.29.1+cpu`. These instructions describe CPU
development. A cloud CUDA environment needs compatible accelerator builds and
its own verification. Direct dependencies are pinned; this is not a complete
transitive lockfile. The cloud notebook inspects and preserves its CUDA stack;
do not apply these CPU installation commands to Colab/Kaggle.

All examples run from the repository root with the environment active. Keep
downloaded weights, source caches, prepared manifests, and experiment outputs out
of Git. Installing dependencies and initially populating source caches require
network access; the model example below explicitly avoids weight downloads.

## Model configuration and weight identity

The public model API is in [src/models/pretrained.py](../src/models/pretrained.py).
`ModelConfig` is an immutable validated dataclass:

```python
from src.models.pretrained import ModelConfig, get_model_spec

config = ModelConfig(
    architecture="convnext_tiny",
    weights=None,
    num_classes=200,
    fine_tuning="frozen",
    head_dropout=0.0,
    freeze_bn_stats=True,
)
spec = get_model_spec(config.architecture)
print(spec.weight_id)
print(spec.native_preprocessing)
```

`weights=None` selects the architecture's pinned pretrained enum. It does not
request random initialization. Pass the exact qualified ID to make the same
choice explicit. Unsupported architectures, weight IDs, class counts, policies,
types, and invalid dropout values are rejected. The output class count is fixed
at 200; labels retain the ordered source class mapping.

| Architecture key | Exact configured weight ID | Native resize / center crop | Native interpolation | Encoder channels | Output stride |
| --- | --- | --- | --- | ---: | ---: |
| `convnext_tiny` | `ConvNeXt_Tiny_Weights.IMAGENET1K_V1` | 236 / 224 | Bilinear | 768 | 32 |
| `efficientnet_b3` | `EfficientNet_B3_Weights.IMAGENET1K_V1` | 320 / 300 | Bicubic | 1,536 | 32 |
| `regnet_y_3_2gf` | `RegNet_Y_3_2GF_Weights.IMAGENET1K_V2` | 232 / 224 | Bilinear | 1,512 | 32 |

The exact native preprocessing dictionary includes `resize_size`, `crop_size`,
`interpolation`, `mean`, `std`, and `antialias`. It is JSON serializable and is
recorded separately from the chosen baseline preprocessing. The native recipe
does not run inside the model and must not be applied to already prepared
tensors. In particular, EfficientNet-B3's native 300-pixel crop is not the
224-pixel full-image baseline. All three native recipes use antialiasing and the
fixed ImageNet normalization constants documented below.

`create_pretrained_model(config, *, load_weights=True, progress=True)` requests
the resolved enum through the native torchvision constructor. Download or load
failures propagate; there is no random-weight fallback. The new task head is
randomly initialized even when encoder weights are loaded.

For offline construction, pass `load_weights=False` explicitly. This dispatches
the constructor with `weights=None`, so the encoder is randomly initialized.
The wrapper exposes `configured_weight_id`, `weights_loaded`, and
`loaded_weight_id`. Offline construction records `False` and `None` for the last
two fields while retaining the configured pinned ID. Its `config` and `spec`
remain available for provenance. Offline construction, enum resolution, or a
mocked constructor call does not establish compatibility with a downloaded
pretrained checkpoint.

## Classification and reusable encoder features

Save this as a Python script at the repository root:

```python
import torch

from src.models.pretrained import ModelConfig, create_pretrained_model


def main():
    model = create_pretrained_model(
        ModelConfig(architecture="convnext_tiny"),
        load_weights=False,
    ).eval()
    images = torch.zeros(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(images)
        features = model.forward_features(images)
    print(logits.shape)               # [2, 200]
    print(features["spatial"].shape)  # [2, 768, 7, 7]
    print(features["pooled"].shape)   # [2, 768]
    print(model.gradcam_target_layer)


if __name__ == "__main__":
    main()
```

`forward(images)` returns logits, without softmax. `forward_features(images)`
returns exactly `spatial` and `pooled`, independently of the task head. At the
224-pixel baseline, spatial tensors are NCHW `[B, D, 7, 7]`, pooled tensors are
`[B, D]`, and logits are `[B, 200]` for all three adapters. The channel count `D`
is shown in the table above. Other input sizes can change the spatial dimensions;
stride 32 describes encoder downsampling rather than a guarantee of exact
integer division for every input size.

The adapter retains ConvNeXt's final encoder normalization. EfficientNet's
pooled embedding is exposed before task-head dropout. The separate `head`
applies the configured dropout and a linear projection to 200 classes. Calling
`forward` and `forward_features` separately performs separate encoder passes;
choose the interface needed by the consumer.

Feature extraction does not unconditionally disable autograd. For a frozen
encoder, set an input tensor's `requires_grad` when a gradient through the
spatial representation is needed. `torch.no_grad()` is an explicit caller choice
for inspection or inference; omit it when computing gradients.

## Fine-tuning policies and mode changes

```python
from src.models.pretrained import ModelConfig, create_pretrained_model

model = create_pretrained_model(
    ModelConfig(architecture="efficientnet_b3", fine_tuning="frozen"),
    load_weights=False,
)
model.train()
model.set_fine_tuning("partial")
model.eval()
model.train()  # Reapplies the selected policy and normalization constraints.
model.set_fine_tuning("full", freeze_bn_stats=False)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
model.set_fine_tuning("frozen", freeze_bn_stats=True)
```

The head remains trainable under every policy. `frozen` freezes the encoder;
`full` enables all encoder parameters. `partial` includes the final transition
and stage, plus the final encoder projection or normalization where applicable:

| Architecture | Trainable encoder portion in `partial` | CAM target reference |
| --- | --- | --- |
| ConvNeXt-Tiny | `encoder[6]`, `encoder[7]`, `encoder_norm` | `encoder.7` |
| EfficientNet-B3 | `encoder[7]`, `encoder[8]` | `encoder.8` |
| RegNetY-3.2GF | `encoder.trunk_output.block4` | `encoder.trunk_output.block4` |

Frozen encoder stages remain in evaluation mode while the model is training,
including their stochastic layers. With `freeze_bn_stats=True` (the default),
BatchNorm running statistics remain fixed even in trainable stages. Its affine
parameters still follow the chosen parameter policy. With
`freeze_bn_stats=False`, BatchNorm statistics may update in active stages during
training; frozen stages remain in evaluation mode. `eval()` puts all modules in
evaluation mode, and a later `train()` restores these policy constraints.
Policy changes clear stale gradients on parameters that become frozen. If a
future training loop changes which parameters an optimizer should own, update
its parameter groups as part of that loop. `model.config` is replaced with an
updated immutable configuration when the policy changes.

`gradcam_target_layer` resolves an existing spatial module rather than
registering a second copy. `get_model_spec(...).gradcam_target_path` records its
path. This supports external hooks and gradient-based inspection; Stage 1A does
not implement a CAM renderer, heatmap overlays, or an explainability experiment.

## Baseline preprocessing and normalization

The data helper reuses [the classification transforms](classification_data.md#preprocessing-and-configurable-normalization).
It decodes a source image lazily, applies EXIF orientation, composites transparency
on white, and returns independent RGB8 pixels. It fits the complete image into a
centered white 224×224 canvas using bilinear interpolation, with aspect ratio
preserved within pixel rounding. It never crops by default or changes source
assets. Only training adds a horizontal flip with probability 0.5; validation
is deterministic.

```python
import json

from src.models.pretrained import ModelConfig
from src.training.data import PreprocessingConfig

config = ModelConfig(architecture="efficientnet_b3")
default = PreprocessingConfig()
plain = PreprocessingConfig(normalization=None)
custom = PreprocessingConfig(
    normalization=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
)
train_transform = default.build_transform(training=True)
val_transform = default.build_transform(training=False)
print(json.dumps(default.describe(config), indent=2))
```

The default uses fixed ImageNet means `(0.485, 0.456, 0.406)` and standard
deviations `(0.229, 0.224, 0.225)`. Normalization is applied once, after conversion
to floating-point RGB in `[0, 1]`; disabled normalization leaves that range intact.
A custom pair requires three finite means and three finite, positive standard
deviations. `image_size` can explicitly change the output square size. Record
the same chosen geometry and constants for training and evaluation. Estimate any
future dataset-derived constants from the 4,794 derived training records only.

`describe(model_config)` records the actual baseline geometry, white padding,
normalization, training flip probability, and validation behavior, alongside the
native weight identity/recipe with `applied=False`. Changing architecture does
not silently select its native crop recipe. The model expects already prepared
tensors and does not normalize a second time.

## Prepared Dataset and DataLoader integration

Use a complete existing preparation. Either extract the separately distributed
Task 1.1 delivery ZIP as described in [the EDA handoff](eda_handoff.md#reports-and-delivery-bundle)
and select its `preparation/` folder, or reproduce it with
`prepare_classification(audit_dir, output_dir, val_fraction=0.20, seed=42)` as
described in [the classification guide](classification_data.md#reproduce-the-preparation).
The ZIP's packaged audit summaries alone are not a complete row-level input for
regenerating preparation. Prepared metadata does not include the source images;
the pinned official-training source must also be cached or accessible.

Replace the explicit placeholder below with the directory containing
`preparation.json` and `classification_manifest.jsonl`. Save it as a script so
the main guard also supports Windows workers:

```python
import random
from pathlib import Path

import numpy as np
import torch

from src.models.pretrained import ModelConfig, create_pretrained_model
from src.training.data import PreprocessingConfig, make_pretrained_loaders


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    prepared_dir = Path("path/to/verified/preparation")  # Replace this placeholder.
    config = ModelConfig(architecture="convnext_tiny", fine_tuning="frozen")
    preprocessing = PreprocessingConfig()
    loaders = make_pretrained_loaders(
        prepared_dir,
        config,
        preprocessing=preprocessing,
        batch_size=2,
        num_workers=0,
        seed=seed,
        pin_memory=False,
    )
    batch = next(iter(loaders["val"]))
    model = create_pretrained_model(config, load_weights=False).eval()
    with torch.no_grad():
        logits = model(batch["image"])
    print(batch["image"].shape, batch["label"].dtype)
    print(batch["image_id"][0], logits.shape)
    print(len(loaders["train"].dataset), len(loaders["val"].dataset))


if __name__ == "__main__":
    main()
```

`make_pretrained_loaders(prepared_dir, model_config, *, preprocessing=None,
source=None, batch_size=32, num_workers=0, seed=42, pin_memory=False)` returns
`{"train": DataLoader, "val": DataLoader}`. It validates the complete preparation
with `load_classification_manifest()` before opening a source or constructing
Datasets. Without an injected `source`, it opens only `load_cub_split("train")`;
both derived partitions use that official-training source. A supplied pinned
`CubSource` supports already-open or offline sources and receives the same
identity/mapping checks. No official-test source is fetched, and no split is
created or silently changed.

Samples retain exactly `image`, `label`, and `image_id`. Default batches contain
`float32` `[B, 3, 224, 224]` images, `int64` `[B]` labels in 0–199, and a list of
stable source IDs. Source class order comes from the original ordered
`ClassLabel.names`, never sorted class names. IDs retain the
`hf:<repository>@<revision>:<split>:<row>` namespace. Indexed reads validate source
identity against the manifest and raise on failure rather than skip samples.

The canonical preparation has **4,794 train, 1,200 validation, and 5,794 official
test rows**, with no exclusions. Both exact-duplicate Pigeon Guillemot rows stay
together in validation; neither is deleted. The helper consumes only train/val
members and preserves all test assignments for later explicit final evaluation.
Caption data, source assets, class IDs, and the seed-42 group-aware split remain
unchanged. See [the preparation policy](classification_data.md#preparation-and-cleaning-policy)
for complete eligibility and duplicate rules.

Only training shuffles. Each partition uses the existing loader's generator;
workers seed Python and NumPy from PyTorch's worker seed. With zero workers,
training augmentation uses the caller's RNG, so seed it before iteration as
above. Recreate loaders and reset caller seeds to reproduce a run. Worker count,
library versions, and iteration schedule remain part of reproducibility. The
data helper supplies no checkpointed RNG restoration or training loop; the
classification trainer below owns those responsibilities.

## Classification training and recovery

Run [02_train_convnext.ipynb](../notebooks/pretrained/02_train_convnext.ipynb) in an
online Colab GPU runtime. Set the full published training PR SHA; use a fresh
kernel and the verified canonical metadata ZIP. Setup verifies exact clean Git
HEAD before project imports, preserves CUDA packages and global source/weight
caches, installs only missing non-Torch dependencies and prints `pip check`
conflicts. Metadata validation requires manifest SHA-256
`a8bd05e048febf32df6d84a6e1bc16eddf6416be723e12dfc89621044962736b` before pixel access.
Only the pinned official-training source supplies train/validation images.

The notebook loads `configs/pretrained_acceptance.json` through `TrainConfig.from_dict`
and calls the same API as `python -m src.cli train --task classification`:

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

The initial acceptance freezes ConvNeXt-Tiny for two epochs with physical batch 8,
accumulation 2, AdamW LR 0.001/decay 0.0001, seed 42, FP16, workers 0 and frozen
BN statistics. Full-image white 224 fitting, ImageNet normalization and train-only
flips are unchanged. Expect 600 successful updates only if all accumulation steps
succeed; actual history and skipped AMP updates decide the count. This is a
bounded acceptance experiment, with no tuned accuracy or duration promise.

Colab uses the stable Drive path
`/content/drive/MyDrive/featherlens/experiments/convnext-frozen-acceptance-v1`.
Preserve the **entire run directory**. To recover, restore the same published code,
canonical preparation and config, then set `resume=True` (CLI: `--resume`) and the
same directory/tracking mode. One writer only. A fresh call refuses nonempty runs;
changed configs need a new run name. `latest.pt` is authoritative at completed
epochs, with optimizer/scaler/scheduler/RNG state; partial epochs are replayed.
Before the first completed checkpoint, retain failure evidence and choose a new
name. Completed runs regenerate reports without duplicate training. CPU recovery
tests do not establish fresh-session GPU or cross-version restoration.

`run_summary.json` records actual best/latest validation aggregates, successful and
skipped updates, runtime, checkpoint hashes, reload proof and tracking ID/status.
`history.json`/`.csv`, validation JSON, image-ID prediction CSVs and `run.log` remain
independent of MLflow. A genuine training completion reloads the best checkpoint
and verifies finite 200-class logits on a validation batch; terminal recovery can
instead report `state_loaded_without_forward`. Inspect the actual proof and failures.
Return `run_summary.json` and `evidence.zip`; the ZIP includes a SHA inventory but
excludes checkpoints, datasets, caches and `mlruns`. Keep latest/best `.pt` files,
their SHA sidecars and local tracking separately in Drive. JSON-only tracking is
available through `tracking="json"` / `--tracking json`; MLflow uses the local
`run_dir/mlruns` FileStore without a server or autologging. Logging failures are
visible and can be reconciled with the same run ID on resume. FileStore artifact
location relocation across machines is not guaranteed.

Kaggle online uses Internet + GPU, `CLOUD="kaggle"`, attached metadata under
`/kaggle/input`, and runs under `/kaggle/working/experiments`. **Save Version** is
required to retain session outputs; download and verify the entire saved run.
No offline training package or cloud storage durability guarantee is provided.

Local trainer/checkpoint/controller tests use synthetic CPU fixtures; real CUB
training metrics, actual updates, cloud checkpoints and fresh-session restoration
remain **PENDING** until the user returns cloud evidence.

## Provenance, verification, and limits

Keep source revisions and ordered class mapping from the pinned source contract;
keep manifest SHA-256, audit fingerprints, policy/split versions, counts, and seed
from `preparation.json`. Record model configuration, exact configured and loaded
weight identity, and `PreprocessingConfig.describe(...)` with future experiment
artifacts. The preprocessing description is a recipe, not evidence of a weight
download, source audit, or executed training run. Prepared validation checks
metadata and stored audit evidence; it does not rehash every image during loading.

Stage 1A verification uses CPU execution, synthetic tensors/prepared fixtures,
and explicit `load_weights=False`. Model tests cover 224-pixel shapes, separate
feature access, gradients, CAM target hooks, parameter masks, BatchNorm buffers,
repeated mode transitions, invalid configurations, offline construction, and
constructor dispatch/error propagation. Data integration tests exercise the
existing Dataset/DataLoader and preparation validator, fit geometry, normalization,
IDs/labels, deterministic validation, and absence of an implicit test-source fetch.
Mocked weight dispatch verifies delegation only. Randomly initialized execution
does not verify real pretrained-checkpoint compatibility or classification quality.

On October 25, 2026, the complete combined suite passed **614 tests with no
failures or skips in 90.57 seconds** on Windows CPU. This included the existing
cached real-data opt-in with `FEATHERLENS_PREPARED_DIR` set and Hugging Face in
offline mode. A separate smoke check connected the existing data pipeline to all
three adapters using real cached train/validation images and random encoder
initialization:

| Adapter | Spatial features | Pooled features | Logits |
| --- | --- | --- | --- |
| ConvNeXt-Tiny | `[2, 768, 7, 7]` | `[2, 768]` | `[2, 200]` |
| EfficientNet-B3 | `[2, 1536, 7, 7]` | `[2, 1536]` | `[2, 200]` |
| RegNetY-3.2GF | `[2, 1512, 7, 7]` | `[2, 1512]` | `[2, 200]` |

The check retained 4,794 training and 1,200 validation records, opened only the
official-training source, and reproduced validation pixels and logits. All
three models recorded `weights_loaded=False` and `loaded_weight_id=None`.
Compatibility checks passed for the 53 dependency packages without requirements
changes. This verifies CPU integration with real data and random initialization;
this historical Stage 1A check did not verify actual pretrained weights or GPU
execution. Subsequent Stage 1B1 evidence below establishes those checks separately.

Run the CPU suite from the active environment:

```bash
python -m pytest -q
```

The opt-in cached real-data test requires a populated pinned source cache and
`FEATHERLENS_PREPARED_DIR`; otherwise it is skipped. Existing Task 1.1 real-data
and EDA measurements remain documented in their original handoffs. They do not
constitute a new pretrained-model or GPU experiment.

Stage 1B1 returned T4 evidence verified all three official weights and all 36
FP32/FP16 checks. A later targeted automatic-precision run passed 12/12 checks
and storage readback; PR 13 merged as
`6ad1adfe47f339414693c2034658a32bc6df2ca6`. See the
[GPU evidence guide](gpu_preflight.md) for the preserved BF16 failure and limits.
Preflight performed zero optimizer updates. Training/recovery/tracking now exist,
while genuine canonical training evidence, prediction CLI, model comparison,
search, CAM rendering, scratch CNNs and downstream tasks remain pending.
