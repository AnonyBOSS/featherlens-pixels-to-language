# Classification data pipeline — Stage 3

This pipeline turns the [pinned source contract](data_contract.md) and [complete integrity audit](integrity_audit.md) into classification decisions, reproducible partitions, and lazy PyTorch batches. It preserves source assets, caption records, original labels, and official test assignments.

Integration foundation: [PR #4](https://github.com/AnonyBOSS/featherlens-pixels-to-language/pull/4) (integrity audit) and [PR #5](https://github.com/AnonyBOSS/featherlens-pixels-to-language/pull/5) (native image access) are merged into main at `51f422d976e34e87a9bb962a2598c3385d2eab2b`. Stage 3 builds on that foundation without changing the Stage 1 source or Stage 2 audit contracts.

## Preparation and cleaning policy

`prepare_classification(audit_dir, output_dir, *, val_fraction=0.20, seed=42)` requires a completed Stage 2 snapshot covering all 5,994 official training rows and 5,794 official test rows. It validates revisions, source identity, the ordered 200-class mapping, flags, image hashes, duplicate evidence, and reconciled summary counts. Missing coverage or unsupported flags stop preparation.

The `classification-v1` policy makes these decisions:

- A classification image needs a valid class reference, successful image decode, and no non-caption metadata error. Ineligible official training rows receive `partition="excluded"` and explicit `decision_codes`.
- Caption errors, including a missing description, do not exclude an otherwise valid classification image. Captions remain available to other tasks.
- The reviewed warnings `caption.count_mismatch` and `duplicate.exact` are permitted. Other warnings require a reviewed policy before preparation can continue.
- Both members of an exact duplicate group are retained. Eligible group members form one indivisible split unit.
- Cross-official-split exact matches and conflicting labels within an exact group stop preparation for review.
- Every original official test row keeps `partition="test"`, even if ineligible. The test Dataset rejects an ineligible population rather than silently evaluating a smaller benchmark.

The split algorithm `stratified-exact-group-v1` works only on eligible official training rows. Within each class, sorted group identities are shuffled using a local Python RNG with seed 42. The validation target is `floor(class_size * 0.20 + 0.5)`. A subset-sum search selects whole groups nearest that target, choosing the smaller count on a tie. Both training and validation must be nonempty for every class. A class that cannot meet that requirement stops preparation. No official test image is reassigned.

### Measured pinned-source preparation

The October 22, 2026 final preparation accounted for all 11,788 audited rows:

| Partition | Rows |
| --- | ---: |
| Training | 4,794 |
| Validation | 1,200 |
| Official test | 5,794 |
| Excluded | 0 |

All 200 classes are represented in every partition. Each has six validation images; 194 classes have 24 training images and six classes have 23. Official test counts range from 11 to 30 per class and are preserved exactly. The complete per-class counts are stored in `preparation.json` under `eligible_class_counts` (class IDs are JSON string keys). The two class-57 Pigeon Guillemot records at official training source rows 1712 and 1731 both belong to **validation**. Their group is `exact:09134262c62225392edf25319d3c64e8c54bee67e86a97dc4bd872fe4a325c16`. Retaining both preserves source coverage but does not make them independent observations.

These are measured counts for the pinned revisions and the stated policy, seed, and split version. A future policy or source change requires preparation and verification again.

## Reproduce the preparation

Use Python 3.12 and the pinned `requirements-classification.txt` environment. The verified Windows CPU installation used Python 3.12.15, PyTorch `2.14.1+cpu`, and torchvision `0.29.1+cpu`; all 53 installed packages passed dependency compatibility checks. The requirements include the EDA dependency pins and `fsspec==2026.7.0`.

With Python 3.12 available as `python`, run these commands from the repository root in PowerShell. Install the CPU builds first, then the remaining pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-classification.txt
.\.venv\Scripts\python.exe -m pip check
```

Use `.\.venv\Scripts\python.exe` to run the examples. For GPU training, choose matching PyTorch/torchvision accelerator builds separately before installing the requirements; the CPU environment verifies the data interfaces and does not benchmark GPU training.

The requirements pin direct dependencies, rather than providing a complete transitive lockfile with wheel hashes. Fresh installation is verified for Windows/Python 3.12 CPU; other operating systems and accelerator builds need their own compatibility check. A cold setup requires access to both the official PyTorch CPU index and PyPI, followed by Hugging Face access to populate the pinned source cache.

Keep caches, environments, reports, and manifests outside tracked source files; `outputs/` and `.venv/` are ignored.

First generate a complete read-only audit, or use an existing verified Stage 2 snapshot. Save this as a Python script when using Windows:

```python
from src.data.integrity_audit import audit_pinned_sources


def main():
    summary = audit_pinned_sources("outputs/integrity-audit")
    if not summary["coverage_complete"]:
        raise RuntimeError("Inspect the incomplete audit before preparing data")
    print(summary["processed_counts"])


if __name__ == "__main__":
    main()
```

A fresh full audit downloads/caches both official source splits for integrity checks. It does not use test images to fit training parameters. The source shards total about 1.07 GiB; allow 3–5 GiB for cache and temporary space. Cached runtime depends on the machine.

Preparation consumes the saved audit; it does not fetch images or decode them again:

```python
from src.data.preparation import (
    load_classification_manifest,
    prepare_classification,
)

summary = prepare_classification(
    "outputs/integrity-audit",
    "outputs/classification-v1",
    val_fraction=0.20,
    seed=42,
)
rows = load_classification_manifest("outputs/classification-v1")
print(summary["partition_counts"])
print(len(rows))
```

Audit and preparation directories must be separate, with neither containing the other. Use one writer per output directory. Publication marks the summary incomplete first and completes it only after publishing the manifest.

| Generated file | Consumer contract |
| --- | --- |
| `classification_manifest.jsonl` | One record for every source row. Retains audited identity, class/split references, quality flags, hashes, and duplicate memberships; adds `eligible`, `decision_codes`, and `partition`. |
| `preparation.json` | Completion marker, schema/policy/split versions, source revisions/counts, seed, validation fraction, ordered class names, audit fingerprints, manifest SHA-256, partition/class counts, and exact-group constraints. |

Always use `load_classification_manifest()` before constructing Datasets. It checks completion, the manifest fingerprint, identity, decisions, group isolation, counts, and membership against the declared split recipe. It returns the stored decisions and rejects inconsistencies; it never silently creates a different split. The Dataset accepts a sequence of rows and cannot independently certify an omitted preparation summary.

## Training and validation handoff

Save the following as `classification_batch_example.py` at the repository root and run `.\.venv\Scripts\python.exe classification_batch_example.py`. It opens only the official training source; both derived partitions share that source. It assumes preparation already exists.

```python
import random

import numpy as np
import torch

from src.data.classification import (
    CubClassificationDataset,
    make_classification_loaders,
)
from src.data.preparation import load_classification_manifest
from src.data.source import load_cub_split
from src.data.transforms import classification_transform


def main():
    seed = 42
    # With num_workers=0, augmentation uses the caller's torch RNG.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    rows = load_classification_manifest("outputs/classification-v1")
    source = load_cub_split("train")  # Indexed access; streaming is unsupported.

    datasets = {
        "train": CubClassificationDataset(
            source,
            rows,
            partition="train",
            transform=classification_transform(training=True),
        ),
        "val": CubClassificationDataset(
            source,
            rows,
            partition="val",
            transform=classification_transform(training=False),
        ),
    }
    loaders = make_classification_loaders(
        datasets,
        batch_size=32,
        num_workers=0,
        seed=seed,
        pin_memory=False,
    )
    batch = next(iter(loaders["train"]))
    print(batch["image"].shape, batch["label"].dtype)
    print(batch["image_id"][0])
    print(source.class_names[batch["label"][0].item()])


if __name__ == "__main__":
    main()
```

The main guard is required when enabling Windows worker processes. Start with `num_workers=0`, then benchmark a small worker count such as two against available CPU, memory, and storage.

A sample contains exactly `{"image": tensor, "label": class_id, "image_id": stable_id}`. Default batching produces:

| Key | Batched value |
| --- | --- |
| `image` | `float32` tensor shaped `[B, 3, 224, 224]` with default transforms |
| `label` | `int64` tensor shaped `[B]`, containing the original class IDs 0–199 |
| `image_id` | List of stable source identifiers, length `B` |

Dataset construction examines manifest metadata. Each indexed read fetches and decodes one source image on demand, checks source label/ID/filename against the manifest, and returns the transformed image. There is no decoded-image cache. Failures raise with stable identity and source-row context; samples are never silently skipped.

Loaders shuffle only training, keep the final partial batch, and open no additional sources. Each partition has its own generator: seeds `seed`, `seed+1`, and `seed+2` for train, validation, and test. Workers seed Python and NumPy from PyTorch's worker seed. The factory does not reset global RNGs. Seed the caller once before iteration for zero-worker augmentation; recreate loaders and reset caller seeds to reproduce a run. Reproducibility assumes the same library versions, worker count, and iteration schedule; this API does not implement checkpointed RNG restoration.

### Preprocessing and configurable normalization

`load_rgb(reference)` accepts the bytes/path mapping exposed by Hugging Face `Image(decode=False)`. Embedded bytes take precedence over a path. It fully decodes, applies EXIF orientation, composites transparency on white, and returns an independent RGB8 Pillow image. Direct callers must close it. It performs no resizing, ICC color-management conversion, hashing, or source modification.

The default `classification_transform(training=..., geometry="fit")` uses bilinear interpolation to fit the entire RGB image inside a centered white 224-pixel square. It preserves aspect ratio within integer pixel rounding and performs no cropping. Training adds a horizontal flip with probability 0.5; validation/test geometry is deterministic. `image_size` changes the output square size.

This default follows review of 12 training originals with two random crops each: the original crop profile removed head content at source rows 1429 and 305. The approved fit profile preserves full-image content for train, validation, and test. Extreme aspect ratios leave more padding and fewer object pixels, so compare model results using the fixed validation partition. Geometry changes do not alter source assets, audit fingerprints, or split membership.

The original profile remains available explicitly as `geometry="crop"`. It resizes the shorter side to 256 pixels with antialiased bilinear interpolation, then applies a random 224-pixel training crop or a deterministic evaluation center crop. Training also flips horizontally with probability 0.5. At another `image_size`, its shorter-side resize scales by `256/224` with half-up rounding. Use the same chosen geometry for training and evaluation.

Normalization is a model choice:

```python
# Default: full-image fit and fixed external ImageNet constants.
default = classification_transform(training=False)

# Unnormalized RGB floats in [0, 1], including white padding.
plain = classification_transform(training=False, normalization=None)

# A model-specific fixed mean/std pair, applied after image preparation.
custom = classification_transform(
    training=False,
    normalization=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
)

# Optional original crop profile: choose it explicitly for both phases.
crop_train = classification_transform(training=True, geometry="crop")
crop_eval = classification_transform(training=False, geometry="crop")
```

ImageNet means are `(0.485, 0.456, 0.406)` and standard deviations are `(0.229, 0.224, 0.225)`. Means/stds must have three finite values; standard deviations must be positive. Use compatible transforms for training and evaluation. If future experiments estimate dataset statistics, estimate them from the derived training partition only and record the resulting constants. Changing preprocessing does not require changing saved split memberships.

Preview augmented training images for the selected model geometry before a long run. The optional crop profile can remove peripheral content, especially on extreme aspect ratios; the fit profile retains all image content. Neither profile uses bird bounding boxes.

### Explicit final evaluation

After model selection is complete, use a separate evaluation script. Save the following as `classification_test_example.py` at the repository root and run `.\.venv\Scripts\python.exe classification_test_example.py`. Reuse the geometry and normalization fixed for the trained model; this example uses the documented defaults. Training/validation examples never open the official test source.

```python
import json
from pathlib import Path

from src.data.classification import (
    CubClassificationDataset,
    make_classification_loaders,
)
from src.data.preparation import load_classification_manifest
from src.data.source import load_cub_split
from src.data.transforms import classification_transform


def main():
    prepared_dir = Path("outputs/classification-v1")
    rows = load_classification_manifest(prepared_dir)
    preparation = json.loads(
        (prepared_dir / "preparation.json").read_text(encoding="utf-8")
    )
    test_source = load_cub_split("test")
    if list(test_source.class_names) != preparation["class_names"]:
        raise RuntimeError("Official test class mapping differs from preparation")
    test_dataset = CubClassificationDataset(
        test_source,
        rows,
        partition="test",
        transform=classification_transform(training=False, geometry="fit"),
    )
    if len(test_dataset) != 5794:
        raise RuntimeError("Official test population must remain complete")
    loader = make_classification_loaders(
        {"test": test_dataset}, batch_size=32, num_workers=0, seed=42,
    )["test"]
    batch = next(iter(loader))
    print(len(test_dataset), batch["image"].shape, batch["label"].dtype)
    print(batch["image_id"][0])


if __name__ == "__main__":
    main()
```

This checks the data interface only. It does not evaluate model quality or select preprocessing using test data. Preserve every official test assignment and reuse the unchanged class-ID mapping.

## Verification boundary and limits

Checkpoint A tests cover malformed/incomplete audit coverage, provenance and class validation, caption-only exceptions, explicit exclusions, duplicate evidence, cross-split/label-conflict stops, deterministic group-aware partitions, invalid configuration, publication failure, and manifest/summary tampering. The combined suite at that checkpoint passed 213 tests. Real preparation produced the measured counts above.

At final review, a complete 11,788-row re-audit took 29.001 seconds on the development machine. Image, caption, and exact-group reports were byte-identical to the Stage 2 reports. This verifies preservation of the audit's `exif-rgb8-white:v1` pixel policy.

Tests in `tests/data/test_image_io.py`, `test_transforms.py`, and `test_classification.py` exercise image loading, transform configuration, lazy sample contracts, and loader behavior. The final combined suite includes the published native-image adapter, the opt-in real-data test, and regressions rejecting unexplained unchecked images while retaining malformed source rows with explicit decisions. **280 tests passed in 70.00 seconds**, with no failures or skips in that opted-in run.

To reproduce that combined check in PowerShell with the pinned sources already cached and preparation available:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:FEATHERLENS_PREPARED_DIR = "outputs/classification-v1"
.\.venv\Scripts\python.exe -m pytest -q
```

`pytest.ini` limits automatic discovery to `tests/`, so ignored output backups are never collected as source tests. Ordinary test runs skip the opt-in CUB test unless `FEATHERLENS_PREPARED_DIR` is set. Offline mode prevents downloads; populate the pinned source cache first. Clear these environment variables when returning to online source setup if necessary.

The final real-data check used the approved fit geometry, cached sources, CPU execution, and two PyTorch threads:

| Check | Measured result |
| --- | --- |
| One 32-image training batch | 0.118 seconds |
| One 32-image validation batch | 0.106 seconds |
| One 32-image official test batch, explicitly requested | 0.104 seconds |
| Windows two-worker startup and first eight-image validation batch | 13.174 seconds; matches zero-worker pixels and IDs |
| Small classifier forward/backward check | `[32, 200]` logits; finite loss 5.61725; backward passed |
| Full re-audit, preparation twice, batches, workers and previews | 46.024 seconds |
| Parent-process peak working set for that full verification | 1,484.80 MiB; includes source memory maps and excludes workers |

Preparation and validation twice took 2.542 seconds; the two generated preparations were byte-identical, including split membership. Cross-partition checks found zero shared stable IDs, encoded-byte hashes, or pixel hashes. A restarted seeded training loader reproduced both order and transformed pixels; evaluation repeated exactly under different RNG state.

These measurements describe this machine and cache state; they are not runtime or memory guarantees. Worker startup dominated the two-worker check, so the zero-worker default remains appropriate until a longer workload demonstrates a benefit.

Visual inspection compared 12 training originals with 24 fit/flipped views. Heads and beaks remained visible, including at source rows 1429 and 305 where the original crop profile removed head content. This is a targeted preview check, not a bounding-box guarantee for every image. Switching to fit changed no source asset, audit fingerprint, or split membership.

All four fit/crop training/evaluation transform profiles passed serialization checks for worker use. Independent final review's unchecked-image validation finding was reproduced, fixed, and covered by two regressions. The development environment's 53 packages passed compatibility checks, and its offline requirements dry run required no changes.

Before publication, a fresh standard Python 3.12.15 venv included pip 25.0.1 and successfully installed the 53 dependency packages from cache. Both pip and uv compatibility checks passed for all 54 distributions including pip. A clean export containing only 59 candidate repository files then passed the entire opted-in suite: **280 tests in 76.08 seconds**, with no failures or skips.

The fresh install used the cached official CPU torch wheel URL and PyPI dependency cache. Separate bounded metadata requests confirmed the documented CPU-index torch/torchvision versions for Python 3.12 Windows. The complete cold online pip download/install sequence was not repeated. Native torchvision CPU operations, Hugging Face/Pillow image decoding, project transforms and a tiny forward/backward check also passed in the fresh environment.

Git diff/whitespace and ignored-artifact checks passed. Generated audits, prepared manifests, previews, caches, and environments remain excluded from the intended source changes.

Preparation keeps compact records for all 11,788 rows in memory. Group retrieval uses hash buckets and union-find, with a per-class reachable-count search for splitting. It avoids pairwise image comparisons and repeated preparation-time decoding. During training, one decoded image must fit in memory; each worker also needs its own decoder and transform working space.

Training access relies on the pinned source and validated metadata; it does not recompute image hashes on every sample. Re-audit if cache integrity is in doubt. Exact hashing does not identify all perceptual near-duplicates or annotation mistakes. pHash, caption-content analysis, EDA visualizations, model training, checkpointing, and a classification CLI are outside this stage.
