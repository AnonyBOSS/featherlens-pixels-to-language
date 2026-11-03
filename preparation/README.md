# Shared classification preparation

This directory contains the reviewed EDA team's fixed classification preparation
from `FeatherLens-CNN-Handoff`. Both the manual CNN and transfer-learning models
should consume this same snapshot. No new split is needed.

## Included files

| File | Purpose |
| --- | --- |
| `classification_manifest.jsonl` | Original image identities, labels, eligibility decisions, hashes, duplicate groups, and partition memberships. |
| `preparation.json` | Completion marker, ordered class names, source revisions, split policy, seed, population counts, and manifest fingerprint. |
| `provenance.json` | Original EDA handoff provenance, preserved unchanged. |
| `classification_batch_example.py` | EDA team's example, adapted to read the preparation files from this directory. |
| `inventory.json` | Byte sizes and SHA-256 fingerprints of the other files in this directory. |

The manifest, preparation summary, and provenance are byte-for-byte copies of
the EDA handoff. The large EDA PDF, original image assets, caches, and trained
models are not included. The historical report references in `provenance.json`
describe the original handoff; its repository commit describes the EDA snapshot,
not the current model implementation.

## Fixed populations and boundaries

| Partition | Images | Use |
| --- | ---: | --- |
| Training | 4,794 | Weight optimization and fitting learned preprocessing. |
| Validation | 1,200 | Model selection; six records per class. |
| Official test | 5,794 | Final evaluation after model selection. |

All 200 classes are represented in every partition. There are zero exclusions.
Preparation uses seed 42 and `stratified-exact-group-v1`. The exact duplicate pair
at official-training rows 1712 and 1731 remains together in validation; those
observations are dependent. Preserve memberships and the ordered class mapping.
No exact encoded-byte or normalized-pixel hashes cross the prepared partitions;
this does not certify the absence of perceptual near-duplicates.

## Load the prepared metadata

Run from the repository root in the project's classification environment:

```python
from src.data.preparation import load_classification_manifest

rows = load_classification_manifest("preparation")
print(len(rows))  # 11788
```

Always use this loader before building a Dataset. It checks completion,
fingerprints, pinned identities, eligibility decisions, class counts, split
reproducibility, and duplicate isolation. This directory also serves as
`data.prepared_dir` in `configs/resnet50.yaml`.

## Read real training and validation batches

See [the classification environment guide](../docs/classification_data.md) for
the verified Python 3.12 setup and pinned dependencies. The guide's CPU wheel
commands validate loading; GPU training requires matching CUDA-enabled PyTorch
and torchvision builds.

After installing the classification dependencies, run from the repository root:

```powershell
python preparation/classification_batch_example.py
```

The example validates the manifest, opens only the pinned official-training
source, and prints training/validation lengths of 4,794 and 1,200. It reads a
batch from each partition with shapes `[4, 3, 224, 224]`, float32 images, int64
labels, and stable image IDs. A first source load downloads/caches the images
from Hugging Face. This package does not contain the photographs.

The shared default transform applies EXIF/RGB handling, aspect-ratio-preserving
fit into a centered white 224-by-224 canvas, ImageNet normalization, and a
training-only horizontal flip with probability 0.5. Evaluation is deterministic.
Fit new data-derived preprocessing parameters only on the 4,794 training rows.

## Regeneration and artifacts

Regenerate alternative preparation into an ignored `outputs/` directory using
the full audit and [the preparation guide](../docs/classification_data.md).
The handoff's summary fingerprints do not replace the full audit files required
for regeneration. Replace this shared snapshot only through a reviewed change.
Keep checkpoints, image caches, experiment logs, and generated previews outside
this directory. Trained models are uploaded to Google Drive separately.

When intentionally changing any included file, regenerate `inventory.json` from
the files in this directory, excluding the inventory itself.
