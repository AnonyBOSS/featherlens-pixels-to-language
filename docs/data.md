# CUB-200-2011 Data Pipeline

All dataset loading, integrity checks, and classification splits live under `src/data/` and share one fixed train/validation/test partition (`preparation/`).

## Pinned Hugging Face sources

| Official split | Repository | Commit SHA | Rows |
| --- | --- | --- | ---: |
| `train` | `Multimodal-Fatima/CUB_train` | `bd3988995e6db193427255abdebd522afe24ce76` | 5,994 |
| `test` | `Multimodal-Fatima/CUB_test` | `82447a42c5e05c7fe21cd36accfdf399148e8668` | 5,794 |

Required dataset columns are `image`, `description`, `label`, `file_name`, and `id`. Class labels (`0`–`199`) preserve the exact `ClassLabel.names` order from Hugging Face rather than sorting alphabetically. Each sample gets a deterministic ID:

```text
hf:<repo-id>@<commit-sha>:<split>:<zero-based-row>
```

### Loading records and decoding images

- `src.data.source.load_cub_split(split, streaming=False)` and `load_cub_sources()` open the pinned Hugging Face splits with `Image(decode=False)` so metadata scans and caption parsing never decode pixels unnecessarily. `parse_captions()` keeps blank lines with their original zero-based `caption_index` while stripping/collapsing whitespace in `clean_caption`.
- `src.data.image_io.load_rgb()` and `read_source_image(source, source_row)` handle decoding: Pillow opens the first frame, applies EXIF orientation (`ImageOps.exif_transpose`), composites any alpha channel onto solid white (`255, 255, 255`), and converts to 8-bit `RGB` without ICC color profile transforms.

## Integrity audit

`src.data.integrity_audit.audit_pinned_sources("outputs/integrity-audit")` checks both splits before preparation (~1.07 GiB across Parquet shards; budget 3–5 GiB free space for Hugging Face cache). Full audit results on the pinned commits:

| Check | Result |
| --- | --- |
| Official train rows accounted for | 5,994 / 5,994 |
| Official test rows accounted for | 5,794 / 5,794 |
| Metadata, fetch, or decode errors | 0 |
| Parsed / usable captions (10 per image) | 117,880 / 117,880 |
| Cross-split exact duplicates | 0 |
| Within-split exact duplicate groups / images | 1 / 2 |

The only duplicate warning is a pair of exact-identical `Pigeon Guillemot` (class 57) images in official train at rows `1712` (`Pigeon_Guillemot_0018_40195.jpg`) and `1731` (`Pigeon_Guillemot_0081_40339.jpg`), grouped as `exact:09134262c62225392edf25319d3c64e8c54bee67e86a97dc4bd872fe4a325c16` using pixel hash policy `exif-rgb8-white:v1`.

## Classification preparation and splits

`src.data.preparation.prepare_classification(audit_dir, output_dir, val_fraction=0.20, seed=42)` filters eligible images (`classification-v1`) and assigns train/validation partitions (`stratified-exact-group-v1`) while keeping exact duplicate groups in the same split:

| Partition | Rows | Notes |
| --- | ---: | --- |
| `train` | 4,794 | 194 classes with 24 images, 6 classes with 23 |
| `val` | 1,200 | 6 images per class across all 200 species (includes both Pigeon Guillemot duplicates) |
| `test` | 5,794 | Official test split preserved untouched (11–30 per class) |
| `excluded` | 0 | No decode or metadata errors |

The tracked manifest snapshot sits in [`preparation/`](../preparation/README.md) with manifest SHA-256:

```text
a8bd05e048febf32df6d84a6e1bc16eddf6416be723e12dfc89621044962736b
```

Always read manifests with `load_classification_manifest(prepared_dir)`, which verifies `preparation.json`, the manifest SHA-256, duplicate group isolation, and split reproducibility before exposing rows.

## PyTorch datasets and transforms

`src.data.classification.CubClassificationDataset` and `make_classification_loaders()` build lazy loaders that only touch official `train` when loading `train`/`val`, and only shuffle `train`. Default preprocessing (`src.data.transforms.classification_transform`) fits the whole EXIF-corrected RGB image inside a centered white `224x224` square (preserving aspect ratio), normalizes with ImageNet mean/std `(0.485, 0.456, 0.406)` / `(0.229, 0.224, 0.225)`, and applies random horizontal flipping (`p=0.5`) during training only.

Quick verification against the tracked preparation:

```python
from src.data.preparation import load_classification_manifest

rows = load_classification_manifest("preparation")
print(len(rows))  # 11788
```

Or run the batch smoke script:

```powershell
python preparation/classification_batch_example.py
```
