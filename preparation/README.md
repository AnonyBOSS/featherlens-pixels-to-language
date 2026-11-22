# Prepared Classification Split

Tracked CUB-200-2011 split snapshot used across all models (`stratified-exact-group-v1`, policy `classification-v1`, seed `42`). See [`docs/data.md`](../docs/data.md) for the full source and audit details.

## Files

| File | Purpose |
| --- | --- |
| `classification_manifest.jsonl` | Per-row IDs, labels, hashes, duplicate group IDs, eligibility, and split assignments (`11,788` rows). |
| `preparation.json` | Split summary, ordered 200 class names, source commits, counts, and manifest SHA-256. |
| `provenance.json` | Original EDA snapshot metadata (kept unchanged). |
| `classification_batch_example.py` | Minimal script loading one batch from `train` and `val`. |
| `inventory.json` | File sizes and SHA-256 checksums for the files in this directory. |

## Splits

- **Train:** `4,794` images (`194` classes × 24, `6` classes × 23)
- **Validation:** `1,200` images (`6` per class across all 200 classes; keeps the duplicate Pigeon Guillemot pair at train rows `1712` and `1731` together)
- **Test:** `5,794` images (official test split, untouched)
- **Excluded:** `0`
- **Manifest SHA-256:** `a8bd05e048febf32df6d84a6e1bc16eddf6416be723e12dfc89621044962736b`

## Usage

```python
from src.data.preparation import load_classification_manifest

rows = load_classification_manifest("preparation")
print(len(rows))  # 11788
```

This directory is the default `data.prepared_dir` in `configs/resnet50/inaturalist.yaml`. To test real DataLoader batches from the repository root (downloads/caches official training images from Hugging Face on first run):

```powershell
python preparation/classification_batch_example.py
```

When regenerating alternative splits or audits during experiments, write outputs under `outputs/` rather than modifying this directory. If any tracked file here is intentionally updated, update `inventory.json` to match.
