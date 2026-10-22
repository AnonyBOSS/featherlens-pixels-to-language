# Statistical EDA handoff

The approved EDA scope is complete. It covers the 5,994 raw official-training
images and 59,940 captions, plus descriptive summaries of the verified 4,794-image
training and 1,200-image validation partitions. The 5,794 official test records
contribute identity/assignment checks only. They do not inform EDA recommendations.

See [findings](eda_findings.md), [image methods](eda_stage2.md), and
[caption/prepared-data methods](eda_stage3.md) for definitions and interpretation.

## Environment and checks

Use Python 3.12. From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-eda.txt
.\.venv\Scripts\python.exe -m pytest -q tests/analysis
```

For the complete repository suite, install the CPU classification dependencies
using [the shared setup](classification_data.md#reproduce-the-preparation).
Then set the cache and preparation paths:

```powershell
$env:HF_HOME=Join-Path (Get-Location).Path '.venv\hf-cache'
$env:HF_DATASETS_CACHE=Join-Path $env:HF_HOME 'datasets'
$env:HF_HUB_CACHE=Join-Path $env:HF_HOME 'hub'
$env:MPLBACKEND='Agg'
$env:MPLCONFIGDIR=Join-Path (Get-Location).Path '.venv\mpl-cache'
$env:HF_HUB_OFFLINE='1'
$env:HF_DATASETS_OFFLINE='1'
$env:FEATHERLENS_PREPARED_DIR=(Resolve-Path 'outputs/eda-stage3/classification-v1').Path
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q tests
```

Offline execution requires the pinned source cache and verified preparation
already present. Without `FEATHERLENS_PREPARED_DIR`, the real-data test is skipped.
The complete suite passed 533 tests with no failures or skips in a source-only
export using the existing environment/cache. The EDA-only command passed 384
tests. Public metadata, caption and prepared-data commands reproduced the complete
reports; the documented figure recipe reproduced all 13 groups. The unchanged
image CLI passed a two-record smoke check; its full 5,994-image scan was verified
during Stage 3. This was not a fresh dependency installation. Direct requirements are pinned; transitive dependencies are not
fully locked. The verified runtime was Python 3.12.0, torch 2.14.1+cpu and
torchvision 0.29.1+cpu on Windows.

## Reports and delivery bundle

The local delivery is `outputs/eda-handoff-2026-10-22.zip`. It contains one matching
report set, 13 PNG/PDF pairs, the two existing preparation files, documentation,
verification records and a SHA-256 inventory. Reports come from the integrated
October 22 verification; historical outputs remain preserved separately.
Generated artifacts are transferred separately and are not committed to Git.

| Report under `reports/` | Schema | Content |
| --- | --- | --- |
| `train-full.json` | `1.1` | Class counts, caption lengths, vocabulary and repetition |
| `train-images-full.json` | `image-eda-1.0` | Native geometry and eligible-image RGB/brightness/contrast/saturation |
| `train-caption-details-full.json` | `caption-details-1.0` | Declared language mentions, per-image diversity and overlap |
| `prepared-training-eda.json` | `prepared-eda-1.0` | Verified membership, quality flags and train/validation summaries |

The figure groups under `figures/` cover class distribution; caption lengths,
vocabulary and repetition; image dimensions, appearance, RGB, galleries and
extremes; caption language, diversity and class variation; and prepared class
coverage. Every group has PNG and PDF exports. `inventory.json` lists their exact
names and hashes. PDF timestamps can change on regeneration; delivery hashes
identify the actual supplied bytes.

`prepared-training-eda.json` binds its image/caption inputs and preparation files
by SHA-256. Keep those files together; do not substitute an older image report.
The bundle's `preparation/` contains `preparation.json` and
`classification_manifest.jsonl`, consumed through the shared manifest loader.
Audit fingerprints are recorded and were checked against the retained audit;
the four audit files are not included. Obtain or regenerate them through the
[shared audit](integrity_audit.md) if independent audit reproduction is needed.

Source revisions are fixed by [the contract](data_contract.md):

- Training: `bd3988995e6db193427255abdebd522afe24ce76`.
- Test: `82447a42c5e05c7fe21cd36accfdf399148e8668`.
- Preparation: seed 42, validation fraction 0.20, `classification-v1`,
  `stratified-exact-group-v1`, pixel policy `exif-rgb8-white:v1`.

## Reproduce reports

Use the cache variables above, a verified preparation directory, and a fresh
output directory. The metadata command streams the training source and may need
network access even when an indexed cache exists. The remaining source commands
use indexed loading. None opens the official test source for EDA.

```powershell
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
Remove-Item Env:HF_DATASETS_OFFLINE -ErrorAction SilentlyContinue
$run='outputs/eda-final-reproduction'
if (Test-Path -LiteralPath $run) { throw 'Choose a fresh output directory' }
$reports=Join-Path $run 'reports'
$figures=Join-Path $run 'figures'
$prepared='outputs/eda-stage3/classification-v1'
.\.venv\Scripts\python.exe -m src.analysis.explore --full-training --split train --output "$reports/train-full.json"
.\.venv\Scripts\python.exe -m src.visualization.eda_plots --report "$reports/train-full.json" --output-dir $figures
.\.venv\Scripts\python.exe -m src.analysis.explore_images --output "$reports/train-images-full.json" --figures-dir $figures
.\.venv\Scripts\python.exe -m src.analysis.explore_caption_details --output "$reports/train-caption-details-full.json" --figures-dir $figures
.\.venv\Scripts\python.exe -m src.analysis.explore_prepared --prepared-dir $prepared --image-report "$reports/train-images-full.json" --caption-report "$reports/train-caption-details-full.json" --output "$reports/prepared-training-eda.json" --figures-dir $figures
```

Run one command at a time and stop if it fails. These commands preserve source
assets and split membership. Regenerating image measurements reads all training
pixels; it is unnecessary when only figures are needed.

## Regenerate figures from supplied reports

Extract the delivery ZIP into `outputs/eda-handoff-2026-10-22/`. With the pinned
training source cached, run this Python block from the repository root. It writes
to a separate figure directory. Only the image galleries reopen a bounded set
of source images; the other figures use saved report values.

```python
import json
from pathlib import Path
from src.data.source import load_cub_split
from src.analysis.explore_images import prepare_gallery
from src.visualization.eda_plots import generate_training_figures
from src.visualization.image_plots import generate_image_figures
from src.visualization.caption_detail_plots import generate_caption_detail_figures
from src.visualization.prepared_eda_plots import generate_prepared_eda_figures

reports = Path("outputs/eda-handoff-2026-10-22/reports")
out = Path("outputs/eda-figure-check")
read = lambda name: json.loads((reports / name).read_text(encoding="utf-8"))
generate_training_figures(read("train-full.json"), out)
images = read("train-images-full.json")
pixels = prepare_gallery(images, load_cub_split("train"))
generate_image_figures(images, out, image_getter=pixels)
generate_caption_detail_figures(read("train-caption-details-full.json"), out)
generate_prepared_eda_figures(read("prepared-training-eda.json"), out)
```

## Recommendations and limits

- Keep the fixed train/validation membership. Both recorded exact-duplicate
  Pigeon Guillemot images remain in validation; they are not independent examples.
- Fit future preprocessing or vocabulary settings on the 4,794 derived training
  images only. The 5,994-image raw summaries include validation members.
- Native image summaries weight each eligible image equally. They include
  backgrounds, apply no ICC correction, and are not pooled normalization values
  or statistics of padded/transformed training tensors.
- Retain source captions. Analysis tokens, lexical mentions and Jaccard overlap
  do not establish semantic accuracy or prescribe a model tokenizer. Inspect
  directional language in context before proposing caption-aware augmentation.

Deferred by agreement: blur/sharpness scoring; formal between-species image
comparisons; new image-caption association studies; and exhaustive spelling or
semantic-quality audits. Descriptive image means already exist for every class.
Contrast is not a blur measurement. Supplied integrity flags do not certify
annotation correctness or perceptual near-duplicate absence.

The [README contribution proposal](eda_readme_proposal.md) is ready for
coordination. The shared README, loader, cleaning, splitting and transform
implementations are unchanged by this EDA handoff.
