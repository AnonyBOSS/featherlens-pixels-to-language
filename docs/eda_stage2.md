# Statistical EDA: full training analysis and figures

This stage measures the complete official training source through the existing
`load_cub_split("train", streaming=True)` interface. The source contract, loader,
caption parser, shared README, official splits, and cleaning policies are unchanged.
It also measures actual training images through the shared
`src.data.image_io.read_source_image` interface and generates image figures and
captioned galleries. The approved numerical definitions remain unchanged.

## Reproduce the measured training report

Use the environment from [Stage 1 setup](eda_stage1.md). From this worktree root:

```powershell
$env:HF_HOME=Join-Path (Get-Location).Path '.venv\hf-cache'
.\.venv\Scripts\python.exe -m src.analysis.explore --full-training --split train
```

The command exhausts the official training iterator without a row limit. It
compares observed rows with the documented 5,994-row reference, rather than
substituting that value for measurement. A count mismatch remains visible in
the report and leaves `full_split_verified` false. Iterator failures propagate
without returning a completed report. Both command modes reject `--split test`
before opening a source, and both report builders reject nontraining sources
before iterating records. Full-training mode also rejects any explicit `--limit`.

The report is saved to `outputs/eda/reports/train-full.json`. It records the
source revision, actual package versions, complete-versus-prefix scope, class
counts, caption coverage, lengths, token frequencies, repetition, bounded text
examples, and limitations. Streaming avoids image decoding in this analysis,
but may transfer training Parquet blocks containing image bytes. The official
test source is not opened.

## Regenerate the figures without reading the dataset

```powershell
$env:MPLBACKEND='Agg'
$env:MPLCONFIGDIR=Join-Path (Get-Location).Path '.venv\mpl-cache'
.\.venv\Scripts\python.exe -m src.visualization.eda_plots --report outputs/eda/reports/train-full.json --output-dir outputs/eda/figures
```

The plotting module consumes saved counts and frequency tables, not source
records. It does not recalculate tokenization, percentiles, cleaning decisions,
or model parameters. It accepts report schema `1.1` and checks required plotting
fields, finite values, training provenance, selection scope, and reconciliation
of saved count tables before creating an output directory or rendering files.
Historical schema `1.0` reports lack the required distributions; regenerate them
with the current analysis command rather than filling in missing values.

All four figures are rendered as PNG and vector PDF in a temporary staging
directory before publication. A rendering failure leaves existing outputs
untouched. Publication replaces only the eight named files and restores their
previous contents if a replacement fails; unrelated files are preserved. This
rollback protects against caught export errors, not process termination or
concurrent writers during publication.

| Figure stem | Purpose and interpretation |
| --- | --- |
| `training_class_distribution` | Show all 200 class counts by rank and readable largest/smallest examples. Count axes start at zero so a one-image difference is not visually exaggerated. Class ties use source class-ID order. |
| `training_caption_lengths` | Show caption word/character length distributions and reported median/p95 markers, alongside caption coverage. The unit is a nonempty caption. Uniform image coverage is displayed as an annotation instead of an uninformative one-bar chart. |
| `training_caption_vocabulary` | Show frequent words with stopwords retained and the vocabulary rank/frequency curve. Hapax token share uses all tokens as its denominator. |
| `training_caption_repetition` | Distinguish overlapping within-image/across-image repeated-text scopes and show exact descriptions shared by many images. Repeated text alone does not establish image duplication or leakage. |

Generated files are Git-ignored. Interpretations and recommendations are tracked
in [EDA findings](eda_findings.md). Source class names are retained verbatim;
selected class examples are not an exhaustive ranking table.

## Statistical computations

The existing pure dataset/caption functions remain independent of plotting.
Caption summaries now also save ascending frequency tables for word length,
character length, and nonempty captions per image. Their counts reconcile to
the explicit caption/image denominators. Global caption repetition records
unique clean texts and extra occurrences beyond one per exact text. These
extras equal within-image extra occurrences plus additional distinct-image
appearances across images; the two component counts have different units.

No stopwords, rare words, repeated captions, or images are removed. The source
parser's raw/clean fields and the documented analysis tokenizer are unchanged.
Full-data descriptive statistics are not confidence intervals and do not verify
annotation accuracy. Captions attached to one image are related observations.

## Approved image measurement functions

`src.analysis.image_stats.measure_resolution(width, height)` returns dimensions,
pixel count, megapixels (`width * height / 1e6`), and aspect ratio (`width / height`).

`measure_rgb_array(rgb)` accepts a nonempty native-resolution `H x W x 3` NumPy
array: either uint8 encoded RGB, scaled internally by 255, or finite floating RGB
values already in `[0, 1]`. It uses every pixel and never modifies the input.

| Measurement | Definition |
| --- | --- |
| RGB | Per-channel mean and population standard deviation, `ddof=0`, in `[0,1]` units |
| Brightness | Mean of `Y = 0.299R + 0.587G + 0.114B` on encoded RGB values; a luma proxy rather than physical luminance |
| Contrast | Population standard deviation of the same `Y` values |
| Saturation | Mean HSV `S = (max(R,G,B) - min(R,G,B)) / max(R,G,B)`, with black pixels assigned zero |

These functions do not decode files, apply EXIF orientation, resolve transparency,
convert color modes, identify corrupt images, or choose validity thresholds.
Their dimensions describe the supplied array. The shared adapter preserves native
geometry and separately reports original versus oriented dimensions. The image
consumer records both, including any header geometry retained on failures.

Future dataset summaries should weight each eligible image equally. That is
different from pooling every pixel to estimate normalization parameters. Any
normalization, vocabulary, threshold, or augmentation policy must be fitted or
selected using the eventual derived training partition after validation splitting.

## Run the training-image analysis

The [shared image-access contract](image_access.md) is now available. From the
repository root, using the same EDA environment and cache:

```powershell
$env:HF_HOME=Join-Path (Get-Location).Path '.venv\hf-cache'
$env:HF_DATASETS_CACHE=Join-Path $env:HF_HOME 'datasets'
$env:HF_HUB_CACHE=Join-Path $env:HF_HOME 'hub'
$env:MPLBACKEND='Agg'
$env:MPLCONFIGDIR=Join-Path (Get-Location).Path '.venv\mpl-cache'
.\.venv\Scripts\python.exe -m src.analysis.explore_images --split train --figures-dir outputs/eda/figures
```

The default visits every indexed official training row. Use `--limit 32` for an
explicitly nonrandom preliminary prefix. Both the command and report builder
reject held-out sources before reading images. Iterator/adapter programming
errors abort completion; recoverable shared row/fetch/decode failures remain
records with their original statuses and reasons.

The full report is `outputs/eda/reports/train-images-full.json`, using schema
`image-eda-1.0`. It retains every attempted identity, class metadata, shared
status, original/oriented dimensions, mode, EXIF/transparency/ICC facts, analysis
eligibility, per-image measurements, source provenance, and aggregate summaries.
It is saved before optional gallery rendering, so a rendering failure does not
discard a completed measurement scan. JSON publication uses a temporary sibling
file and replacement; invalid serialization or failed replacement preserves an
existing report.

Before rendering, saved reports must reconcile the declared indexed source size,
complete or prefix row coverage, per-row source provenance, canonical image IDs,
and class mapping. Gallery preparation also checks the current source size and
class mapping against the report before reading selected images.

### Shared pixels and analysis eligibility

Fetching, complete decoding, EXIF correction, and RGB conversion belong to the
shared adapter. The analysis does not implement a second decoder or change any
shared source, contract, validity flag, cleaning rule, or split.

- The adapter returns native oriented RGB8 pixels without resizing or ICC color
  management. Its white-alpha display policy is preserved, not reapplied.
- Color measurements require shared status `ok`, no original transparency
  metadata, and original mode `1`, `L`, `P`, `RGB`, `CMYK`, or `YCbCr`.
- Transparent or unsupported numeric modes are retained with analysis status
  `unsupported_mode` and a specific exclusion reason. This is a color-analysis
  eligibility decision, not an image-invalidity or classification-exclusion rule.
- Original and oriented resolution summaries include every available dimension
  pair, including header facts retained on failures. Each denominator is explicit.
- Missing class metadata remains in global image summaries and has a separate
  unclassified count; it is not assigned to an invented species.
- Every returned PIL image is closed. Only small selected gallery arrays remain
  in memory after measurement; no dataset-wide decoded-image cache is built.

Global appearance summaries give each eligible image equal weight. The means of
per-image RGB means or standard deviations are descriptive; they are not pooled
pixel normalization estimates. Raw source duplicates remain in this population.

### Image figure groups

Each group is exported as PNG and PDF using the same staged-render/publication
helper as the class/caption figures. All ten new image-figure files render before
any of their existing targets are replaced; unrelated figures remain untouched.

| Figure stem | Purpose |
| --- | --- |
| `training_image_dimensions` | Original width/height scatter, oriented aspect ratios, and native megapixels, with geometry-availability denominators. |
| `training_image_appearance` | Per-image brightness, contrast, and saturation distributions with fixed, documented units. |
| `training_image_rgb` | R/G/B per-image mean distributions on matched 0–1 axes. |
| `training_image_gallery` | Illustrative images from evenly spaced observed class IDs, using the first eligible source row in each selected class. |
| `training_image_extremes` | Lowest/highest brightness, contrast, and saturation examples, with source-order tie breaking. Images can recur. |

Galleries use actual shared-adapter pixels and a source caption with its original
caption index. The controller verifies saved identity, class labels, and metrics
against the selected images before publication. Galleries are illustrative, not
random samples or semantic-quality audits. Extreme measurements are not automatic
quality failures.

To regenerate figures from saved measurements without scanning every image again,
run this Python code from the repository root in the EDA environment:

```python
import json
from pathlib import Path
from src.data.source import load_cub_split
from src.analysis.explore_images import prepare_gallery
from src.visualization.image_plots import generate_image_figures

report = json.loads(Path("outputs/eda/reports/train-images-full.json").read_text(encoding="utf-8"))
source = load_cub_split("train")
pixels = prepare_gallery(report, source)
generate_image_figures(report, "outputs/eda/figures", image_getter=pixels)
```

Only the selected gallery images are decoded and checked again. Plotting functions
consume the saved measurements, and do not fit thresholds or preprocessing.

Audit quality flags and future derived train/validation manifests remain later
integration inputs. The official test split remains reserved for final evaluation.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Synthetic tests check full iterator exhaustion, reference-count mismatches,
interrupted scans, training-only command and builder guards, saved-frequency
reconciliation, plotted counts and labels, report compatibility, preservation of
existing figures after invalid inputs or export failures, figure cleanup, and the approved
image formulas on hand-computable arrays. Image-consumer tests use the actual
shared adapter on tiny in-memory images, checking native EXIF dimensions, explicit
failures, color exclusions, nullable classes, source identity, image closure,
gallery consistency, and retention of a report after figure failure. They do not
download the dataset.
Actual image-record validation is performed separately by the complete training
scan and selected-record numerical checks. Synthetic tests do not establish
semantic image/caption accuracy or suitability of every possible source mode.
