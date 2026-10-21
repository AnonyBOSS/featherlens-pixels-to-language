# Statistical EDA: full training analysis and figures

This stage measures the complete official training source through the existing
`load_cub_split("train", streaming=True)` interface. The source contract, loader,
caption parser, shared README, official splits, and cleaning policies are unchanged.
It also provides pure image-statistics functions using the approved definitions;
source-image integration remains pending the decoding and failure-handling policy.

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
Their dimensions describe the supplied array; an adapter must preserve native
geometry and separately record original versus oriented dimensions. No source
image measurements or image-derived preprocessing recommendations are claimed
in this stage's findings.

Future dataset summaries should weight each eligible image equally. That is
different from pooling every pixel to estimate normalization parameters. Any
normalization, vocabulary, threshold, or augmentation policy must be fitted or
selected using the eventual derived training partition after validation splitting.

## Remaining data-interface agreement

Before running pixel analysis, agree on an image-access view providing:

- Canonical `image_id` and existing class/source provenance.
- Original dimensions, original mode, orientation handling, and any RGB conversion.
- Explicit decoding status and reason; failed images remain in attempted counts.
- Transparency/color-mode eligibility policy and available shared quality flags.

This is a consumer requirement for the existing contract, not a second source
loader or an implemented corruption/cleaning pipeline. No shared interface has
been changed. Image integration, quality flags, and later derived split manifests
remain coordination points with the data pipeline.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Synthetic tests check full iterator exhaustion, reference-count mismatches,
interrupted scans, training-only command and builder guards, saved-frequency
reconciliation, plotted counts and labels, report compatibility, preservation of
existing figures after invalid inputs or export failures, figure cleanup, and the approved
image formulas on hand-computable arrays. They do not download the dataset.
Actual training-record validation is performed separately by the complete scan.
Image-formula tests establish numerical behavior, not compatibility with every
source image mode or decoding policy.
