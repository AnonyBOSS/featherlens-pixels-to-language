# Statistical EDA: Stage 1

Stage 1 provides independent image-metadata and caption statistics. It consumes
the existing [source contract](data_contract.md) and does not change source
loading, caption parsing, cleaning policies, official splits, or image assets.
Pixel statistics and EDA figure generation are a later stage.

## Setup and tests

From the repository root, using Python 3.12 on Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-eda.txt
.\.venv\Scripts\python.exe -m pytest -q tests/analysis
```

On another platform, use the corresponding virtual-environment Python path
(`.venv/bin/python` on Linux/macOS). The EDA dependency file pins direct packages;
the generated report also records the actual runtime package versions. These
pins do not lock every transitive dependency or replace the full project list.
`multiprocess` is pinned to 0.70.17 because 0.70.19 produced a ResourceTracker
shutdown traceback on the installed Windows Python 3.12.0 interpreter. Its
dependency constraints remain compatible with the pinned `datasets` version.
These EDA tests use synthetic records and do not download CUB. For the complete
repository suite, install the classification dependencies and follow the
[handoff instructions](eda_handoff.md#environment-and-checks).

## Bounded source exploration

```powershell
$env:HF_HOME=Join-Path (Get-Location).Path '.venv\hf-cache'
.\.venv\Scripts\python.exe -m src.analysis.explore --split train --limit 200
```

The command uses `load_cub_split(..., streaming=True)` from the shared loader.
It saves a JSON report to `outputs/eda/reports/train-prefix-200.json` by default.
Use `--output path/to/report.json` to select a different path. Generated outputs
are ignored by Git. Tracked interpretations live in [eda_findings.md](eda_findings.md).

This is a **nonrandom source-order prefix**, not a representative sample or a
full audit. No image pixels are decoded, but streaming can transfer embedded
image bytes in Parquet blocks. The record limit is not a network byte limit.
Do not infer full class imbalance, unseen-species absence, or full vocabulary
coverage from this report. Other official splits showing zero selected rows
have not been audited. Both report builders and the command accept only the
official training split; held-out sources are rejected before record iteration.
The pure statistical functions below remain usable with synthetic records.

## Public statistical functions

- `src.analysis.dataset_stats.summarize_dataset(images, class_names=...)` counts
  image records, unique IDs, source splits, class coverage, missing optional
  metadata, and revision provenance. It preserves the source class order and
  includes zero-count classes. Repeated identical metadata rows are counted and
  flagged; conflicting metadata for one identity raises an error. This is not
  duplicate-image detection. The imbalance ratio is null if any expected class
  has zero observations, rather than silently excluding that class.
- `src.analysis.caption_stats.summarize_captions(records)` accepts the existing
  `(ImageRecord, tuple[CaptionRecord, ...])` pairs. It reports image, caption-line,
  nonempty-caption, and word-token denominators separately. Missing caption
  groups retain their image observation. Duplicate image inputs and captions
  linked to the wrong image raise errors rather than inflating summaries.
- `src.analysis.caption_stats.tokenize_caption(text)` creates analysis tokens
  on a derived copy. It applies Unicode NFC normalization and casefolding,
  normalizes curly apostrophes, and counts alphabetic Unicode words. Internal
  apostrophes and hyphens (ASCII, U+2010, U+2011) stay within words; digits and
  other punctuation separate words. Hyphen variants remain distinct tokens.
  Stopwords remain; there is no stemming, lemmatization, or spelling correction.
- `src.analysis.reporting.summarize_numbers(values)` returns count, minimum,
  maximum, mean, median, and p05/p25/p75/p95/p99. Percentiles use linear
  interpolation at index `(n - 1) * p`. Empty numerical samples have null
  measurements; JSON never contains NaN or infinity.

## Interpretation conventions

Caption lengths describe **nonempty `clean_caption` strings**. Character length
counts Unicode code points. A punctuation-only nonempty caption has zero words
and remains in the length denominator. Blank caption records are counted
separately, not silently removed from quality coverage. Zero parsed lines cannot
distinguish an absent description from an empty description; that distinction
requires the original source field and is not inferred here.

The documented expectation of ten captions is a reference for mismatch counts,
not a rule that automatically excludes images. Whitespace anomalies count raw
lines that differ from their trimmed, whitespace-collapsed form. Original
strings remain untouched.

Repeated captions use case-sensitive `clean_caption` text, excluding blanks:

- Within-image extra occurrences are `count - 1` in each image/text group.
- Across-image extra occurrences are `distinct_image_count - 1` for each text.
  These count additional images, excluding repetitions already counted within
  an image. The report also records repeated distinct texts and bounded examples.

Repeated generic language alone does not establish data leakage. Hapax word
types occur once in the analyzed corpus; `hapax_token_share` divides that number
by all word tokens, not by vocabulary size. Corpus size affects both vocabulary
and rare-word counts, so do not compare differently sized corpora as if they had
equal coverage. No training vocabulary or preprocessing threshold is fitted.

## Integration boundaries

Use `image_id` for joins and retain `source_split` as official train/test identity.
The shared classification manifest retains derived partition identity separately;
[Stage 3](eda_stage3.md) consumes it without changing official split identity. Image
access, decoding failures, quality flags, cleaning, duplicate algorithms,
augmentation, and the shared CLI remain with the data pipeline. Extend the
approved contract before adding dependent image measurements. Shared README
updates should be integrated alongside these verified commands.
