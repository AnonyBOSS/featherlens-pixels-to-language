# Caption language and prepared-training EDA

For final reproduction, delivery files and deferred work, see the
[EDA handoff](eda_handoff.md).

This stage adds training-caption term mentions, within-image lexical diversity,
and class-level caption-length comparisons. It also joins saved image/caption
measurements to the existing classification manifest. Statistical functions,
source orchestration, and plotting remain separate. No cleaning, splitting,
transforms, model vocabulary, or fitted normalization is implemented here.

## Required shared interfaces

Use the existing EDA environment and the shared APIs documented in
[the source contract](data_contract.md), [integrity audit](integrity_audit.md),
and [classification preparation](classification_data.md). The prepared-data
command requires the merged `src.data.preparation` module as well as the Stage 2
image EDA modules; it imports no PyTorch or torchvision functionality.

The October 22 integrated verification used main commit
`69cc3660ced392e248d8f353ae8cebe665b246b5` and the Stage 3 working sources in
the same checkout. The shared pipeline is available directly; no copied runtime,
alternate loader, or pending shared PR is required. An earlier isolated check
used shared commit `ada1404e34dabaf8fac567e91238d2d5062f5ce5` with the Stage 2
EDA base `432a83608bf753fc6c89f721c6437bce56721913`.

The EDA commands use `requirements-eda.txt`. Running the complete integrated
repository suite also requires `requirements-classification.txt` for the shared
PyTorch tests; those dependencies do not become imports of the statistical code.

## Run the caption analysis

From the repository root:

```powershell
$env:HF_HOME=Join-Path (Get-Location).Path '.venv\hf-cache'
$env:HF_DATASETS_CACHE=Join-Path $env:HF_HOME 'datasets'
$env:HF_HUB_CACHE=Join-Path $env:HF_HOME 'hub'
$env:MPLBACKEND='Agg'
$env:MPLCONFIGDIR=Join-Path (Get-Location).Path '.venv\mpl-cache'
.\.venv\Scripts\python.exe -m src.analysis.explore_caption_details --split train --figures-dir outputs/eda/figures
```

The runner opens `load_cub_split("train", streaming=False)` and consumes its
metadata/caption iterator. It never decodes image pixels or opens the official
test source. Default execution accounts for every indexed training row and
compares the measured population with the 5,994-row reference. `--limit N`
produces an explicitly nonrandom prefix, even if it reaches the source end.
Interrupted or prematurely terminated iterators cannot certify complete coverage.

The report is `outputs/eda/reports/train-caption-details-full.json`, with schema
`caption-details-1.0`. Existing reports and their schemas are preserved. The new
report retains canonical provenance, lexicon definitions, population scope,
per-image measurements, aggregate counts, per-class summaries, and bounded
source-caption examples. It does not save a second full caption corpus.

### Term mentions

`src.analysis.caption_details.TERM_LEXICON` declares the small
`caption-terms-v1` color, anatomy, and direction lexicon. Variants are explicit:
for example gray/grey, beak/bill, and breast/chest. Existing analysis tokenization
is reused unchanged. Matching also considers components separated by ASCII,
U+2010, or U+2011 hyphens, so `blue-green` can mention two color terms while
remaining one token for the existing vocabulary/length analysis.

Each term counts at most once per caption and once per image. Category counts
use the union of matching terms, rather than their sum. Captions and categories
can overlap. Saved prevalence denominators distinguish all nonempty captions,
images with nonempty captions, and all images. Zero denominators yield null.
Punctuation-only nonempty captions remain in the nonempty-caption denominator.

These are lexical mentions, not verified bird attributes. Negation, polysemy,
species names, and unlisted variants limit interpretation. No stemming,
autocorrection, source rewriting, or stopword removal is applied. Saved reports
must match both the version and exact declared lexicon before plotting.

### Within-image lexical diversity and overlap

For each image, pooled type/token ratio is the number of distinct analysis
tokens divided by all analysis tokens across its captions. It is null when
there are no tokens. TTR depends on text length and is not a quality score.

For each unordered pair of token-bearing captions, Jaccard overlap is the
intersection size divided by the union size of their token sets. Repeated
captions and stopwords remain. Captions without words are excluded only from
this pair calculation; fewer than two token-bearing captions yield no pairs and
a null mean. Word order and semantic agreement are not measured. Ten
token-bearing captions yield 45 pairs.

Overall and class summaries weight each available image equally. Caption-length
class comparisons summarize per-image mean word counts, including zero-word
nonempty captions. All source classes remain in the saved mapping, including
absent classes. No significance tests treat an image's captions as independent
images, and no class difficulty or annotation-quality ranking is inferred.

## Reuse or regenerate classification preparation

The EDA consumer needs the existing `classification_manifest.jsonl` and
`preparation.json` together in a prepared directory. Flags and exact-group
memberships are already carried by that manifest. For independent audit
provenance verification or regeneration, retain the four audit files as well:
`summary.json`, `images.jsonl`, `captions.jsonl`, and `exact_groups.jsonl`.

The verified local directories are:

- `outputs/eda-stage3/integrity-audit/`
- `outputs/eda-stage3/classification-v1/`

Use a new, empty directory for regeneration; preserve existing completed
artifacts. The shared audit reads both pinned official sources solely for
integrity coverage and preparation. It does not fit model parameters. EDA
statistics and preprocessing recommendations use training rows.

```python
from pathlib import Path
from src.data.integrity_audit import audit_pinned_sources
from src.data.preparation import prepare_classification, load_classification_manifest

def main():
    audit_dir = Path("outputs/eda-stage3/integrity-audit")
    prepared_dir = Path("outputs/eda-stage3/classification-v1")
    if audit_dir.exists() or prepared_dir.exists():
        raise FileExistsError("Use new directories or validate and reuse completed artifacts")
    audit = audit_pinned_sources(audit_dir)
    if not audit["coverage_complete"]:
        raise RuntimeError("Inspect incomplete audit coverage before continuing")
    preparation = prepare_classification(audit_dir, prepared_dir, seed=42, val_fraction=0.20)
    rows = load_classification_manifest(prepared_dir)
    assert preparation["partition_counts"] == {"train": 4794, "val": 1200, "test": 5794, "excluded": 0}
    assert len(rows) == 11788

if __name__ == "__main__":
    main()
```

The source revisions, audit completion/counts, preparation versions, seed,
validation fraction, fingerprints, class coverage, and group constraints must
be verified before relying on these expected counts. The actual run verified
all of them; expected values were not substituted for measurements. The shared
loader validates the manifest and split recipe. EDA never reimplements them.

## Join the prepared membership to saved training measurements

```powershell
.\.venv\Scripts\python.exe -m src.analysis.explore_prepared --prepared-dir outputs/eda-stage3/classification-v1 --figures-dir outputs/eda/figures
```

This reads the complete Stage 2 image report and the new complete caption-detail
report, calls `load_classification_manifest`, and writes
`outputs/eda/reports/prepared-training-eda.json` (`prepared-eda-1.0`). Input
report SHA-256 values, preparation SHA-256, manifest SHA-256, and audit
fingerprints preserve the measured snapshot relationship.

Canonical IDs, training provenance, source metadata, class mapping, full source
coverage, and partition counts must reconcile across inputs. Prefix reports
cannot substitute for full training measurements. Preparation metadata is checked
before and after loading/analysis. Missing or inconsistent joins fail explicitly.
Source IDs and filenames are join metadata; they are not content-identity proof.

Counts distinguish raw training, derived training, validation, exclusions, and
unknown-class exclusions. Quality counts are distinct affected training images
per supplied error/warning code, overall, by class, and by partition. Overlapping
flags are not added into an invented total of invalid images. A paired-null class
is retained only for an excluded row, without imputation from the EDA mapping.

Train/validation checks inspect the supplied canonical IDs, encoded/pixel
hashes, and exact-group memberships. They do not compute new fingerprints or
near-duplicate candidates. Official test records contribute identity and
assignment-preservation counts only; test appearance, captions, flags, and
per-class statistics do not enter this EDA report.

Partition image metrics use the recorded native-pixel representation and explicit
RGB-analysis eligibility. Caption metrics use available per-image measurements.
These image-weighted summaries are descriptive, not fitted normalization or
post-transform tensor statistics. Future fitted preprocessing must use the
derived training partition only. Validation summaries describe the supplied split.

## Figures and export safety

Each group is saved as PNG and PDF:

| Figure stem | Purpose |
| --- | --- |
| `training_caption_language` | Top declared term mentions by category, with caption denominators. The rare-direction panel uses a clearly labeled finer percentage scale and precise nonzero counts. |
| `training_caption_diversity` | Per-image TTR, mean pairwise Jaccard, and TTR versus amount of text; no quality thresholds. |
| `training_class_caption_variation` | Ranked class mean caption length and readable selected species; image weighting is explicit. |
| `training_prepared_class_coverage` | Derived train/validation counts in class order and saved distributions of per-class counts. |

Plotting consumes saved results and validates schema, provenance, selection scope,
class/count reconciliation, lexicon, and measurement ranges before rendering.
The existing export helper stages all figure pairs in a bundle before publishing
and restores prior files on caught publication errors. Process termination and
concurrent writers are outside that rollback guarantee. Unrelated figures survive.
Reports use the existing strict, atomic JSON writer and remain saved if optional
figure rendering fails. Quality-by-class counts are reported as data and text;
the two duplicate warnings do not justify an otherwise empty quality chart.

Figures can be regenerated by loading the saved JSON and calling
`generate_caption_detail_figures` or `generate_prepared_eda_figures`. Neither
plotting module reads source records or manifests.

## Validation

The complete integrated repository suite passed **533 tests**, with no failures
or skips, using Windows Python 3.12.0, torch 2.14.1+cpu and torchvision 0.29.1+cpu.
`python -m pip check` found no broken requirements. With both pinned sources
already cached and the verified preparation available, reproduce this check after
setting the cache and Matplotlib variables above:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:HF_DATASETS_OFFLINE='1'
$env:FEATHERLENS_PREPARED_DIR=(Resolve-Path 'outputs/eda-stage3/classification-v1').Path
.\.venv\Scripts\python.exe -m pytest -q tests
```

This includes the shared preparation, transforms, lazy datasets, Windows worker
checks and opt-in cached train/validation batch/loss/backward check. It does not
optimize a model or open the official test source for statistical EDA. Without
`FEATHERLENS_PREPARED_DIR`, the real-data test is skipped; synthetic tests need no
dataset download.

Stage 3 tests cover hand-counted mention unions, declared variants, hyphens,
empty/zero-word captions, Jaccard, equal-image class weighting, source/caption
identity, honest scope, strict saved-report validation, preservation of old
outputs, authoritative manifest-loader use, prepared joins, group isolation,
unknown-class exclusions, and figure counts/labels.

Actual caption validation covered all 5,994 training images and 59,940 captions,
with independent reductions on 12 selected records. Fourteen train/validation
numeric summaries were checked independently, including all recorded percentiles,
against NumPy reductions within absolute tolerance `1e-12`. The authorized shared
audit/preparation accounted for all 11,788 official rows with the verified source
versions and seed. All four new PNG figure groups were visually inspected.

After updating to main, all four complete-training reports were regenerated in
`outputs/eda-stage3/integrated-69cc366/`, using the native shared modules. Every
measured result matches the earlier reports. All 13 PNG figure groups are
byte-identical; the 13 PDFs differ only in export timestamps. Independent checks
reduced all 14 train/validation numeric summaries within `1e-12` and checked RGB,
brightness, contrast, saturation and resolution on 16 actual records, including
extremes and every grayscale source image, within `1e-10`.

The fresh image report includes main's corrected population wording: raw training
contains the later derived-validation members. Its measurements are unchanged;
the regenerated prepared report records the new input-report hash. Reports in
that verification directory form a consistent input/output set. All 37 earlier
EDA, audit and preparation artifacts were preserved byte-for-byte, including the
historical image report. The earlier isolated run had preserved 28 EDA outputs
while adding the prepared report and coverage figure.

Stage 3 changes do not modify the shared pipeline or README. Generated reports,
figures, manifests, test records, caches, backups and runtime snapshots are excluded
from Git. Findings and limitations are in [the EDA findings](eda_findings.md).
