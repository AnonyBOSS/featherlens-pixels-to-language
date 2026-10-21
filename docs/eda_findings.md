# EDA findings

## Stage 2: complete official training analysis

Analysis date: October 21, 2026. This section supersedes the preliminary prefix
for training-wide class/caption conclusions. Reproduction and measurement
definitions are in [Stage 2 instructions](eda_stage2.md). The historical Stage 1
observations below remain explicitly limited to their original 200-row subset.

Source: `Multimodal-Fatima/CUB_train` at revision
`bd3988995e6db193427255abdebd522afe24ce76`, official `train` split. The iterator was
exhausted and yielded **5,994 records**, matching the documented reference. The
report is `outputs/eda/reports/train-full.json` (generated and Git-ignored).
No official test records or source image pixels were analyzed. Shared loading,
caption parsing, source text, and split definitions remain unchanged.

### Class distribution and metadata coverage

**Question:** How balanced is the complete official training split?

**Method:** Count image metadata records by the ordered source class mapping.
Keep class counts independent of the number of captions. Compare observed rows
with the reference only after exhausting the source iterator.

**Measured finding:** All 200 species are represented. There are 194 classes
with 30 images and six with 29. Minimum/maximum counts are 29/30, the mean is
29.97, the median is 30, and the maximum/minimum imbalance ratio is approximately
1.0345. The six 29-image classes are Common Raven, Nelson Sharp tailed Sparrow,
Bank Swallow, Artic Tern, Red cockaded Woodpecker, and House Wren, using source
names verbatim. All records have a source ID and filename; all 5,994 canonical
row-based image IDs are unique.

**Interpretation:** Class-count imbalance in this raw training source is small.
The seven-species concentration observed in Stage 1 was a selection effect of
the source prefix, not evidence that the remaining species were absent.

**Recommendation:** Preserve class coverage when creating a validation split.
These counts alone do not justify aggressive class rebalancing. Recompute class
counts after the lead engineer's quality exclusions and derived split manifest
before discussing sampling changes.

**Limitations:** Canonical IDs identify source rows; they do not verify image
content uniqueness or filename uniqueness. Full corruption, duplicate-image,
leakage, and cleaning audits remain separate. Counts do not measure class
difficulty or image quality.

**Figure:** `outputs/eda/figures/training_class_distribution.png`.

### Caption coverage and information content

**Question:** Does every training image have ten usable descriptions?

**Method:** Count parsed caption lines and nonempty cleaned strings separately
from images. Examine saved shortest-caption examples without changing source text.

**Measured finding:** All 5,994 images have ten nonempty caption lines, yielding
59,940 nonempty captions. No parsed blank lines, zero-word nonempty captions, or
raw-line whitespace anomalies were observed under the implemented definitions.
However, a shortest example is the single word `this` (source row 1171, caption
index 7). Exactly three captions have fewer than five analysis words: lengths
one, three, and four, with one occurrence each.

**Interpretation:** Structural completeness does not imply semantic usefulness
or ten distinct descriptions. A standalone demonstrative does not describe a
bird; other short captions may still be informative.

**Recommendation:** Separate structural validity from caption informativeness.
Flag specific sparse examples for review rather than applying an automatic
minimum-length exclusion rule. A weak caption should not automatically remove
its image from the classification population.

**Limitations:** This is not an exhaustive semantic or spelling audit. Nonempty
strings can still contain incorrect, generic, or inconsistent descriptions.

### Caption length and vocabulary

**Question:** What are the training-wide length and vocabulary distributions?

**Method:** Use the unchanged analysis tokenizer, with stopwords retained and
no spelling correction. Save frequency tables separately from plotted figures.
Summarize nonempty captions using the documented linear percentile convention.

**Measured finding:** Word lengths have mean 15.2504, median 14, p95 24, p99 32,
and range 1-65. Character lengths have mean 77.4612, median 70, p95 129, p99 168,
and range 4-332. There are 914,107 word tokens and 4,701 distinct word types;
2,308 types occur once. Those singleton types represent about 49.10% of the
vocabulary types but only 0.2525% of all tokens.

**Interpretation:** The distribution has a right tail, and the complete-source
upper-tail values differ from the preliminary prefix (p95 24 rather than 25,
maximum 65 rather than 55). Many rare types account for little total token mass.
They may mix specialized terminology with annotation noise; rarity is not a
reliable error label.

**Recommendation:** Preserve raw text and investigate rare forms in context.
Retain these distributions for later model-team discussion, while recomputing
any fitted vocabulary or sequence-length configuration on the eventual derived
training partition using the eventual model tokenizer. Do not choose a hard
truncation limit from these analysis-word percentiles alone.

**Limitations:** Vocabulary size depends on tokenization and corpus size.
Analysis words are not model subword tokens. Caption observations from one
image are related; these descriptive percentiles are not confidence intervals.
No image-caption correctness claim follows from frequent color/anatomy words.

**Figures:** `training_caption_lengths.png` and
`training_caption_vocabulary.png` in `outputs/eda/figures/`.

### Exact caption repetition

**Question:** How often do the ten descriptions repeat within an image, and how
often do descriptions recur across different training images?

**Method:** Compare case-sensitive whitespace-cleaned texts, excluding blanks.
Count distinct repeated texts separately from extra occurrences and distinct
images. Preserve bounded source-linked examples for inspection.

**Measured finding:** There are 50,477 distinct clean texts among 59,940 caption
occurrences, leaving 9,463 occurrences beyond one per exact text (15.7875% of all
caption occurrences). Within-image repetition affects 583 images (9.7264%) and
588 image/text groups, with 639 extra occurrences involving 153 distinct texts.
Across-image repetition involves 1,534 distinct texts and 8,824 additional
distinct-image appearances. The components reconcile: `639 + 8,824 = 9,463`.
The most widely shared saved example appears 601 times across 501 images.

**Interpretation:** Ten nonempty caption slots do not always provide ten distinct
descriptions. Generic descriptions can apply to many birds; exact text repetition
alone does not demonstrate duplicate images, split leakage, or invalid captions.

**Recommendation:** Share source-linked repetition examples for cleaning-policy
discussion. Keep raw source lines and their indices unchanged. Any deduplication
policy should distinguish within-image repetition from generic descriptions
shared across images and should document its impact on caption counts.

**Limitations:** Matching preserves case and punctuation and does not detect
paraphrases. Within/across repeated-text scopes overlap. The global extra share
is a redundancy definition, not a percentage of incorrect captions.

**Figure:** `outputs/eda/figures/training_caption_repetition.png`.

### Image measurement status and remaining work

RGB, brightness, contrast, saturation, and resolution definitions were approved.
Pure NumPy measurement functions and synthetic formula tests are implemented.
They accept native-resolution RGB arrays and do not decode images or apply
color-mode, transparency, orientation, or validity policies. Their numerical
tests do not constitute an actual source-image audit.

Actual pixel distributions and image-derived preprocessing recommendations
remain pending agreement on the existing pipeline's image-access and decoding
status view. No source-wide image quality, resizing, normalization, or color-jitter
conclusions are claimed from the class/caption analysis. The official test split
remains reserved; future fitted preprocessing must use the derived training
partition after validation splitting.

### Validation

The initial expanded synthetic suite passed **101 tests**. After PR-preparation
safeguards, the complete suite passed **142 tests**, including existing source
contract tests, training-only command/builder guards, complete-scan safeguards,
report compatibility and count reconciliation, preservation of existing figures
after invalid inputs or export failures, plot counts, and pure image formulas.
Actual class/caption validation
used all 5,994 training records, with 59,940 nonempty captions. Four figure groups
were generated from the saved report as PNG and PDF; plotting does not reopen
the source. The source decoder, actual pixel measurements, and pipeline cleaning
policies are not validated by these synthetic image-formula tests.

During PR preparation, both saved reports were preserved byte-for-byte against
a verified backup. The full-training figure pairs were regenerated from the
saved schema `1.1` report, with unchanged PNG pixels and measured conclusions.
The historical prefix report uses schema `1.0`; it remains preserved as evidence
but is rejected before figure publication because it lacks required plotting
distributions. Regenerate it with the current builder before plotting. These
checks reused saved evidence; they did not repeat the dataset scan.

## Stage 1: preliminary training-prefix exploration

Analysis date: October 21, 2026. These are measured observations from **the first
200 official training records**, not conclusions about the complete CUB dataset.

- Source: `Multimodal-Fatima/CUB_train`.
- Revision: `bd3988995e6db193427255abdebd522afe24ce76`.
- Source rows: 0 through 199, selected in source order without shuffling.
- Report: `outputs/eda/reports/train-prefix-200.json` (generated, Git-ignored).
- Methods and runnable instructions: [Stage 1 analysis](eda_stage1.md).
- Contract: [shared source interface](data_contract.md).

The generated JSON records source provenance, runtime versions, selection limits,
statistical definitions, and bounded examples with image IDs and caption indices.
Source captions and images were not modified. No test records or image pixels
were analyzed. No cleaning, normalization, augmentation, vocabulary configuration,
or sequence-length threshold was selected.

### 1. Population and class coverage

**Question:** What population did we actually examine, and is it representative?

**Method:** Count one image metadata record per source row, preserving the
ordered class mapping. Count unique canonical image IDs separately from rows.

**Measured finding:** There are 200 records with 200 unique canonical IDs and
seven observed species. Class IDs 0 through 5 have 30 selected records each;
class ID 6 has 20. These are the three albatross species, Groove billed Ani,
Crested Auklet, Least Auklet, and Parakeet Auklet in the source mapping. No selected
record lacks `source_id` or `file_name`.

**Interpretation:** This prefix is concentrated in early class IDs. Its coverage
does not support conclusions about the balance or absence of species in the
full source. The full count of the seventh selected class remains unverified.

**Recommendation:** Use a complete metadata scan for class distributions and
coverage, or a deliberately sampled training subset for preliminary comparisons.
Keep canonical image IDs as the join key and preserve class-map order.

**Limitations:** Uniqueness here concerns source-row identity, not image content.
Corruption, duplicate images, leakage, and full metadata validity remain outside
this analysis. Unselected classes and the test split have not been audited.

### 2. Caption coverage

**Question:** Does the documented ten-caption structure hold in these records?

**Method:** Consume the existing caption parser; count parsed lines, nonempty
cleaned captions, and images separately. Compare counts with the documented
expectation without using a mismatch as an automatic exclusion rule.

**Measured finding:** The 200 images yield 2,000 caption lines and 2,000 nonempty
captions. Every selected image has ten nonempty captions. There are no empty
caption lines, zero-word captions, or raw-line whitespace anomalies in this
prefix.

**Interpretation:** The shared parser and statistical interfaces work together
on real source data, and this subset matches the documented caption count.

**Recommendation:** Retain separate image, line, and nonempty-caption
denominators in the full audit. Preserve images when caption data are absent.

**Limitations:** This does not prove that all images have ten valid descriptions.
Nonempty captions are not necessarily accurate or informative. Zero parsed lines
would not distinguish missing descriptions from empty descriptions in this
interface.

### 3. Caption length

**Question:** How much does caption length vary in this subset?

**Method:** Count derived alphabetic word tokens in each nonempty cleaned
caption, retaining internal apostrophes and hyphens. Character counts use Unicode
code points. Percentiles use linear interpolation at `(n - 1) * p`.

**Measured finding:** Among 2,000 captions, word lengths have a mean of 15.0715,
median of 14, minimum of 5, p95 of 25, p99 of 34, and maximum of 55. Cleaned
character lengths have a mean of 75.411, median of 67, and range of 23 to 284.

**Interpretation:** A typical caption is much shorter than the longest examples;
the upper tail deserves inspection before later captioning configuration.

**Recommendation:** Recompute training-wide distributions and inspect upper-tail
examples before proposing a sequence-length policy. These analysis word tokens
are not the eventual model tokenizer.

**Limitations:** The prefix covers seven species only. Captions from one image
are related observations; these percentiles are descriptive, not confidence
intervals. They do not justify a truncation threshold on their own.

### 4. Vocabulary and suspected text noise

**Question:** What vocabulary is observed, and do rare words merit inspection?

**Method:** Use derived NFC-normalized, casefolded word tokens, retaining
stopwords. Inspect the saved longest-caption examples manually. Apply no
spelling correction or linguistic rewriting to source captions.

**Measured finding:** The prefix contains 30,143 word tokens and 890 distinct
token types. Of these types, 426 occur once, contributing about 1.4133% of all
tokens. Frequent content words include `bird`, `black`, `white`, `beak`, and
`bill`. The longest-caption examples include strings such as `hthe`, `cheeck`,
and `whte`, which appear to be spelling errors.

**Interpretation:** Rare types mix potentially legitimate specialized vocabulary
with possible annotation noise. A low-frequency token is not automatically an
error. Anatomy and color terms are visibly relevant in these examples, but this
is not a quantified semantic-quality assessment.

**Recommendation:** Keep raw descriptions and contextual examples for review.
Separate clearly documented analysis normalization from any proposed shared
cleaning policy. Do not autocorrect anatomy or species terminology. Measure
training-wide vocabulary before discussing coverage or rare-token policies.

**Limitations:** The vocabulary is specific to this sample and tokenization
policy. Apparent errors were identified in a few saved examples, not by an
exhaustive spelling audit. Numerals are not word tokens under this policy.

### 5. Repeated captions

**Question:** Is exact caption repetition present within and across images?

**Method:** Group case-sensitive `clean_caption` text, excluding blanks.
Within-image extras count additional occurrences in each image/text group.
Across-image extras count additional distinct image appearances for each text.
These scopes partition repeated occurrences without double-counting them.

**Measured finding:** Within-image repetition affects 20 image/text groups,
involving 11 distinct texts and 24 extra occurrences. Across-image repetition
involves 61 distinct texts and 218 additional image appearances. One generic
description appears 35 times across 32 images.

**Interpretation:** Ten nonempty caption lines do not necessarily mean ten
distinct descriptions. Generic repeated language may describe multiple images
appropriately; repetition alone does not establish image duplication or leakage.

**Recommendation:** Share identified text/image references with the lead
engineer for policy discussion. Preserve repetitions in the source while
reporting their scope. Any future deduplication policy must be explicit and
separate from image-content duplicate detection.

**Limitations:** Matching is exact after the shared whitespace normalization,
with case and punctuation preserved. It does not detect paraphrases or semantic
similarity, and these rates cannot be generalized to unexamined species.

## Validation and next discussion

Synthetic tests cover known class/split counts, absent classes, missing optional
metadata, conflicting identities, caption denominators, Unicode tokenization,
repetition scopes, empty inputs, quantiles, strict JSON serialization, and the
bounded report adapter. The existing source/schema/parser tests remain included.

Validation on Windows Python 3.12.0: `python -m pytest -q` passed **51 tests**
(11 existing data-layer tests and 40 new analysis cases). `python -m pip check`
reported no broken requirements. A third-party ResourceTracker shutdown error
with `multiprocess` 0.70.19 was resolved by the compatibility pin in
`requirements-eda.txt`; the tests and real exploration then completed without
that traceback. The public data request retried once after a server disconnect
and succeeded. The report records the final package versions.

Before the next substantial stage, discuss training-wide metadata access and
the image-decoding/measurement interface with the lead engineer. Priorities are
complete descriptive coverage, agreed image-statistic definitions, and useful
Matplotlib figures. Pixel quality, cleaning, duplicate detection, split creation,
and model work are not implemented by Stage 1.
