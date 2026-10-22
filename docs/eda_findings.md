# EDA findings

## Stage 3: caption language and prepared-training integration

Analysis date: October 22, 2026. The caption population is the complete official
training source at revision `bd3988995e6db193427255abdebd522afe24ce76`: 5,994 images
and 59,940 nonempty, token-bearing captions. Methods and commands are in
[Stage 3 instructions](eda_stage3.md). Earlier Stage 1/2 measurements below remain
preserved; derived partitions are new views of the same raw training rows.

### Color, anatomy, and directional language

**Question:** What vocabulary categories do training captions actually mention?

**Method:** Use the declared `caption-terms-v1` lexicon on the existing analysis
tokens plus explicit hyphen components. Count each term/category once per caption
and once per image. Category totals use unions. Source descriptions stay unchanged.

**Measured finding:**

| Category | Captions with a mention | Share of 59,940 captions | Images with a mention |
| --- | ---: | ---: | ---: |
| Color | 58,438 | 97.494% | 5,994 |
| Anatomy | 58,278 | 97.227% | 5,994 |
| Direction/position | 520 | 0.868% | 482 |

The declared directional category appears for 8.041% of images. It includes terms
such as `above`, `upper`, and `lower`; it is not a count of horizontal orientation.
Separately, `left` occurs in three captions across three images, and `right` in
20 captions across 19 images. These term counts are not added into a claimed
count of distinct left/right captions. The saved lexicon groups gray/grey,
beak/bill, breast/chest, and other explicitly listed variants.

**Interpretation:** Color and anatomy descriptions are common under this matching
rule, with at least one of each in every image's caption set. This establishes
lexical coverage, not discriminative color information or visual correctness.
Directional mentions are less frequent and mix body-part position with orientation.

**Recommendation:** Preserve these source-linked examples for later captioning
review. Inspect left/right context before choosing future text-aware geometric
augmentation. This analysis does not change the classification flip policy.
Discuss strong color transformations before captioning experiments, without
assuming that a whole-image RGB mean verifies a mentioned bird color.

**Limitations:** The hand-declared lexicon is incomplete. Negation, context,
polysemy, species names, and unlisted compounds can affect matches. No semantic
image-caption validation, automatic caption correction, or augmentation decision
was performed.

### Within-image diversity, overlap, and class variation

**Question:** How much lexical variety do an image's captions provide, and does
caption length vary across species?

**Method:** Pool analysis tokens within each image for type/token ratio. Average
unordered token-set Jaccard over its token-bearing caption pairs, retaining
stopwords and repeated captions. Summarize per-image means by class with equal
image weighting; no independence assumption is made for an image's ten captions.

**Measured finding:** All 5,994 images provide 45 caption pairs each, for
269,730 pairs. Median per-image TTR is 0.3351 (p05 0.2739, p95 0.4000). Median
per-image mean Jaccard is 0.3047 (p05 0.2383, p95 0.3850). The medians are
151 analysis tokens and 51 distinct token types per image; these are separate
marginal summaries.
Per-image mean caption length averages 15.2504 words.

Across the 200 class means, Fish Crow is lowest at 12.9133 words and European
Goldfinch highest at 17.3433 words; both have 30 source images. Selected examples
include exact repeated captions with Jaccard 1 and differently worded captions
of the same image with Jaccard 0. The report retains bounded examples with original
caption indices, including repeated descriptions at source row 20.

**Interpretation:** Captions contain both shared vocabulary and differing lexical
details. Jaccard 1 does not prove identical meaning, and Jaccard 0 does not prove
contradictory descriptions. Class-length differences may reflect descriptions,
annotator style, or visual content; this study does not identify their cause.

**Recommendation:** Use the distributions and examples for reporting and review.
Keep TTR alongside token counts, and keep repeated captions visible. Do not use
these summaries to automatically remove captions or rank species by difficulty.

**Limitations:** TTR depends on text length. Jaccard ignores word order and meaning,
and common words inflate overlap. Class comparisons are descriptive; no hypothesis
tests, semantic model, annotation-quality labels, or fitted thresholds are included.

### Prepared split coverage and quality-flag integration

**Question:** Do the verified classification partitions preserve source coverage,
class representation, and exact-group isolation, and how do saved training
measurements distribute across them?

**Method:** Regenerate the shared audit and preparation with pinned source
revisions, seed 42, validation fraction 0.20, policy `classification-v1`, and
split version `stratified-exact-group-v1`. Validate completeness and fingerprints
through the existing shared APIs. Join canonical training IDs to saved image and
caption measurements; do not implement another cleaner or split algorithm.

**Measured finding:** The audit accounts for all 11,788 official source rows.
Preparation yields 4,794 training, 1,200 validation, 5,794 official test, and zero
excluded rows. All 200 classes remain in training and validation. There are 194
training classes with 24 images and six with 23; every validation class has six.
All 5,994 training-source rows join to both EDA reports, without unknown classes
or missing numeric observations in this run.

No training-source error flags are present. The two `duplicate.exact` warning
images are Pigeon Guillemot source rows 1712 and 1731, both in validation and in
one recorded exact group. There is no shared canonical ID, encoded-byte hash,
normalized-pixel hash, or exact group between derived training and validation.
These are checks of supplied manifest evidence, not new duplicate algorithms.

| Equal-image mean | Derived training, n=4,794 | Validation, n=1,200 |
| --- | ---: | ---: |
| Native brightness | 0.4868 | 0.4910 |
| Native contrast | 0.1774 | 0.1800 |
| Native saturation | 0.3137 | 0.3154 |
| Mean caption words per image | 15.2484 | 15.2582 |
| Per-image TTR | 0.3350 | 0.3360 |
| Per-image mean Jaccard | 0.3079 | 0.3064 |

**Interpretation:** The supplied split preserves the measured class coverage and
recorded exact-group constraint. These selected means are numerically close, but
that does not establish equality of all distributions or model suitability. The
duplicate pair contributes two validation rows despite matching image content;
retaining it follows the approved shared policy and does not make it independent.

**Recommendation:** Keep the fixed memberships and document the validation pair
when interpreting later metrics. Any future fitted normalization or vocabulary
policy must use derived training membership, with the intended image pipeline and
weighting. No splitting, exclusions, preprocessing constants, or augmentation
settings are changed by this EDA.

**Limitations:** Essential audit flags do not verify label/caption semantics or
perceptual near-duplicates. Native-pixel summaries are not transformed/padded
tensor statistics. Official test rows were used only by the authorized integrity
audit and for identity/assignment counts here; test appearance and captions were
not used for statistical EDA or preprocessing recommendations.

### Stage 3 validation and artifacts

After integration with main `69cc366`, the complete repository suite passed
**533 tests**, with no failures or skips, including the shared pipeline and cached
train/validation integration checks. The earlier EDA-only checkpoint passed 435.
Actual caption results reconcile to the earlier 5,994-image/59,940-caption report, including total
token counts and every class count. Twelve selected actual caption records were
reduced independently. Fourteen train/validation metric summaries, including all
saved percentiles, agree with independent NumPy reductions within `1e-12`.

The shared manifest loader independently validates the real preparation before
analysis. Source revisions, audit completeness, preparation metadata, expected
counts, class coverage, and exact-group isolation were checked. All four new
figure groups were visually inspected. Same-checkout regeneration reproduced
all measured results and all 13 figure groups: PNGs are byte-identical and PDFs
differ only in export timestamps. Sixteen actual image records independently
confirm the native RGB, brightness, contrast, saturation and resolution formulas
within `1e-10`. Actual pixel measurements are complete for all 5,994 raw training
images; they are not fitted or post-transform normalization statistics.

All 37 earlier EDA/audit/preparation artifacts remain byte-identical. Fresh,
consistent reports and figures are under `outputs/eda-stage3/integrated-69cc366/`;
their image-report scope wording includes the 1,200 derived-validation members
of raw training. The earlier isolated prepared integration preserved its 28
preexisting EDA outputs. Stage 3-owned changes leave the shared pipeline and
README untouched; merged upstream changes arrived through the branch update.

New generated reports are `outputs/eda/reports/train-caption-details-full.json`
and `outputs/eda/reports/prepared-training-eda.json`. New PNG/PDF stems are
`training_caption_language`, `training_caption_diversity`,
`training_class_caption_variation`, and `training_prepared_class_coverage`.
All generated reports, figures, audit/preparation manifests, and caches remain
Git-ignored. The full findings do not certify semantic annotation quality or fit
any model/preprocessing policy.

## Stage 2: complete official training analysis

Analysis date: October 21, 2026. This section supersedes the preliminary prefix
for training-wide class/caption conclusions. Reproduction and measurement
definitions are in [Stage 2 instructions](eda_stage2.md). The historical Stage 1
observations below remain explicitly limited to their original 200-row subset.

Source: `Multimodal-Fatima/CUB_train` at revision
`bd3988995e6db193427255abdebd522afe24ce76`, official `train` split. The iterator was
exhausted and yielded **5,994 records**, matching the documented reference. The
report is `outputs/eda/reports/train-full.json` (generated and Git-ignored).
No official test records were analyzed. The class/caption calculations below do
not decode image pixels; the image section adds a separate complete training
scan through the shared native-image adapter. Shared loading, caption parsing,
source text, and split definitions remain unchanged.

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

### Image access, coverage, and source representation

Analysis date: October 22, 2026. These image findings complete the image portion
of Stage 2. Commands and policies are in [Stage 2 instructions](eda_stage2.md).
The generated report is `outputs/eda/reports/train-images-full.json`.

**Question:** Can every official training image be accounted for using the shared
native-image interface, and what representation do the measurements describe?

**Method:** Visit every indexed row through
`src.data.image_io.read_source_image`. Retain canonical identity, metadata errors,
the original shared row/fetch/decode status, original/oriented geometry, mode,
EXIF, transparency, and ICC facts. Apply the approved numerical functions to the
adapter's native RGB8 arrays, giving each eligible image equal weight. No second
decoder, source cleaning, or duplicate filtering is introduced.

**Measured finding:** All **5,994 training rows** were attempted, decoded, and
eligible for color measurement. There were zero row/fetch/decode failures, zero
color exclusions, zero metadata errors, and zero unclassified rows. All 200
species retain the same counts as the metadata report. Original modes were RGB
for 5,990 images and L for four; the shared adapter converts the latter to RGB.
No transparency metadata was observed. An ICC profile key was present in 3,004
images and absent in 2,990; the shared display policy applies no ICC transform.
Recorded EXIF orientations were 0 for 490 images, 1 for two, and absent for 5,502.
No original/oriented dimension differences were observed.

**Interpretation:** The raw training population is fully accounted for under
this decoder and measurement policy. The measurements describe encoded display
RGB, including the shared conversion of grayscale images. ICC-key presence is
not evidence of profile validity or calibrated color. Zero decoding failures
does not establish annotation correctness or overall training eligibility.

**Recommendation:** Preserve the mode/profile facts and use an explicitly
consistent color representation in later preprocessing. If the team changes
ICC handling or the display policy, recompute appearance statistics for that
representation. Keep missing/error categories explicit on later data versions.

**Limitations:** The adapter's first-frame, EXIF, RGB8, and no-ICC policies define
these results. The scan does not independently audit duplicates, leakage,
semantic quality, or physical camera orientation. All raw source rows remain.

### Image dimensions and resolution

**Question:** How variable is native geometry, and what does that imply for
future resizing discussions?

**Method:** Summarize recorded original and oriented width/height pairs, pixel
counts, megapixels, and width/height aspect ratios. Include every known pair,
independently of RGB eligibility. No crop, resize, or model input size is selected.

**Measured finding:** All 5,994 images have both dimension pairs. Original and
oriented summaries agree in this scan:

| Measurement | Minimum | Median | p95 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Width, pixels | 121 | 500 | 500 | 500 |
| Height, pixels | 120 | 375 | 500 | 500 |
| Aspect ratio, width/height | 0.4880 | 1.3333 | 1.5337 | 3.5971 |
| Megapixels | 0.01946 | 0.17850 | 0.23650 | 0.25000 |

The width and height medians are separate marginal summaries, not an observed
joint median image shape. Mean image size is approximately 0.17940 megapixels.

**Interpretation:** Source geometry varies, including portrait and wide views.
A direct square stretch would change proportions by different amounts across
images. The observed width and height are each bounded by 500 pixels in this
source representation; no larger native detail can be inferred from these files.

**Recommendation:** Discuss aspect-preserving resizing and padding with the data
engineer, and inspect bird coverage before selecting a crop policy. Retain native
source geometry in reports. The measurements do not select a model resolution.

**Limitations:** Global dimensions do not measure the bird's bounding box,
effective detail, or semantic completeness. No resize/crop experiment was run.

**Figure:** `outputs/eda/figures/training_image_dimensions.png`.

### Image brightness, contrast, saturation, and RGB

**Question:** What appearance variation is present before model preprocessing?

**Method:** Measure all native pixels in each eligible image. Brightness is mean
encoded luma `Y = .299R + .587G + .114B`; contrast is population `std(Y)`; saturation
is mean HSV S. RGB values are scaled by 255, and per-image RGB means and population
standard deviations are recorded separately. Image-level summaries have equal
image weighting and use the same linear percentile convention as caption EDA.

**Measured finding:** Each distribution contains 5,994 image observations:

| Measurement | Mean | Median | p05 | p95 |
| --- | ---: | ---: | ---: | ---: |
| Brightness, mean Y | 0.4876 | 0.4834 | 0.2872 | 0.7032 |
| Contrast, std(Y) | 0.1779 | 0.1737 | 0.0942 | 0.2767 |
| Saturation, mean HSV S | 0.3140 | 0.2945 | 0.0801 | 0.6128 |

Brightness ranges from 0.0823 to 0.9577, contrast from 0.0381 to 0.4166, and
saturation from 0 to 0.9537. The averages of per-image R/G/B means are
`[0.4856, 0.4994, 0.4324]`; the averages of their per-image standard deviations are
`[0.1817, 0.1811, 0.1927]`.

**Interpretation:** Images span a broad range of global appearance. These values
combine birds, backgrounds, text, and other visible content. The channel averages
describe this corpus under the specified display policy, not species-specific
color or calibrated reflectance. Mean per-image standard deviation omits
between-image mean differences and is not the corpus standard deviation.

**Recommendation:** Keep the distributions and source-linked examples for
preprocessing discussion. Recompute any fitted normalization on the eventual
derived training partition, with its intended pixel weighting and image pipeline.
Do not copy these descriptive image-weighted RGB summaries directly into model
normalization or choose cleaning/jitter thresholds from these distributions alone.

**Limitations:** Encoded luma is not physical luminance, and HSV saturation is not
a perceptual color calibration. No bird mask, ICC transform, class-discriminative
color study, augmentation experiment, or fitted threshold is included.

**Figures:** `training_image_appearance.png` and `training_image_rgb.png` in
`outputs/eda/figures/`.

### Image examples and interpretation of extremes

**Question:** What do selected images and numerical extremes actually look like?

**Method:** Select the first eligible source row from six evenly spaced observed
class groups. Separately select the minimum/maximum brightness, contrast, and
saturation, breaking ties by source row and ID. Display actual shared-adapter
pixels with a source caption and record its original caption index. Recheck
identity, species labels, and measurements before gallery export.

**Measured finding:** The class-spanning examples use source rows 0, 1200, 2400,
3569, 4766, and 5964. Appearance extremes include:

- Row 1196, Olive sided Flycatcher: highest brightness, 0.9577; the displayed
  image has a large pale background around a darker bird.
- Row 4270, Caspian Tern: lowest brightness, 0.0823; a bright bird is visible
  against a predominantly dark background.
- Row 4259, Caspian Tern: lowest contrast, 0.0381; the displayed bird is visible
  against a relatively uniform blue sky.
- Row 281, Red winged Blackbird: highest contrast, 0.4166.
- Row 726, Pelagic Cormorant: lowest saturation, 0; the displayed image is grayscale.
- Row 464, Painted Bunting: highest saturation, 0.9537; both the bird and the
  prominent green background contribute to the displayed color.

**Interpretation:** Global brightness/contrast can largely reflect background
composition. A low score alone does not establish that the bird is unusable.
The galleries also contain visible text/watermarks in some examples; this is a
sample observation, not a measured dataset-wide watermark frequency.

**Recommendation:** Share specific source IDs for human review before proposing
quality exclusions. Preserve original captions and avoid inferring caption errors
from a whole-image color mismatch. Use any later shared audit flags separately.

**Limitations:** These selections are deterministic illustrations, not random
samples, prevalence estimates, or a semantic validation of captions. Images may
recur across different extremes. No automated quality labels are assigned.

**Figures:** `training_image_gallery.png` and `training_image_extremes.png` in
`outputs/eda/figures/`.

### Validation

The initial expanded synthetic suite passed **101 tests**. After PR-preparation
safeguards, the complete suite passed **142 tests**, including existing source
contract tests, training-only command/builder guards, complete-scan safeguards,
report compatibility and count reconciliation, preservation of existing figures
after invalid inputs or export failures, plot counts, and pure image formulas.
Actual class/caption validation
used all 5,994 training records, with 59,940 nonempty captions. Four figure groups
were generated from the saved report as PNG and PDF; those class/caption plotting
functions do not reopen the source.

After integrating the completed shared image interface, the base suite passed
182 tests. With the image consumer, aggregation, plotting, and regression cases,
the complete suite passed **316 tests**, including final source-coverage,
identity, and class-mapping regression cases. The actual image scan accounted for all
5,994 training rows and generated five additional PNG/PDF figure groups. An
independent numerical check reread 15 selected training records, including fixed
rows, gallery examples, and extrema. Alternative float64 reduction formulas agreed
within absolute tolerance `1e-10`; the largest observed difference was approximately
`1.88e-12`. All saved aggregate summaries reconcile to the recorded measurement
rows, and class counts match the earlier training metadata report.

All five image PNGs were visually inspected for readable labels, displayed source
pixels, and caption layout. The two prior JSON reports and eight prior figure
files remain byte-identical to the pre-image-stage backup. The official test split
was not opened. These checks do not establish semantic caption correctness,
absence of near duplicates/leakage, or model suitability of a cleaning policy.

During the earlier class/caption PR preparation, both saved reports were preserved byte-for-byte against
a verified backup. The full-training figure pairs were regenerated from the
saved schema `1.1` report, with unchanged PNG pixels and measured conclusions.
The historical prefix report uses schema `1.0`; it remains preserved as evidence
but is rejected before figure publication because it lacks required plotting
distributions. Regenerate it with the current builder before plotting. These
checks reused saved evidence; they did not repeat that metadata/caption dataset scan.

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
