# CUB integrity audit — Stage 2

This is a read-only audit of the two revision-pinned CUB sources in [data_contract.md](data_contract.md). It records observations and quality flags separately from source records. It does not remove images, change the official train/test split, set cleaning policy, perform statistical EDA, or create a PyTorch dataset.

## Run the essential audit

From the repository root in an environment with `datasets`, `Pillow`, and `fsspec` installed:

```python
from src.data.integrity_audit import audit_pinned_sources

summary = audit_pinned_sources("outputs/integrity-audit")
print(summary["run_status"], summary["processed_counts"])
```

The default loads both pinned splits into the Hugging Face cache for indexed access. Check free disk space before a full run; the four source Parquet shards total about 1.07 GiB, and the prepared cache plus temporary files require additional space. `outputs/` is ignored by Git. A streamed audit is available with `streaming=True`, but an iterator failure can prevent later rows from being reached. The run summary marks such coverage incomplete.

`audit_sources(sources, output_dir, expected_counts=...)` accepts supplied `CubSource`-like objects for synthetic tests or another pinned source setup. The official wrapper expects 5,994 train and 5,794 test rows. For indexed data, it attempts every source index, preserving a stable `image_id` even when a row cannot be read or inspected. An inaccessible row is represented by a placeholder and makes coverage incomplete. A corrupt image that was read and classified as a decode failure still counts as an accounted-for row.

## Output contract

| File | Key and purpose |
| --- | --- |
| `images.jsonl` | One record per attempted source row, keyed by `image_id`; preserves source revision, split, row, source ID, filename, class reference, validity states, flags, hashes, and exact group membership. |
| `captions.jsonl` | One record per parsed source caption line, keyed by `(image_id, caption_index)`; retains `raw_caption`, `clean_caption`, `usable`, and line-level flags. |
| `exact_groups.jsonl` | One connected component of byte-identical or pixel-identical images per record, with deterministic `group_id`, sorted `image_ids`, hash evidence, cross-split indicator, and label-conflict indicator. |
| `summary.json` | Expected, observed, and processed rows per split; coverage status; error and warning counts (including by split); caption and duplicate totals; revisions, library versions, normalization version, and run times. |

In `images.jsonl`, `metadata_valid`, `image_valid`, and `caption_structural_valid` are separate facts. `image_valid` means the image bytes were fetched and fully decoded; it is not a classification eligibility decision. `audit_status` is `invalid` if any error flag exists, `warning` if only warning flags exist, or `valid` otherwise. `error_flags` and `warning_flags` are stable, machine-readable dotted codes such as `metadata.invalid_label`, `caption.empty`, `image.fetch_failed`, `image.decode_failed`, `duplicate.exact`, and `leakage.exact_cross_split`. Error messages are diagnostic text and should not be used as programmatic keys.

A caption is `usable` only if its minimally normalized text is nonempty and its raw line has no Unicode control character in category `Cc` other than tab. Blank lines keep their original `caption_index`. A description with other than ten parsed lines receives a structural warning; no caption is dropped. This audit does not analyze vocabulary or caption content.

`coverage_complete` means both expected row counts were accounted for with no row-read, row-inspection, source-load, or iterator interruption. It does **not** mean every image or caption is valid. Stage 3 should inspect the separate flags and adopt an explicit policy before excluding or transforming anything.

Use a distinct output directory for each run, and require `coverage_complete=true` before consuming its manifests. A run atomically marks its summary incomplete before scanning or replacing files, publishes the manifests, and publishes the final summary last. An interrupted rerun therefore cannot leave a previous complete summary certifying partially replaced files. Concurrent writers must use separate directories.

Caption reports are streamed to a temporary JSONL file during scanning. The audit retains only compact image observations and hash indexes in memory; encoded image data uses an 8 MiB spool limit and pixel hashing uses 64-row strips. The decoded image itself must fit in memory. Duplicate grouping uses hash buckets and union-find rather than pairwise pixel comparisons.

## Image fingerprints and exact groups

`encoded_sha256` is SHA-256 of the fetched encoded bytes. `pixel_sha256` is SHA-256 over the domain tag `featherlens:pixel:exif-rgb8-white:v1\0`, orientation-corrected width and height as two unsigned 64-bit big-endian integers, and row-major 8-bit RGB pixels. The audit fully decodes with Pillow, applies EXIF orientation, and composites transparency over opaque white before RGB conversion. Thus byte-different lossless encodings with the same displayed pixels can share a pixel hash, while visually similar JPEG re-encodings usually require a later near-duplicate screen. Source format and mode are recorded before normalization; recorded dimensions describe the normalized image. No resizing or ICC profile color-management transform is applied; equality means equality under this versioned normalization.

Equal byte or pixel hashes are joined transitively into one exact component. Its `group_id` is `exact:` plus SHA-256 of the sorted member `image_id` values separated by newlines. Each image carries that identifier in `exact_group_ids` (empty or one member in this version). A group spanning official train and test is flagged as possible evaluation leakage. The report does not automatically modify either official split. Filename or source-provided ID equality alone does not establish image identity.

## Stage 3 consumption

Treat `images.jsonl` as the source-identity and quality-flag manifest. A classification cleaning policy can combine a valid class reference, successful image decode, and reviewed flags; captioning can select `captions.jsonl` records with `usable=true` while preserving raw text and original indexes. Use `exact_group_ids` as a group constraint when later deriving a validation split from official training data. Review every cross-split exact group before evaluation. Keep the original source records and this audit output separate so decisions can be revised without redownloading or rewriting CUB.

The optional perceptual-hash candidate phase is not part of this essential audit. Its thresholds must be selected from synthetic cases or training-side review, not by optimizing against the official test set.

## Verified full run — 2026-10-21

The final essential audit used the exact revisions in the source contract with Python 3.11.17, datasets 5.1.0, Pillow 13.0.0, and fsspec 2026.7.0.

| Check | Result |
| --- | --- |
| Official train rows accounted for | 5,994 / 5,994 |
| Official test rows accounted for | 5,794 / 5,794 |
| Row-read or row-inspection failures | 0 |
| Metadata, caption, fetch, or decode error flags | 0 |
| Parsed / usable caption lines | 117,880 / 117,880 |
| Exact duplicate groups / images | 1 / 2 |
| Exact groups crossing official train/test | 0 |
| Conflicting labels within exact groups | 0 |

The sole warning concerns these two official training records, both class 57 (`Pigeon Guillemot`):

| Source row | Source filename |
| --- | --- |
| 1712 | `Pigeon_Guillemot_0018_40195.jpg` |
| 1731 | `Pigeon_Guillemot_0081_40339.jpg` |

Both encoded bytes and normalized pixels match. Their exact group is `exact:09134262c62225392edf25319d3c64e8c54bee67e86a97dc4bd872fe4a325c16`. They remain in the source and the audit manifest. A later training/validation split must keep them in the same partition or apply an approved deduplication policy.

The initial 200-row streamed benchmark took 158.75 seconds, with about 0.51 seconds spent decoding and hashing; remote reads dominated. Caching all source rows subsequently took about 6.1 minutes. The final cached full audit took 32.719 seconds, including 4.273 seconds loading cached sources and 28.446 seconds scanning and reporting. Generated reports totaled 54,049,156 bytes (about 51.55 MiB). These are measurements on this machine, not runtime guarantees; budget 3–5 GiB of cache and temporary space for a fresh run.

No invalid record was identified by these checks. This establishes full coverage for the pinned sources and normalization version, not semantic correctness of every label or caption. Perceptual near-duplicates, human annotation mistakes, and model-specific eligibility remain outside the essential audit.
