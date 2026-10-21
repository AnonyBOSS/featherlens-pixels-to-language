# CUB source data contract — Stage 1

This document defines the shared source interface for the CUB-200-2011 image and caption data. It covers source loading and identity only. It does not define cleaned data, generated train/validation manifests, image quality decisions, augmentation, or task-specific PyTorch datasets.

## Pinned sources

| Official source split | Hugging Face repository | Pinned commit |
| --- | --- | --- |
| Train | `Multimodal-Fatima/CUB_train` | `bd3988995e6db193427255abdebd522afe24ce76` |
| Test | `Multimodal-Fatima/CUB_test` | `82447a42c5e05c7fe21cd36accfdf399148e8668` |

Both repositories use the `default` configuration. The public `load_cub_split()` function passes the exact repository, official split, and commit to `datasets.load_dataset`. The commits are part of the source contract: changing either one requires a new verification pass and can change row-based identifiers.

The source schema requires these Hugging Face features: `image: Image`, `description: Value("string")`, `label: ClassLabel`, `file_name: Value("string")`, and `id: Value("int64")`. Additional source fields are allowed. Labels are the zero-based positions in the ordered `ClassLabel.names` sequence; class names are never sorted or derived from filenames. The loader requires 200 unique class names, and `load_cub_sources()` rejects a train/test mapping difference even when both sides have 200 classes.

## Public interface

```python
from src.data.source import load_cub_sources

sources = load_cub_sources(streaming=True)  # useful for a small inspection
train = sources["train"]
image, captions = next(train.iter_records())
print(image.image_id, image.class_id, image.class_name)
print(captions[0].raw_caption, captions[0].clean_caption)
```

`load_cub_split("train")` or `load_cub_split("test")` opens one split. `load_cub_sources()` opens both and verifies the ordered class mapping. The default is a regular Hugging Face `Dataset`, suitable for later random access but requiring the source data to be downloaded and cached. `streaming=True` returns an iterable source for inspection or sequential analysis; callers should not assume random access in this mode.

Each returned `CubSource` exposes `spec`, `class_names`, `dataset`, and `iter_records()`. `dataset` is a Hugging Face source view with `Image(decode=False)`, so its image values remain source path/bytes references. `iter_records()` uses a metadata-only projection and does not decode image pixels. Later task adapters can decode images explicitly from `dataset`; Stage 1 does not create image tensors or alter source assets.

## Image identity and metadata

`ImageRecord.image_id` has this exact form:

```text
hf:<repository-id>@<full-commit-sha>:<official-source-split>:<zero-based-source-row>
```

For example, the first train row is `hf:Multimodal-Fatima/CUB_train@bd3988995e6db193427255abdebd522afe24ce76:train:0`. The Hugging Face `id` and `file_name` are retained for traceability, but neither is used as the primary key. The source `id` begins at zero in each repository, and whole-dataset filename uniqueness has not been verified.

`ImageRecord` fields are `image_id`, `source_repo`, `source_revision`, `source_split`, `source_row`, `source_id`, `file_name`, `class_id`, and `class_name`. Records are immutable Python dataclasses. The loader checks that each read label is an integer within the ordered class mapping; other per-row validity checks belong to the later audit stage.

The row number is stable for the pinned repository commit and split. A new commit means a new identity namespace even if a filename happens to be unchanged. This avoids accidental joins across different source versions.

## Caption relationship

`CaptionRecord` fields are `image_id`, zero-based `caption_index`, `raw_caption`, and `clean_caption`. `caption_index` preserves the line position in the original `description` field, including internal blank lines. `raw_caption` preserves the line content except for its CR, LF, or CRLF separator; other whitespace within a line does not create another caption. `clean_caption` only trims and collapses whitespace; case, punctuation, bird terminology, and spelling remain unchanged. A trailing newline does not create a phantom caption. A missing description (`None`) yields no caption records and does not remove its image record.

The original `description` remains in the Hugging Face source view. Classification consumers can use image metadata without consuming captions, while later captioning work can join caption records on `image_id`.

## Verification boundary

Live Hugging Face metadata reports 5,994 train and 5,794 test rows and matching five-field schemas. A small streamed sample from each pinned commit loaded with `datasets.load_dataset` and showed 200 classes and ten caption lines. These observations do not establish whole-dataset ID or filename uniqueness, ten valid captions for every image, image decodability, class balance after exclusions, or absence of duplicates and leakage. Those require a full audit in a later stage.

The Stage 1 synthetic tests cover identity, caption parsing, schema checks, ordered class-map agreement, pinned loader arguments, undecoded image references, and invalid labels. No full dataset scan is part of Stage 1.

On October 21, 2026, a live `load_cub_sources(streaming=True)` smoke test read the first row from each pinned revision. Both exposed the same ordered 200-class mapping, produced ten linked caption records, and kept `image.decode` false. This is a two-row check, not a full dataset audit.
