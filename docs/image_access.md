# Native CUB image-access contract

This interface is a small extension to the Stage 1 [source contract](data_contract.md). It requires Pillow, fsspec, and the existing pinned loader. It has no PyTorch, classification-preparation, or Stage 2 audit dependency.

## Availability and integration

The implementation branch is `feat/shared-image-access`, based on main commit `3926d3e2499d721baac042a2a1c07948c8ed791c`. It can be reviewed and consumed independently of the classification pipeline and integrity-audit PR. Existing Stage 1 record fields and Stage 2 hash policies are unchanged.

## Read training images

```python
import numpy as np

from src.data.image_io import read_source_image
from src.data.source import load_cub_split

source = load_cub_split("train")  # Indexed, revision-pinned; opens no test source.
failures = []
for source_row in range(len(source.dataset)):
    view = read_source_image(source, source_row)
    if view.status != "ok":
        failures.append({
            "image_id": view.image_id,
            "source_row": view.source_row,
            "status": view.status,
            "reason": view.error_code,
        })
        continue
    with view.image as rgb:  # The consumer owns and closes this PIL image.
        pixels = np.asarray(rgb)
        # RGB uint8, shape (oriented_height, oriented_width, 3), range 0..255.
        # Run measurements here; no crop, resize, tensor normalization or filtering.
        print(view.image_id, view.class_id, view.class_name, pixels.shape)
```

For an initial check, read just `read_source_image(source, 0)` and close its image. The complete loop deliberately visits every raw official training row. It returns failures instead of dropping those rows. Source iteration is indexed: `streaming=True`, boolean indices, negative indices, and out-of-range indices raise `ValueError` rather than inventing a source identity. Stable IDs use the pinned source revision and zero-based source row.

The lower-level `load_rgb(reference)` still accepts a Hugging Face `Image(decode=False)` bytes/path mapping and returns a caller-owned PIL RGB image, raising explicit fetch/decoder exceptions. Prefer `read_source_image` when metadata and recoverable row status are needed.

## Returned SourceImageView

| Fields | Meaning |
| --- | --- |
| `image_id`, `source_repo`, `source_revision`, `source_split`, `source_row` | Canonical source identity, retained even when a real row fails to read or decode. |
| `source_id`, `file_name`, `class_id`, `class_name` | Original source references and ordered class mapping; invalid values become `None` with a metadata error. Valid labels remain the original zero-based class IDs. |
| `status` | `ok`, `row_error`, `fetch_error`, or `decode_error`. `ok` describes decoded pixels, not overall data quality or classification eligibility. |
| `error_code`, `error_message` | Machine-readable failure reason and diagnostic text; both `None` on successful decoding. |
| `metadata_errors` | Tuple of ID, filename, or class validation errors, independent of pixel status. This is not complete schema/caption validation or a replacement for the integrity audit. |
| `image` | Fully loaded, independent PIL RGB image on success; `None` on image/row failure. |
| `original_size` | Encoded image dimensions before EXIF correction, `(width, height)`. |
| `oriented_size` | Native dimensions after EXIF correction, `(width, height)`. No resize is performed. |
| `original_mode`, `image_format` | Mode and encoded format before RGB conversion, such as `L` and `JPEG`. |
| `exif_orientation` | Original orientation tag 274, or `None` when absent. |
| `has_transparency` | Whether the encoded image has an alpha band or transparency information, before white compositing. |
| `icc_profile_present` | Whether encoded metadata contains an ICC profile key. Presence does not establish profile validity or color-managed output. |

Header facts can remain available on a truncated-image decoding failure. Facts not reached during a failure are `None`. `metadata_errors` does not inspect caption descriptions, duplicate IDs across rows, filenames across rows, or semantic correctness. A malformed class label does not erase otherwise decodable pixels: resolution analysis can still account for the image while class-based analysis handles the metadata error explicitly.

## Pixel policies

- Embedded bytes take precedence over paths. Local and fsspec paths use bounded encoded-data spooling; decoded pixels for one image must fit in memory.
- Pillow fully decodes the first image frame; this interface does not iterate animated/multiframe assets. Decompression-bomb limits remain enabled.
- EXIF orientation is applied before exposing pixels. Rotations can swap width and height; EXIF orientation is removed from the returned display image.
- Pixels are converted with Pillow to opaque RGB8. Conventional grayscale, palette, and CMYK conversions follow Pillow's conversion behavior. Unsupported conversions fail explicitly. Numeric/scientific modes should not be assumed suitable for color statistics merely because RGB conversion succeeds.
- Transparency is composited over white before RGB conversion, using the same `exif-rgb8-white:v1` display policy as the audit. Original transparency remains visible through `has_transparency`.
- No ICC color-management transform, gamut calibration, resizing, augmentation, tensor conversion, normalization, hashing, or source-file modification occurs.
- NumPy conversion in the example yields `uint8` values from 0 through 255. Dividing by 255 or applying model normalization is a consumer decision, not this interface's pixel contract.

For color summaries, the EDA consumer should apply its approved original-mode/transparency policy explicitly. Do not treat white-composited alpha pixels as original opaque-scene color. Mode/profile/transparency facts are observations; the adapter does not choose the teammate's measurement policy. Preserve resolution accounting and report color exclusions with reasons keyed by `image_id`.

## Failure and quality joins

`row_error` uses `source.row_read_failed` or `metadata.invalid_row`. Fetch errors include `invalid_reference`, `missing_reference`, and `fetch_failed`. Decoder errors include `decode_failed` and `decompression_bomb`. Metadata errors use `metadata.missing_id`, `metadata.invalid_source_id`, `metadata.missing_file_name`, `metadata.invalid_filename`, `metadata.missing_label`, and `metadata.invalid_label`.

This is a **raw, unfiltered view**: it performs no cleaning, splitting, classification eligibility selection, or silent skipping. Consumers must record each failed/excluded row and their own analysis-specific exclusion reasons. Caption quality does not affect pixel access.

If a Stage 2 image audit is available, join its original error/warning flags and exact-group IDs by `image_id`:

```python
import json

with open("outputs/training_quality_flags.jsonl", encoding="utf-8") as handle:
    quality_by_id = {row["image_id"]: row for row in map(json.loads, handle)}

# After reading a view:
quality = quality_by_id.get(view.image_id)  # None means unknown, not "clean".
```

The optional sidecar is generated audit data and must remain outside Git. It is not needed to decode images, and it must match the same pinned revision. Original audit status and flags should retain their names and meanings. The adapter's current decode status is a fresh observation, separate from a previously saved audit.

## Verification

Synthetic tests cover identity-linked native RGB access, class metadata, EXIF dimensions, grayscale/palette conversion, original alpha/ICC facts, malformed metadata with valid pixels, row-read errors, fetch/decode failures, truncated-image header facts, invalid indices, streaming rejection, and `load_rgb` pixel equivalence. The complete current-main suite passed 162 tests, including 20 new adapter tests. A cached real-data scan accounted for all 5,994 official training rows: all decoded successfully, with zero identity/class/pixel-hash mismatches against the original audit and no metadata errors. Scanning and pixel-parity verification took 11.721 seconds on the development machine; the first 32 rows took 0.103 seconds. Torch was not imported. These are measurements, not performance guarantees. No official test image is used for EDA parameter selection.
