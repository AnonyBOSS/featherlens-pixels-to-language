"""Shared descriptive statistics, JSON writing, caption tokenization, and pixel/resolution metrics."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
from statistics import fmean
import tempfile
from typing import Any
import unicodedata

import numpy as np

from src.data.source import CaptionRecord, ImageRecord


_EXAMPLE_LIMIT = 5
_INTERNAL_JOINERS = {"'", "-", "\u2010", "\u2011"}


def summarize_numbers(values: Iterable[Real]) -> dict[str, int | float | None]:
    """Compute min/max/mean/median and p05/p25/p75/p95/p99 using (n - 1) * p interpolation."""
    numbers = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("summary values must be real numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("summary values must be finite")
        numbers.append(number)
    numbers.sort()

    def quantile(probability: float) -> float | None:
        if not numbers:
            return None
        position = (len(numbers) - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        return numbers[lower] + (numbers[upper] - numbers[lower]) * (position - lower)

    return {
        "count": len(numbers),
        "min": numbers[0] if numbers else None,
        "max": numbers[-1] if numbers else None,
        "mean": fmean(numbers) if numbers else None,
        "median": quantile(0.5),
        **{name: quantile(p) for name, p in (
            ("p05", 0.05), ("p25", 0.25), ("p75", 0.75),
            ("p95", 0.95), ("p99", 0.99),
        )},
    }


def write_json_report(report: Mapping[str, Any], output: str | Path) -> Path:
    """Atomically write a UTF-8 JSON report, rejecting NaN/Infinity."""
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}-", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(serialized + "\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def _distribution(counts: list[int], class_names: tuple[str, ...]) -> list[dict[str, Any]]:
    total = sum(counts)
    return [
        {
            "class_id": class_id,
            "class_name": class_name,
            "count": counts[class_id],
            "proportion": counts[class_id] / total if total else 0.0,
        }
        for class_id, class_name in enumerate(class_names)
    ]


def summarize_dataset(
    images: Iterable[ImageRecord], *, class_names: Sequence[str]
) -> dict[str, Any]:
    """Count image records per class and split without deduplicating."""
    if isinstance(class_names, (str, bytes)):
        raise ValueError("class_names must be an ordered sequence of class names")
    names = tuple(class_names)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("class_names must contain nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("class_names must be unique")

    counts = [0] * len(names)
    split_counts = {"train": [0] * len(names), "test": [0] * len(names)}
    identities: dict[str, ImageRecord] = {}
    provenance: Counter[tuple[str, str, str]] = Counter()
    missing_source_id = 0
    missing_file_name = 0
    total_records = 0

    for image in images:
        class_id = image.class_id
        if isinstance(class_id, bool) or not isinstance(class_id, int) or not 0 <= class_id < len(names):
            raise ValueError(f"Invalid class_id for image {image.image_id!r}: {class_id!r}")
        if image.class_name != names[class_id]:
            raise ValueError(f"Class name does not match class_id for image {image.image_id!r}")
        if image.source_split not in split_counts:
            raise ValueError(f"Unknown official source_split: {image.source_split!r}")
        previous = identities.get(image.image_id)
        if previous is not None and previous != image:
            raise ValueError(f"Conflicting metadata for image_id {image.image_id!r}")
        identities[image.image_id] = image

        counts[class_id] += 1
        split_counts[image.source_split][class_id] += 1
        provenance[(image.source_repo, image.source_revision, image.source_split)] += 1
        missing_source_id += image.source_id is None
        missing_file_name += image.file_name is None or not image.file_name.strip()
        total_records += 1

    source_splits = {}
    for split, values in split_counts.items():
        observed = sum(count > 0 for count in values)
        source_splits[split] = {
            "total_records": sum(values),
            "expected_classes": len(names),
            "observed_classes": observed,
            "class_coverage": observed / len(names) if names else None,
            "missing_class_ids": [class_id for class_id, count in enumerate(values) if count == 0],
            "class_distribution": _distribution(values, names),
        }

    return {
        "total_records": total_records,
        "unique_image_ids": len(identities),
        "repeated_image_id_rows": total_records - len(identities),
        "expected_classes": len(names),
        "observed_classes": sum(count > 0 for count in counts),
        "missing_class_ids": [class_id for class_id, count in enumerate(counts) if count == 0],
        "class_distribution": _distribution(counts, names),
        "class_count_summary": summarize_numbers(counts),
        "imbalance_ratio": max(counts) / min(counts) if counts and min(counts) > 0 else None,
        "source_splits": source_splits,
        "missing_metadata": {"source_id": missing_source_id, "file_name": missing_file_name},
        "source_provenance": [
            {
                "source_repo": repo,
                "source_revision": revision,
                "source_split": split,
                "record_count": count,
            }
            for (repo, revision, split), count in sorted(provenance.items())
        ],
    }


def tokenize_caption(text: str) -> tuple[str, ...]:
    """Tokenize into lowercase alphabetic words, keeping internal apostrophes and hyphens."""
    normalized = unicodedata.normalize("NFC", text).casefold()
    normalized = unicodedata.normalize("NFC", normalized)
    normalized = normalized.replace("\u2018", "'").replace("\u2019", "'")
    tokens: list[str] = []
    current: list[str] = []
    for index, character in enumerate(normalized):
        if character.isalpha():
            current.append(character)
        elif (
            character in _INTERNAL_JOINERS
            and current
            and current[-1].isalpha()
            and index + 1 < len(normalized)
            and normalized[index + 1].isalpha()
        ):
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tuple(tokens)


def summarize_captions(
    records: Iterable[tuple[ImageRecord, Sequence[CaptionRecord]]],
    *,
    expected_captions_per_image: int = 10,
    top_k: int = 20,
) -> dict:
    """Compute caption counts, length distributions, vocabulary, and exact repetition stats."""
    if expected_captions_per_image < 0:
        raise ValueError("expected_captions_per_image must be non-negative")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")

    image_ids: set[str] = set()
    line_counts: list[int] = []
    nonempty_counts: list[int] = []
    character_lengths: list[int] = []
    word_lengths: list[int] = []
    token_counts: Counter[str] = Counter()
    occurrences: dict[str, Counter[str]] = defaultdict(Counter)
    examples: list[dict] = []
    whitespace_anomaly_count = 0

    for image, captions in records:
        if image.image_id in image_ids:
            raise ValueError(f"image_id must be unique: {image.image_id}")
        image_ids.add(image.image_id)
        line_counts.append(len(captions))
        nonempty_count = 0

        for caption in captions:
            if caption.image_id != image.image_id:
                raise ValueError("caption image_id must match its ImageRecord")
            if caption.raw_caption != " ".join(caption.raw_caption.split()):
                whitespace_anomaly_count += 1
            if not caption.clean_caption:
                continue

            nonempty_count += 1
            text = caption.clean_caption
            tokens = tokenize_caption(text)
            character_lengths.append(len(text))
            word_lengths.append(len(tokens))
            token_counts.update(tokens)
            occurrences[text][image.image_id] += 1
            examples.append(
                {
                    "image_id": image.image_id,
                    "caption_index": caption.caption_index,
                    "text": text,
                    "word_count": len(tokens),
                }
            )

        nonempty_counts.append(nonempty_count)

    total_tokens = sum(token_counts.values())
    hapax_word_types = sum(count == 1 for count in token_counts.values())
    ranked_tokens = sorted(token_counts.items(), key=lambda item: (-item[1], item[0]))

    def repeated_summary(*, across_images: bool) -> dict:
        repeated: list[tuple[str, Counter[str], int]] = []
        repeated_image_text_groups = 0
        affected_image_ids: set[str] = set()
        for text, counts_by_image in occurrences.items():
            if across_images:
                extra = len(counts_by_image) - 1
            else:
                extra = sum(count - 1 for count in counts_by_image.values())
                repeated_image_text_groups += sum(
                    count > 1 for count in counts_by_image.values()
                )
                affected_image_ids.update(
                    image_id for image_id, count in counts_by_image.items() if count > 1
                )
            if extra:
                repeated.append((text, counts_by_image, extra))
        repeated.sort(key=lambda item: (-item[2], item[0]))
        result = {
            "repeated_distinct_texts": len(repeated),
            "extra_occurrences": sum(extra for _, _, extra in repeated),
            "examples": [
                {
                    "text": text,
                    "occurrence_count": sum(counts_by_image.values()),
                    "image_count": len(counts_by_image),
                    "image_ids": sorted(counts_by_image)[:_EXAMPLE_LIMIT],
                    "extra_occurrences": extra,
                }
                for text, counts_by_image, extra in repeated[:_EXAMPLE_LIMIT]
            ],
        }
        if not across_images:
            result["repeated_image_text_groups"] = repeated_image_text_groups
            result["affected_image_count"] = len(affected_image_ids)
        return result

    shortest = sorted(
        examples,
        key=lambda row: (
            row["word_count"], len(row["text"]), row["image_id"], row["caption_index"]
        ),
    )[:_EXAMPLE_LIMIT]
    longest = sorted(
        examples,
        key=lambda row: (
            -row["word_count"], -len(row["text"]), row["image_id"], row["caption_index"]
        ),
    )[:_EXAMPLE_LIMIT]
    nonempty_caption_count = sum(nonempty_counts)
    caption_line_count = sum(line_counts)

    return {
        "image_count": len(image_ids),
        "caption_line_count": caption_line_count,
        "nonempty_caption_count": nonempty_caption_count,
        "empty_caption_count": caption_line_count - nonempty_caption_count,
        "images_without_caption_lines": line_counts.count(0),
        "images_without_nonempty_captions": nonempty_counts.count(0),
        "expected_captions_per_image": expected_captions_per_image,
        "per_image_line_counts": summarize_numbers(line_counts),
        "per_image_nonempty_counts": summarize_numbers(nonempty_counts),
        "line_count_mismatch_images": sum(
            count != expected_captions_per_image for count in line_counts
        ),
        "nonempty_count_mismatch_images": sum(
            count != expected_captions_per_image for count in nonempty_counts
        ),
        "character_lengths": summarize_numbers(character_lengths),
        "word_lengths": summarize_numbers(word_lengths),
        "word_length_distribution": [
            {"word_count": length, "caption_count": count}
            for length, count in sorted(Counter(word_lengths).items())
        ],
        "character_length_distribution": [
            {"character_count": length, "caption_count": count}
            for length, count in sorted(Counter(character_lengths).items())
        ],
        "per_image_nonempty_count_distribution": [
            {"caption_count": caption_count, "image_count": image_count}
            for caption_count, image_count in sorted(Counter(nonempty_counts).items())
        ],
        "zero_word_caption_count": word_lengths.count(0),
        "vocabulary_size": len(token_counts),
        "total_tokens": total_tokens,
        "token_frequencies": dict(sorted(token_counts.items())),
        "top_tokens": [
            {"token": token, "count": count} for token, count in ranked_tokens[:top_k]
        ],
        "hapax_word_types": hapax_word_types,
        "hapax_token_share": hapax_word_types / total_tokens if total_tokens else None,
        "whitespace_anomaly_count": whitespace_anomaly_count,
        "repetition": {
            "unique_nonempty_texts": len(occurrences),
            "global_extra_occurrences": nonempty_caption_count - len(occurrences),
            "within_images": repeated_summary(across_images=False),
            "across_images": repeated_summary(across_images=True),
        },
        "shortest_examples": shortest,
        "longest_examples": longest,
    }


def measure_resolution(width: int, height: int) -> dict:
    """Return dimensions, pixel count, megapixels, and width/height aspect ratio."""
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, Integral)
        or not isinstance(height, Integral)
    ):
        raise TypeError("width and height must be integers")
    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    width = int(width)
    height = int(height)
    pixel_count = width * height
    return {
        "width": width,
        "height": height,
        "pixel_count": pixel_count,
        "megapixels": pixel_count / 1_000_000,
        "aspect_ratio": width / height,
    }


def measure_rgb_array(rgb: np.ndarray) -> dict:
    """Measure per-channel RGB, luma brightness/contrast, and HSV saturation on an HxWx3 array."""
    if not isinstance(rgb, np.ndarray):
        raise TypeError("rgb must be a NumPy array")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] < 1 or rgb.shape[1] < 1:
        raise ValueError("rgb must have nonempty shape H x W x 3")
    if rgb.dtype == np.uint8:
        values = rgb.astype(np.float64) / 255.0
    elif np.issubdtype(rgb.dtype, np.floating):
        values = rgb.astype(np.float64, copy=False)
    else:
        raise TypeError("rgb dtype must be uint8 or floating point")
    if not np.isfinite(values).all():
        raise ValueError("rgb values must be finite")
    if (values < 0).any() or (values > 1).any():
        raise ValueError("floating rgb values must be in [0, 1]")

    height, width = values.shape[:2]
    luma = 0.299 * values[..., 0] + 0.587 * values[..., 1] + 0.114 * values[..., 2]
    maximum = values.max(axis=2)
    minimum = values.min(axis=2)
    saturation = np.divide(
        maximum - minimum,
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 0,
    )
    return {
        **measure_resolution(width, height),
        "rgb_mean": values.mean(axis=(0, 1)).tolist(),
        "rgb_std": values.std(axis=(0, 1), ddof=0).tolist(),
        "brightness_mean": float(luma.mean()),
        "contrast_std": float(luma.std(ddof=0)),
        "saturation_mean": float(saturation.mean()),
    }
