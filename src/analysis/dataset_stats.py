"""Dataset-level summaries over the shared image metadata records."""

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from src.data.schemas import ImageRecord

from .reporting import summarize_numbers


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
    """Count image records without deduplication or caption-based expansion.

    Class order comes from the shared source mapping, including absent classes.
    Repeated identical image records remain in every row-count denominator and
    are reported separately. A repeated identity with different metadata is an
    error because its class and provenance would otherwise be ambiguous.
    """
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
