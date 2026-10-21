"""Run metadata/caption exploration through the shared CUB loader.

This is a standalone analysis entry point, not a replacement source loader or
the shared training CLI. No image pixels are decoded. Streaming may still
transfer image-containing Parquet row groups; the record limit is not a byte cap.
"""

import argparse
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from itertools import islice
from pathlib import Path
import platform
import sys
from typing import TYPE_CHECKING, Any

from .caption_stats import summarize_captions
from .dataset_stats import summarize_dataset
from .reporting import write_json_report

if TYPE_CHECKING:
    from src.data.source import CubSource


def _runtime_versions() -> dict[str, Any]:
    packages = {}
    for package in (
        "datasets", "huggingface-hub", "pyarrow", "multiprocess", "dill",
        "numpy", "Pillow", "matplotlib", "pytest"
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {"python": platform.python_version(), "packages": packages}


def _build_report(source: "CubSource", records: list, population: dict) -> dict[str, Any]:
    """Keep the statistical computation identical across selection strategies."""
    return {
        "schema_version": "1.1",
        "analysis_stage": (
            "stage2_training_metadata_and_captions"
            if population["selection_method"] == "complete_official_training_split"
            else "stage1_metadata_and_captions"
        ),
        "source": asdict(source.spec),
        "runtime": _runtime_versions(),
        "population": population,
        "methods": {
            "quantiles": "linear interpolation at index (n - 1) * p",
            "class_counts": "one observation per ImageRecord, never per caption",
            "class_count_summary": (
                "all expected source classes, including zero observations; "
                "interpret coverage according to the recorded selection scope"
            ),
            "source_splits": "counts in analyzed rows only; other splits are not audited",
            "word_tokens": (
                "NFC and casefold on a derived copy; alphabetic Unicode words; "
                "curly apostrophes normalized; internal apostrophes and hyphens "
                "(ASCII, U+2010, U+2011) retained; digits excluded"
            ),
            "caption_lengths": "nonempty clean captions; zero-word captions retained",
            "repetition": (
                "case-sensitive clean captions, excluding blanks; within-image extras "
                "count caption occurrences, across-image extras count additional images"
            ),
            "source_text": "raw_caption and clean_caption are never modified",
        },
        "dataset": summarize_dataset(
            (image for image, _ in records), class_names=source.class_names,
        ),
        "captions": summarize_captions(records),
    }


_LIMITATIONS = [
    "No image integrity, duplicate-image, or leakage audit was performed.",
    "Zero caption lines do not distinguish missing from empty descriptions.",
    "No pixel statistics, cleaning thresholds, or model vocabulary were fitted.",
    "No official test records were used for full-training analysis.",
    "Future preprocessing estimates must be fitted on the derived training partition.",
]


def _require_training_source(source: "CubSource") -> None:
    if source.spec.source_split != "train":
        raise ValueError("Statistical EDA accepts only the official train split")


def build_exploration_report(source: "CubSource", *, limit: int = 200) -> dict[str, Any]:
    """Summarize the first N source rows, explicitly retaining selection limits."""
    _require_training_source(source)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    records = list(islice(source.iter_records(), limit))
    population = {
        "selection_method": "source_order_prefix",
        "requested_limit": limit,
        "records_analyzed": len(records),
        "random_sample": False,
        "full_split_verified": False,
        "limitations": [
            "A source-order prefix is not representative of the full split.",
            "Class coverage, repetition, and vocabulary describe only these rows.",
            *_LIMITATIONS[:3],
        ],
    }
    return _build_report(source, records, population)


def build_training_report(
    source: "CubSource", *, expected_records: int = 5994
) -> dict[str, Any]:
    """Exhaust the official training source and compare its count to a reference.

    The default reference comes from the assignment; it never substitutes for
    the measured count. A mismatch is reported and leaves full_split_verified
    false. Iterator failures propagate without returning a completed report.
    """
    _require_training_source(source)
    if isinstance(expected_records, bool) or not isinstance(expected_records, int) or expected_records < 1:
        raise ValueError("expected_records must be a positive integer")
    records = list(source.iter_records())
    matches = len(records) == expected_records
    population = {
        "selection_method": "complete_official_training_split",
        "requested_limit": None,
        "source_exhausted": True,
        "records_analyzed": len(records),
        "reference_record_count": expected_records,
        "matches_reference_count": matches,
        "random_sample": False,
        "full_split_verified": matches,
        "limitations": list(_LIMITATIONS),
    }
    if not matches:
        population["limitations"].append(
            "Observed rows differ from the reference; investigate coverage before relying on full-split claims."
        )
    return _build_report(source, records, population)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--full-training", action="store_true")
    parser.add_argument("--expected-records", type=int, default=5994)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.full_training and args.limit is not None:
        parser.error("--full-training cannot be combined with --limit")
    if args.expected_records < 1:
        parser.error("--expected-records must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    # Import only for an actual source run; pure statistical tests remain offline.
    from src.data.source import load_cub_split

    source = load_cub_split(args.split, streaming=True)
    if args.full_training:
        print("Reading the complete official training source; no image decoding.", flush=True)
        report = build_training_report(source, expected_records=args.expected_records)
        default_name = "train-full.json"
    else:
        limit = args.limit if args.limit is not None else 200
        report = build_exploration_report(source, limit=limit)
        default_name = f"{args.split}-prefix-{limit}.json"
    output = args.output or Path("outputs/eda/reports") / default_name
    saved = write_json_report(report, output)
    print(f"Saved {report['population']['records_analyzed']} source rows to {saved}")
    if not args.full_training:
        print("This nonrandom prefix does not establish whole-split statistics.")
    elif not report["population"]["full_split_verified"]:
        print("Observed count differs from the reference; inspect report limitations.", file=sys.stderr)


if __name__ == "__main__":
    main()
