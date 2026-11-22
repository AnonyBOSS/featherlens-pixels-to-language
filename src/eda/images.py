"""Training-image EDA: per-image measurements, report aggregation, and gallery loading."""

import argparse
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np

from .explore import _runtime_versions
from .stats import measure_rgb_array, summarize_numbers, write_json_report


_STATUSES = ("ok", "row_error", "fetch_error", "decode_error", "unsupported_mode")
_SOURCE_STATUSES = ("ok", "row_error", "fetch_error", "decode_error")
_OPAQUE_MODES = {"1", "L", "P", "RGB", "CMYK", "YCbCr"}
_SCALAR_METRICS = (
    "width", "height", "pixel_count", "megapixels", "aspect_ratio",
    "brightness_mean", "contrast_std", "saturation_mean",
)
_CLASS_METRICS = ("brightness_mean", "contrast_std", "saturation_mean", "aspect_ratio")
_CHANNELS = ("red", "green", "blue")


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{path} must be an integer >= {minimum}")
    return int(value)


def _number(value: Any, path: str, *, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{path} must be a finite nonnegative number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{path} must be a finite nonnegative number") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{path} must be a finite nonnegative number")
    if maximum is not None and number > maximum + 1e-12:
        raise ValueError(f"{path} must not exceed {maximum}")
    return number


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a nonempty string")
    return value


def _field(row: Mapping, key: str, path: str) -> Any:
    if key not in row:
        raise ValueError(f"Missing required field: {path}.{key}")
    return row[key]


def _dimensions(row: dict, prefix: str, path: str) -> tuple[int, int] | None:
    width = _field(row, f"{prefix}_width", path)
    height = _field(row, f"{prefix}_height", path)
    if width is None and height is None:
        return None
    if width is None or height is None:
        raise ValueError(f"{path}.{prefix} dimensions must be both known or both null")
    width = _integer(width, f"{path}.{prefix}_width", minimum=1)
    height = _integer(height, f"{path}.{prefix}_height", minimum=1)
    row[f"{prefix}_width"] = width
    row[f"{prefix}_height"] = height
    return width, height


def _metrics(value: Any, oriented: tuple[int, int], path: str) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object for status=ok")
    metrics = deepcopy(dict(value))
    width = _integer(_field(metrics, "width", path), f"{path}.width", minimum=1)
    height = _integer(_field(metrics, "height", path), f"{path}.height", minimum=1)
    pixels = _integer(_field(metrics, "pixel_count", path), f"{path}.pixel_count", minimum=1)
    if (width, height) != oriented or pixels != width * height:
        raise ValueError(f"{path} geometry must match oriented dimensions and width * height")
    metrics.update(width=width, height=height, pixel_count=pixels)
    for name, expected in (("megapixels", pixels / 1_000_000), ("aspect_ratio", width / height)):
        number = _number(_field(metrics, name, path), f"{path}.{name}")
        if not math.isclose(number, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{path}.{name} does not match measured dimensions")
        metrics[name] = number
    for name, maximum in (("brightness_mean", 1.0), ("contrast_std", 0.5), ("saturation_mean", 1.0)):
        metrics[name] = _number(_field(metrics, name, path), f"{path}.{name}", maximum=maximum)
    for name, maximum in (("rgb_mean", 1.0), ("rgb_std", 0.5)):
        vector = _field(metrics, name, path)
        if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence) or len(vector) != 3:
            raise ValueError(f"{path}.{name} must contain three RGB channel values")
        metrics[name] = [_number(number, f"{path}.{name}[{index}]", maximum=maximum) for index, number in enumerate(vector)]
    return metrics


def _resolution_summaries(rows: list[dict], prefix: str) -> dict:
    dimensions = [(row[f"{prefix}_width"], row[f"{prefix}_height"]) for row in rows]
    return {
        "width": summarize_numbers(width for width, _ in dimensions),
        "height": summarize_numbers(height for _, height in dimensions),
        "pixel_count": summarize_numbers(width * height for width, height in dimensions),
        "megapixels": summarize_numbers(width * height / 1_000_000 for width, height in dimensions),
        "aspect_ratio": summarize_numbers(width / height for width, height in dimensions),
    }


def summarize_image_measurements(
    rows: Iterable[Mapping[str, Any]], *, class_names: Sequence[str]
) -> dict[str, Any]:
    """Validate and aggregate per-image measurement rows with equal image weighting."""
    if isinstance(class_names, (str, bytes)):
        raise ValueError("class_names must be an ordered sequence of names")
    names = tuple(class_names)
    for index, name in enumerate(names):
        _string(name, f"class_names[{index}]")
    if len(set(names)) != len(names):
        raise ValueError("class_names must be unique")
    measurements = []
    identities = set()
    source_rows = set()
    for index, source_row in enumerate(rows):
        path = f"measurements[{index}]"
        if not isinstance(source_row, Mapping):
            raise ValueError(f"{path} must be an object")
        row = deepcopy(dict(source_row))
        identity = _string(_field(row, "image_id", path), f"{path}.image_id")
        position = _integer(_field(row, "source_row", path), f"{path}.source_row")
        if identity in identities or position in source_rows:
            raise ValueError("image_id and source_row must each be unique")
        identities.add(identity)
        source_rows.add(position)
        row["source_row"] = position
        class_id = _field(row, "class_id", path)
        class_name = _field(row, "class_name", path)
        if class_id is None or class_name is None:
            if class_id is not None or class_name is not None:
                raise ValueError(f"{path} class_id/class_name must be paired null values")
        else:
            class_id = _integer(class_id, f"{path}.class_id")
            if class_id >= len(names) or class_name != names[class_id]:
                raise ValueError(f"{path} class_id/class_name must match the ordered class mapping")
        row["class_id"] = class_id
        status = _field(row, "status", path)
        if status not in _STATUSES:
            raise ValueError(f"{path}.status must be one of {_STATUSES}")
        if "source_status" in row:
            source_status = row["source_status"]
            expected = "ok" if status in {"ok", "unsupported_mode"} else status
            if source_status not in _SOURCE_STATUSES or source_status != expected:
                raise ValueError(f"{path}.source_status must retain the original source status independently of analysis exclusions")
        reason = _field(row, "reason", path)
        if reason is not None and not isinstance(reason, str):
            raise ValueError(f"{path}.reason must be a string or null")
        mode = _field(row, "original_mode", path)
        if mode is not None:
            _string(mode, f"{path}.original_mode")
        original = _dimensions(row, "original", path)
        oriented = _dimensions(row, "oriented", path)
        orientation = _field(row, "exif_orientation", path)
        if orientation is not None:
            if isinstance(orientation, bool) or not isinstance(orientation, Integral):
                raise ValueError(f"{path}.exif_orientation must be an integer or null")
            row["exif_orientation"] = int(orientation)
        for field in ("icc_profile_present", "has_transparency"):
            if field in row and row[field] is not None and not isinstance(row[field], bool):
                raise ValueError(f"{path}.{field} must be a boolean or null")
        if "metadata_errors" in row:
            errors = row["metadata_errors"]
            if isinstance(errors, (str, bytes)) or not isinstance(errors, Sequence):
                raise ValueError(f"{path}.metadata_errors must be a sequence of error strings")
            row["metadata_errors"] = [_string(error, f"{path}.metadata_errors") for error in errors]
        metrics = _field(row, "metrics", path)
        if status in {"ok", "unsupported_mode"} and (original is None or oriented is None or mode is None):
            raise ValueError(f"{path} decoded rows require known mode and original/oriented dimensions")
        if status == "ok":
            row["metrics"] = _metrics(metrics, oriented, f"{path}.metrics")
        elif metrics is not None:
            raise ValueError(f"{path}.metrics must be null for status={status}")
        try:
            json.dumps(row, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path} must contain JSON-safe finite values") from error
        measurements.append(row)

    measurements.sort(key=lambda row: row["source_row"])
    status_counts = {status: sum(row["status"] == status for row in measurements) for status in _STATUSES}
    source_statuses = Counter(
        row.get("source_status", "ok" if row["status"] == "unsupported_mode" else row["status"])
        for row in measurements
    )
    source_fact_counts = {
        fact: {
            "true": sum(row.get(fact) is True for row in measurements),
            "false": sum(row.get(fact) is False for row in measurements),
            "unknown": sum(row.get(fact) is None for row in measurements),
        }
        for fact in ("has_transparency", "icc_profile_present")
    }
    orientation_counts = Counter(
        str(row["exif_orientation"]) if row["exif_orientation"] is not None else "missing"
        for row in measurements
    )
    size_difference_count = sum(
        row["original_width"] is not None and row["oriented_width"] is not None
        and (row["original_width"], row["original_height"]) != (row["oriented_width"], row["oriented_height"])
        for row in measurements
    )
    exclusion_reasons = Counter(
        row["reason"] if row["reason"] is not None else "unknown"
        for row in measurements if row["status"] == "unsupported_mode"
    )
    mode_counts = Counter(row["original_mode"] if row["original_mode"] is not None else "unknown" for row in measurements)
    eligible = [row for row in measurements if row["status"] == "ok"]
    decoded = [row for row in measurements if row["status"] in {"ok", "unsupported_mode"}]
    original_known = [row for row in measurements if row["original_width"] is not None]
    oriented_known = [row for row in measurements if row["oriented_width"] is not None]
    unclassified = [row for row in measurements if row["class_id"] is None]
    metadata_error_counts = Counter(error for row in measurements for error in row.get("metadata_errors", ()))
    class_summaries = []
    for class_id, class_name in enumerate(names):
        selected = [row for row in measurements if row["class_id"] == class_id]
        selected_eligible = [row for row in selected if row["status"] == "ok"]
        class_summaries.append({
            "class_id": class_id,
            "class_name": class_name,
            "attempted_image_count": len(selected),
            "eligible_image_count": len(selected_eligible),
            "status_counts": {status: sum(row["status"] == status for row in selected) for status in _STATUSES},
            "metric_means": {
                metric: summarize_numbers(row["metrics"][metric] for row in selected_eligible)["mean"]
                for metric in _CLASS_METRICS
            },
        })
    return {
        "weighting": "equal_eligible_image",
        "attempted_image_count": len(measurements),
        "decoded_image_count": len(decoded),
        "eligible_image_count": len(eligible),
        "ineligible_image_count": len(measurements) - len(eligible),
        "row_error_count": status_counts["row_error"],
        "fetch_error_count": status_counts["fetch_error"],
        "decode_error_count": status_counts["decode_error"],
        "unsupported_mode_count": status_counts["unsupported_mode"],
        "status_counts": status_counts,
        "source_status_counts": {status: source_statuses[status] for status in _SOURCE_STATUSES},
        "source_fact_counts": source_fact_counts,
        "exif_orientation_counts": dict(sorted(orientation_counts.items())),
        "original_oriented_size_difference_count": size_difference_count,
        "color_exclusion_reason_counts": dict(sorted(exclusion_reasons.items())),
        "unclassified_image_count": len(unclassified),
        "unclassified_eligible_image_count": sum(row["status"] == "ok" for row in unclassified),
        "unclassified_status_counts": {status: sum(row["status"] == status for row in unclassified) for status in _STATUSES},
        "metadata_error_image_count": sum(bool(row.get("metadata_errors")) for row in measurements),
        "metadata_error_count": sum(metadata_error_counts.values()),
        "metadata_error_counts": dict(sorted(metadata_error_counts.items())),
        "original_mode_counts": dict(sorted(mode_counts.items())),
        "original_dimensions_available_count": len(original_known),
        "original_resolution": _resolution_summaries(original_known, "original"),
        "oriented_dimensions_available_count": len(oriented_known),
        "oriented_resolution": _resolution_summaries(oriented_known, "oriented"),
        "metric_summaries": {
            **{metric: summarize_numbers(row["metrics"][metric] for row in eligible) for metric in _SCALAR_METRICS},
            **{
                metric: {
                    channel: summarize_numbers(row["metrics"][metric][index] for row in eligible)
                    for index, channel in enumerate(_CHANNELS)
                }
                for metric in ("rgb_mean", "rgb_std")
            },
        },
        "class_summaries": class_summaries,
        "measurements": measurements,
    }


def _training_source(source):
    if source.spec.source_split != "train":
        raise ValueError("Image EDA accepts only the official train split")
    try:
        length = len(source.dataset)
    except (TypeError, AttributeError) as error:
        raise ValueError("Image EDA requires the shared indexed, nonstreaming source") from error
    if not callable(getattr(source.dataset, "__getitem__", None)):
        raise ValueError("Image EDA requires the shared indexed, nonstreaming source")
    return length


def _check_identity(view, source, source_row):
    expected = {
        "image_id": source.spec.image_id(source_row), "source_row": source_row,
        "source_repo": source.spec.repo_id, "source_revision": source.spec.revision,
        "source_split": "train",
    }
    if any(getattr(view, key) != value for key, value in expected.items()):
        raise ValueError("Shared image view identity does not match the requested training row")
    if view.status not in _SOURCE_STATUSES:
        raise ValueError("Unknown shared image status")


def _measurement_row(view):
    original = view.original_size or (None, None)
    oriented = view.oriented_size or (None, None)
    row = {
        "image_id": view.image_id, "source_repo": view.source_repo,
        "source_revision": view.source_revision, "source_split": view.source_split,
        "source_row": view.source_row, "source_id": view.source_id,
        "file_name": view.file_name, "class_id": view.class_id, "class_name": view.class_name,
        "source_status": view.status, "status": view.status,
        "reason": view.error_code, "error_code": view.error_code,
        "error_message": view.error_message, "metadata_errors": list(view.metadata_errors),
        "original_width": original[0], "original_height": original[1],
        "oriented_width": oriented[0], "oriented_height": oriented[1],
        "original_mode": view.original_mode, "image_format": view.image_format,
        "exif_orientation": view.exif_orientation, "has_transparency": view.has_transparency,
        "icc_profile_present": view.icc_profile_present, "metrics": None,
    }
    if view.status != "ok":
        return row
    if view.image is None or view.image.mode != "RGB" or view.image.size != view.oriented_size:
        raise ValueError("Successful shared view must expose native oriented RGB pixels")
    if view.has_transparency is not False:
        row.update(status="unsupported_mode", reason="transparency_metadata_present" if view.has_transparency else "unknown_transparency_policy")
    elif view.original_mode not in _OPAQUE_MODES:
        row.update(status="unsupported_mode", reason="unsupported_original_mode")
    else:
        row["metrics"] = measure_rgb_array(np.asarray(view.image))
    return row


def build_image_report(source, *, limit: int | None = None, expected_records: int = 5994,
                       reader: Callable | None = None, progress: Callable | None = None) -> dict:
    """Measure every requested training image via ``read_source_image`` without dropping failures."""
    size = _training_source(source)
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer")
    if isinstance(expected_records, bool) or not isinstance(expected_records, int) or expected_records < 1:
        raise ValueError("expected_records must be a positive integer")
    if reader is None:
        from src.data.image_io import read_source_image
        reader = read_source_image
    count = size if limit is None else min(size, limit)
    rows = []
    for source_row in range(count):
        view = reader(source, source_row)
        try:
            _check_identity(view, source, source_row)
            rows.append(_measurement_row(view))
        finally:
            if view.image is not None:
                view.image.close()
        if progress is not None:
            progress(source_row + 1, count)
    images = summarize_image_measurements(rows, class_names=source.class_names)
    full = limit is None
    matches = len(rows) == expected_records
    return {
        "schema_version": "image-eda-1.0", "source": asdict(source.spec),
        "runtime": _runtime_versions(), "class_names": list(source.class_names),
        "population": {
            "selection_method": "complete_official_training_split" if full else "source_order_prefix",
            "requested_limit": limit, "records_analyzed": len(rows), "indexed_source_size": size,
            "source_exhausted": full, "reference_record_count": expected_records,
            "matches_reference_count": matches, "full_split_verified": full and matches,
            "random_sample": False,
        },
        "methods": {
            "source_pixel_policy": "shared EXIF-corrected native RGB8; exif-rgb8-white:v1; no resize or ICC transform",
            "resolution": "original dimensions wherever known; oriented dimensions wherever known, including preserved failure headers",
            "color_eligibility": "shared status ok, no original transparency metadata, conventional opaque modes 1/L/P/RGB/CMYK/YCbCr",
            "rgb": "encoded RGB / 255; per-image means and population standard deviations using every pixel",
            "brightness": "mean Y = .299 R + .587 G + .114 B on encoded RGB",
            "contrast": "population standard deviation of Y",
            "saturation": "mean HSV S = (max-min)/max, zero for black pixels",
            "weighting": "one equal-weight eligible image; these are not pooled-pixel normalization estimates",
            "quantiles": "linear interpolation at (n - 1) * p",
        },
        "images": images,
        "limitations": [
            "Raw source rows, including duplicates, remain in the descriptive population.",
            "Decoded does not mean clean or classification-eligible; source metadata errors remain independent.",
            "RGB conversion is not ICC color management; color statistics include backgrounds.",
            "Transparency metadata excludes color measurement even if an alpha image might be fully opaque.",
            "Extremes are examples, not automatic quality failures or preprocessing thresholds.",
            "No derived-partition filtering is applied. A complete scan of the pinned CUB "
            "official training set covers all 5,994 raw images, including the 1,200 "
            "derived-validation members of classification-v1 (seed 42, fraction 0.20); "
            "limited scans cover only their requested raw-source prefix. "
            "No official test data was analyzed and no preprocessing parameters were fitted.",
            "Future fitted preprocessing must use the eventual derived training partition.",
        ],
    }


def prepare_gallery(report, source):
    """Attach source captions to selected gallery rows and return a row->RGB lookup."""
    source_size = _training_source(source)
    if asdict(source.spec) != report["source"]:
        raise ValueError("Gallery source must match the measured source revision")
    if source_size != report["population"]["indexed_source_size"]:
        raise ValueError("Gallery source size must match the saved report")
    if list(source.class_names) != report["class_names"]:
        raise ValueError("Gallery class mapping must match the saved report")
    from src.data.image_io import read_source_image
    from src.data.source import parse_captions
    from .image_plots import select_extreme_rows, select_representative_rows
    selected = select_representative_rows(report)
    selected += [row for _, _, row in select_extreme_rows(report)]
    by_id = {row["image_id"]: row for row in selected}
    pixels = {}
    for identity, row in by_id.items():
        view = read_source_image(source, row["source_row"])
        try:
            _check_identity(view, source, row["source_row"])
            if view.image_id != identity:
                raise ValueError("Gallery image identity must match the saved measurement")
            measured = _measurement_row(view)
            if measured["status"] != "ok" or measured["metrics"] != row["metrics"]:
                raise ValueError("Gallery pixels no longer match the saved image measurements")
            if (view.class_id, view.class_name) != (row["class_id"], row["class_name"]):
                raise ValueError("Gallery class metadata no longer matches the saved report")
            pixels[identity] = np.asarray(view.image).copy()
        finally:
            if view.image is not None:
                view.image.close()
        raw = source.dataset[row["source_row"]]
        description = raw.get("description")
        captions = parse_captions(description, image_id=identity) if isinstance(description, str) else ()
        selected_caption = next((caption for caption in captions if caption.clean_caption), None)
        row["caption_text"] = selected_caption.clean_caption if selected_caption is not None else None
        row["caption_index"] = selected_caption.caption_index if selected_caption is not None else None
    return lambda row: pixels[row["image_id"]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--expected-records", type=int, default=5994)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--figures-dir", type=Path)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.expected_records < 1:
        parser.error("--expected-records must be positive")
    from src.data.source import load_cub_split
    source = load_cub_split("train", streaming=False)
    def progress(done, total):
        if done % 500 == 0 or done == total:
            print(f"Measured {done:,}/{total:,} attempted training images", flush=True)
    report = build_image_report(source, limit=args.limit, expected_records=args.expected_records, progress=progress)
    name = "train-images-full.json" if args.limit is None else f"train-images-prefix-{args.limit}.json"
    output = args.output or Path("outputs/eda/reports") / name
    saved = write_json_report(report, output)
    print(f"Saved image report to {saved}", flush=True)
    if args.figures_dir is not None:
        from .image_plots import generate_image_figures
        getter = prepare_gallery(report, source)
        write_json_report(report, output)  # retain gallery captions in JSON
        paths = generate_image_figures(report, args.figures_dir, image_getter=getter)
        print(f"Saved {len(paths)} image figure groups as PNG/PDF", flush=True)
    print(f"Eligible color images: {report['images']['eligible_image_count']:,}/{report['images']['attempted_image_count']:,}")


if __name__ == "__main__":
    main()
