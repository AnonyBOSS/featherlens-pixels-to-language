"""Pure image EDA aggregation over caller-supplied measurement records.

Every attempted identity remains in the saved rows and status denominators.
Original and oriented geometry each use all known dimension pairs, including
facts retained on failures. Pixel summaries use status=ok only,
giving each eligible image one equal-weight observation. In particular, mean
per-image RGB means/stds are descriptive values, not pooled-pixel normalization
parameters. The caller owns decoding, orientation, conversion, and eligibility.
"""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
import json
import math
from numbers import Integral, Real
from typing import Any

from .reporting import summarize_numbers


_STATUSES = ("ok", "row_error", "fetch_error", "decode_error", "unsupported_mode")
_SOURCE_STATUSES = ("ok", "row_error", "fetch_error", "decode_error")
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
    """Validate, retain, and aggregate independent per-image measurements.

    Repeated image IDs or source rows raise instead of being silently dropped.
    Failure rows require metrics=None. Decoded rows retain original/oriented
    dimension pairs even when their modes are ineligible for pixel analysis.
    Null class identity is paired and retained globally, including valid pixels;
    unclassified rows are counted explicitly outside known per-class groups.
    Source status and metadata errors retain their independent meanings.
    Raw EXIF orientation integers are preserved without imposing a tag policy.
    Unknown fields are retained if JSON-safe; source assets are never opened.
    """
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
