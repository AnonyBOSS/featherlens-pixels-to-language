"""Training image EDA figures from saved measurements and supplied gallery pixels.

Geometry includes every recorded original and oriented dimension pair, including
failure rows and modes ineligible for RGB statistics. Appearance plots
give each eligible image one observation; they never pool pixels or fit a policy.
Gallery callbacks supply actual native oriented RGB views. This module does not
decode files, convert modes, resize measurement arrays, or assign quality flags.
"""

from collections.abc import Callable, Mapping, Sequence
import math
from pathlib import Path
import textwrap
from typing import Any

import numpy as np

from src.data.schemas import SourceSpec

from .eda_plots import _BLUE, _GOLD, _INK, _MUTED, _STYLE, _TEAL, _count_axis, _empty, _pyplot
from .figure_export import save_figure_bundle


_STATUSES = {"ok", "unsupported_mode", "row_error", "fetch_error", "decode_error"}
_APPEARANCE = (
    ("brightness_mean", "Mean encoded luma Y", "Per-image mean Y (0–1)", (0, 1)),
    ("contrast_std", "Contrast: encoded luma standard deviation", "Per-image population std(Y) (0–0.5)", (0, 0.5)),
    ("saturation_mean", "Mean HSV saturation", "Per-image mean HSV S (0–1)", (0, 1)),
)


def _count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _number(value: Any, label: str, *, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite nonnegative number")
    if not math.isfinite(value) or value < 0 or (maximum is not None and value > maximum + 1e-12):
        raise ValueError(f"{label} is outside its measurement range")
    return float(value)


def _rows(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Validate plotting dependencies before calling the shared export helper."""
    try:
        if report["schema_version"] != "image-eda-1.0":
            raise ValueError("Image figures require schema_version 'image-eda-1.0'")
        source = report["source"]
        if source["source_split"] != "train":
            raise ValueError("Image figures accept only the official training source")
        if any(not isinstance(source[key], str) or not source[key].strip() for key in ("repo_id", "revision")):
            raise ValueError("Image figures require source repository and revision provenance")
        spec = SourceSpec(source["repo_id"], source["source_split"], source["revision"])
        class_names = report["class_names"]
        if (
            isinstance(class_names, (str, bytes))
            or not isinstance(class_names, Sequence)
            or any(not isinstance(name, str) or not name.strip() for name in class_names)
            or len(set(class_names)) != len(class_names)
        ):
            raise ValueError("class_names must be an ordered sequence of unique nonempty names")
        population = report["population"]
        selection = population["selection_method"]
        if selection not in {"complete_official_training_split", "source_order_prefix"}:
            raise ValueError("Unsupported image-analysis selection_method")
        if not isinstance(population["full_split_verified"], bool):
            raise ValueError("full_split_verified must be a boolean")
        images = report["images"]
        attempted = _count(images["attempted_image_count"], "attempted_image_count")
        if _count(population["records_analyzed"], "records_analyzed") != attempted:
            raise ValueError("Attempted images must equal records_analyzed")
        indexed_size = _count(population["indexed_source_size"], "indexed_source_size")
        if population["random_sample"] is not False:
            raise ValueError("The declared image selection must have random_sample=False")
        if selection == "complete_official_training_split":
            reference = _count(population["reference_record_count"], "reference_record_count")
            matches = population["matches_reference_count"]
            if (
                population["source_exhausted"] is not True
                or population["requested_limit"] is not None
                or reference < 1
                or not isinstance(matches, bool)
                or matches != (attempted == reference)
                or population["full_split_verified"] != matches
                or attempted != indexed_size
            ):
                raise ValueError("Complete image analysis requires exhaustion, indexed source coverage, and consistent reference counts")
        else:
            limit = _count(population["requested_limit"], "requested_limit")
            if limit < 1 or attempted != min(limit, indexed_size) or population["full_split_verified"]:
                raise ValueError("Prefix image analysis requires exact indexed prefix coverage and unverified scope")
        rows = images["measurements"]
        if not isinstance(rows, list) or len(rows) != attempted:
            raise ValueError("Saved measurement rows must equal attempted_image_count")
        identities = set()
        source_rows = set()
        status_counts = dict.fromkeys(_STATUSES, 0)
        original_count = 0
        oriented_count = 0
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("Each measurement must be an object")
            identity = row["image_id"]
            if not isinstance(identity, str) or not identity or identity in identities:
                raise ValueError("Image IDs must be nonempty and unique")
            identities.add(identity)
            source_row = _count(row["source_row"], "source_row")
            if source_row in source_rows:
                raise ValueError("Source rows must be unique")
            source_rows.add(source_row)
            if (
                row["source_repo"] != spec.repo_id
                or row["source_revision"] != spec.revision
                or row["source_split"] != spec.source_split
            ):
                raise ValueError("Measurement source provenance must match the declared training source")
            if identity != spec.image_id(source_row):
                raise ValueError("Image IDs must match the canonical source identity and row")
            if row["class_id"] is None or row["class_name"] is None:
                if row["class_id"] is not None or row["class_name"] is not None:
                    raise ValueError("class_id/class_name must be both known or both null")
            else:
                class_id = _count(row["class_id"], "class_id")
                if class_id >= len(class_names) or row["class_name"] != class_names[class_id]:
                    raise ValueError("class_id/class_name must match the report's ordered class mapping")
            status = row["status"]
            if status not in _STATUSES:
                raise ValueError("Unknown image measurement status")
            status_counts[status] += 1
            for prefix in ("original", "oriented"):
                width, height = row[f"{prefix}_width"], row[f"{prefix}_height"]
                if width is None and height is None:
                    if status in {"ok", "unsupported_mode"}:
                        raise ValueError("Decoded images require original and oriented dimensions")
                    continue
                if width is None or height is None or _count(width, "width") < 1 or _count(height, "height") < 1:
                    raise ValueError("Dimension pairs must be positive integers or both null")
                if prefix == "original":
                    original_count += 1
                else:
                    oriented_count += 1
            if status == "ok":
                metrics = row["metrics"]
                if not isinstance(metrics, Mapping):
                    raise ValueError("Eligible images require saved RGB measurements")
                if _count(metrics["width"], "metrics.width") < 1 or _count(metrics["height"], "metrics.height") < 1:
                    raise ValueError("Metric dimensions must be positive")
                if (metrics["width"], metrics["height"]) != (row["oriented_width"], row["oriented_height"]):
                    raise ValueError("Metric dimensions must match the oriented view")
                for key, maximum in (("brightness_mean", 1), ("contrast_std", 0.5), ("saturation_mean", 1)):
                    _number(metrics[key], key, maximum=maximum)
                means = metrics["rgb_mean"]
                if not isinstance(means, list) or len(means) != 3:
                    raise ValueError("rgb_mean must contain three saved channel means")
                for mean in means:
                    _number(mean, "rgb_mean", maximum=1)
            elif row["metrics"] is not None:
                raise ValueError("Ineligible images must not have RGB measurements")
            if row.get("caption_text") is not None and not isinstance(row["caption_text"], str):
                raise ValueError("caption_text must be a string or null")
        if source_rows != set(range(attempted)):
            raise ValueError("Source rows must cover the exact indexed source-order range")
        expected = {
            "eligible_image_count": status_counts["ok"],
            "decoded_image_count": status_counts["ok"] + status_counts["unsupported_mode"],
            "decode_error_count": status_counts["decode_error"],
            "row_error_count": status_counts["row_error"],
            "fetch_error_count": status_counts["fetch_error"],
            "unsupported_mode_count": status_counts["unsupported_mode"],
            "original_dimensions_available_count": original_count,
            "oriented_dimensions_available_count": oriented_count,
        }
        for key, expected_count in expected.items():
            if _count(images[key], key) != expected_count:
                raise ValueError(f"{key} disagrees with saved measurement rows")
        return sorted(rows, key=lambda row: (row["source_row"], row["image_id"]))
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing or malformed image plotting field: {error}") from error


def _title(figure, report, title: str, denominator: str) -> None:
    population = report["population"]
    if population["full_split_verified"]:
        scope = "complete split"
    elif population["selection_method"] == "source_order_prefix":
        scope = "nonrandom source-order prefix"
    else:
        scope = "complete iterator; reference count unverified"
    figure.suptitle(
        f"{title}\nOfficial TRAINING | {scope}\n{denominator}",
        fontsize=13, color=_INK,
    )


def _eligible(report):
    return [row for row in _rows(report) if row["status"] == "ok"]


def _pixel_denominator(report) -> str:
    images = report["images"]
    failures = images["row_error_count"] + images["fetch_error_count"] + images["decode_error_count"]
    return (
        f"{images['eligible_image_count']:,} eligible / {images['attempted_image_count']:,} attempted images; "
        f"{images['unsupported_mode_count']:,} mode/transparency exclusions; {failures:,} row/fetch/decode failures"
    )


def create_image_dimensions_figure(report):
    """Use all known original/oriented geometry, independently of RGB eligibility."""
    rows = _rows(report)
    original = [row for row in rows if row["original_width"] is not None]
    oriented = [row for row in rows if row["oriented_width"] is not None]
    plt = _pyplot()
    with plt.rc_context(_STYLE):
        figure, axes = plt.subplots(1, 3, figsize=(16, 5.5), layout="constrained")
        try:
            axes[0].scatter([row["original_width"] for row in original], [row["original_height"] for row in original], s=13, alpha=0.35, color=_BLUE)
            axes[0].set(title=f"Original geometry (n={len(original):,})", xlabel="Original width (pixels)", ylabel="Original height (pixels)")
            axes[0].set_xlim(left=0)
            axes[0].set_ylim(bottom=0)
            axes[0].grid(alpha=0.7)
            for axis, values, title, xlabel, color in (
                (axes[1], [row["oriented_width"] / row["oriented_height"] for row in oriented], "Oriented-view aspect ratio", "Oriented width / height", _TEAL),
                (axes[2], [row["oriented_width"] * row["oriented_height"] / 1_000_000 for row in oriented], "Native resolution", "Megapixels (1 MP = 1,000,000 pixels)", _GOLD),
            ):
                axis.hist(values, bins=30, color=color)
                axis.set(title=f"{title} (n={len(oriented):,})", xlabel=xlabel, ylabel="Images with oriented dimensions")
                _count_axis(axis)
                if not values:
                    _empty(axis, "No oriented dimensions available")
            if not original:
                _empty(axes[0], "No original dimensions available")
            _title(figure, report, "Image dimensions and resolution", f"{len(original):,} original dimension pairs; {len(oriented):,} oriented dimension pairs / {report['images']['attempted_image_count']:,} attempted")
            figure.supxlabel("All known geometry is retained, including mode exclusions and failure rows with recorded dimensions; no resizing.", fontsize=9, color=_MUTED)
            return figure
        except BaseException:
            plt.close(figure)
            raise


def create_image_appearance_figure(report):
    """Plot one saved scalar per eligible image, rather than pooled pixels."""
    rows = _eligible(report)
    plt = _pyplot()
    with plt.rc_context(_STYLE):
        figure, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True, layout="constrained")
        try:
            for axis, (metric, title, xlabel, limits), color in zip(axes, _APPEARANCE, (_BLUE, _TEAL, _GOLD)):
                values = [row["metrics"][metric] for row in rows]
                axis.hist(values, bins=30, range=limits, color=color)
                axis.set(title=title, xlabel=xlabel, xlim=limits, ylabel="Eligible image count")
                _count_axis(axis)
                if not values:
                    _empty(axis, "No eligible RGB measurements")
            _title(figure, report, "Image appearance distributions", _pixel_denominator(report))
            figure.supxlabel("Encoded RGB: Y = .299R + .587G + .114B; population std(Y); HSV S. Each image has equal weight; no invalidity thresholds.", fontsize=9, color=_MUTED)
            return figure
        except BaseException:
            plt.close(figure)
            raise


def create_image_rgb_figure(report):
    """Show native-image channel means on matched 0–1 channel axes."""
    rows = _eligible(report)
    plt = _pyplot()
    with plt.rc_context(_STYLE):
        figure, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True, layout="constrained")
        try:
            for index, (axis, channel, color) in enumerate(zip(axes, ("Red", "Green", "Blue"), ("#bd4147", "#378b62", _BLUE))):
                values = [row["metrics"]["rgb_mean"][index] for row in rows]
                axis.hist(values, bins=30, range=(0, 1), color=color)
                axis.set(title=f"{channel} channel", xlabel="Per-image mean encoded channel value (0–1)", xlim=(0, 1), ylabel="Eligible image count")
                _count_axis(axis)
                if not values:
                    _empty(axis, "No eligible RGB measurements")
            _title(figure, report, "Per-image RGB means", _pixel_denominator(report))
            figure.supxlabel("Each channel mean uses every pixel in one native oriented RGB view. The histogram counts images; backgrounds also contribute.", fontsize=9, color=_MUTED)
            return figure
        except BaseException:
            plt.close(figure)
            raise


def select_representative_rows(report, *, count: int = 6) -> list[Mapping]:
    """Choose first source rows from evenly spaced observed class IDs.

    These illustrative examples are deterministic, not a random or statistically
    representative sample. At most one image per selected class group is returned;
    paired-null class metadata forms an explicitly unassigned final group.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Gallery count must be a positive integer")
    first_by_class = {}
    for row in _eligible(report):
        first_by_class.setdefault(row["class_id"], row)
    classes = sorted(first_by_class, key=lambda value: (value is None, value if value is not None else 0))
    selected_count = min(count, len(classes))
    if not selected_count:
        return []
    positions = [0] if selected_count == 1 else [round(index * (len(classes) - 1) / (selected_count - 1)) for index in range(selected_count)]
    return [first_by_class[classes[position]] for position in positions]


def select_extreme_rows(report) -> list[tuple[str, str, Mapping]]:
    """Select low/high appearance extrema; ties use source row and identity."""
    rows = _eligible(report)
    if not rows:
        return []
    selected = []
    for metric, title, _, _ in _APPEARANCE:
        low = min(rows, key=lambda row: (row["metrics"][metric], row["source_row"], row["image_id"]))
        high = min(rows, key=lambda row: (-row["metrics"][metric], row["source_row"], row["image_id"]))
        selected.extend(((f"Lowest {title.lower()}", metric, low), (f"Highest {title.lower()}", metric, high)))
    return selected


def _gallery(report, selections, image_getter: Callable[[Mapping], np.ndarray], *, extremes: bool):
    plt = _pyplot()
    columns = min(3, max(1, len(selections)))
    row_count = max(1, math.ceil(len(selections) / columns))
    with plt.rc_context(_STYLE):
        figure, axes = plt.subplots(row_count, columns, figsize=(5 * columns, 6 * row_count), squeeze=False, layout="constrained")
        try:
            for axis in axes.flat:
                axis.set_axis_off()
            for axis, (label, metric, row) in zip(axes.flat, selections):
                rgb = image_getter(row)
                shape = (row["metrics"]["height"], row["metrics"]["width"], 3)
                if not isinstance(rgb, np.ndarray) or rgb.shape != shape:
                    raise ValueError("Gallery callback must supply the saved native oriented RGB shape")
                if rgb.dtype != np.uint8 and not np.issubdtype(rgb.dtype, np.floating):
                    raise ValueError("Gallery RGB arrays must be uint8 or normalized floating point")
                if np.issubdtype(rgb.dtype, np.floating) and (not np.isfinite(rgb).all() or (rgb < 0).any() or (rgb > 1).any()):
                    raise ValueError("Floating gallery RGB values must be finite and in [0, 1]")
                axis.imshow(rgb)
                title = f"{row['class_name'] or 'Unassigned class'} | source row {row['source_row']}"
                if metric:
                    title = f"{label}: {row['metrics'][metric]:.4f}\n{title}"
                axis.set_title(textwrap.fill(title, 58, replace_whitespace=False), fontsize=10, color=_INK)
                caption = row.get("caption_text") or "No nonempty reference caption available."
                axis.text(0.5, -0.045, textwrap.fill(caption, 60), transform=axis.transAxes, va="top", ha="center", fontsize=8, color=_MUTED)
            if not selections:
                _empty(axes.flat[0], "No eligible images for this gallery")
            title = "Descriptive appearance extrema" if extremes else "Selected bird images and reference captions"
            _title(figure, report, title, _pixel_denominator(report))
            note = (
                "Low/high saved scalar values; ties use source order. Images may recur. These examples are not quality flags."
                if extremes else "Illustrative selection: evenly spaced observed class groups (unassigned last), first eligible source row per group; not a random sample."
            )
            figure.supxlabel(note, fontsize=9, color=_MUTED)
            return figure
        except BaseException:
            plt.close(figure)
            raise


def create_image_gallery_figure(report, image_getter: Callable[[Mapping], np.ndarray]):
    selections = [("", None, row) for row in select_representative_rows(report)]
    return _gallery(report, selections, image_getter, extremes=False)


def create_image_extremes_figure(report, image_getter: Callable[[Mapping], np.ndarray]):
    return _gallery(report, select_extreme_rows(report), image_getter, extremes=True)


def generate_image_figures(report, output_dir: str | Path, *, image_getter: Callable[[Mapping], np.ndarray]) -> list[Path]:
    """Publish three statistical groups and two actual-pixel galleries together."""
    _rows(report)  # Validate every plotting field before the export helper writes.
    if not callable(image_getter):
        raise ValueError("image_getter must supply actual source RGB arrays")
    return save_figure_bundle(
        (
            ("training_image_dimensions", lambda: create_image_dimensions_figure(report)),
            ("training_image_appearance", lambda: create_image_appearance_figure(report)),
            ("training_image_rgb", lambda: create_image_rgb_figure(report)),
            ("training_image_gallery", lambda: create_image_gallery_figure(report, image_getter)),
            ("training_image_extremes", lambda: create_image_extremes_figure(report, image_getter)),
        ),
        output_dir,
    )
