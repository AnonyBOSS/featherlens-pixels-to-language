"""Training-image EDA through the shared native-image access contract.

The shared adapter owns fetching, decoding, EXIF correction and RGB conversion.
This consumer measures its pixels and records analysis eligibility explicitly.
No test source, cleaning, duplicate audit, fitted preprocessing, or model is used.
"""

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

from .explore import _runtime_versions
from .image_stats import measure_rgb_array
from .image_summary import summarize_image_measurements
from .reporting import write_json_report


_OPAQUE_MODES = {"1", "L", "P", "RGB", "CMYK", "YCbCr"}
_SOURCE_STATUSES = {"ok", "row_error", "fetch_error", "decode_error"}


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
    """Visit every requested raw row without dropping failures or missing classes.

    The default exhausts the indexed official training source. A supplied limit
    produces an explicitly nonrandom prefix, even when it reaches the end.
    Unexpected adapter/programming failures propagate rather than certify a scan.
    """
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
            "No official test data or derived validation partition was analyzed.",
            "Future fitted preprocessing must use the eventual derived training partition.",
        ],
    }


def prepare_gallery(report, source):
    """Add source captions and return a bounded getter for selected actual pixels."""
    source_size = _training_source(source)
    if asdict(source.spec) != report["source"]:
        raise ValueError("Gallery source must match the measured source revision")
    if source_size != report["population"]["indexed_source_size"]:
        raise ValueError("Gallery source size must match the saved report")
    if list(source.class_names) != report["class_names"]:
        raise ValueError("Gallery class mapping must match the saved report")
    from src.data.captions import parse_captions
    from src.data.image_io import read_source_image
    from src.visualization.image_plots import select_extreme_rows, select_representative_rows
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
        from src.visualization.image_plots import generate_image_figures
        getter = prepare_gallery(report, source)
        write_json_report(report, output)  # Retain source-linked gallery captions.
        paths = generate_image_figures(report, args.figures_dir, image_getter=getter)
        print(f"Saved {len(paths)} image figure groups as PNG/PDF", flush=True)
    print(f"Eligible color images: {report['images']['eligible_image_count']:,}/{report['images']['attempted_image_count']:,}")


if __name__ == "__main__":
    main()
