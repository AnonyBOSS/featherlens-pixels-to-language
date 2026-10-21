"""Read-only, row-level integrity audit for the pinned CUB sources.

The outputs are observations keyed by Stage 1 image IDs, not cleaning decisions.
Every accessible source row is retained even when its metadata or image is invalid.
"""

from collections import Counter, defaultdict
from collections.abc import Mapping
from hashlib import sha256
import json
from pathlib import Path
import time
import unicodedata
from typing import Any

import datasets
import PIL

from .captions import parse_captions
from .image_integrity import ImageInspection, inspect_image
from .source import load_cub_split


EXPECTED_COUNTS = {"train": 5994, "test": 5794}
_REQUIRED_FIELDS = ("image", "description", "label", "file_name", "id")
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_PIXEL_HASH_ALGORITHM = "exif-rgb8-white:v1"


def _error(report: dict[str, Any], flag: str, message: str | None = None) -> None:
    if flag not in report["error_flags"]:
        report["error_flags"].append(flag)
    if message is not None:
        report["error_messages"][flag] = message


def _warning(report: dict[str, Any], flag: str) -> None:
    if flag not in report["warning_flags"]:
        report["warning_flags"].append(flag)


def _base_image(source: Any, source_row: int) -> dict[str, Any]:
    spec = source.spec
    return {
        "image_id": spec.image_id(source_row),
        "source_repo": spec.repo_id,
        "source_revision": spec.revision,
        "source_split": spec.source_split,
        "source_row": source_row,
        "source_id": None,
        "file_name": None,
        "class_id": None,
        "class_name": None,
        "metadata_valid": False,
        "image_valid": False,
        "image_status": "not_checked",
        "image_error_code": None,
        "image_format": None,
        "image_mode": None,
        "width": None,
        "height": None,
        "encoded_sha256": None,
        "pixel_sha256": None,
        "caption_count": None,
        "usable_caption_count": 0,
        "caption_structural_valid": False,
        "exact_group_ids": [],
        "error_flags": [],
        "warning_flags": [],
        "error_messages": {},
        "audit_status": "invalid",
    }


def _caption_is_usable(raw: str, clean: str) -> tuple[bool, list[str]]:
    flags: list[str] = []
    if not clean:
        flags.append("caption.empty")
    # Tabs are allowed because Stage 1 normalizes them without losing raw text.
    if any(unicodedata.category(character) == "Cc" and character != "\t" for character in raw):
        flags.append("caption.control_character")
    return not flags, flags


def _inspect_row(
    source: Any,
    source_row: int,
    row: Any,
    image_inspector: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = _base_image(source, source_row)
    caption_reports: list[dict[str, Any]] = []
    if not isinstance(row, Mapping):
        _error(report, "metadata.invalid_row", f"Expected a mapping, got {type(row).__name__}")
        return report, caption_reports

    for field in _REQUIRED_FIELDS:
        if field not in row:
            _error(report, f"metadata.missing_{field}")

    label = row.get("label")
    if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(source.class_names):
        _error(report, "metadata.invalid_label", repr(label))
    else:
        report["class_id"] = label
        report["class_name"] = source.class_names[label]

    source_id = row.get("id")
    if isinstance(source_id, bool) or not isinstance(source_id, int):
        _error(report, "metadata.invalid_source_id", repr(source_id))
    else:
        report["source_id"] = source_id
        if source_id < 0:
            _warning(report, "metadata.negative_source_id")

    file_name = row.get("file_name")
    if not isinstance(file_name, str) or not file_name.strip():
        _error(report, "metadata.invalid_filename", repr(file_name))
    else:
        report["file_name"] = file_name
        if "/" in file_name or "\\" in file_name or "\x00" in file_name:
            _warning(report, "metadata.suspicious_filename")
        if Path(file_name).suffix.lower() not in _IMAGE_EXTENSIONS:
            _warning(report, "metadata.unusual_extension")

    description = row.get("description")
    if description is None:
        _error(report, "caption.missing_description")
    elif not isinstance(description, str):
        _error(report, "caption.invalid_description", f"Got {type(description).__name__}")
    else:
        captions = parse_captions(description, image_id=report["image_id"])
        report["caption_count"] = len(captions)
        if len(captions) != 10:
            _warning(report, "caption.count_mismatch")
        for caption in captions:
            usable, flags = _caption_is_usable(caption.raw_caption, caption.clean_caption)
            caption_reports.append(
                {
                    "image_id": caption.image_id,
                    "caption_index": caption.caption_index,
                    "raw_caption": caption.raw_caption,
                    "clean_caption": caption.clean_caption,
                    "usable": usable,
                    "error_flags": flags,
                    "warning_flags": [],
                }
            )
            if not usable:
                for flag in flags:
                    _error(report, flag)
            else:
                report["usable_caption_count"] += 1
        report["caption_structural_valid"] = len(captions) == 10 and all(
            caption["usable"] for caption in caption_reports
        )

    reference = row.get("image")
    if not isinstance(reference, Mapping) or not (reference.get("bytes") is not None or reference.get("path")):
        _error(report, "metadata.invalid_image_reference")
    try:
        image: ImageInspection = image_inspector(reference)
    except Exception as error:
        report["image_status"] = "decode_error"
        _error(report, "image.decode_failed", f"Unexpected inspector error: {type(error).__name__}: {error}")
    else:
        report["image_status"] = image.status
        report["image_valid"] = image.status == "ok"
        report["image_error_code"] = image.error_code
        report["image_format"] = image.format
        report["image_mode"] = image.mode
        report["width"] = image.width
        report["height"] = image.height
        report["encoded_sha256"] = image.encoded_sha256
        report["pixel_sha256"] = image.pixel_sha256
        if image.status == "fetch_error":
            _error(report, "image.fetch_failed", image.error_message)
        elif image.status == "decode_error":
            _error(report, "image.decode_failed", image.error_message)

    report["metadata_valid"] = not any(flag.startswith("metadata.") for flag in report["error_flags"])
    return report, caption_reports


def _record_duplicate_values(reports: list[dict[str, Any]]) -> None:
    for split in {report["source_split"] for report in reports}:
        members = [report for report in reports if report["source_split"] == split]
        for field, flag in (("source_id", "metadata.duplicate_source_id"), ("file_name", "metadata.duplicate_filename")):
            by_value: dict[Any, list[dict[str, Any]]] = defaultdict(list)
            for report in members:
                if report[field] is not None:
                    by_value[report[field]].append(report)
            for matching in by_value.values():
                if len(matching) > 1:
                    for report in matching:
                        _warning(report, flag)


def _exact_groups(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parent = list(range(len(reports)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    duplicate_hashes: list[tuple[str, str, int]] = []
    for field in ("encoded_sha256", "pixel_sha256"):
        values: dict[str, list[int]] = defaultdict(list)
        for index, report in enumerate(reports):
            if report[field] is not None:
                values[report[field]].append(index)
        for digest, matching in values.items():
            if len(matching) > 1:
                duplicate_hashes.append((field, digest, matching[0]))
                for index in matching[1:]:
                    union(matching[0], index)

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(reports)):
        components[find(index)].append(index)
    evidence_by_root: dict[int, list[dict[str, str]]] = defaultdict(list)
    for field, digest, first_member in duplicate_hashes:
        evidence_by_root[find(first_member)].append({"kind": field, "sha256": digest})

    groups: list[dict[str, Any]] = []
    for root, members in components.items():
        if len(members) < 2:
            continue
        image_ids = sorted(reports[index]["image_id"] for index in members)
        group_id = "exact:" + sha256("\n".join(image_ids).encode("utf-8")).hexdigest()
        evidence = sorted(
            evidence_by_root[root], key=lambda item: (item["kind"], item["sha256"])
        )
        kinds = {item["kind"] for item in evidence}
        splits = sorted({reports[index]["source_split"] for index in members})
        labels = sorted({reports[index]["class_id"] for index in members if reports[index]["class_id"] is not None})
        crosses_split = "train" in splits and "test" in splits
        label_conflict = len(labels) > 1
        for index in members:
            report = reports[index]
            report["exact_group_ids"] = [group_id]
            _warning(report, "duplicate.exact")
            if crosses_split:
                _warning(report, "leakage.exact_cross_split")
            if label_conflict:
                _warning(report, "duplicate.label_conflict")
        groups.append(
            {
                "group_id": group_id,
                "image_ids": image_ids,
                "hash_kind": next(iter(kinds)) if len(kinds) == 1 else "mixed",
                "hash_evidence": evidence,
                "source_splits": splits,
                "class_ids": labels,
                "crosses_official_split": crosses_split,
                "label_conflict": label_conflict,
            }
        )
    return sorted(groups, key=lambda group: group["group_id"])


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _mark_incomplete(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_summary(
        output_dir / "summary.json",
        {
            "schema_version": 1,
            "run_status": "incomplete",
            "coverage_complete": False,
            "fatal_errors": ["Audit did not finish publishing all manifests"],
        },
    )


def _write_caption_records(handle: Any, records: list[dict[str, Any]]) -> tuple[int, int]:
    for record in records:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return len(records), sum(record["usable"] for record in records)


def audit_sources(
    sources: Mapping[str, Any],
    output_dir: str | Path,
    *,
    expected_counts: Mapping[str, int],
    image_inspector: Any = inspect_image,
    initial_fatal_errors: list[str] | None = None,
) -> dict[str, Any]:
    """Audit source rows and write Stage 3-ready, immutable observations.

    An indexed dataset is read by source row so a failed row can be reported
    and later rows can still be inspected. Iterable sources are supported as a
    fallback, but an iterator failure is fatal for remaining-row coverage.
    """
    started = time.perf_counter()
    output_dir = Path(output_dir)
    _mark_incomplete(output_dir)
    reports: list[dict[str, Any]] = []
    caption_count = 0
    usable_caption_count = 0
    processed_counts = {split: 0 for split in expected_counts}
    observed_counts: dict[str, int | None] = {split: None for split in expected_counts}
    row_read_errors = {split: 0 for split in expected_counts}
    row_audit_errors = {split: 0 for split in expected_counts}
    fatal_errors: list[str] = list(initial_fatal_errors or [])
    caption_temporary = output_dir / "captions.jsonl.tmp"

    with caption_temporary.open("w", encoding="utf-8", newline="\n") as caption_handle:
        for split in expected_counts:
            source = sources.get(split)
            if source is None:
                fatal_errors.append(f"{split}: source could not be loaded")
                continue
            if source.spec.source_split != split:
                fatal_errors.append(f"{split}: source specification declares {source.spec.source_split!r}")
            dataset = source.dataset
            try:
                row_count = len(dataset)
            except Exception:
                # A readable iterator may survive an unavailable length/index.
                row_count = None
            observed_counts[split] = row_count
            if row_count is not None and hasattr(dataset, "__getitem__"):
                for source_row in range(row_count):
                    try:
                        row = dataset[source_row]
                    except Exception as error:
                        report = _base_image(source, source_row)
                        _error(report, "coverage.row_read_failed", f"{type(error).__name__}: {error}")
                        reports.append(report)
                        row_read_errors[split] += 1
                    else:
                        try:
                            report, captions = _inspect_row(source, source_row, row, image_inspector)
                        except Exception as error:
                            report = _base_image(source, source_row)
                            _error(report, "coverage.row_audit_failed", f"{type(error).__name__}: {error}")
                            reports.append(report)
                            row_audit_errors[split] += 1
                        else:
                            reports.append(report)
                            added, usable = _write_caption_records(caption_handle, captions)
                            caption_count += added
                            usable_caption_count += usable
                    processed_counts[split] += 1
            else:
                try:
                    for source_row, row in enumerate(dataset):
                        try:
                            report, captions = _inspect_row(source, source_row, row, image_inspector)
                        except Exception as error:
                            report = _base_image(source, source_row)
                            _error(report, "coverage.row_audit_failed", f"{type(error).__name__}: {error}")
                            reports.append(report)
                            row_audit_errors[split] += 1
                        else:
                            reports.append(report)
                            added, usable = _write_caption_records(caption_handle, captions)
                            caption_count += added
                            usable_caption_count += usable
                        processed_counts[split] += 1
                except Exception as error:
                    fatal_errors.append(f"{split}: iterator stopped after {processed_counts[split]} rows: {type(error).__name__}: {error}")
                observed_counts[split] = processed_counts[split]

    _record_duplicate_values(reports)
    groups = _exact_groups(reports)
    for report in reports:
        report["error_flags"].sort()
        report["warning_flags"].sort()
        report["audit_status"] = (
            "invalid" if report["error_flags"] else "warning" if report["warning_flags"] else "valid"
        )

    coverage_complete = (
        not fatal_errors
        and all(processed_counts[split] == expected_counts[split] for split in expected_counts)
        and all(row_read_errors[split] == 0 for split in expected_counts)
        and all(row_audit_errors[split] == 0 for split in expected_counts)
    )
    error_counts = Counter(flag for report in reports for flag in report["error_flags"])
    warning_counts = Counter(flag for report in reports for flag in report["warning_flags"])
    error_counts_by_split = {
        split: dict(sorted(Counter(
            flag for report in reports if report["source_split"] == split
            for flag in report["error_flags"]
        ).items()))
        for split in expected_counts
    }
    warning_counts_by_split = {
        split: dict(sorted(Counter(
            flag for report in reports if report["source_split"] == split
            for flag in report["warning_flags"]
        ).items()))
        for split in expected_counts
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_status": "complete" if coverage_complete else "incomplete",
        "coverage_complete": coverage_complete,
        "expected_counts": dict(expected_counts),
        "observed_counts": observed_counts,
        "processed_counts": processed_counts,
        "row_read_errors": row_read_errors,
        "row_audit_errors": row_audit_errors,
        "source_revisions": {split: source.spec.revision for split, source in sources.items()},
        "fatal_errors": fatal_errors,
        "audit_status_counts": dict(Counter(report["audit_status"] for report in reports)),
        "error_flag_counts": dict(sorted(error_counts.items())),
        "warning_flag_counts": dict(sorted(warning_counts.items())),
        "error_flag_counts_by_split": error_counts_by_split,
        "warning_flag_counts_by_split": warning_counts_by_split,
        "caption_records": caption_count,
        "usable_caption_records": usable_caption_count,
        "invalid_caption_records": caption_count - usable_caption_count,
        "exact_duplicate_groups": len(groups),
        "exact_duplicate_images": sum(bool(report["exact_group_ids"]) for report in reports),
        "cross_split_exact_groups": sum(group["crosses_official_split"] for group in groups),
        "pixel_hash_algorithm": _PIXEL_HASH_ALGORITHM,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "library_versions": {"datasets": datasets.__version__, "Pillow": PIL.__version__},
    }
    _write_jsonl(output_dir / "images.jsonl", reports)
    caption_temporary.replace(output_dir / "captions.jsonl")
    _write_jsonl(output_dir / "exact_groups.jsonl", groups)
    _write_summary(output_dir / "summary.json", summary)
    return summary


def audit_pinned_sources(output_dir: str | Path, *, streaming: bool = False) -> dict[str, Any]:
    """Load both pinned official splits and run the essential integrity audit."""
    started = time.perf_counter()
    output_dir = Path(output_dir)
    _mark_incomplete(output_dir)
    sources = {}
    load_errors = []
    for split in EXPECTED_COUNTS:
        try:
            sources[split] = load_cub_split(split, streaming=streaming)
        except Exception as error:
            load_errors.append(f"{split}: load failed: {type(error).__name__}: {error}")
    if len(sources) == 2 and sources["train"].class_names != sources["test"].class_names:
        load_errors.append("Official train and test ordered class mappings differ")
    summary = audit_sources(
        sources, output_dir, expected_counts=EXPECTED_COUNTS, initial_fatal_errors=load_errors
    )
    summary["scan_seconds"] = summary["runtime_seconds"]
    summary["runtime_seconds"] = round(time.perf_counter() - started, 3)
    summary["load_seconds"] = round(summary["runtime_seconds"] - summary["scan_seconds"], 3)
    _write_summary(output_dir / "summary.json", summary)
    return summary
