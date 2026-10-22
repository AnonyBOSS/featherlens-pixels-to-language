"""Audited classification eligibility and reproducible, group-constrained splits.

Preparation never edits source assets or the audit. A completed manifest is a
versioned decision snapshot, reusable with different model preprocessing.
"""

from collections import Counter, defaultdict
from collections.abc import Mapping
from hashlib import sha256
import json
import math
from pathlib import Path
from random import Random
from typing import Any

from .integrity_audit import EXPECTED_COUNTS
from .source import _SOURCES


_ERROR_CODES = {
    "metadata.invalid_row", "metadata.missing_image", "metadata.missing_description",
    "metadata.missing_label", "metadata.missing_file_name", "metadata.missing_id",
    "metadata.invalid_label", "metadata.invalid_source_id", "metadata.invalid_filename",
    "metadata.invalid_image_reference", "caption.missing_description",
    "caption.invalid_description", "caption.empty", "caption.control_character",
    "image.fetch_failed", "image.decode_failed", "coverage.row_read_failed",
    "coverage.row_audit_failed",
}
_WARNING_CODES = {
    "metadata.negative_source_id", "metadata.suspicious_filename",
    "metadata.unusual_extension", "metadata.duplicate_source_id",
    "metadata.duplicate_filename", "caption.count_mismatch", "duplicate.exact",
    "leakage.exact_cross_split", "duplicate.label_conflict",
}
_ALLOWED_WARNINGS = {"caption.count_mismatch", "duplicate.exact"}
_PIXEL_VERSION = "exif-rgb8-white:v1"
_POLICY_VERSION = "classification-v1"
_SPLIT_VERSION = "stratified-exact-group-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid JSONL at {path.name}:{line_number}") from error
            _require(isinstance(row, dict), f"Expected an object at {path.name}:{line_number}")
            rows.append(row)
    return rows


def _flags(row: Mapping[str, Any], key: str, supported: set[str]) -> list[str]:
    flags = row.get(key)
    _require(isinstance(flags, list) and all(isinstance(flag, str) for flag in flags), f"Invalid {key}")
    _require(len(flags) == len(set(flags)), f"Repeated {key}")
    _require(not (set(flags) - supported), f"Unrecognized {key}: {set(flags) - supported}; review required")
    return flags


def _validate_rows(rows: list[dict[str, Any]]) -> list[str]:
    _require(len(rows) == sum(EXPECTED_COUNTS.values()), "Manifest does not account for all expected source rows")
    seen: set[str] = set()
    row_numbers: dict[str, set[int]] = {split: set() for split in EXPECTED_COUNTS}
    classes: dict[int, str] = {}
    for row in rows:
        split = row.get("source_split")
        _require(isinstance(split, str) and split in _SOURCES, "Unknown official source split")
        spec = _SOURCES[split]
        index = row.get("source_row")
        _require(_integer(index) and 0 <= index < EXPECTED_COUNTS[split], "Invalid source row")
        _require(row.get("source_repo") == spec.repo_id and row.get("source_revision") == spec.revision, "Source provenance does not match pinned revision")
        identifier = spec.image_id(index)
        _require(row.get("image_id") == identifier and identifier not in seen, "Invalid or duplicated stable image_id")
        seen.add(identifier)
        _require(index not in row_numbers[split], "Duplicated source row")
        row_numbers[split].add(index)
        errors = _flags(row, "error_flags", _ERROR_CODES)
        _flags(row, "warning_flags", _WARNING_CODES)
        _require(not any(flag.startswith("coverage.") for flag in errors), "Audit has uninspected source rows")
        metadata_valid = row.get("metadata_valid")
        _require(isinstance(metadata_valid, bool) and metadata_valid == (not any(flag.startswith("metadata.") for flag in errors)), "Inconsistent metadata validity")
        for field, missing_codes in (
            ("source_id", {"metadata.missing_id", "metadata.invalid_source_id", "metadata.invalid_row"}),
            ("file_name", {"metadata.missing_file_name", "metadata.invalid_filename", "metadata.invalid_row"}),
        ):
            value = row.get(field)
            if value is None:
                _require(bool(set(errors) & missing_codes), f"Missing {field} without a source error")
            else:
                valid_value = _integer(value) if field == "source_id" else isinstance(value, str) and bool(value.strip())
                _require(valid_value, f"Invalid {field}")
        label, name = row.get("class_id"), row.get("class_name")
        if label is None:
            _require(name is None and bool(set(errors) & {"metadata.invalid_label", "metadata.missing_label", "metadata.invalid_row"}), "Invalid class reference without a source error")
        else:
            _require(_integer(label) and 0 <= label < 200 and isinstance(name, str) and bool(name.strip()), "Invalid class reference")
            _require(label not in classes or classes[label] == name, "Inconsistent ordered class mapping")
            classes[label] = name
        status = row.get("image_status")
        _require(status in {"ok", "fetch_error", "decode_error", "not_checked"}, "Invalid image status")
        _require(status != "not_checked" or "metadata.invalid_row" in errors, "Unexplained unchecked image; audit must inspect every accessible source row")
        valid = row.get("image_valid")
        _require(isinstance(valid, bool) and valid == (status == "ok"), "Inconsistent image validity")
        for field in ("encoded_sha256", "pixel_sha256"):
            _require(row.get(field) is None or _digest(row[field]), f"Invalid {field}")
        if valid:
            _require(all(_digest(row.get(field)) for field in ("encoded_sha256", "pixel_sha256")), "Decoded image lacks valid hashes")
            _require(all(_integer(row.get(field)) and row[field] > 0 for field in ("width", "height")), "Decoded image has invalid dimensions")
            _require(not any(flag.startswith("image.") for flag in errors), "Decoded image has failure flags")
        else:
            _require(row.get("pixel_sha256") is None, "Failed image has a normalized-pixel hash")
            _require(status == "not_checked" or f"image.{'fetch' if status == 'fetch_error' else 'decode'}_failed" in errors, "Image failure lacks a reason")
        memberships = row.get("exact_group_ids")
        _require(isinstance(memberships, list) and len(memberships) <= 1 and all(isinstance(value, str) for value in memberships), "Invalid exact group membership")
        expected_status = "invalid" if errors else "warning" if row["warning_flags"] else "valid"
        _require(row.get("audit_status") == expected_status, "Inconsistent audit status")
    for split, count in EXPECTED_COUNTS.items():
        _require(row_numbers[split] == set(range(count)), f"Incomplete source rows for {split}")
    _require(set(classes) == set(range(200)) and len(set(classes.values())) == 200, "Audit does not provide a consistent complete 200-class mapping")
    return [classes[index] for index in range(200)]


def _fingerprint_groups(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Verify grouping from saved hashes only; source images are never decoded."""
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    duplicates = []
    for field in ("encoded_sha256", "pixel_sha256"):
        buckets: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            if row.get(field) is not None:
                buckets[row[field]].append(index)
        for digest, members in buckets.items():
            if len(members) > 1:
                duplicates.append((field, digest, members[0]))
                for other in members[1:]:
                    parents[find(other)] = find(members[0])
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[find(index)].append(index)
    evidence: dict[int, list[dict[str, str]]] = defaultdict(list)
    for field, digest, member in duplicates:
        evidence[find(member)].append({"kind": field, "sha256": digest})
    groups = {}
    for root, members in components.items():
        if len(members) < 2:
            continue
        identifiers = sorted(rows[index]["image_id"] for index in members)
        group_id = "exact:" + sha256("\n".join(identifiers).encode("utf-8")).hexdigest()
        splits = sorted({rows[index]["source_split"] for index in members})
        labels = sorted({rows[index]["class_id"] for index in members if rows[index]["class_id"] is not None})
        _require(len(splits) == 1, f"Exact group crosses official splits: {group_id}; review required")
        _require(len(labels) <= 1, f"Exact group has conflicting class labels: {group_id}; review required")
        hash_evidence = sorted(evidence[root], key=lambda entry: (entry["kind"], entry["sha256"]))
        kinds = {entry["kind"] for entry in hash_evidence}
        groups[group_id] = {
            "group_id": group_id, "image_ids": identifiers,
            "source_splits": splits, "class_ids": labels,
            "crosses_official_split": False, "label_conflict": False,
            "hash_kind": next(iter(kinds)) if len(kinds) == 1 else "mixed",
            "hash_evidence": hash_evidence,
        }
    row_groups = {identifier: group_id for group_id, group in groups.items() for identifier in group["image_ids"]}
    for row in rows:
        expected = [row_groups[row["image_id"]]] if row["image_id"] in row_groups else []
        _require(row["exact_group_ids"] == expected, "Exact group membership disagrees with saved hashes")
        _require(("duplicate.exact" in row["warning_flags"]) == bool(expected), "Exact duplicate flags disagree with group membership")
    return groups


def _load_audit(directory: Path) -> tuple[list[dict[str, Any]], list[str], dict[str, Any], dict[str, str]]:
    files = {name: directory / name for name in ("summary.json", "images.jsonl", "exact_groups.jsonl")}
    before = {name: _file_hash(path) for name, path in files.items()}
    summary = json.loads(files["summary.json"].read_text(encoding="utf-8"))
    _require(isinstance(summary, dict), "Invalid audit summary")
    _require(_integer(summary.get("schema_version")) and summary.get("schema_version") == 1 and summary.get("pixel_hash_algorithm") == _PIXEL_VERSION, "Unsupported audit schema or normalization")
    _require(summary.get("run_status") == "complete" and summary.get("coverage_complete") is True and summary.get("fatal_errors") == [], "Audit coverage is incomplete")
    for key in ("expected_counts", "observed_counts", "processed_counts"):
        _require(summary.get(key) == EXPECTED_COUNTS, f"Audit {key} differs from pinned source coverage")
    for key in ("row_read_errors", "row_audit_errors"):
        counters = summary.get(key)
        _require(isinstance(counters, dict) and counters == {"train": 0, "test": 0} and all(_integer(value) for value in counters.values()), f"Audit {key} prevents complete coverage")
    _require(summary.get("source_revisions") == {split: spec.revision for split, spec in _SOURCES.items()}, "Audit source revisions differ from pinned sources")
    rows = _read_jsonl(files["images.jsonl"])
    classes = _validate_rows(rows)
    expected_groups = _fingerprint_groups(rows)
    groups = _read_jsonl(files["exact_groups.jsonl"])
    _require(len(groups) == len(expected_groups) and len({group.get("group_id") for group in groups}) == len(groups), "Missing or duplicated exact groups")
    for group in groups:
        _require(group.get("group_id") in expected_groups and group == expected_groups[group["group_id"]], "Exact group evidence does not match member records")
    for key, expected in (
        ("audit_status_counts", dict(Counter(row["audit_status"] for row in rows))),
        ("error_flag_counts", dict(Counter(flag for row in rows for flag in row["error_flags"]))),
        ("warning_flag_counts", dict(Counter(flag for row in rows for flag in row["warning_flags"]))),
        ("error_flag_counts_by_split", {split: dict(Counter(flag for row in rows if row["source_split"] == split for flag in row["error_flags"])) for split in EXPECTED_COUNTS}),
        ("warning_flag_counts_by_split", {split: dict(Counter(flag for row in rows if row["source_split"] == split for flag in row["warning_flags"])) for split in EXPECTED_COUNTS}),
        ("exact_duplicate_groups", len(groups)),
        ("exact_duplicate_images", sum(bool(row["exact_group_ids"]) for row in rows)),
        ("cross_split_exact_groups", 0),
    ):
        _require(summary.get(key) == expected, f"Audit summary {key} disagrees with manifests")
    _require(before == {name: _file_hash(path) for name, path in files.items()}, "Audit changed while reading; retry after the audit finishes")
    return rows, classes, expected_groups, before


def _eligibility(row: Mapping[str, Any]) -> tuple[bool, list[str]]:
    unreviewed = set(row["warning_flags"]) - _ALLOWED_WARNINGS
    _require(not unreviewed, f"Warnings require a reviewed classification policy: {sorted(unreviewed)}")
    reasons = {
        flag for flag in row["error_flags"]
        if not flag.startswith("caption.") and flag != "metadata.missing_description"
    }
    if not row["image_valid"]:
        reasons.add("image.not_decodable")
    if row["class_id"] is None:
        reasons.add("class.invalid")
    return not reasons, sorted(reasons)


def _assign_partitions(rows: list[dict[str, Any]], fraction: float, seed: int) -> None:
    random = Random(seed)
    by_class: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["source_split"] == "test":
            row["partition"] = "test"  # Preserve all official assignments, even invalid rows.
        elif not row["eligible"]:
            row["partition"] = "excluded"
        else:
            key = next(iter(row["exact_group_ids"]), row["image_id"])
            by_class[row["class_id"]][key].append(row)
    for class_id in range(200):
        groups = [by_class[class_id][key] for key in sorted(by_class[class_id])]
        random.shuffle(groups)
        size = sum(len(group) for group in groups)
        target = math.floor(size * fraction + 0.5)
        reachable: dict[int, tuple[int, ...]] = {0: ()}
        for group_index, group in enumerate(groups):
            for count, chosen in list(reachable.items()):
                new_count = count + len(group)
                if new_count not in reachable:
                    reachable[new_count] = chosen + (group_index,)
        feasible = [count for count in reachable if 0 < count < size]
        _require(bool(feasible), f"Class {class_id} cannot form nonempty train/val using whole groups")
        selected_count = min(feasible, key=lambda count: (abs(count - target), count))
        validation = set(reachable[selected_count])
        for index, group in enumerate(groups):
            for row in group:
                row["partition"] = "val" if index in validation else "train"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)



def _validate_configuration(fraction: Any, seed: Any) -> None:
    _require(isinstance(fraction, (int, float)) and not isinstance(fraction, bool) and math.isfinite(fraction) and 0 < fraction < 1, "val_fraction must be finite and strictly between zero and one")
    _require(_integer(seed) and 0 <= seed < 2**32, "seed must be an integer in [0, 2**32)")


def _population_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    partitions = Counter(row["partition"] for row in rows)
    eligible = Counter(row["source_split"] for row in rows if row["eligible"])
    classes: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        if row["eligible"]:
            classes[row["partition"]][row["class_id"]] += 1
    return {
        "partition_counts": {key: partitions[key] for key in ("train", "val", "test", "excluded")},
        "eligible_counts": {key: eligible[key] for key in EXPECTED_COUNTS},
        "eligible_class_counts": {
            partition: {str(class_id): classes[partition][class_id] for class_id in range(200)}
            for partition in ("train", "val", "test")
        },
    }


def prepare_classification(
    audit_dir: str | Path,
    output_dir: str | Path,
    *,
    val_fraction: float = 0.20,
    seed: int = 42,
) -> dict[str, Any]:
    """Validate a pinned audit and publish classification decisions and splits.

    Both exact duplicate training rows are retained as one split unit. Caption
    failures do not exclude a valid classification image. Unreviewed warnings
    stop preparation. Official test assignments are preserved even for ineligible
    rows; evaluation consumers must reject incomplete eligible test populations.
    """
    _validate_configuration(val_fraction, seed)
    audit_dir, output_dir = Path(audit_dir).resolve(), Path(output_dir).resolve()
    _require(output_dir != audit_dir and audit_dir not in output_dir.parents and output_dir not in audit_dir.parents, "Use a separate preparation directory outside the audit")
    source_rows, classes, groups, fingerprints = _load_audit(audit_dir)
    rows = []
    for source_row in source_rows:
        eligible, reasons = _eligibility(source_row)
        rows.append({**source_row, "eligible": eligible, "decision_codes": reasons})
    _assign_partitions(rows, float(val_fraction), seed)
    rows.sort(key=lambda row: (row["source_split"] == "test", row["source_row"]))
    counts = _population_counts(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "preparation.json"
    _write_json(summary_path, {"schema_version": 1, "complete": False, "error": "Preparation did not finish publishing"})
    manifest_path = output_dir / "classification_manifest.jsonl"
    _write_manifest(manifest_path, rows)
    result = {
        "schema_version": 1, "complete": True,
        "policy_version": _POLICY_VERSION, "split_version": _SPLIT_VERSION,
        "seed": seed, "val_fraction": float(val_fraction),
        "pixel_hash_algorithm": _PIXEL_VERSION,
        "source_revisions": {split: spec.revision for split, spec in _SOURCES.items()},
        "source_counts": dict(EXPECTED_COUNTS), "class_names": classes,
        "audit_fingerprints": fingerprints,
        "manifest_sha256": _file_hash(manifest_path),
        **counts,
        "group_constraints": {key: group["image_ids"] for key, group in groups.items()},
        "decision_counts": dict(Counter(code for row in rows for code in row["decision_codes"])),
    }
    _write_json(summary_path, result)
    return result


def load_classification_manifest(prepared_dir: str | Path) -> list[dict[str, Any]]:
    """Read a completed, fingerprint-checked preparation; never silently resplit."""
    directory = Path(prepared_dir)
    summary_path = directory / "preparation.json"
    initial_summary = summary_path.read_bytes()
    summary = json.loads(initial_summary)
    _require(isinstance(summary, dict) and summary.get("complete") is True and _integer(summary.get("schema_version")) and summary.get("schema_version") == 1, "Preparation is incomplete or unsupported")
    _require(summary.get("policy_version") == _POLICY_VERSION and summary.get("split_version") == _SPLIT_VERSION and summary.get("pixel_hash_algorithm") == _PIXEL_VERSION, "Unsupported preparation policy/split version")
    _require(summary.get("source_counts") == EXPECTED_COUNTS and summary.get("source_revisions") == {split: spec.revision for split, spec in _SOURCES.items()}, "Prepared source references differ from pinned sources")
    _validate_configuration(summary.get("val_fraction"), summary.get("seed"))
    manifest_path = directory / "classification_manifest.jsonl"
    _require(_file_hash(manifest_path) == summary.get("manifest_sha256"), "Preparation manifest hash mismatch")
    rows = _read_jsonl(manifest_path)
    classes = _validate_rows(rows)
    _require(classes == summary.get("class_names"), "Prepared class mapping mismatch")
    groups = _fingerprint_groups(rows)
    _require(summary.get("group_constraints") == {key: group["image_ids"] for key, group in groups.items()}, "Prepared group constraints mismatch")
    for row in rows:
        eligible, reasons = _eligibility(row)
        _require(row.get("eligible") is eligible and row.get("decision_codes") == reasons, "Prepared eligibility decision mismatch")
        partition = row.get("partition")
        allowed = {"test"} if row["source_split"] == "test" else {"train", "val"} if eligible else {"excluded"}
        _require(partition in allowed, "Prepared partition violates source/eligibility policy")
    by_id = {row["image_id"]: row for row in rows}
    for group in groups.values():
        partitions = {by_id[identifier]["partition"] for identifier in group["image_ids"] if by_id[identifier]["eligible"]}
        _require(len(partitions) <= 1, "Exact group crosses derived partitions")
    expected_rows = [dict(row) for row in rows]
    _assign_partitions(expected_rows, summary["val_fraction"], summary["seed"])
    _require(all(row["partition"] == expected["partition"] for row, expected in zip(rows, expected_rows)), "Stored split membership disagrees with the declared seed and split recipe")
    for key, expected in _population_counts(rows).items():
        _require(expected == summary.get(key), f"Preparation manifest {key} mismatch")
    _require(_file_hash(manifest_path) == summary["manifest_sha256"] and summary_path.read_bytes() == initial_summary, "Preparation changed while reading")
    return rows
