"""Describe verified classification membership using existing training EDA.

The shared manifest loader owns integrity verification, eligibility decisions,
and split construction. This consumer never parses its JSONL, re-splits data,
decodes images, or computes duplicate fingerprints. Isolation checks below are
observations of identifiers/hashes/group memberships already in that manifest.
Official test rows are checked only for identity/assignment preservation and
never enter image, caption, flag, class, or training-group analysis.
"""

from collections import Counter, defaultdict
from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

from src.data.schemas import SourceSpec

from .reporting import summarize_numbers


_PARTITIONS = ("train", "val", "excluded")
_VERSIONS = {
    "policy_version": "classification-v1",
    "split_version": "stratified-exact-group-v1",
    "pixel_hash_algorithm": "exif-rgb8-white:v1",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, *, minimum=0):
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum, f"{name} must be an integer >= {minimum}")
    return value


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _metadata(summary):
    _require(isinstance(summary, Mapping) and type(summary.get("schema_version")) is int and summary["schema_version"] == 1 and summary.get("complete") is True, "Preparation must be complete with supported schema version 1")
    for field, version in _VERSIONS.items():
        _require(summary.get(field) == version, f"Unsupported preparation {field}: {summary.get(field)!r}")
    _integer(summary.get("seed"), "preparation.seed")
    _require(summary["seed"] < 2**32, "preparation.seed must be less than 2**32")
    fraction = summary.get("val_fraction")
    _require(isinstance(fraction, (int, float)) and not isinstance(fraction, bool) and math.isfinite(fraction) and 0 < fraction < 1, "preparation.val_fraction must be finite and strictly between zero and one")
    for field, keys in (("source_counts", {"train", "test"}), ("partition_counts", {"train", "val", "test", "excluded"})):
        values = summary.get(field)
        _require(isinstance(values, Mapping) and set(values) == keys, f"Invalid preparation.{field}")
        for key, value in values.items():
            _integer(value, f"preparation.{field}.{key}")
    revisions = summary.get("source_revisions")
    _require(isinstance(revisions, Mapping) and set(revisions) == {"train", "test"} and all(isinstance(value, str) and value.strip() for value in revisions.values()), "Preparation source revisions must identify train and test")
    _require(_digest(summary.get("manifest_sha256")), "Preparation must record a valid manifest SHA256")
    fingerprints = summary.get("audit_fingerprints")
    _require(isinstance(fingerprints, Mapping) and fingerprints and all(isinstance(key, str) and _digest(value) for key, value in fingerprints.items()), "Preparation audit fingerprints are missing or invalid")
    _require(isinstance(summary.get("group_constraints"), Mapping), "Preparation group constraints must be an object")


def _report_rows(image_report, caption_report):
    # Reuse existing saved-report checks; importing this helper renders nothing.
    from src.visualization.image_plots import _rows as validate_image_rows
    from .explore_caption_details import validate_caption_details_report

    images = validate_image_rows(image_report)
    captions = validate_caption_details_report(caption_report)["images"]
    for name, report in (("image", image_report), ("caption", caption_report)):
        population = report["population"]
        _require(population["selection_method"] == "complete_official_training_split" and population["full_split_verified"] is True, f"Prepared EDA requires a verified complete official training {name} report")
    _require(image_report["source"] == caption_report["source"], "Image and caption report source revisions differ")
    _require(image_report["class_names"] == caption_report["class_names"], "Image and caption report class mappings differ")
    _require(image_report["population"] == caption_report["population"], "Image and caption report populations differ")
    image_by_id = {row["image_id"]: row for row in images}
    caption_by_id = {row["image_id"]: row for row in captions}
    _require(len(image_by_id) == len(images) and len(caption_by_id) == len(captions), "EDA reports contain duplicate training identities")
    _require(set(image_by_id) == set(caption_by_id), "Image and caption reports do not cover the same canonical training identities")
    return image_by_id, caption_by_id


def _flags(row, field):
    flags = row.get(field)
    _require(isinstance(flags, list) and all(isinstance(code, str) and code.strip() for code in flags), f"Invalid {field} for {row.get('image_id')!r}")
    return set(flags)


def _quality(rows):
    return {
        "image_count": len(rows),
        "unknown_class_image_count": sum(row["class_id"] is None for row in rows),
        "error_code_image_counts": dict(sorted(Counter(code for row in rows for code in _flags(row, "error_flags")).items())),
        "warning_code_image_counts": dict(sorted(Counter(code for row in rows for code in _flags(row, "warning_flags")).items())),
    }


def _partition_statistics(rows, image_by_id, caption_by_id):
    pixels = [image_by_id[row["image_id"]] for row in rows if image_by_id[row["image_id"]]["status"] == "ok"]
    captions = [caption_by_id[row["image_id"]] for row in rows]
    image_metrics = {}
    for field, maximum in (("brightness_mean", 1), ("contrast_std", 0.5), ("saturation_mean", 1), ("aspect_ratio", None)):
        values = [row["metrics"][field] for row in pixels]
        for row, value in zip(pixels, values):
            _require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 and (maximum is None or value <= maximum + 1e-12), f"Invalid saved image metric {field}")
            _require(field != "aspect_ratio" or value > 0, "Image aspect ratios must be positive")
            if field == "aspect_ratio":
                _require(math.isclose(value, row["oriented_width"] / row["oriented_height"], rel_tol=1e-12, abs_tol=1e-12), f"Saved aspect_ratio differs from oriented dimensions: {row['image_id']}")
        image_metrics[field] = summarize_numbers(values)
    return {
        "image_count": len(rows),
        "eligible_pixel_image_count": len(pixels),
        "pixel_ineligible_image_count": len(rows) - len(pixels),
        "captioned_image_count": sum(row["nonempty_caption_count"] > 0 for row in captions),
        "token_bearing_image_count": sum(row["token_count"] > 0 for row in captions),
        "pairwise_overlap_image_count": sum(row["pairwise_jaccard_mean"] is not None for row in captions),
        "image_metrics": image_metrics,
        "caption_metrics": {
            field: summarize_numbers(row[field] for row in captions if row[field] is not None)
            for field in ("mean_caption_words", "type_token_ratio", "pairwise_jaccard_mean")
        },
    }


def _groups_and_isolation(training, all_rows, constraints):
    memberships = defaultdict(list)
    all_by_id = {row["image_id"]: row for row in all_rows}
    for row in all_rows:
        groups = row.get("exact_group_ids")
        _require(isinstance(groups, list) and len(groups) <= 1 and all(isinstance(group, str) and group for group in groups), f"Invalid supplied exact-group membership for {row['image_id']}")
        for group in groups:
            memberships[group].append(row["image_id"])
    _require(set(memberships) == set(constraints), "Preparation group constraints disagree with manifest memberships")
    for group, identities in memberships.items():
        declared = constraints[group]
        _require(isinstance(declared, list) and len(identities) >= 2 and len(declared) == len(set(declared)) and set(declared) == set(identities), f"Exact-group membership differs from preparation: {group}")
        _require(len({all_by_id[identity]["source_split"] for identity in identities}) == 1, f"Supplied exact group crosses official sources: {group}")

    for field in ("encoded_sha256", "pixel_sha256", "exact_group_ids"):
        observed = defaultdict(set)
        for row in training:
            if row["partition"] not in {"train", "val"}:
                continue
            values = row[field] if field == "exact_group_ids" else [row.get(field)]
            for value in values:
                if value is not None:
                    observed[value].add(row["partition"])
        for value, partitions in observed.items():
            _require(len(partitions) <= 1, f"Supplied {field} membership overlaps derived train and val: {value}")

    training_ids = {row["image_id"] for row in training}
    groups = []
    for group, identities in sorted(memberships.items()):
        if not set(identities) & training_ids:
            continue
        members = sorted((all_by_id[identity] for identity in identities), key=lambda row: row["source_row"])
        groups.append({
            "group_id": group,
            "image_count": len(members),
            "partition_counts": {partition: sum(row["partition"] == partition for row in members) for partition in _PARTITIONS},
            "members": [{key: row[key] for key in ("image_id", "source_row", "class_id", "class_name", "partition")} for row in members],
        })
    return {
        "training_group_count": len(groups),
        "training_group_image_count": sum(group["image_count"] for group in groups),
        "groups": groups,
        "train_val_overlap_violations": [],
        "isolation_fields_checked": ["image_id", "encoded_sha256", "pixel_sha256", "exact_group_ids"],
    }


def build_prepared_training_report(prepared_dir, *, image_report, caption_report, manifest_loader=None):
    """Join verified preparation membership to complete raw training EDA only.

    A paired-null class is retained only for an excluded training row and counted
    as unknown, never borrowed from the EDA class map. Every known class must
    agree across the manifest and both EDA snapshots. No output files are written.
    """
    directory = Path(prepared_dir)
    summary_path = directory / "preparation.json"
    before = summary_path.read_bytes()
    summary = json.loads(before)
    _metadata(summary)
    image_by_id, caption_by_id = _report_rows(image_report, caption_report)
    names = list(image_report["class_names"])
    source = image_report["source"]
    _require(summary.get("class_names") == names, "Preparation and EDA class mappings differ")
    _require(summary["source_revisions"]["train"] == source["revision"], "Preparation and EDA training revisions differ")
    _require(summary["source_counts"]["train"] == len(image_by_id), "Preparation training count differs from the complete EDA population")
    if manifest_loader is None:
        from src.data.preparation import load_classification_manifest
        manifest_loader = load_classification_manifest
    _require(callable(manifest_loader), "manifest_loader must be callable")
    rows = manifest_loader(directory)
    _require(summary_path.read_bytes() == before, "Preparation changed while loading the verified manifest")
    _require(isinstance(rows, list) and all(isinstance(row, Mapping) for row in rows), "Shared manifest loader must return row objects")

    identities = set()
    source_positions = {"train": set(), "test": set()}
    training = []
    for row in rows:
        split = row.get("source_split")
        _require(split in source_positions, "Manifest has an unknown official source split")
        position = _integer(row.get("source_row"), "manifest.source_row")
        identity = row.get("image_id")
        repo, revision = row.get("source_repo"), row.get("source_revision")
        _require(isinstance(repo, str) and repo.strip() and revision == summary["source_revisions"][split], f"Manifest source revision mismatch: {identity}")
        _require(identity == SourceSpec(repo, split, revision).image_id(position) and identity not in identities and position not in source_positions[split], f"Missing, noncanonical, or duplicated manifest identity: {identity}")
        identities.add(identity)
        source_positions[split].add(position)
        _require(isinstance(row.get("eligible"), bool), f"Manifest eligibility must be explicit for {identity}")
        partition = row.get("partition")
        if split == "test":
            _require(partition == "test", f"Official test identity changed partition: {identity}")
            continue  # No test flags, caption values, or pixel measurements enter EDA.
        _require(repo == source["repo_id"], f"Manifest training repository differs from EDA: {identity}")
        _require(identity in image_by_id and identity in caption_by_id, f"Manifest training identity is missing from EDA: {identity}")
        _require(partition in _PARTITIONS and (partition in {"train", "val"}) == row["eligible"], f"Manifest training assignment violates its eligibility: {identity}")
        class_id, class_name = row.get("class_id"), row.get("class_name")
        if class_id is None or class_name is None:
            _require(class_id is None and class_name is None and partition == "excluded", f"Unknown class is allowed only for excluded training rows: {identity}")
        else:
            _integer(class_id, "manifest.class_id")
            _require(class_id < len(names) and class_name == names[class_id], f"Manifest class differs from the ordered mapping: {identity}")
            for kind, observed in (("image", image_by_id[identity]), ("caption", caption_by_id[identity])):
                _require((observed["class_id"], observed["class_name"]) == (class_id, class_name), f"Manifest/{kind} class mismatch for {identity}: manifest={(class_id, class_name)!r}, report={(observed['class_id'], observed['class_name'])!r}")
        for field in ("source_id", "file_name"):
            _require(row.get(field) == image_by_id[identity].get(field) == caption_by_id[identity].get(field), f"Manifest/EDA {field} mismatch for {identity}")
        for field in ("encoded_sha256", "pixel_sha256"):
            _require(row.get(field) is None or _digest(row[field]), f"Invalid supplied {field}: {identity}")
        _flags(row, "error_flags")
        _flags(row, "warning_flags")
        training.append(row)

    for split in ("train", "test"):
        _require(source_positions[split] == set(range(summary["source_counts"][split])), f"Manifest does not preserve complete {split} source-row identities")
    _require({row["image_id"] for row in training} == set(image_by_id), "Manifest and EDA do not cover exactly the same training identities")
    actual_partitions = {partition: sum(row["partition"] == partition for row in rows) for partition in ("train", "val", "test", "excluded")}
    _require(actual_partitions == summary["partition_counts"], "Manifest partition counts disagree with preparation metadata")
    source_counts = {split: sum(row["source_split"] == split for row in rows) for split in ("train", "test")}
    _require(source_counts == summary["source_counts"], "Manifest source counts disagree with preparation metadata")
    grouped = {partition: [row for row in training if row["partition"] == partition] for partition in _PARTITIONS}
    groups = _groups_and_isolation(training, rows, summary["group_constraints"])
    provenance = {key: deepcopy(summary[key]) for key in ("schema_version", "policy_version", "split_version", "pixel_hash_algorithm", "seed", "val_fraction", "source_revisions", "source_counts", "partition_counts", "audit_fingerprints", "manifest_sha256")}
    provenance["preparation_sha256"] = sha256(before).hexdigest()
    classes = [
        {"class_id": class_id, "class_name": class_name, "partition_counts": {partition: sum(row["class_id"] == class_id for row in grouped[partition]) for partition in _PARTITIONS}}
        for class_id, class_name in enumerate(names)
    ]
    class_distributions = {}
    for scope in ("raw", "train", "val"):
        counts = [sum(row["partition_counts"].values()) if scope == "raw" else row["partition_counts"][scope] for row in classes]
        class_distributions[scope] = [
            {"images_per_class": count, "class_count": frequency}
            for count, frequency in sorted(Counter(counts).items())
        ]
    result = {
        "schema_version": "prepared-eda-1.0",
        "source": deepcopy(source),
        "class_names": names,
        "preparation": provenance,
        "population": {
            "raw_training_image_count": len(training),
            "derived_training_image_count": len(grouped["train"]),
            "validation_image_count": len(grouped["val"]),
            "excluded_training_image_count": len(grouped["excluded"]),
            "unknown_class_training_image_count": sum(row["class_id"] is None for row in training),
        },
        "official_test": {"identity_count": source_counts["test"], "preserved_test_partition_count": actual_partitions["test"], "identities_preserved": True, "image_caption_statistics_used": False},
        "classes": classes,
        "class_count_distributions": class_distributions,
        "quality": {
            "overall": _quality(training),
            "by_partition": {partition: _quality(selected) for partition, selected in grouped.items()},
            "by_class": [{"class_id": class_id, "class_name": class_name, **_quality([row for row in training if row["class_id"] == class_id])} for class_id, class_name in enumerate(names)],
        },
        "exact_groups": groups,
        "partitions": {partition: _partition_statistics(grouped[partition], image_by_id, caption_by_id) for partition in ("train", "val")},
        "methods": {
            "membership": "Existing shared classification manifest loader; no re-splitting or policy decisions",
            "quality_counts": "Distinct affected official-training image IDs per supplied error/warning code; flags may overlap and do not themselves exclude rows",
            "numeric_weighting": "One available value per image; RGB-analysis-ineligible rows excluded only from pixel summaries, not partition counts",
            "unknown_classes": "Paired-null classes retained only in excluded membership and unknown counts; no class is imputed from EDA",
            "isolation": "Canonical image identity and supplied hash/exact-group membership checks; source IDs are join metadata, not content equivalence; no duplicate-detection algorithms or image hashing",
            "validation": "Derived validation summaries are descriptive split audits; future fitted preprocessing uses derived training membership only",
        },
        "limitations": ["No normalization, thresholds, vocabulary settings, augmentation, hypothesis tests, or cleaning decisions are fitted.", "Recorded flags and duplicate memberships inherit the verified preparation snapshot's definitions.", "Official test records contribute identity/assignment counts only."],
    }
    _require(summary_path.read_bytes() == before, "Preparation changed while analyzing the verified manifest")
    json.dumps(result, allow_nan=False)
    return result
