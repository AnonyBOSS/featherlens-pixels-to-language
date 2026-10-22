"""Behavioral tests for preparation of the pinned CUB classification population."""

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import importlib
import json
from pathlib import Path

import pytest


COUNTS = {"train": 5994, "test": 5794}
PINS = {
    "train": ("Multimodal-Fatima/CUB_train", "bd3988995e6db193427255abdebd522afe24ce76"),
    "test": ("Multimodal-Fatima/CUB_test", "82447a42c5e05c7fe21cd36accfdf399148e8668"),
}


def preparation():
    return importlib.import_module("src.data.preparation")


def _group(images, members):
    selected = [images[index] for index in members]
    identifiers = sorted(row["image_id"] for row in selected)
    group_id = "exact:" + sha256("\n".join(identifiers).encode()).hexdigest()
    digest = selected[0]["encoded_sha256"]
    pixel_digest = selected[0]["pixel_sha256"]
    splits = sorted({row["source_split"] for row in selected})
    labels = sorted({row["class_id"] for row in selected if row["class_id"] is not None})
    for row in selected:
        row["encoded_sha256"] = digest
        row["pixel_sha256"] = pixel_digest
        row["exact_group_ids"] = [group_id]
        row["warning_flags"] = ["duplicate.exact"]
        if len(splits) > 1:
            row["warning_flags"].append("leakage.exact_cross_split")
        if len(labels) > 1:
            row["warning_flags"].append("duplicate.label_conflict")
    return {
        "group_id": group_id, "image_ids": identifiers, "source_splits": splits,
        "class_ids": labels, "crosses_official_split": len(splits) > 1,
        "label_conflict": len(labels) > 1, "hash_kind": "mixed",
        "hash_evidence": [
            {"kind": "encoded_sha256", "sha256": digest},
            {"kind": "pixel_sha256", "sha256": pixel_digest},
        ],
    }


@pytest.fixture(scope="module")
def base_images():
    images = []
    for split, count in COUNTS.items():
        sizes = [30] * 194 + [29] * 6 if split == "train" else [29] * 194 + [28] * 6
        labels = [class_id for class_id, size in enumerate(sizes) for _ in range(size)]
        repo, revision = PINS[split]
        for index in range(count):
            digest = f"{len(images) + 1:064x}"
            label = labels[index]
            images.append({
                "image_id": f"hf:{repo}@{revision}:{split}:{index}",
                "source_repo": repo, "source_revision": revision,
                "source_split": split, "source_row": index, "source_id": index,
                "file_name": f"{split}_{index}.jpg", "class_id": label,
                "class_name": f"Bird {label}", "metadata_valid": True,
                "image_valid": True, "image_status": "ok", "image_error_code": None,
                "image_format": "JPEG", "image_mode": "RGB", "width": 500, "height": 333,
                "encoded_sha256": digest, "pixel_sha256": digest,
                "caption_count": 10, "usable_caption_count": 10,
                "caption_structural_valid": True, "exact_group_ids": [],
                "error_flags": [], "warning_flags": [], "error_messages": {},
                "audit_status": "valid",
            })
    return images


def write_audit(path, images, groups=(), summary_changes=None):
    path.mkdir()
    for row in images:
        row["audit_status"] = "invalid" if row["error_flags"] else "warning" if row["warning_flags"] else "valid"
    summary = {
        "schema_version": 1, "run_status": "complete", "coverage_complete": True,
        "expected_counts": COUNTS, "observed_counts": COUNTS, "processed_counts": COUNTS,
        "source_revisions": {key: value[1] for key, value in PINS.items()},
        "row_read_errors": {"train": 0, "test": 0}, "row_audit_errors": {"train": 0, "test": 0},
        "fatal_errors": [], "pixel_hash_algorithm": "exif-rgb8-white:v1",
        "audit_status_counts": dict(Counter(row["audit_status"] for row in images)),
        "error_flag_counts": dict(Counter(flag for row in images for flag in row["error_flags"])),
        "warning_flag_counts": dict(Counter(flag for row in images for flag in row["warning_flags"])),
        "error_flag_counts_by_split": {split: dict(Counter(flag for row in images if row["source_split"] == split for flag in row["error_flags"])) for split in COUNTS},
        "warning_flag_counts_by_split": {split: dict(Counter(flag for row in images if row["source_split"] == split for flag in row["warning_flags"])) for split in COUNTS},
        "exact_duplicate_groups": len(groups),
        "exact_duplicate_images": sum(bool(row["exact_group_ids"]) for row in images),
        "cross_split_exact_groups": sum(group["crosses_official_split"] for group in groups),
    }
    if summary_changes:
        summary.update(summary_changes)
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (path / "images.jsonl").write_text("".join(json.dumps(row) + "\n" for row in images), encoding="utf-8")
    (path / "exact_groups.jsonl").write_text("".join(json.dumps(row) + "\n" for row in groups), encoding="utf-8")
    return path


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_preparation_accounts_for_every_row_and_keeps_duplicates_together(tmp_path, base_images):
    images = deepcopy(base_images)
    group = _group(images, [1712, 1731])
    audit = write_audit(tmp_path / "audit", images, [group])
    result = preparation().prepare_classification(audit, tmp_path / "prepared")
    rows = preparation().load_classification_manifest(tmp_path / "prepared")

    assert result["complete"] is True
    assert result["partition_counts"] == {"train": 4794, "val": 1200, "test": 5794, "excluded": 0}
    assert len(rows) == 11788
    assert len({row["image_id"] for row in rows}) == 11788
    pair = [row for row in rows if row["image_id"] in group["image_ids"]]
    assert len(pair) == 2 and pair[0]["partition"] == pair[1]["partition"]
    assert all(row["eligible"] for row in pair)
    assert all(row["partition"] == "test" for row in rows if row["source_split"] == "test")
    assert Counter(row["class_id"] for row in rows if row["partition"] == "val") == {i: 6 for i in range(200)}
    assert result["class_names"] == [f"Bird {i}" for i in range(200)]


def test_membership_is_independent_of_audit_row_order_and_changes_with_seed(tmp_path, base_images):
    images = deepcopy(base_images)
    group = _group(images, [1712, 1731])
    audit = write_audit(tmp_path / "audit", images, [group])
    prepare = preparation().prepare_classification
    prepare(audit, tmp_path / "first", seed=42)
    (audit / "images.jsonl").write_text("".join(json.dumps(row) + "\n" for row in reversed(images)), encoding="utf-8")
    prepare(audit, tmp_path / "reordered", seed=42)
    prepare(audit, tmp_path / "different", seed=43)
    assert (tmp_path / "first/classification_manifest.jsonl").read_bytes() == (tmp_path / "reordered/classification_manifest.jsonl").read_bytes()
    a = {r["image_id"] for r in read_rows(tmp_path / "first/classification_manifest.jsonl") if r["partition"] == "val"}
    b = {r["image_id"] for r in read_rows(tmp_path / "different/classification_manifest.jsonl") if r["partition"] == "val"}
    assert a != b


def test_caption_errors_including_missing_description_do_not_exclude_classification(tmp_path, base_images):
    images = deepcopy(base_images)
    images[0].update(
        metadata_valid=False, caption_structural_valid=False, caption_count=None,
        usable_caption_count=0,
        error_flags=["metadata.missing_description", "caption.missing_description"],
    )
    images[1].update(caption_structural_valid=False, usable_caption_count=9, error_flags=["caption.empty"])
    audit = write_audit(tmp_path / "audit", images)
    preparation().prepare_classification(audit, tmp_path / "out")
    rows = preparation().load_classification_manifest(tmp_path / "out")
    assert all(row["eligible"] and row["partition"] in {"train", "val"} for row in rows[:2])
    assert "metadata.missing_description" in rows[0]["error_flags"]


def test_image_and_class_errors_have_explicit_exclusion_decisions(tmp_path, base_images):
    images = deepcopy(base_images)
    images[0].update(image_valid=False, image_status="decode_error", pixel_sha256=None, error_flags=["image.decode_failed"])
    images[1].update(metadata_valid=False, class_id=None, class_name=None, error_flags=["metadata.invalid_label"])
    audit = write_audit(tmp_path / "audit", images)
    result = preparation().prepare_classification(audit, tmp_path / "out")
    rows = preparation().load_classification_manifest(tmp_path / "out")
    assert result["eligible_counts"]["train"] == 5992
    assert [row["partition"] for row in rows[:2]] == ["excluded", "excluded"]
    assert "image.decode_failed" in rows[0]["decision_codes"]
    assert "metadata.invalid_label" in rows[1]["decision_codes"]


def test_invalid_test_image_retains_its_official_assignment(tmp_path, base_images):
    images = deepcopy(base_images)
    images[5994].update(image_valid=False, image_status="fetch_error", encoded_sha256=None, pixel_sha256=None, error_flags=["image.fetch_failed"])
    audit = write_audit(tmp_path / "audit", images)
    result = preparation().prepare_classification(audit, tmp_path / "out")
    row = preparation().load_classification_manifest(tmp_path / "out")[5994]
    assert row["source_split"] == row["partition"] == "test"
    assert row["eligible"] is False
    assert result["partition_counts"]["test"] == 5794
    assert result["eligible_counts"]["test"] == 5793


@pytest.mark.parametrize("changes", [
    {"coverage_complete": False}, {"run_status": "incomplete"},
    {"fatal_errors": ["interruption"]},
    {"processed_counts": {"train": 5993, "test": 5794}},
    {"row_read_errors": {"train": 1, "test": 0}},
    {"source_revisions": {"train": "a" * 40, "test": PINS["test"][1]}},
    {"schema_version": 99}, {"pixel_hash_algorithm": "unknown"},
])
def test_incomplete_or_mixed_audit_cannot_publish_preparation(tmp_path, base_images, changes):
    audit = write_audit(tmp_path / "audit", deepcopy(base_images), summary_changes=changes)
    with pytest.raises(ValueError):
        preparation().prepare_classification(audit, tmp_path / "out")
    assert not (tmp_path / "out/classification_manifest.jsonl").exists()


@pytest.mark.parametrize("mutation", ["missing_row", "duplicate_row", "identity", "class_name", "unknown_error", "unknown_warning", "bad_flags", "bad_hash"])
def test_malformed_source_identity_metadata_or_flags_are_rejected(tmp_path, base_images, mutation):
    images = deepcopy(base_images)
    if mutation == "missing_row":
        images.pop()
    elif mutation == "duplicate_row":
        images[-1] = deepcopy(images[-2])
    elif mutation == "identity":
        images[0]["image_id"] += ":other"
    elif mutation == "class_name":
        images[0]["class_name"] = "Wrong mapping"
    elif mutation == "unknown_error":
        images[0]["error_flags"] = ["caption.future_unknown"]
    elif mutation == "unknown_warning":
        images[0]["warning_flags"] = ["quality.future_unknown"]
    elif mutation == "bad_flags":
        images[0]["error_flags"] = "image.decode_failed"
    else:
        images[0]["pixel_sha256"] = "not-a-digest"
    audit = write_audit(tmp_path / "audit", images)
    with pytest.raises(ValueError):
        preparation().prepare_classification(audit, tmp_path / "out")


@pytest.mark.parametrize("mutation", ["missing_group", "extra_member", "label_conflict", "cross_split", "unreported_hash_match"])
def test_inconsistent_or_leaking_exact_groups_are_rejected(tmp_path, base_images, mutation):
    images = deepcopy(base_images)
    group = _group(images, [1712, 1731])
    groups = [group]
    if mutation == "missing_group":
        groups = []
    elif mutation == "extra_member":
        group["image_ids"].append("missing:image")
    elif mutation == "label_conflict":
        images[1731].update(class_id=58, class_name="Bird 58")
    elif mutation == "cross_split":
        images[1731]["exact_group_ids"] = []
        images[1731]["warning_flags"] = []
        images[1731]["encoded_sha256"] = f"{1732:064x}"
        images[1731]["pixel_sha256"] = f"{1732:064x}"
        groups = [_group(images, [1712, 5994 + 1653])]
        groups[0]["crosses_official_split"] = False  # Member evidence still exposes overlap.
    else:
        images[0]["encoded_sha256"] = images[1]["encoded_sha256"]
    audit = write_audit(tmp_path / "audit", images, groups)
    with pytest.raises(ValueError):
        preparation().prepare_classification(audit, tmp_path / "out")


def test_a_class_with_one_indivisible_group_fails_instead_of_leaking(tmp_path, base_images):
    images = deepcopy(base_images)
    members = [i for i, row in enumerate(images) if row["source_split"] == "train" and row["class_id"] == 57]
    group = _group(images, members)
    audit = write_audit(tmp_path / "audit", images, [group])
    with pytest.raises(ValueError, match="class|split|group"):
        preparation().prepare_classification(audit, tmp_path / "out")


@pytest.mark.parametrize("kwargs", [
    {"val_fraction": float("nan")}, {"val_fraction": 0}, {"val_fraction": 1},
    {"seed": True}, {"seed": -1}, {"seed": 2**32}, {"seed": 0.5},
])
def test_invalid_configuration_is_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        preparation().prepare_classification(tmp_path / "missing", tmp_path / "out", **kwargs)


def test_manifest_tampering_is_rejected_by_reader(tmp_path, base_images):
    audit = write_audit(tmp_path / "audit", deepcopy(base_images))
    preparation().prepare_classification(audit, tmp_path / "out")
    manifest = tmp_path / "out/classification_manifest.jsonl"
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash|fingerprint|manifest"):
        preparation().load_classification_manifest(tmp_path / "out")


def test_publication_failure_invalidates_a_prior_complete_summary(tmp_path, base_images, monkeypatch):
    audit = write_audit(tmp_path / "audit", deepcopy(base_images))
    out = tmp_path / "out"
    out.mkdir()
    (out / "preparation.json").write_text('{"complete": true}', encoding="utf-8")

    def fail_write(*args, **kwargs):
        raise OSError("synthetic output failure")

    monkeypatch.setattr(preparation(), "_write_manifest", fail_write)
    with pytest.raises(OSError):
        preparation().prepare_classification(audit, out)
    assert json.loads((out / "preparation.json").read_text(encoding="utf-8"))["complete"] is False

@pytest.mark.parametrize("field", ["eligible_counts", "eligible_class_counts", "seed"])
def test_reader_rejects_inconsistent_preparation_summary(tmp_path, base_images, field):
    audit = write_audit(tmp_path / "audit", deepcopy(base_images))
    preparation().prepare_classification(audit, tmp_path / "out")
    path = tmp_path / "out/preparation.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    if field == "seed":
        summary[field] = True
    elif field == "eligible_counts":
        summary[field]["train"] = 0
    else:
        summary[field]["val"]["0"] = 0
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError):
        preparation().load_classification_manifest(tmp_path / "out")

@pytest.mark.parametrize("field,value", [
    ("source_id", None), ("source_id", True), ("source_id", []),
    ("file_name", None), ("file_name", False), ("file_name", []),
])
def test_impossible_valid_provenance_is_rejected(tmp_path, base_images, field, value):
    images = deepcopy(base_images)
    images[0][field] = value
    audit = write_audit(tmp_path / "audit", images)
    with pytest.raises(ValueError):
        preparation().prepare_classification(audit, tmp_path / "out")


@pytest.mark.parametrize("changes", [
    {"schema_version": True},
    {"row_read_errors": {"train": False, "test": 0}},
])
def test_boolean_summary_counts_or_version_are_not_integers(tmp_path, base_images, changes):
    audit = write_audit(tmp_path / "audit", deepcopy(base_images), summary_changes=changes)
    with pytest.raises(ValueError):
        preparation().prepare_classification(audit, tmp_path / "out")


@pytest.mark.parametrize("mutation", ["seed", "membership"])
def test_reader_checks_declared_split_recipe(tmp_path, base_images, mutation):
    audit = write_audit(tmp_path / "audit", deepcopy(base_images))
    preparation().prepare_classification(audit, tmp_path / "out")
    path = tmp_path / "out/preparation.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "seed":
        summary["seed"] = 43
    else:
        manifest = tmp_path / "out/classification_manifest.jsonl"
        rows = read_rows(manifest)
        train = next(row for row in rows if row["class_id"] == 0 and row["partition"] == "train")
        val = next(row for row in rows if row["class_id"] == 0 and row["partition"] == "val")
        train["partition"], val["partition"] = "val", "train"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        summary["manifest_sha256"] = sha256(manifest.read_bytes()).hexdigest()
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="split|seed|membership"):
        preparation().load_classification_manifest(tmp_path / "out")


@pytest.mark.parametrize("key", ["error_flag_counts_by_split", "warning_flag_counts_by_split"])
def test_audit_split_counters_must_match_rows(tmp_path, base_images, key):
    audit = write_audit(tmp_path / "audit", deepcopy(base_images), summary_changes={key: {"train": {"image.decode_failed": 700}, "test": {}}})
    with pytest.raises(ValueError, match="summary|split"):
        preparation().prepare_classification(audit, tmp_path / "prepared")


def test_unchecked_image_without_source_failure_is_rejected(tmp_path, base_images):
    images = deepcopy(base_images)
    images[0].update(
        image_valid=False, image_status="not_checked",
        encoded_sha256=None, pixel_sha256=None, width=None, height=None,
    )
    audit = write_audit(tmp_path / "audit", images)
    with pytest.raises(ValueError, match="unchecked|not.checked|uninspected"):
        preparation().prepare_classification(audit, tmp_path / "prepared")
    assert not (tmp_path / "prepared/classification_manifest.jsonl").exists()


def test_malformed_source_row_remains_accounted_for_with_explicit_exclusion(tmp_path, base_images):
    images = deepcopy(base_images)
    images[0].update(
        source_id=None, file_name=None, class_id=None, class_name=None,
        metadata_valid=False, image_valid=False, image_status="not_checked",
        encoded_sha256=None, pixel_sha256=None, width=None, height=None,
        image_format=None, image_mode=None,
        caption_count=None, usable_caption_count=0, caption_structural_valid=False,
        error_flags=["metadata.invalid_row"],
    )
    audit = write_audit(tmp_path / "audit", images)
    preparation().prepare_classification(audit, tmp_path / "prepared")
    rows = preparation().load_classification_manifest(tmp_path / "prepared")
    assert len(rows) == 11788
    assert rows[0]["partition"] == "excluded" and rows[0]["eligible"] is False
    assert rows[0]["decision_codes"] == ["class.invalid", "image.not_decodable", "metadata.invalid_row"]
