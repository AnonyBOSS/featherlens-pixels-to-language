"""Synthetic membership joins using real raw-report builders and injected verification."""

from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from src.analysis.explore_caption_details import build_caption_details_report
from src.analysis.explore_images import build_image_report
from src.analysis.prepared_eda import build_prepared_training_report
from src.data.captions import parse_captions
from src.data.schemas import ImageRecord, SourceSpec


NAMES = ("Red bird", "Blue bird")


def encoded(color):
    stream = BytesIO()
    with Image.new("RGB", (2, 1), color) as image:
        image.save(stream, format="PNG")
    return {"bytes": stream.getvalue(), "path": None}


def raw_reports(source):
    return (
        build_image_report(source, expected_records=len(source.dataset)),
        build_caption_details_report(source, expected_records=len(source.dataset)),
    )


@pytest.fixture
def preparation(tmp_path):
    spec = SourceSpec("synthetic/cub", "train", "training-revision")
    dataset = [
        {"image": encoded(color), "label": label, "id": index, "file_name": f"bird-{index}.png", "description": description}
        for index, (color, label, description) in enumerate((
            ((255, 0, 0), 0, "red bird\nred wing"),
            ((255, 0, 0), 0, "red bird\nred wing"),
            ((0, 0, 255), 1, "blue bird\nblue wing"),
            ((128, 128, 128), 1, None),
        ))
    ]

    def records():
        for position, row in enumerate(dataset):
            identity = spec.image_id(position)
            record = ImageRecord(identity, spec.repo_id, spec.revision, "train", position, row["id"], row["file_name"], row["label"], NAMES[row["label"]])
            yield record, parse_captions(row["description"], image_id=identity)

    source = SimpleNamespace(spec=spec, class_names=NAMES, dataset=dataset, iter_records=records)
    image_report, caption_report = raw_reports(source)
    group_ids = [spec.image_id(0), spec.image_id(1)]
    group_id = "exact:" + sha256("\n".join(sorted(group_ids)).encode()).hexdigest()
    rows = []
    for position, partition in enumerate(("train", "train", "val", "excluded")):
        row = dataset[position]
        rows.append({
            "image_id": spec.image_id(position), "source_repo": spec.repo_id,
            "source_revision": spec.revision, "source_split": "train", "source_row": position,
            "source_id": row["id"], "file_name": row["file_name"],
            "class_id": row["label"], "class_name": NAMES[row["label"]],
            "partition": partition, "eligible": partition != "excluded",
            "error_flags": ["image.decode_failed"] if partition == "excluded" else [],
            "warning_flags": (["duplicate.exact", "caption.count_mismatch"] if position == 0 else ["duplicate.exact"] if position == 1 else []),
            "exact_group_ids": [group_id] if position < 2 else [],
            "encoded_sha256": sha256(row["image"]["bytes"]).hexdigest(),
            "pixel_sha256": sha256(b"synthetic-pixel-evidence" + row["image"]["bytes"]).hexdigest() if partition != "excluded" else None,
        })
    test_spec = SourceSpec("synthetic/cub-test", "test", "test-revision")
    for position in range(2):
        rows.append({
            "image_id": test_spec.image_id(position), "source_repo": test_spec.repo_id,
            "source_revision": test_spec.revision, "source_split": "test", "source_row": position,
            "source_id": position, "file_name": f"test-{position}.png", "class_id": None, "class_name": None,
            "partition": "test", "eligible": False, "exact_group_ids": [],
            "error_flags": ["metadata.missing_label"], "warning_flags": [],
            "encoded_sha256": None, "pixel_sha256": None,
        })
    summary = {
        "schema_version": 1, "complete": True,
        "policy_version": "classification-v1", "split_version": "stratified-exact-group-v1",
        "pixel_hash_algorithm": "exif-rgb8-white:v1", "seed": 42, "val_fraction": 0.25,
        "source_revisions": {"train": spec.revision, "test": test_spec.revision},
        "source_counts": {"train": 4, "test": 2}, "class_names": list(NAMES),
        "partition_counts": {"train": 2, "val": 1, "test": 2, "excluded": 1},
        "manifest_sha256": "a" * 64,
        "audit_fingerprints": {"summary.json": "b" * 64, "images.jsonl": "c" * 64, "exact_groups.jsonl": "d" * 64},
        "group_constraints": {group_id: group_ids},
    }
    path = tmp_path / "preparation.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return SimpleNamespace(directory=tmp_path, path=path, summary=summary, rows=rows, source=source, image_report=image_report, caption_report=caption_report)


def analyze(data, *, loader=None):
    return build_prepared_training_report(
        data.directory, image_report=data.image_report, caption_report=data.caption_report,
        manifest_loader=(lambda directory: data.rows) if loader is None else loader,
    )


def save_summary(data):
    data.path.write_text(json.dumps(data.summary), encoding="utf-8")


def test_membership_counts_quality_groups_and_numeric_denominators_reconcile(preparation):
    data = preparation
    before = deepcopy((data.rows, data.image_report, data.caption_report))
    calls = []

    def loader(directory):
        calls.append(directory)
        return data.rows

    report = analyze(data, loader=loader)

    assert calls == [data.directory]
    assert report["schema_version"] == "prepared-eda-1.0"
    assert report["population"] == {
        "raw_training_image_count": 4, "derived_training_image_count": 2,
        "validation_image_count": 1, "excluded_training_image_count": 1,
        "unknown_class_training_image_count": 0,
    }
    assert report["classes"][0]["partition_counts"] == {"train": 2, "val": 0, "excluded": 0}
    assert report["classes"][1]["partition_counts"] == {"train": 0, "val": 1, "excluded": 1}
    assert report["class_count_distributions"] == {
        "raw": [{"images_per_class": 2, "class_count": 2}],
        "train": [{"images_per_class": 0, "class_count": 1}, {"images_per_class": 2, "class_count": 1}],
        "val": [{"images_per_class": 0, "class_count": 1}, {"images_per_class": 1, "class_count": 1}],
    }
    assert report["quality"]["overall"]["error_code_image_counts"] == {"image.decode_failed": 1}
    assert report["quality"]["overall"]["warning_code_image_counts"] == {"caption.count_mismatch": 1, "duplicate.exact": 2}
    assert report["quality"]["by_partition"]["train"]["warning_code_image_counts"]["duplicate.exact"] == 2
    assert report["exact_groups"]["training_group_count"] == 1
    assert report["exact_groups"]["training_group_image_count"] == 2
    assert [member["source_row"] for member in report["exact_groups"]["groups"][0]["members"]] == [0, 1]
    assert report["exact_groups"]["groups"][0]["partition_counts"] == {"train": 2, "val": 0, "excluded": 0}
    assert report["partitions"]["train"]["image_count"] == report["partitions"]["train"]["eligible_pixel_image_count"] == 2
    assert report["partitions"]["train"]["image_metrics"]["brightness_mean"]["mean"] == pytest.approx(.299)
    assert report["partitions"]["val"]["image_metrics"]["brightness_mean"]["mean"] == pytest.approx(.114)
    assert report["partitions"]["train"]["caption_metrics"]["mean_caption_words"]["mean"] == 2
    assert report["partitions"]["train"]["caption_metrics"]["type_token_ratio"]["mean"] == pytest.approx(.75)
    assert report["official_test"] == {"identity_count": 2, "preserved_test_partition_count": 2, "identities_preserved": True, "image_caption_statistics_used": False}
    assert report["preparation"]["manifest_sha256"] == "a" * 64
    assert report["preparation"]["preparation_sha256"] == sha256(data.path.read_bytes()).hexdigest()
    assert "test" not in report["partitions"]
    assert before == (data.rows, data.image_report, data.caption_report)
    json.dumps(report, allow_nan=False)


def test_default_public_loader_is_imported_lazily_and_used(preparation, monkeypatch):
    data = preparation
    calls = []
    monkeypatch.setitem(sys.modules, "src.data.preparation", SimpleNamespace(load_classification_manifest=lambda directory: calls.append(directory) or data.rows))

    report = build_prepared_training_report(data.directory, image_report=data.image_report, caption_report=data.caption_report)

    assert calls == [data.directory]
    assert report["population"]["derived_training_image_count"] == 2


def test_excluded_unknown_class_is_retained_without_imputing_eda_species(preparation):
    data = preparation
    data.rows[3].update(class_id=None, class_name=None)
    data.rows[3]["error_flags"].append("metadata.missing_label")

    report = analyze(data)

    assert report["population"]["unknown_class_training_image_count"] == 1
    assert report["quality"]["by_partition"]["excluded"]["unknown_class_image_count"] == 1
    assert sum(row["partition_counts"]["excluded"] for row in report["classes"]) == 0
    assert report["population"]["excluded_training_image_count"] == 1
    assert report["class_count_distributions"]["raw"] == [{"images_per_class": 1, "class_count": 1}, {"images_per_class": 2, "class_count": 1}]


@pytest.mark.parametrize("case", ["missing", "duplicate", "revision", "class", "counts"])
def test_inconsistent_manifest_joins_or_counts_fail(preparation, case):
    data = preparation
    if case == "missing":
        data.rows.pop(3)
    elif case == "duplicate":
        data.rows.append(deepcopy(data.rows[0]))
    elif case == "revision":
        data.rows[2]["source_revision"] = "different-revision"
    elif case == "class":
        data.rows[2].update(class_id=0, class_name=NAMES[0])
    else:
        data.summary["partition_counts"].update(train=3, val=0)
        save_summary(data)

    with pytest.raises(ValueError):
        analyze(data)


def test_official_test_partition_is_preserved_even_for_ineligible_rows(preparation):
    data = preparation
    data.rows[4]["partition"] = "excluded"
    with pytest.raises(ValueError, match="Official test identity"):
        analyze(data)


def test_shared_exact_group_cannot_straddle_train_and_validation(preparation):
    data = preparation
    data.rows[1]["partition"] = "val"
    data.summary["partition_counts"].update(train=1, val=2)
    save_summary(data)
    with pytest.raises(ValueError, match="overlaps derived train and val"):
        analyze(data)


@pytest.mark.parametrize("field", ["encoded_sha256", "pixel_sha256"])
def test_supplied_hash_train_val_overlap_is_rejected_without_decoding(preparation, field):
    data = preparation
    data.rows[2][field] = data.rows[0][field]
    with pytest.raises(ValueError, match=f"{field} membership overlaps"):
        analyze(data)


def test_equal_source_ids_across_partitions_do_not_imply_duplicate_content(preparation):
    data = preparation
    data.source.dataset[2]["id"] = 0
    data.rows[2]["source_id"] = 0
    data.image_report, data.caption_report = raw_reports(data.source)
    report = analyze(data)

    assert report["population"]["derived_training_image_count"] == 2
    assert report["population"]["validation_image_count"] == 1
    assert "source_id" not in report["exact_groups"]["isolation_fields_checked"]
    assert "image_id" in report["exact_groups"]["isolation_fields_checked"]


@pytest.mark.parametrize("field,replacement", [("source_id", 999), ("file_name", "different.png")])
def test_source_id_and_filename_must_still_match_the_same_canonical_eda_row(preparation, field, replacement):
    data = preparation
    data.rows[2][field] = replacement
    with pytest.raises(ValueError, match=f"Manifest/EDA {field} mismatch"):
        analyze(data)


def test_preparation_changed_by_loader_is_not_certified(preparation):
    data = preparation

    def changing_loader(directory):
        data.path.write_bytes(data.path.read_bytes() + b" ")
        return data.rows

    with pytest.raises(ValueError, match="changed while loading"):
        analyze(data, loader=changing_loader)


def test_incomplete_metadata_and_nonfull_reports_fail_before_loader(preparation):
    data = preparation
    forbidden = lambda _: pytest.fail("Unexpected manifest access")
    data.summary["complete"] = False
    save_summary(data)
    with pytest.raises(ValueError, match="complete"):
        analyze(data, loader=forbidden)
    data.summary["complete"] = True
    save_summary(data)
    data.caption_report = build_caption_details_report(data.source, limit=2, expected_records=4)
    with pytest.raises(ValueError, match="complete official training"):
        analyze(data, loader=forbidden)


def test_public_verifier_failure_propagates_without_new_decisions(preparation):
    def rejecting_loader(directory):
        raise ValueError("shared verification failed")

    with pytest.raises(ValueError, match="shared verification failed"):
        analyze(preparation, loader=rejecting_loader)


def test_saved_aspect_ratio_must_match_oriented_geometry(preparation):
    data = preparation
    data.image_report["images"]["measurements"][0]["metrics"]["aspect_ratio"] = 1.25
    with pytest.raises(ValueError, match="aspect_ratio differs from oriented dimensions"):
        analyze(data)


def test_class_count_distributions_preserve_zero_classes_and_reconcile_known_counts(preparation):
    data = preparation
    data.source.class_names = (*NAMES, "Absent bird")
    data.summary["class_names"] = list(data.source.class_names)
    save_summary(data)
    data.image_report, data.caption_report = raw_reports(data.source)

    report = analyze(data)

    assert report["class_count_distributions"] == {
        "raw": [{"images_per_class": 0, "class_count": 1}, {"images_per_class": 2, "class_count": 2}],
        "train": [{"images_per_class": 0, "class_count": 2}, {"images_per_class": 2, "class_count": 1}],
        "val": [{"images_per_class": 0, "class_count": 2}, {"images_per_class": 1, "class_count": 1}],
    }
    for scope, expected_images in (("raw", 4), ("train", 2), ("val", 1)):
        distribution = report["class_count_distributions"][scope]
        assert sum(row["class_count"] for row in distribution) == 3
        assert sum(row["images_per_class"] * row["class_count"] for row in distribution) == expected_images
