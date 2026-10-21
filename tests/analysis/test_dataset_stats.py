"""Small metadata fixtures with independently known image counts."""

from dataclasses import replace
import json

import pytest

from src.analysis.dataset_stats import summarize_dataset
from src.data.schemas import CaptionRecord, ImageRecord


NAMES = ("Red bird", "Blue bird", "Green bird")


def image(row, class_id=0, split="train", **changes):
    record = ImageRecord(
        image_id=f"hf:birds/{split}@revision:{split}:{row}",
        source_repo=f"birds/{split}",
        source_revision="revision",
        source_split=split,
        source_row=row,
        source_id=row,
        file_name=f"bird_{row}.jpg",
        class_id=class_id,
        class_name=NAMES[class_id] if 0 <= class_id < len(NAMES) else "Unknown",
    )
    return replace(record, **changes)


def test_full_mapping_includes_zeros_and_split_totals_reconcile():
    records = [image(0), image(1), image(2, class_id=1), image(0, class_id=1, split="test")]

    result = summarize_dataset(iter(records), class_names=NAMES)

    assert result["total_records"] == result["unique_image_ids"] == 4
    assert result["repeated_image_id_rows"] == 0
    assert result["expected_classes"] == 3
    assert result["observed_classes"] == 2
    assert result["missing_class_ids"] == [2]
    assert result["class_distribution"] == [
        {"class_id": 0, "class_name": "Red bird", "count": 2, "proportion": 0.5},
        {"class_id": 1, "class_name": "Blue bird", "count": 2, "proportion": 0.5},
        {"class_id": 2, "class_name": "Green bird", "count": 0, "proportion": 0.0},
    ]
    assert result["class_count_summary"]["count"] == 3
    assert result["class_count_summary"]["min"] == 0
    assert result["class_count_summary"]["max"] == 2
    assert result["class_count_summary"]["mean"] == pytest.approx(4 / 3)
    assert result["imbalance_ratio"] is None
    assert [entry["count"] for entry in result["source_splits"]["train"]["class_distribution"]] == [2, 1, 0]
    assert [entry["count"] for entry in result["source_splits"]["test"]["class_distribution"]] == [0, 1, 0]
    assert result["source_splits"]["train"]["class_coverage"] == pytest.approx(2 / 3)
    assert result["source_splits"]["test"]["missing_class_ids"] == [0, 2]
    assert sum(split["total_records"] for split in result["source_splits"].values()) == 4
    assert sum(entry["count"] for entry in result["class_distribution"]) == 4
    assert sum(entry["record_count"] for entry in result["source_provenance"]) == 4
    json.dumps(result, allow_nan=False)


def test_imbalance_ratio_uses_expected_classes_when_all_are_observed():
    records = [image(0), image(1), image(2), image(3, class_id=1), image(4, class_id=2)]

    result = summarize_dataset(records, class_names=NAMES)

    assert result["imbalance_ratio"] == 3.0
    assert result["missing_class_ids"] == []


def test_empty_images_keep_expected_class_order_and_zero_counts():
    result = summarize_dataset([], class_names=NAMES)

    assert result["total_records"] == result["unique_image_ids"] == 0
    assert result["observed_classes"] == 0
    assert result["missing_class_ids"] == [0, 1, 2]
    assert all(entry["count"] == 0 and entry["proportion"] == 0.0 for entry in result["class_distribution"])
    assert result["class_count_summary"]["mean"] == 0.0
    assert result["imbalance_ratio"] is None
    assert result["source_provenance"] == []
    assert result["missing_metadata"] == {"source_id": 0, "file_name": 0}
    assert set(result["source_splits"]) == {"train", "test"}
    json.dumps(result, allow_nan=False)


def test_empty_mapping_and_input_have_no_numeric_class_observations():
    result = summarize_dataset([], class_names=())

    assert result["class_distribution"] == []
    assert result["class_count_summary"]["count"] == 0
    assert result["class_count_summary"]["mean"] is None
    assert result["imbalance_ratio"] is None
    assert result["source_splits"]["train"]["class_coverage"] is None


def test_caption_counts_do_not_weight_image_distribution():
    paired_records = [
        (image(0), ()),
        (image(1), (CaptionRecord("image-1", 0, "red bird", "red bird"),)),
        (image(2, class_id=1), tuple(CaptionRecord("image-2", i, "blue bird", "blue bird") for i in range(10))),
    ]

    result = summarize_dataset((record for record, captions in paired_records), class_names=NAMES)

    assert result["total_records"] == 3
    assert [entry["count"] for entry in result["class_distribution"]] == [2, 1, 0]


def test_repeated_metadata_rows_are_reported_without_silent_deduplication():
    record = image(0)

    result = summarize_dataset([record, record], class_names=NAMES)

    assert result["total_records"] == 2
    assert result["unique_image_ids"] == 1
    assert result["repeated_image_id_rows"] == 1
    assert result["class_distribution"][0]["count"] == 2
    assert result["source_splits"]["train"]["total_records"] == 2


def test_missing_metadata_counts_none_and_blank_names_without_treating_zero_id_as_missing():
    records = [
        image(0),
        image(1, source_id=None, file_name=None),
        image(2, file_name="  \t"),
        image(3, file_name=""),
    ]

    result = summarize_dataset(records, class_names=NAMES)

    assert result["missing_metadata"] == {"source_id": 1, "file_name": 3}
    assert result["source_provenance"] == [
        {"source_repo": "birds/train", "source_revision": "revision", "source_split": "train", "record_count": 4}
    ]


@pytest.mark.parametrize("names", [("Bird", "Bird"), ("Bird", ""), ("Bird", None), "Bird"])
def test_invalid_class_mapping_is_rejected(names):
    with pytest.raises(ValueError, match="class_names"):
        summarize_dataset([], class_names=names)


@pytest.mark.parametrize("class_id", [-1, 3, True, 0.5])
def test_invalid_labels_are_rejected(class_id):
    record = replace(image(0), class_id=class_id)

    with pytest.raises(ValueError, match="class_id"):
        summarize_dataset([record], class_names=NAMES)


def test_class_name_must_match_ordered_source_mapping():
    with pytest.raises(ValueError, match="Class name"):
        summarize_dataset([image(0, class_name="Blue bird")], class_names=NAMES)


def test_conflicting_metadata_for_same_image_id_is_rejected():
    record = image(0)

    with pytest.raises(ValueError, match="Conflicting metadata"):
        summarize_dataset([record, replace(record, source_id=42)], class_names=NAMES)


def test_generated_validation_split_cannot_be_confused_with_official_source_split():
    with pytest.raises(ValueError, match="official source_split"):
        summarize_dataset([image(0, source_split="validation")], class_names=NAMES)
