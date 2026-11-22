import json
import pytest
from src.eda.stats import summarize_numbers, write_json_report
import src.eda.stats as reporting
from dataclasses import replace
from src.eda.stats import summarize_dataset
from src.data.source import CaptionRecord, ImageRecord
import copy
from src.eda.stats import summarize_captions, tokenize_caption
from src.data.source import parse_captions
from src.data.source import ImageRecord
import numpy as np
from src.eda.stats import measure_resolution, measure_rgb_array


def test_summary_uses_linear_quantiles_and_handles_singletons():
    result = summarize_numbers([4, 1, 3, 2])
    assert result["count"] == 4
    assert result["mean"] == result["median"] == 2.5
    assert result["min"] == 1
    assert result["max"] == 4
    assert result["p05"] == pytest.approx(1.15)
    assert result["p25"] == pytest.approx(1.75)
    assert result["p75"] == pytest.approx(3.25)
    assert result["p95"] == pytest.approx(3.85)
    assert result["p99"] == pytest.approx(3.97)
    single = summarize_numbers([7])
    assert single["count"] == 1
    assert all(value == 7 for key, value in single.items() if key != "count")


def test_empty_summary_has_null_measurements_and_strict_json():
    result = summarize_numbers([])
    assert result["count"] == 0
    assert all(value is None for key, value in result.items() if key != "count")
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_values_are_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        summarize_numbers([value])


def test_json_output_roundtrips_unicode_and_rejects_nan_before_writing(tmp_path):
    output = tmp_path / "nested" / "report.json"
    report = {"word": "caf\u00e9", "summary": summarize_numbers([1, 2])}
    assert write_json_report(report, output) == output
    assert json.loads(output.read_text(encoding="utf-8")) == report
    with pytest.raises(ValueError):
        write_json_report({"bad": float("nan")}, output)
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_failed_report_publication_keeps_previous_report_and_cleans_temporary_file(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    write_json_report({"valid": "previous"}, output)
    before = output.read_bytes()
    def fail_replace(*args):
        raise OSError("simulated publication failure")
    monkeypatch.setattr(reporting.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated publication"):
        write_json_report({"valid": "new"}, output)
    assert output.read_bytes() == before
    assert list(tmp_path.iterdir()) == [output]


def test_invalid_report_does_not_create_a_directory(tmp_path):
    directory = tmp_path / "absent"
    with pytest.raises(ValueError):
        write_json_report({"bad": float("inf")}, directory / "report.json")
    assert not directory.exists()


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


def image_and_captions(image_id, description):
    image = ImageRecord(
        image_id=image_id,
        source_repo="synthetic",
        source_revision="test",
        source_split="train",
        source_row=0,
        source_id=None,
        file_name=None,
        class_id=0,
        class_name="Synthetic bird",
    )
    return image, parse_captions(description, image_id=image_id)


def test_tokenization_handles_unicode_words_joiners_and_numeric_separators():
    text = "CAFÉ Cafe\u0301 bird’s blue-green R2D2 123 foo_bar 'wings' naïve\u2011bird"

    assert tokenize_caption(text) == (
        "café", "café", "bird's", "blue-green", "r", "d", "foo", "bar", "wings",
        "naïve\u2011bird",
    )
    assert tokenize_caption("!!! 123 ___ -- ''") == ()
    assert tokenize_caption("bird--like ‘crest’") == ("bird", "like", "crest")


def test_hand_counted_denominators_lengths_vocabulary_and_whitespace():
    records = [
        image_and_captions("image-a", "  Red  bird.\nRed bird.\n\n!!!"),
        image_and_captions("image-b", "Red bird.\nBlue bird"),
        image_and_captions("image-c", None),
    ]

    summary = summarize_captions(records, expected_captions_per_image=2, top_k=2)

    assert summary["image_count"] == 3
    assert summary["caption_line_count"] == 6
    assert summary["nonempty_caption_count"] == 5
    assert summary["empty_caption_count"] == 1
    assert summary["images_without_caption_lines"] == 1
    assert summary["images_without_nonempty_captions"] == 1
    assert summary["per_image_line_counts"]["mean"] == 2
    assert summary["per_image_nonempty_counts"]["mean"] == pytest.approx(5 / 3)
    assert summary["line_count_mismatch_images"] == 2
    assert summary["nonempty_count_mismatch_images"] == 2
    assert summary["character_lengths"]["count"] == 5
    assert summary["character_lengths"]["mean"] == pytest.approx(7.8)
    assert summary["word_lengths"]["mean"] == pytest.approx(1.6)
    assert summary["word_lengths"]["min"] == 0
    assert summary["zero_word_caption_count"] == 1
    assert summary["word_length_distribution"] == [
        {"word_count": 0, "caption_count": 1},
        {"word_count": 2, "caption_count": 4},
    ]
    assert summary["character_length_distribution"] == [
        {"character_count": 3, "caption_count": 1},
        {"character_count": 9, "caption_count": 4},
    ]
    assert summary["per_image_nonempty_count_distribution"] == [
        {"caption_count": 0, "image_count": 1},
        {"caption_count": 2, "image_count": 1},
        {"caption_count": 3, "image_count": 1},
    ]
    for distribution in ("word_length_distribution", "character_length_distribution"):
        assert sum(row["caption_count"] for row in summary[distribution]) == summary["nonempty_caption_count"]
    image_distribution = summary["per_image_nonempty_count_distribution"]
    assert sum(row["image_count"] for row in image_distribution) == summary["image_count"]
    assert sum(row["caption_count"] * row["image_count"] for row in image_distribution) == summary["nonempty_caption_count"]
    assert summary["total_tokens"] == 8
    assert sum(
        row["word_count"] * row["caption_count"]
        for row in summary["word_length_distribution"]
    ) == summary["total_tokens"]
    assert summary["vocabulary_size"] == 3
    assert summary["token_frequencies"] == {"bird": 4, "blue": 1, "red": 3}
    assert summary["top_tokens"] == [
        {"token": "bird", "count": 4}, {"token": "red", "count": 3}
    ]
    assert summary["hapax_word_types"] == 1
    assert summary["hapax_token_share"] == pytest.approx(1 / 8)
    assert summary["whitespace_anomaly_count"] == 1
    assert summary["shortest_examples"][0] == {
        "image_id": "image-a", "caption_index": 3, "text": "!!!", "word_count": 0
    }


def test_repetition_separates_within_image_from_across_image_occurrences():
    records = [
        image_and_captions("image-a", "Same\nSame\nOnly\nOnly\n\n"),
        image_and_captions("image-b", "Same\nOnly\nOther"),
        image_and_captions("image-c", "Same\nOnly\nonly"),
    ]

    summary = summarize_captions(records)
    within = summary["repetition"]["within_images"]
    across = summary["repetition"]["across_images"]

    assert within["repeated_distinct_texts"] == 2
    assert within["repeated_image_text_groups"] == 2
    assert within["affected_image_count"] == 1  # Two repeating texts in one image.
    assert within["extra_occurrences"] == 2
    assert across["repeated_distinct_texts"] == 2
    assert across["extra_occurrences"] == 4
    assert [example["text"] for example in across["examples"]] == ["Only", "Same"]
    assert across["examples"][0] == {
        "text": "Only",
        "occurrence_count": 4,
        "image_count": 3,
        "image_ids": ["image-a", "image-b", "image-c"],
        "extra_occurrences": 2,
    }
    assert within["extra_occurrences"] + across["extra_occurrences"] == 6
    assert summary["repetition"]["unique_nonempty_texts"] == 4
    assert summary["repetition"]["global_extra_occurrences"] == 6
    assert (
        summary["repetition"]["global_extra_occurrences"]
        == summary["nonempty_caption_count"] - summary["repetition"]["unique_nonempty_texts"]
        == within["extra_occurrences"] + across["extra_occurrences"]
    )


def test_missing_blank_and_punctuation_only_captions_remain_distinguishable():
    summary = summarize_captions(
        [
            image_and_captions("missing", None),
            image_and_captions("blank", " \t\n"),
            image_and_captions("punctuation", "!!!"),
        ]
    )

    assert summary["image_count"] == 3
    assert summary["caption_line_count"] == 2
    assert summary["empty_caption_count"] == 1
    assert summary["nonempty_caption_count"] == 1
    assert summary["images_without_caption_lines"] == 1
    assert summary["images_without_nonempty_captions"] == 2
    assert summary["word_lengths"]["count"] == 1
    assert summary["word_lengths"]["mean"] == 0
    assert summary["total_tokens"] == 0
    assert summary["hapax_token_share"] is None
    assert summary["whitespace_anomaly_count"] == 1
    assert summary["repetition"]["within_images"]["repeated_distinct_texts"] == 0
    assert summary["repetition"]["within_images"]["affected_image_count"] == 0


def test_empty_input_is_json_safe_and_has_no_invented_observations():
    summary = summarize_captions(iter(()), top_k=0)

    assert summary["image_count"] == 0
    assert summary["word_lengths"]["count"] == 0
    assert summary["word_lengths"]["mean"] is None
    assert summary["per_image_line_counts"]["min"] is None
    assert summary["line_count_mismatch_images"] == 0
    assert summary["top_tokens"] == []
    assert summary["shortest_examples"] == []
    assert summary["longest_examples"] == []
    assert summary["hapax_token_share"] is None
    assert summary["word_length_distribution"] == []
    assert summary["character_length_distribution"] == []
    assert summary["per_image_nonempty_count_distribution"] == []
    assert summary["repetition"]["unique_nonempty_texts"] == 0
    assert summary["repetition"]["global_extra_occurrences"] == 0
    assert summary["repetition"]["within_images"]["affected_image_count"] == 0
    json.dumps(summary, allow_nan=False)


def test_summary_is_order_independent_and_does_not_mutate_source_records():
    records = [
        image_and_captions("image-b", "  Bird’s  wing \nblue-green"),
        image_and_captions("image-a", "Bird’s wing\nBlue-green"),
    ]
    before = copy.deepcopy(records)

    summary = summarize_captions(iter(records))

    assert records == before
    assert summary == summarize_captions(reversed(records))
    assert summary["token_frequencies"] == {"bird's": 2, "blue-green": 2, "wing": 2}
    assert summary["top_tokens"] == [
        {"token": "bird's", "count": 2},
        {"token": "blue-green", "count": 2},
        {"token": "wing", "count": 2},
    ]
    assert summary["repetition"]["across_images"]["repeated_distinct_texts"] == 1
    json.dumps(summary, allow_nan=False)


def test_duplicate_images_and_mislinked_captions_are_rejected():
    record = image_and_captions("image-a", "Bird")
    with pytest.raises(ValueError, match="unique"):
        summarize_captions([record, record])

    image, captions = record
    mislinked = replace(captions[0], image_id="another-image")
    with pytest.raises(ValueError, match="match"):
        summarize_captions([(image, [mislinked])])


@pytest.mark.parametrize("kwargs", [{"top_k": -1}, {"expected_captions_per_image": -1}])
def test_negative_summary_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError, match="non-negative"):
        summarize_captions([], **kwargs)


def test_resolution_reports_nonsquare_native_dimensions_and_pixel_units():
    assert measure_resolution(4, 2) == {
        "width": 4,
        "height": 2,
        "pixel_count": 8,
        "megapixels": 0.000008,
        "aspect_ratio": 2.0,
    }
    json.dumps(measure_resolution(np.int64(4), np.int64(2)), allow_nan=False)


@pytest.mark.parametrize(
    "color,brightness,saturation",
    [
        ([0, 0, 0], 0.0, 0.0),
        ([255, 255, 255], 1.0, 0.0),
        ([255, 0, 0], 0.299, 1.0),
        ([0, 255, 0], 0.587, 1.0),
        ([0, 0, 255], 0.114, 1.0),
    ],
)
def test_constant_black_white_and_primary_colors_have_known_statistics(
    color, brightness, saturation
):
    rgb = np.array([[color]], dtype=np.uint8)

    result = measure_rgb_array(rgb)

    assert result["rgb_mean"] == pytest.approx([channel / 255 for channel in color])
    assert result["rgb_std"] == [0.0, 0.0, 0.0]
    assert result["brightness_mean"] == pytest.approx(brightness)
    assert result["contrast_std"] == 0.0
    assert result["saturation_mean"] == pytest.approx(saturation)
    assert result["pixel_count"] == 1
    json.dumps(result, allow_nan=False)


def test_two_tone_grayscale_uses_population_contrast_and_has_no_saturation():
    rgb = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)

    result = measure_rgb_array(rgb)

    assert result["rgb_mean"] == pytest.approx([0.5, 0.5, 0.5])
    assert result["rgb_std"] == pytest.approx([0.5, 0.5, 0.5])
    assert result["brightness_mean"] == pytest.approx(0.5)
    assert result["contrast_std"] == pytest.approx(0.5)
    assert result["saturation_mean"] == 0.0


def test_nonsquare_red_blue_image_measures_every_pixel_with_explicit_luma_weights():
    red = [255, 0, 0]
    blue = [0, 0, 255]
    rgb = np.array([[red, red, blue], [blue, red, blue]], dtype=np.uint8)

    result = measure_rgb_array(rgb)

    assert result["width"] == 3
    assert result["height"] == 2
    assert result["pixel_count"] == 6
    assert result["megapixels"] == pytest.approx(0.000006)
    assert result["aspect_ratio"] == 1.5
    assert result["rgb_mean"] == pytest.approx([0.5, 0.0, 0.5])
    assert result["rgb_std"] == pytest.approx([0.5, 0.0, 0.5])
    assert result["brightness_mean"] == pytest.approx((0.299 + 0.114) / 2)
    assert result["contrast_std"] == pytest.approx((0.299 - 0.114) / 2)
    assert result["saturation_mean"] == 1.0


def test_uint8_and_normalized_float_views_agree_without_mutating_inputs():
    encoded = np.array(
        [[[0, 120, 255], [20, 30, 40]], [[255, 255, 255], [0, 0, 0]]],
        dtype=np.uint8,
    )
    normalized = encoded.astype(np.float64) / 255
    encoded_before = encoded.copy()
    normalized_before = normalized.copy()

    encoded_result = measure_rgb_array(encoded)
    normalized_result = measure_rgb_array(normalized)

    for key, value in encoded_result.items():
        assert value == pytest.approx(normalized_result[key])
    np.testing.assert_array_equal(encoded, encoded_before)
    np.testing.assert_array_equal(normalized, normalized_before)
    json.dumps(normalized_result, allow_nan=False)


@pytest.mark.parametrize(
    "rgb",
    [
        np.array([[[np.nan, 0, 0]]]),
        np.array([[[np.inf, 0, 0]]]),
        np.array([[[-0.01, 0, 0]]]),
        np.array([[[1.01, 0, 0]]]),
    ],
)
def test_nonfinite_and_out_of_range_float_values_are_rejected(rgb):
    with pytest.raises(ValueError, match="finite|\\[0, 1\\]"):
        measure_rgb_array(rgb)


@pytest.mark.parametrize(
    "rgb",
    [
        np.zeros((2, 2)),
        np.zeros((2, 2, 4)),
        np.zeros((1, 2, 2, 3)),
        np.empty((0, 2, 3)),
        np.empty((2, 0, 3)),
    ],
)
def test_wrong_shape_and_empty_images_are_rejected(rgb):
    with pytest.raises(ValueError, match="shape"):
        measure_rgb_array(rgb)


@pytest.mark.parametrize("dtype", [np.uint16, np.int32, np.bool_, np.complex128, object])
def test_ambiguous_integer_depths_and_unsupported_dtypes_are_rejected(dtype):
    with pytest.raises(TypeError, match="dtype"):
        measure_rgb_array(np.zeros((1, 1, 3), dtype=dtype))


@pytest.mark.parametrize("width,height", [(0, 1), (1, 0), (-1, 2)])
def test_nonpositive_dimensions_are_rejected(width, height):
    with pytest.raises(ValueError, match="positive"):
        measure_resolution(width, height)


@pytest.mark.parametrize("width,height", [(True, 1), (1, False), (1.5, 2), (2, "3")])
def test_noninteger_dimensions_are_rejected(width, height):
    with pytest.raises(TypeError, match="integers"):
        measure_resolution(width, height)
