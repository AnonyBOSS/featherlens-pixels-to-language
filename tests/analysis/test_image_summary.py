"""Hand-computable image-record summaries without decoding source assets."""

from copy import deepcopy
import json

import pytest

from src.analysis.image_summary import summarize_image_measurements


NAMES = ("Bird A", "Bird B", "Unseen bird")


def measurement(position, *, class_id=0, status="ok", white=False):
    width, height = (3, 2) if white else (1, 1)
    channel = 1.0 if white else 0.0
    return {
        "image_id": f"image-{position}",
        "class_id": class_id,
        "class_name": NAMES[class_id],
        "source_row": position,
        "status": status,
        "source_status": "ok" if status == "unsupported_mode" else status,
        "metadata_errors": (),
        "reason": None if status == "ok" else "synthetic failure or excluded mode",
        "original_mode": "RGB" if status == "ok" else "RGBA",
        "original_width": width,
        "original_height": height,
        "exif_orientation": None,
        "oriented_width": width if status != "decode_error" else None,
        "oriented_height": height if status != "decode_error" else None,
        "metrics": {
            "width": width,
            "height": height,
            "pixel_count": width * height,
            "megapixels": width * height / 1_000_000,
            "aspect_ratio": width / height,
            "rgb_mean": [channel, channel, channel],
            "rgb_std": [0.0, 0.0, 0.0],
            "brightness_mean": channel,
            "contrast_std": 0.0,
            "saturation_mean": 0.0,
        } if status == "ok" else None,
    }


@pytest.fixture
def rows():
    black = measurement(0)
    white = measurement(1, class_id=1, white=True)
    unsupported = measurement(2, status="unsupported_mode")
    unsupported.update(original_width=4, original_height=5, oriented_width=4, oriented_height=5)
    decode_error = measurement(3, class_id=1, status="decode_error")
    decode_error.update(original_mode=None, original_width=8, original_height=4)
    return [black, white, unsupported, decode_error]


def test_attempted_decoded_and_pixel_eligible_denominators_are_separate(rows):
    summary = summarize_image_measurements(rows, class_names=NAMES)

    assert summary["attempted_image_count"] == 4
    assert summary["decoded_image_count"] == 3
    assert summary["eligible_image_count"] == 2
    assert summary["ineligible_image_count"] == 2
    assert summary["decode_error_count"] == 1
    assert summary["unsupported_mode_count"] == 1
    assert summary["status_counts"] == {"ok": 2, "row_error": 0, "fetch_error": 0, "decode_error": 1, "unsupported_mode": 1}
    assert sum(summary["status_counts"].values()) == summary["attempted_image_count"]
    assert summary["original_mode_counts"] == {"RGB": 2, "RGBA": 1, "unknown": 1}
    assert len(summary["measurements"]) == 4


def test_geometry_summaries_have_independent_availability_denominators(rows):
    summary = summarize_image_measurements(rows, class_names=NAMES)

    assert summary["original_dimensions_available_count"] == 4
    assert summary["original_resolution"]["width"]["mean"] == 4.0
    assert summary["original_resolution"]["pixel_count"]["mean"] == 14.75
    assert summary["oriented_dimensions_available_count"] == 3
    assert summary["oriented_resolution"]["width"]["mean"] == pytest.approx(8 / 3)
    assert summary["oriented_resolution"]["pixel_count"]["mean"] == 9.0
    assert summary["metric_summaries"]["pixel_count"]["count"] == 2
    assert summary["metric_summaries"]["pixel_count"]["mean"] == 3.5


def test_equal_image_rgb_and_brightness_are_not_pooled_pixel_normalization(rows):
    summary = summarize_image_measurements(rows, class_names=NAMES)

    assert summary["weighting"] == "equal_eligible_image"
    brightness = summary["metric_summaries"]["brightness_mean"]
    assert brightness["count"] == 2
    assert brightness["mean"] == brightness["median"] == 0.5
    assert brightness["p95"] == pytest.approx(0.95)
    assert brightness["mean"] != pytest.approx(6 / 7)  # White image has six pixels.
    for channel in ("red", "green", "blue"):
        assert summary["metric_summaries"]["rgb_mean"][channel]["mean"] == 0.5
        assert summary["metric_summaries"]["rgb_std"][channel]["mean"] == 0.0


def test_compact_class_summaries_keep_absent_classes_and_failure_counts(rows):
    classes = summarize_image_measurements(rows, class_names=NAMES)["class_summaries"]

    assert [row["class_name"] for row in classes] == list(NAMES)
    assert [row["attempted_image_count"] for row in classes] == [2, 2, 0]
    assert [row["eligible_image_count"] for row in classes] == [1, 1, 0]
    assert classes[0]["status_counts"] == {"ok": 1, "row_error": 0, "fetch_error": 0, "decode_error": 0, "unsupported_mode": 1}
    assert classes[1]["status_counts"] == {"ok": 1, "row_error": 0, "fetch_error": 0, "decode_error": 1, "unsupported_mode": 0}
    assert classes[0]["metric_means"]["brightness_mean"] == 0.0
    assert classes[1]["metric_means"]["brightness_mean"] == 1.0
    assert classes[2]["metric_means"] == {
        "brightness_mean": None, "contrast_std": None,
        "saturation_mean": None, "aspect_ratio": None,
    }


def test_unknown_dimensions_are_excluded_only_from_their_explicit_dimension_count(rows):
    rows[3].update(original_width=None, original_height=None)
    summary = summarize_image_measurements(rows, class_names=NAMES)

    assert summary["attempted_image_count"] == 4
    assert summary["original_dimensions_available_count"] == 3
    assert summary["original_resolution"]["width"]["count"] == 3
    assert summary["measurements"][3]["original_width"] is None


def test_original_and_oriented_geometry_can_differ_without_applying_exif_policy():
    row = measurement(0, white=True)
    row.update(original_width=2, original_height=3, exif_orientation=6)
    summary = summarize_image_measurements([row], class_names=NAMES)

    assert summary["original_resolution"]["aspect_ratio"]["mean"] == pytest.approx(2 / 3)
    assert summary["oriented_resolution"]["aspect_ratio"]["mean"] == 1.5
    assert summary["measurements"][0]["exif_orientation"] == 6


def test_order_independence_json_safety_and_copied_measurement_rows(rows):
    rows[0].update(icc_profile_present=True, adapter_note="kept unchanged", exif_orientation=0)
    before = deepcopy(rows)
    summary = summarize_image_measurements(iter(reversed(rows)), class_names=NAMES)

    assert rows == before
    assert summary == summarize_image_measurements(rows, class_names=NAMES)
    assert [row["source_row"] for row in summary["measurements"]] == [0, 1, 2, 3]
    assert summary["measurements"][0]["icc_profile_present"] is True
    assert summary["measurements"][0]["adapter_note"] == "kept unchanged"
    assert summary["measurements"][0]["exif_orientation"] == 0  # Raw unusual tag is retained.
    json.dumps(summary, allow_nan=False)
    summary["measurements"][0]["metrics"]["rgb_mean"][0] = 0.25
    assert rows == before


@pytest.mark.parametrize("empty", [True, False])
def test_no_eligible_pixels_yields_null_summaries_without_losing_attempts(empty):
    rows = [] if empty else [measurement(0, status="unsupported_mode"), measurement(1, status="decode_error")]
    summary = summarize_image_measurements(rows, class_names=NAMES)

    assert summary["attempted_image_count"] == len(rows)
    assert summary["eligible_image_count"] == 0
    assert summary["metric_summaries"]["brightness_mean"]["count"] == 0
    assert summary["metric_summaries"]["brightness_mean"]["mean"] is None
    assert summary["metric_summaries"]["rgb_mean"]["red"]["mean"] is None
    assert len(summary["measurements"]) == len(rows)
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("identity", ["image_id", "source_row"])
def test_repeated_identities_are_rejected_instead_of_silently_deduplicated(identity):
    first = measurement(0)
    second = measurement(1)
    second[identity] = first[identity]

    with pytest.raises(ValueError, match="unique"):
        summarize_image_measurements([first, second], class_names=NAMES)


@pytest.mark.parametrize(
    "field,value",
    [
        ("image_id", ""), ("source_row", True), ("source_row", -1),
        ("class_id", True), ("class_id", 99), ("class_name", "Wrong bird"),
        ("status", "silently_skipped"), ("reason", {}), ("original_mode", 42),
        ("original_width", 0), ("original_width", None),
        ("oriented_width", 0), ("exif_orientation", 1.5),
        ("icc_profile_present", 1), ("metrics", None),
    ],
)
def test_invalid_record_contract_values_fail_explicitly(field, value):
    row = measurement(0)
    row[field] = value
    with pytest.raises(ValueError):
        summarize_image_measurements([row], class_names=NAMES)


@pytest.mark.parametrize(
    "field,value",
    [
        ("width", 2), ("height", True), ("pixel_count", 2),
        ("megapixels", 1.0), ("aspect_ratio", 0.0),
        ("brightness_mean", float("nan")), ("contrast_std", float("inf")),
        ("saturation_mean", -0.01), ("saturation_mean", 1.01),
        ("contrast_std", 0.6), ("rgb_mean", [0, 0]),
        ("rgb_mean", [0, 0, 1.01]), ("rgb_std", [-0.01, 0, 0]),
        ("rgb_std", [0, 0, 0.6]), ("rgb_std", "RGB"),
    ],
)
def test_malformed_nonfinite_out_of_range_or_inconsistent_metrics_are_rejected(field, value):
    row = measurement(0)
    row["metrics"][field] = value
    with pytest.raises(ValueError):
        summarize_image_measurements([row], class_names=NAMES)


def test_missing_fields_failed_rows_with_metrics_and_non_json_metadata_are_rejected():
    missing = measurement(0)
    del missing["metrics"]["brightness_mean"]
    with pytest.raises(ValueError, match="Missing"):
        summarize_image_measurements([missing], class_names=NAMES)

    failed = measurement(0)
    failed.update(status="decode_error", source_status="decode_error")
    with pytest.raises(ValueError, match="must be null"):
        summarize_image_measurements([failed], class_names=NAMES)

    unsupported = measurement(0, status="unsupported_mode")
    unsupported.update(oriented_width=None, oriented_height=None)
    with pytest.raises(ValueError, match="decoded rows"):
        summarize_image_measurements([unsupported], class_names=NAMES)

    invalid_extra = measurement(0)
    invalid_extra["adapter_note"] = float("nan")
    with pytest.raises(ValueError, match="JSON-safe"):
        summarize_image_measurements([invalid_extra], class_names=NAMES)


@pytest.mark.parametrize("names", ["Bird", ("Bird", "Bird"), ("",)])
def test_invalid_class_name_mappings_are_rejected(names):
    with pytest.raises(ValueError):
        summarize_image_measurements([], class_names=names)


@pytest.mark.parametrize("failure_status", ["row_error", "fetch_error", "decode_error"])
def test_source_failures_are_preserved_separately_with_unknown_header_facts(failure_status):
    row = measurement(0, status=failure_status)
    row.update(
        class_id=None, class_name=None, original_mode=None,
        original_width=None, original_height=None, oriented_width=None,
        oriented_height=None, icc_profile_present=None, has_transparency=None,
        metadata_errors=("metadata.invalid_label",),
    )
    summary = summarize_image_measurements([row], class_names=NAMES)

    assert summary["attempted_image_count"] == 1
    assert summary["decoded_image_count"] == summary["eligible_image_count"] == 0
    assert summary["status_counts"][failure_status] == 1
    assert summary[f"{failure_status}_count"] == 1
    assert sum(summary["status_counts"].values()) == 1
    assert summary["unclassified_image_count"] == 1
    assert summary["unclassified_eligible_image_count"] == 0
    assert summary["unclassified_status_counts"][failure_status] == 1
    assert summary["original_dimensions_available_count"] == 0
    assert summary["oriented_dimensions_available_count"] == 0
    assert summary["measurements"][0]["source_status"] == failure_status
    assert summary["measurements"][0]["icc_profile_present"] is None
    assert summary["measurements"][0]["metadata_errors"] == ["metadata.invalid_label"]
    json.dumps(summary, allow_nan=False)


def test_null_class_keeps_successful_pixels_in_global_statistics_and_metadata_errors():
    classified = measurement(0)
    unclassified = measurement(1, white=True)
    unclassified.update(class_id=None, class_name=None, metadata_errors=("metadata.invalid_label", "metadata.invalid_filename"))
    summary = summarize_image_measurements([classified, unclassified], class_names=NAMES)

    assert summary["eligible_image_count"] == 2
    assert summary["metric_summaries"]["brightness_mean"]["mean"] == 0.5
    assert summary["unclassified_image_count"] == summary["unclassified_eligible_image_count"] == 1
    assert summary["unclassified_status_counts"]["ok"] == 1
    assert sum(group["eligible_image_count"] for group in summary["class_summaries"]) == 1
    assert summary["class_summaries"][0]["metric_means"]["brightness_mean"] == 0.0
    assert summary["metadata_error_image_count"] == 1
    assert summary["metadata_error_count"] == 2
    assert summary["metadata_error_counts"] == {"metadata.invalid_filename": 1, "metadata.invalid_label": 1}


def test_oriented_dimensions_retained_on_failure_are_still_described(rows):
    rows[3].update(original_width=8, original_height=4, oriented_width=4, oriented_height=8)
    summary = summarize_image_measurements(rows, class_names=NAMES)

    assert summary["decoded_image_count"] == 3
    assert summary["oriented_dimensions_available_count"] == 4
    assert summary["oriented_resolution"]["width"]["mean"] == 3.0
    assert summary["oriented_resolution"]["pixel_count"]["mean"] == 14.75
    assert summary["metric_summaries"]["width"]["count"] == 2
    assert summary["measurements"][3]["metrics"] is None


@pytest.mark.parametrize(
    "updates",
    [
        {"class_id": None}, {"class_name": None},
        {"source_status": "decode_error"}, {"source_status": "unsupported_mode"},
        {"has_transparency": 1}, {"metadata_errors": "metadata.invalid_label"},
        {"metadata_errors": [None]},
    ],
)
def test_invalid_class_pairs_source_status_and_optional_facts_are_rejected(updates):
    row = measurement(0)
    row.update(updates)
    with pytest.raises(ValueError):
        summarize_image_measurements([row], class_names=NAMES)


def test_analysis_mode_exclusion_retains_original_success_status():
    row = measurement(0, status="unsupported_mode")
    row.update(reason="transparency_metadata_present", has_transparency=True, icc_profile_present=None)
    summary = summarize_image_measurements([row], class_names=NAMES)

    assert summary["decoded_image_count"] == 1
    assert summary["eligible_image_count"] == 0
    assert summary["unsupported_mode_count"] == 1
    assert summary["measurements"][0]["status"] == "unsupported_mode"
    assert summary["measurements"][0]["source_status"] == "ok"
    assert summary["measurements"][0]["has_transparency"] is True
    assert summary["measurements"][0]["icc_profile_present"] is None


def test_source_fact_counts_distinguish_false_unknown_and_analysis_exclusion(rows):
    rows[0].update(has_transparency=True, icc_profile_present=False)
    rows[1].update(has_transparency=False, icc_profile_present=True)
    rows[2].update(has_transparency=None, icc_profile_present=None, reason="numeric_original_mode")
    del rows[2]["source_status"]  # Legacy consumer row: excluded pixels still decoded.
    summary = summarize_image_measurements(rows, class_names=NAMES)

    assert summary["source_status_counts"] == {"ok": 3, "row_error": 0, "fetch_error": 0, "decode_error": 1}
    assert summary["status_counts"]["ok"] == 2
    assert summary["source_fact_counts"] == {
        "has_transparency": {"true": 1, "false": 1, "unknown": 2},
        "icc_profile_present": {"true": 1, "false": 1, "unknown": 2},
    }
    assert summary["color_exclusion_reason_counts"] == {"numeric_original_mode": 1}
    for counts in summary["source_fact_counts"].values():
        assert sum(counts.values()) == summary["attempted_image_count"]


def test_orientation_tag_counts_and_known_geometry_difference_are_observations():
    rotated = measurement(0, white=True)
    rotated.update(original_width=2, original_height=3, exif_orientation=6)
    unchanged = measurement(1)
    unchanged["exif_orientation"] = 1
    failed = measurement(2, status="decode_error")
    failed.update(original_width=None, original_height=None, exif_orientation=6)
    summary = summarize_image_measurements([rotated, unchanged, failed], class_names=NAMES)

    assert summary["exif_orientation_counts"] == {"1": 1, "6": 2}
    assert summary["original_oriented_size_difference_count"] == 1
    assert summary["source_status_counts"]["decode_error"] == 1
    assert summary["decoded_image_count"] == 2
    assert summary["eligible_image_count"] == 2  # EXIF presence is not a quality exclusion.
    missing = summarize_image_measurements([measurement(0)], class_names=NAMES)
    assert missing["exif_orientation_counts"] == {"missing": 1}
