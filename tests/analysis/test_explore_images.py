"""Exercise the EDA consumer against the real shared adapter on tiny images."""
from dataclasses import replace
from io import BytesIO
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from src.analysis.explore_images import build_image_report, main, prepare_gallery
from src.data.image_io import read_source_image
from src.data.schemas import SourceSpec


def encoded(mode="RGB", color=(255, 0, 0), size=(4, 2), *, orientation=None):
    stream = BytesIO()
    with Image.new(mode, size, color) as image:
        if orientation is not None:
            exif = Image.Exif()
            exif[274] = orientation
            image.save(stream, format="PNG", exif=exif)
        else:
            image.save(stream, format="PNG")
    return {"bytes": stream.getvalue(), "path": "must-not-open.jpg"}


def source(references=None):
    references = references if references is not None else [encoded(), encoded("L", 128)]
    rows = [{"image": reference, "label": index % 2, "id": index,
             "file_name": f"bird-{index}.png", "description": "A red bird.\nSmall wings."}
            for index, reference in enumerate(references)]
    return SimpleNamespace(spec=SourceSpec("synthetic/cub", "train", "revision"),
                           class_names=("Red", "Blue"), dataset=rows)


def test_shared_adapter_is_used_and_native_metrics_are_measured_without_source_changes():
    data = source()
    original = list(data.dataset)
    calls, images = [], []
    def reader(received, index):
        assert received is data
        calls.append(index)
        view = read_source_image(received, index)
        images.append(view.image)
        return view
    report = build_image_report(data, reader=reader, expected_records=2)
    assert calls == [0, 1]
    assert report["population"]["full_split_verified"] is True
    assert report["images"]["attempted_image_count"] == report["images"]["eligible_image_count"] == 2
    assert report["images"]["measurements"][0]["metrics"]["brightness_mean"] == pytest.approx(.299)
    assert report["images"]["measurements"][1]["metrics"]["rgb_mean"] == pytest.approx([128 / 255] * 3)
    assert data.dataset == original
    for image in images:
        with pytest.raises(ValueError):
            np.asarray(image)


def test_shared_orientation_and_alpha_facts_are_consumed_without_reimplementing_decoder():
    data = source([encoded(size=(4, 2), orientation=6), encoded("RGBA", (10, 20, 30, 100))])
    report = build_image_report(data, expected_records=2)
    rotated, alpha = report["images"]["measurements"]
    assert (rotated["original_width"], rotated["original_height"]) == (4, 2)
    assert (rotated["oriented_width"], rotated["oriented_height"]) == (2, 4)
    assert rotated["metrics"]["aspect_ratio"] == .5
    assert alpha["source_status"] == "ok"
    assert alpha["status"] == "unsupported_mode"
    assert alpha["reason"] == "transparency_metadata_present"
    assert alpha["metrics"] is None
    assert report["images"]["decoded_image_count"] == 2
    assert report["images"]["eligible_image_count"] == 1
    assert report["images"]["original_dimensions_available_count"] == 2


def test_decode_and_fetch_failures_retain_identity_and_explicit_denominators():
    data = source([encoded(), {"bytes": b"not an image", "path": None}, {}])
    report = build_image_report(data, expected_records=3)
    rows = report["images"]["measurements"]
    assert [row["source_status"] for row in rows] == ["ok", "decode_error", "fetch_error"]
    assert [row["image_id"] for row in rows] == [data.spec.image_id(i) for i in range(3)]
    assert report["images"]["attempted_image_count"] == 3
    assert report["images"]["eligible_image_count"] == 1
    assert report["images"]["decode_error_count"] == 1
    assert report["images"]["fetch_error_count"] == 1


def test_missing_class_is_retained_in_global_pixels_but_not_assigned_to_a_species():
    data = source([encoded()])
    data.dataset[0]["label"] = "invalid"
    report = build_image_report(data, expected_records=1)
    row = report["images"]["measurements"][0]
    assert row["class_id"] is row["class_name"] is None
    assert "metadata.invalid_label" in row["metadata_errors"]
    assert report["images"]["eligible_image_count"] == 1
    assert report["images"]["unclassified_image_count"] == 1
    assert sum(group["attempted_image_count"] for group in report["images"]["class_summaries"]) == 0


def test_row_access_error_keeps_the_attempt_without_making_up_class_metadata():
    data = source()
    class InterruptedRows(list):
        def __getitem__(self, index):
            if index == 1:
                raise OSError("synthetic row error")
            return super().__getitem__(index)
    data.dataset = InterruptedRows(data.dataset)
    report = build_image_report(data, expected_records=2)
    row = report["images"]["measurements"][1]
    assert row["status"] == row["source_status"] == "row_error"
    assert row["image_id"] == data.spec.image_id(1)
    assert row["class_id"] is None
    assert report["images"]["attempted_image_count"] == 2
    assert report["images"]["row_error_count"] == 1


def test_numeric_original_mode_is_not_treated_as_scene_rgb_and_pixels_are_closed():
    data = source([encoded()])
    view = replace(read_source_image(data, 0), original_mode="F")
    report = build_image_report(data, expected_records=1, reader=lambda *_: view)
    row = report["images"]["measurements"][0]
    assert row["source_status"] == "ok"
    assert row["status"] == "unsupported_mode"
    assert row["reason"] == "unsupported_original_mode"
    assert row["metrics"] is None
    assert report["images"]["decoded_image_count"] == 1
    with pytest.raises(ValueError):
        np.asarray(view.image)


def test_gallery_uses_matching_shared_pixels_and_preserves_reference_caption():
    data = source()
    report = build_image_report(data, expected_records=2)
    getter = prepare_gallery(report, data)
    first = report["images"]["measurements"][0]
    np.testing.assert_array_equal(getter(first), np.full((2, 4, 3), [255, 0, 0], dtype=np.uint8))
    assert first["caption_text"] == "A red bird."
    assert first["caption_index"] == 0


def test_gallery_rejects_a_different_source_revision():
    data = source()
    report = build_image_report(data, expected_records=2)
    data.spec = replace(data.spec, revision="different")
    with pytest.raises(ValueError, match="revision"):
        prepare_gallery(report, data)


@pytest.mark.parametrize("change", ["size", "class_mapping", "image_id"])
def test_gallery_rejects_incompatible_saved_source_before_reading_images(monkeypatch, change):
    import src.data.image_io as image_io
    data = source()
    report = build_image_report(data, expected_records=2)
    if change == "size":
        data.dataset.append(dict(data.dataset[0]))
    elif change == "class_mapping":
        data.class_names = ("Changed", "Blue")
    else:
        report["images"]["measurements"][0]["image_id"] = "wrong-identity"
    monkeypatch.setattr(image_io, "read_source_image", lambda *_: pytest.fail("Incompatible report must fail before image access"))

    with pytest.raises(ValueError):
        prepare_gallery(report, data)
    assert all("caption_text" not in row for row in report["images"]["measurements"])


def test_gallery_rejects_stale_species_labels_even_with_identical_pixels():
    data = source()
    report = build_image_report(data, expected_records=2)
    data.dataset[0]["label"] = 1
    with pytest.raises(ValueError, match="class metadata"):
        prepare_gallery(report, data)


def test_cli_keeps_completed_measurements_when_gallery_rendering_fails(monkeypatch, tmp_path):
    import src.analysis.explore_images as explore_images
    import src.visualization.image_plots as image_plots
    data = source()
    output = tmp_path / "report.json"
    monkeypatch.setitem(sys.modules, "src.data.source", SimpleNamespace(load_cub_split=lambda *args, **kwargs: data))
    monkeypatch.setattr(explore_images, "prepare_gallery", lambda report, source: lambda row: None)
    def fail_render(*args, **kwargs):
        raise OSError("simulated gallery rendering failure")
    monkeypatch.setattr(image_plots, "generate_image_figures", fail_render)
    with pytest.raises(OSError, match="simulated gallery"):
        main(["--expected-records", "2", "--output", str(output), "--figures-dir", str(tmp_path / "figures")])
    import json
    assert json.loads(output.read_text(encoding="utf-8"))["images"]["attempted_image_count"] == 2
    assert not (tmp_path / "figures").exists()


@pytest.mark.parametrize("reference_count", [1, 3])
def test_reference_mismatch_does_not_truncate_or_invent_images(reference_count):
    report = build_image_report(source(), expected_records=reference_count)
    assert report["population"]["records_analyzed"] == 2
    assert report["population"]["source_exhausted"] is True
    assert report["population"]["full_split_verified"] is False


def test_prefix_never_claims_full_split_and_never_reads_past_its_limit():
    data = source()
    visited = []
    def reader(received, index):
        visited.append(index)
        return read_source_image(received, index)
    report = build_image_report(data, limit=1, expected_records=1, reader=reader)
    assert visited == [0]
    assert report["population"]["selection_method"] == "source_order_prefix"
    assert report["population"]["full_split_verified"] is False


def test_nontraining_source_is_rejected_before_reading_images():
    data = source()
    data.spec = SourceSpec("synthetic/test", "test", "revision")
    def forbidden_reader(*args):
        raise AssertionError("Held-out images must never be opened")
    with pytest.raises(ValueError, match="train"):
        build_image_report(data, reader=forbidden_reader)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_limits_fail_before_reading(limit):
    with pytest.raises(ValueError, match="positive integer"):
        build_image_report(source(), limit=limit)


def test_streaming_source_is_rejected():
    data = source()
    data.dataset = iter(data.dataset)
    with pytest.raises(ValueError, match="indexed"):
        build_image_report(data)


def test_bad_identity_closes_pixels_and_does_not_certify_report():
    data = source()
    view = read_source_image(data, 0)
    def incorrect_view(*args):
        return replace(view, image_id="wrong")
    with pytest.raises(ValueError, match="identity"):
        build_image_report(data, reader=incorrect_view)
    with pytest.raises(ValueError):
        np.asarray(view.image)


def test_unexpected_reader_failure_propagates():
    def broken_reader(*args):
        raise RuntimeError("interrupted source")
    with pytest.raises(RuntimeError, match="interrupted"):
        build_image_report(source(), reader=broken_reader)


@pytest.mark.parametrize("arguments", [["--split", "test"], ["--limit", "0"], ["--expected-records", "0"]])
def test_invalid_cli_arguments_never_open_a_source(monkeypatch, arguments):
    def forbidden_loader(*args, **kwargs):
        raise AssertionError("Invalid CLI must fail before loader")
    monkeypatch.setitem(sys.modules, "src.data.source", SimpleNamespace(load_cub_split=forbidden_loader))
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
