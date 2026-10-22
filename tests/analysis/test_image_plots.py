from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.analysis.image_stats import measure_rgb_array
from src.analysis.image_summary import summarize_image_measurements
from src.data.schemas import SourceSpec
import src.visualization.image_plots as image_plots
from src.visualization.image_plots import (
    create_image_appearance_figure,
    create_image_dimensions_figure,
    create_image_extremes_figure,
    create_image_gallery_figure,
    create_image_rgb_figure,
    generate_image_figures,
    select_extreme_rows,
    select_representative_rows,
)


SOURCE = SourceSpec("synthetic", "train", "test")
CLASS_NAMES = ("Red", "Gray", "Black")


def source_fields(source_row):
    return {
        "image_id": SOURCE.image_id(source_row), "source_row": source_row,
        "source_repo": SOURCE.repo_id, "source_revision": SOURCE.revision,
        "source_split": SOURCE.source_split,
    }


@pytest.fixture
def image_data():
    arrays = {
        "red": np.full((1, 2, 3), [255, 0, 0], dtype=np.uint8),
        "blue": np.full((2, 1, 3), [0, 0, 255], dtype=np.uint8),
        "gray": np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8),
        "black": np.zeros((2, 2, 3), dtype=np.uint8),
    }
    rows = []
    for position, (identity, array) in enumerate(arrays.items()):
        height, width = array.shape[:2]
        rows.append({
            "image_id": identity, "source_row": position,
            "class_id": max(0, position - 1), "class_name": ("Red", "Gray", "Black")[max(0, position - 1)],
            "status": "ok", "reason": None, "original_mode": "RGB",
            "original_width": width, "original_height": height,
            "oriented_width": width, "oriented_height": height,
            "exif_orientation": None, "metrics": measure_rgb_array(array),
            "caption_text": f"Reference caption for {identity}.",
        })
    rows.extend((
        {"image_id": "mode", "source_row": 4, "class_id": 0, "class_name": "Red", "status": "unsupported_mode", "reason": "alpha mode", "original_mode": "RGBA", "original_width": 7, "original_height": 5, "oriented_width": 5, "oriented_height": 7, "exif_orientation": 6, "metrics": None},
        {"image_id": "failed", "source_row": 5, "class_id": 0, "class_name": "Red", "status": "decode_error", "reason": "synthetic error", "original_mode": "RGB", "original_width": 11, "original_height": 9, "oriented_width": None, "oriented_height": None, "exif_orientation": None, "metrics": None},
    ))
    for row in rows:
        row.update(source_fields(row["source_row"]))
    arrays.update({SOURCE.image_id(index): array for index, array in enumerate(list(arrays.values()))})
    report = {
        "schema_version": "image-eda-1.0",
        "source": {"repo_id": "synthetic", "source_split": "train", "revision": "test"},
        "class_names": list(CLASS_NAMES),
        "population": {"selection_method": "complete_official_training_split", "requested_limit": None, "records_analyzed": 6, "indexed_source_size": 6, "source_exhausted": True, "reference_record_count": 6, "matches_reference_count": True, "full_split_verified": True, "random_sample": False},
        "images": summarize_image_measurements(rows, class_names=CLASS_NAMES),
    }
    return report, arrays


@pytest.fixture(autouse=True)
def close_new_figures():
    before = set(plt.get_fignums())
    yield
    for number in set(plt.get_fignums()) - before:
        plt.close(number)


def test_geometry_includes_known_failed_headers_and_decoded_unsupported_modes(image_data):
    report, _ = image_data
    figure = create_image_dimensions_figure(report)

    assert len(figure.axes[0].collections[0].get_offsets()) == 6
    assert list(figure.axes[0].collections[0].get_offsets()[-1]) == [11, 9]
    assert sum(bar.get_height() for bar in figure.axes[1].patches) == 5
    assert sum(bar.get_height() for bar in figure.axes[2].patches) == 5
    assert "n=6" in figure.axes[0].get_title()
    assert "n=5" in figure.axes[1].get_title()
    assert "Oriented width / height" == figure.axes[1].get_xlabel()


@pytest.mark.parametrize("builder", [create_image_appearance_figure, create_image_rgb_figure])
def test_pixel_histograms_count_images_equally_and_disclose_exclusions(image_data, builder):
    report, _ = image_data
    figure = builder(report)

    assert [sum(bar.get_height() for bar in axis.patches) for axis in figure.axes] == [4, 4, 4]
    assert "4 eligible / 6 attempted" in figure._suptitle.get_text()
    assert "1 mode/transparency exclusions" in figure._suptitle.get_text()
    assert "1 row/fetch/decode failures" in figure._suptitle.get_text()
    assert all(axis.get_ylabel() == "Eligible image count" for axis in figure.axes)


def test_gallery_selection_is_order_independent_and_extrema_ties_use_source_order(image_data):
    report, _ = image_data
    original = deepcopy(report)
    assert [row["image_id"] for row in select_representative_rows(report)] == [SOURCE.image_id(index) for index in (0, 2, 3)]
    selected = select_extreme_rows(report)
    assert [row["image_id"] for _, _, row in selected] == [SOURCE.image_id(index) for index in (3, 2, 0, 2, 2, 0)]
    report["images"]["measurements"].reverse()
    assert [row["image_id"] for row in select_representative_rows(report)] == [SOURCE.image_id(index) for index in (0, 2, 3)]
    assert [row["image_id"] for _, _, row in select_extreme_rows(report)] == [row["image_id"] for _, _, row in selected]
    report["images"]["measurements"].reverse()
    assert report == original


def test_galleries_display_supplied_actual_pixels_captions_and_descriptive_labels(image_data):
    report, arrays = image_data
    before = {identity: array.copy() for identity, array in arrays.items()}
    calls = []

    def getter(row):
        calls.append(row["image_id"])
        return arrays[row["image_id"]]

    figure = create_image_gallery_figure(report, getter)
    assert calls == [SOURCE.image_id(index) for index in (0, 2, 3)]
    np.testing.assert_array_equal(figure.axes[0].images[0].get_array(), arrays["red"])
    assert "source row 0" in figure.axes[0].get_title()
    assert figure.axes[0].texts[0].get_text() == "Reference caption for red."
    assert "not a random sample" in figure._supxlabel.get_text()
    extremes = create_image_extremes_figure(report, getter)
    assert len(extremes.axes) == 6
    assert "Lowest mean encoded luma" in extremes.axes[0].get_title()
    assert "not quality flags" in extremes._supxlabel.get_text()
    for identity in arrays:
        np.testing.assert_array_equal(arrays[identity], before[identity])


def test_all_five_groups_reuse_shared_bundle_export(image_data, monkeypatch, tmp_path):
    report, arrays = image_data
    captured = []

    def fake_export(figures, output_dir):
        captured.extend(figures)
        assert output_dir == tmp_path
        return [Path(output_dir) / f"{stem}.png" for stem, _ in figures]

    monkeypatch.setattr(image_plots, "save_figure_bundle", fake_export)
    paths = generate_image_figures(report, tmp_path, image_getter=lambda row: arrays[row["image_id"]])

    assert [stem for stem, _ in captured] == [
        "training_image_dimensions", "training_image_appearance", "training_image_rgb",
        "training_image_gallery", "training_image_extremes",
    ]
    assert len(paths) == 5
    assert all(callable(builder) for _, builder in captured)


def test_test_source_and_bad_counts_fail_before_export_or_gallery_access(image_data, monkeypatch, tmp_path):
    report, _ = image_data
    monkeypatch.setattr(image_plots, "save_figure_bundle", lambda *_: pytest.fail("Unexpected export"))
    getter = lambda _: pytest.fail("Unexpected gallery access")
    report["source"]["source_split"] = "test"
    with pytest.raises(ValueError, match="training"):
        generate_image_figures(report, tmp_path, image_getter=getter)
    report["source"]["source_split"] = "train"
    report["images"]["eligible_image_count"] = 5
    with pytest.raises(ValueError, match="eligible_image_count"):
        generate_image_figures(report, tmp_path, image_getter=getter)


def test_gallery_rejects_different_native_view_and_closes_figure(image_data):
    report, _ = image_data
    before = set(plt.get_fignums())
    with pytest.raises(ValueError, match="shape"):
        create_image_gallery_figure(report, lambda _: np.zeros((1, 1, 3), dtype=np.uint8))
    assert set(plt.get_fignums()) == before


def test_empty_and_unverified_prefix_scopes_remain_honest_and_do_not_request_pixels(image_data):
    report, _ = image_data
    report["images"] = summarize_image_measurements([], class_names=("Red", "Gray", "Black"))
    report["population"] = {"selection_method": "source_order_prefix", "records_analyzed": 0, "indexed_source_size": 0, "requested_limit": 10, "full_split_verified": False, "random_sample": False}

    gallery = create_image_gallery_figure(report, lambda _: pytest.fail("No eligible image exists"))
    assert "nonrandom source-order prefix" in gallery._suptitle.get_text()
    assert "0 eligible / 0 attempted" in gallery._suptitle.get_text()
    assert select_extreme_rows(report) == []
    dimensions = create_image_dimensions_figure(report)
    assert not len(dimensions.axes[0].collections[0].get_offsets())


def test_source_failures_and_unassigned_pixels_keep_global_denominators(image_data):
    report, arrays = image_data
    rows = deepcopy(report["images"]["measurements"])
    rows[0].update(class_id=None, class_name=None, source_status="ok")
    rows[5].update(oriented_width=11, oriented_height=9)
    for source_row, status in ((6, "row_error"), (7, "fetch_error")):
        rows.append({
            **source_fields(source_row),
            "class_id": None, "class_name": None,
            "status": status, "source_status": status, "reason": "synthetic source failure",
            "original_mode": None, "original_width": None, "original_height": None,
            "oriented_width": None, "oriented_height": None,
            "exif_orientation": None, "metrics": None,
        })
    report["images"] = summarize_image_measurements(rows, class_names=("Red", "Gray", "Black"))
    report["population"].update(records_analyzed=8, reference_record_count=8, indexed_source_size=8)

    appearance = create_image_appearance_figure(report)
    assert "4 eligible / 8 attempted" in appearance._suptitle.get_text()
    assert "3 row/fetch/decode failures" in appearance._suptitle.get_text()
    assert sum(bar.get_height() for bar in appearance.axes[0].patches) == 4
    geometry = create_image_dimensions_figure(report)
    assert sum(bar.get_height() for bar in geometry.axes[1].patches) == 6
    assert [row["image_id"] for row in select_representative_rows(report)] == [SOURCE.image_id(index) for index in (1, 2, 3, 0)]
    gallery = create_image_gallery_figure(report, lambda row: arrays[row["image_id"]])
    assert "Unassigned class" in gallery.axes[3].get_title()


@pytest.mark.parametrize("changes", [
    {"source_exhausted": False},
    {"requested_limit": 1},
    {"reference_record_count": 0},
    {"matches_reference_count": False},
    {"full_split_verified": False},
    {"random_sample": True},
    {"selection_method": "source_order_prefix", "requested_limit": 5, "full_split_verified": False},
    {"selection_method": "source_order_prefix", "requested_limit": 10, "full_split_verified": True},
])
def test_inconsistent_scope_is_rejected_before_export(image_data, changes, monkeypatch, tmp_path):
    report, _ = image_data
    report["population"].update(changes)
    monkeypatch.setattr(image_plots, "save_figure_bundle", lambda *_: pytest.fail("Unexpected export"))
    with pytest.raises(ValueError):
        generate_image_figures(report, tmp_path, image_getter=lambda _: pytest.fail("Unexpected pixels"))


def test_exhausted_reference_mismatch_is_accepted_without_complete_split_claim(image_data):
    report, _ = image_data
    report["population"].update(reference_record_count=99, matches_reference_count=False, full_split_verified=False)

    figure = create_image_rgb_figure(report)

    assert "reference count unverified" in figure._suptitle.get_text()
    assert "| complete split |" not in figure._suptitle.get_text()


@pytest.mark.parametrize("metric,value", [
    ("brightness_mean", float("nan")),
    ("contrast_std", 0.6),
    ("saturation_mean", -0.01),
    ("rgb_mean", [0, 0, 2]),
])
def test_invalid_pixel_plot_values_are_rejected_before_export(image_data, metric, value, monkeypatch, tmp_path):
    report, _ = image_data
    report["images"]["measurements"][0]["metrics"][metric] = value
    monkeypatch.setattr(image_plots, "save_figure_bundle", lambda *_: pytest.fail("Unexpected export"))
    with pytest.raises(ValueError):
        generate_image_figures(report, tmp_path, image_getter=lambda _: pytest.fail("Unexpected pixels"))


def test_duplicate_source_rows_and_partially_known_class_metadata_are_rejected(image_data):
    report, _ = image_data
    report["images"]["measurements"][1]["source_row"] = 0
    with pytest.raises(ValueError, match="Source rows"):
        create_image_rgb_figure(report)
    report["images"]["measurements"][1]["source_row"] = 1
    report["images"]["measurements"][0]["class_id"] = None
    with pytest.raises(ValueError, match="both known or both null"):
        create_image_rgb_figure(report)


def reject_without_changing_outputs(report, monkeypatch, tmp_path):
    destination = tmp_path / "existing-figures"
    destination.mkdir()
    sentinels = {
        "training_image_dimensions.png": b"prior image figure",
        "training_image_gallery.pdf": b"prior gallery figure",
        "unrelated.txt": b"keep this file",
    }
    for name, content in sentinels.items():
        (destination / name).write_bytes(content)
    monkeypatch.setattr(
        image_plots, "save_figure_bundle",
        lambda *args, **kwargs: pytest.fail("Invalid reports must not reach export"),
    )
    with pytest.raises(ValueError):
        generate_image_figures(
            report, destination,
            image_getter=lambda _: pytest.fail("Invalid reports must not request gallery pixels"),
        )
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == sentinels


@pytest.mark.parametrize("indexed_size", [-1, True, None, 6.5])
def test_invalid_indexed_size_preserves_existing_figures(image_data, indexed_size, monkeypatch, tmp_path):
    report, _ = image_data
    report["population"]["indexed_source_size"] = indexed_size

    reject_without_changing_outputs(report, monkeypatch, tmp_path)


def test_missing_indexed_size_is_not_inferred_from_measured_rows(image_data, monkeypatch, tmp_path):
    report, _ = image_data
    del report["population"]["indexed_source_size"]

    reject_without_changing_outputs(report, monkeypatch, tmp_path)


@pytest.mark.parametrize("indexed_size", [5, 7])
def test_complete_scope_must_cover_declared_indexed_source(image_data, indexed_size, monkeypatch, tmp_path):
    report, _ = image_data
    report["population"]["indexed_source_size"] = indexed_size

    reject_without_changing_outputs(report, monkeypatch, tmp_path)


def test_prefix_cannot_silently_stop_before_its_limit_and_source_end(image_data, monkeypatch, tmp_path):
    report, _ = image_data
    report["population"].update(
        selection_method="source_order_prefix", requested_limit=8,
        indexed_source_size=8, source_exhausted=False, full_split_verified=False,
    )

    reject_without_changing_outputs(report, monkeypatch, tmp_path)


def test_prefix_rows_must_cover_the_exact_source_order_range(image_data, monkeypatch, tmp_path):
    report, _ = image_data
    original = report["images"]["measurements"]
    report["images"] = summarize_image_measurements(
        [original[index] for index in (0, 2, 3)], class_names=CLASS_NAMES,
    )
    report["population"].update(
        selection_method="source_order_prefix", requested_limit=3,
        records_analyzed=3, source_exhausted=False,
        matches_reference_count=False, full_split_verified=False,
    )

    reject_without_changing_outputs(report, monkeypatch, tmp_path)


@pytest.mark.parametrize("changes", [
    {"source_split": "test"},
    {"source_repo": "different/repository"},
    {"source_revision": "different-revision"},
    {"image_id": "noncanonical-or-stale-identity"},
    {"class_id": 999},
    {"class_name": "Wrong species"},
])
def test_saved_row_provenance_and_classes_are_checked_before_export(image_data, changes, monkeypatch, tmp_path):
    report, _ = image_data
    report["images"]["measurements"][0].update(changes)

    reject_without_changing_outputs(report, monkeypatch, tmp_path)


@pytest.mark.parametrize("mapping", [["Red", "Red", "Black"], ["Red", "", "Black"], "Red"])
def test_invalid_report_class_mapping_preserves_existing_figures(image_data, mapping, monkeypatch, tmp_path):
    report, _ = image_data
    report["class_names"] = mapping

    reject_without_changing_outputs(report, monkeypatch, tmp_path)


def test_missing_class_mapping_is_not_reconstructed_from_rows(image_data, monkeypatch, tmp_path):
    report, _ = image_data
    del report["class_names"]

    reject_without_changing_outputs(report, monkeypatch, tmp_path)


@pytest.mark.parametrize("limit", [3, 10])
def test_valid_prefix_uses_smaller_of_limit_and_indexed_size_regardless_of_row_order(image_data, limit):
    report, _ = image_data
    selected = report["images"]["measurements"][:limit]
    report["images"] = summarize_image_measurements(selected, class_names=CLASS_NAMES)
    report["images"]["measurements"].reverse()
    report["population"].update(
        selection_method="source_order_prefix", requested_limit=limit,
        records_analyzed=len(selected), source_exhausted=False,
        matches_reference_count=(len(selected) == 6), full_split_verified=False,
    )

    rows = image_plots._rows(report)

    assert [row["source_row"] for row in rows] == list(range(min(limit, 6)))
