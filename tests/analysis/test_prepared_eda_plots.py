from copy import deepcopy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import src.visualization.prepared_eda_plots as plots


@pytest.fixture
def report():
    return {
        "schema_version": "prepared-eda-1.0", "source": {"repo_id": "synthetic/cub", "source_split": "train", "revision": "revision"},
        "class_names": ["Alpha", "Beta", "Absent"],
        "classes": [{"class_id": index, "class_name": name, "partition_counts": counts} for index, (name, counts) in enumerate((
            ("Alpha", {"train": 2, "val": 1, "excluded": 0}), ("Beta", {"train": 1, "val": 1, "excluded": 0}), ("Absent", {"train": 0, "val": 0, "excluded": 0})))],
        "preparation": {"schema_version": 1, "policy_version": "classification-v1", "split_version": "stratified-exact-group-v1", "pixel_hash_algorithm": "exif-rgb8-white:v1", "manifest_sha256": "a" * 64, "seed": 42, "val_fraction": .4, "source_revisions": {"train": "revision", "test": "test-revision"}, "source_counts": {"train": 5, "test": 2}, "partition_counts": {"train": 3, "val": 2, "test": 2, "excluded": 0}},
        "population": {"raw_training_image_count": 5, "derived_training_image_count": 3, "validation_image_count": 2, "excluded_training_image_count": 0, "unknown_class_training_image_count": 0},
        "official_test": {"identity_count": 2, "preserved_test_partition_count": 2, "identities_preserved": True, "image_caption_statistics_used": False},
        "class_count_distributions": {"raw": [{"images_per_class": 0, "class_count": 1}, {"images_per_class": 2, "class_count": 1}, {"images_per_class": 3, "class_count": 1}], "train": [{"images_per_class": 0, "class_count": 1}, {"images_per_class": 1, "class_count": 1}, {"images_per_class": 2, "class_count": 1}], "val": [{"images_per_class": 0, "class_count": 1}, {"images_per_class": 1, "class_count": 2}]},
    }


def test_class_coverage_uses_saved_counts_and_retains_absent_classes(report):
    before = deepcopy(report)
    figure = plots.create_prepared_class_coverage_figure(report)
    try:
        assert list(figure.axes[0].lines[0].get_ydata()) == [2, 1, 0]
        assert list(figure.axes[0].lines[1].get_ydata()) == [1, 1, 0]
        assert "2/3 classes" in figure.axes[0].lines[0].get_label()
        assert sum(bar.get_height() for bar in figure.axes[1].patches) == 9
        assert "Official TRAINING" in figure._suptitle.get_text()
        assert report == before
    finally:
        plt.close(figure)


@pytest.mark.parametrize("change", ["schema", "test_source", "class_counts", "distribution", "unknown", "test_assignment", "fingerprint", "revision"])
def test_invalid_coverage_cannot_replace_existing_figures(report, change, tmp_path, monkeypatch):
    existing = tmp_path / "training_prepared_class_coverage.png"
    existing.write_bytes(b"preserved figure")
    if change == "schema": report["schema_version"] = "old"
    elif change == "test_source": report["source"]["source_split"] = "test"
    elif change == "class_counts": report["classes"][0]["partition_counts"]["train"] += 1
    elif change == "distribution": report["class_count_distributions"]["val"][0]["class_count"] += 1
    elif change == "unknown": report["population"]["unknown_class_training_image_count"] = 1
    elif change == "test_assignment": report["official_test"]["identities_preserved"] = False
    elif change == "fingerprint": report["preparation"]["manifest_sha256"] = "missing"
    else: report["preparation"]["source_revisions"]["train"] = "other"
    monkeypatch.setattr(plots, "save_figure_bundle", lambda *_: pytest.fail("Unexpected export"))
    monkeypatch.setattr(plots, "_pyplot", lambda: pytest.fail("Unexpected rendering"))
    with pytest.raises(ValueError): plots.generate_prepared_eda_figures(report, tmp_path)
    assert existing.read_bytes() == b"preserved figure"


def test_coverage_export_uses_the_existing_bundle_helper(report, monkeypatch, tmp_path):
    captured = []
    def exporter(figures, output):
        captured.extend(figures)
        assert output == tmp_path
        return [tmp_path / "training_prepared_class_coverage.png"]
    monkeypatch.setattr(plots, "save_figure_bundle", exporter)
    assert len(plots.generate_prepared_eda_figures(report, tmp_path)) == 1
    assert captured[0][0] == "training_prepared_class_coverage"
