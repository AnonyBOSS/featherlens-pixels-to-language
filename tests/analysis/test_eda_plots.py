"""Synthetic report-only checks for plot meaning, saving, and cleanup."""

from copy import deepcopy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pytest

from src.visualization.eda_plots import (
    create_caption_length_figure,
    create_caption_repetition_figure,
    create_class_distribution_figure,
    create_vocabulary_figure,
    generate_training_figures,
    main,
)
import src.visualization.eda_plots as eda_plots
import src.visualization.figure_export as figure_export


@pytest.fixture
def report():
    # Three images, five nonempty captions, and three distinct clean texts.
    return {
        "schema_version": "1.1",
        "source": {"source_split": "train"},
        "population": {
            "selection_method": "complete_official_training_split",
            "records_analyzed": 3,
            "requested_limit": None,
            "source_exhausted": True,
            "random_sample": False,
            "reference_record_count": 3,
            "matches_reference_count": True,
            "full_split_verified": True,
        },
        "dataset": {
            "total_records": 3,
            "class_distribution": [
                {"class_id": 0, "class_name": "Long named blue bird", "count": 2},
                {"class_id": 1, "class_name": "Green bird", "count": 1},
                {"class_id": 2, "class_name": "Unseen species", "count": 0},
            ],
        },
        "captions": {
            "image_count": 3,
            "nonempty_caption_count": 5,
            "expected_captions_per_image": 10,
            "word_lengths": {"median": 2, "p95": 3.6},
            "character_lengths": {"median": 6, "p95": 19.6},
            "word_length_distribution": [
                {"word_count": 0, "caption_count": 1},
                {"word_count": 2, "caption_count": 3},
                {"word_count": 4, "caption_count": 1},
            ],
            "character_length_distribution": [
                {"character_count": 3, "caption_count": 1},
                {"character_count": 6, "caption_count": 3},
                {"character_count": 23, "caption_count": 1},
            ],
            "per_image_nonempty_count_distribution": [
                {"caption_count": 1, "image_count": 1},
                {"caption_count": 2, "image_count": 2},
            ],
            "vocabulary_size": 5,
            "total_tokens": 10,
            "token_frequencies": {"a": 3, "bird": 4, "green": 1, "little": 1, "sings": 1},
            "hapax_word_types": 3,
            "hapax_token_share": 0.3,
            "repetition": {
                "unique_nonempty_texts": 3,
                "global_extra_occurrences": 2,
                "within_images": {"repeated_distinct_texts": 1, "extra_occurrences": 1},
                "across_images": {
                    "repeated_distinct_texts": 1,
                    "extra_occurrences": 1,
                    "examples": [{"text": "a bird", "image_count": 2, "occurrence_count": 3}],
                },
            },
        },
    }


@pytest.fixture(autouse=True)
def close_new_figures():
    previous = set(plt.get_fignums())
    yield
    for figure_number in set(plt.get_fignums()) - previous:
        plt.close(figure_number)


def test_class_plot_counts_images_and_retains_zero_class(report):
    figure = create_class_distribution_figure(report)
    assert list(figure.axes[0].lines[0].get_ydata()) == [2, 1, 0]
    assert [bar.get_width() for bar in figure.axes[1].patches] == [2, 1, 0]
    assert [bar.get_width() for bar in figure.axes[2].patches] == [0, 1, 2]
    assert figure.axes[0].get_ylabel() == "Image count"
    assert "3 images" in figure._suptitle.get_text()
    assert "5 nonempty captions" in figure._suptitle.get_text()


def test_length_bars_use_reported_frequencies_and_summary_markers(report):
    figure = create_caption_length_figure(report)
    words, characters, coverage = figure.axes
    assert [bar.get_height() for bar in words.patches] == [1, 3, 1]
    assert [bar.get_height() for bar in characters.patches] == [1, 3, 1]
    assert [bar.get_height() for bar in coverage.patches] == [1, 2]
    assert [line.get_xdata()[0] for line in words.lines] == [2, 3.6]
    assert words.get_ylabel() == "Nonempty caption count"
    assert coverage.get_ylabel() == "Image count"


def test_vocabulary_retains_stopwords_and_labels_hapax_token_denominator(report):
    figure = create_vocabulary_figure(report)
    top, ranks = figure.axes
    assert [tick.get_text() for tick in top.get_yticklabels()][:2] == ["bird", "a"]
    assert [bar.get_width() for bar in top.patches] == [4, 3, 1, 1, 1]
    assert "stopwords retained" in top.get_title()
    assert list(ranks.lines[0].get_ydata()) == [4, 3, 1, 1, 1]
    assert ranks.get_xscale() == ranks.get_yscale() == "log"
    assert "30.00% of all tokens" in ranks.texts[0].get_text()


def test_repetition_scopes_share_distinct_text_units_and_examples_count_images(report):
    figure = create_caption_repetition_figure(report)
    scopes, examples = figure.axes
    assert [bar.get_width() for bar in scopes.patches] == [1, 1]
    assert "scopes overlap" in scopes.get_title()
    assert scopes.get_xlabel() == "Distinct clean caption texts"
    assert examples.patches[0].get_width() == 2
    assert examples.get_xlabel() == "Distinct images containing this exact text"


def test_prefix_label_and_local_style_do_not_overstate_scope_or_mutate_report(report):
    report["population"].update(selection_method="source_order_prefix", full_split_verified=False, requested_limit=3)
    original_report = deepcopy(report)
    original_style = dict(matplotlib.rcParams)
    figure = create_class_distribution_figure(report)
    assert "nonrandom source-order prefix" in figure._suptitle.get_text()
    assert "complete split" not in figure._suptitle.get_text()
    assert report == original_report
    assert dict(matplotlib.rcParams) == original_style


def test_test_split_is_rejected_before_creating_output(report, tmp_path):
    report["source"]["source_split"] = "test"
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="training report"):
        generate_training_figures(report, output)
    assert not output.exists()


def test_generate_saves_four_png_and_pdf_pairs_and_closes_figures(report, tmp_path):
    previous = set(plt.get_fignums())
    paths = generate_training_figures(report, tmp_path)
    assert len(paths) == 4
    assert all(path.suffix == ".png" and path.stat().st_size > 0 for path in paths)
    assert all(path.with_suffix(".pdf").stat().st_size > 0 for path in paths)
    assert set(plt.get_fignums()) == previous


def test_save_failure_still_closes_figure(report, tmp_path, monkeypatch):
    previous = set(plt.get_fignums())

    def fail_save(*args, **kwargs):
        raise OSError("simulated save error")

    monkeypatch.setattr(Figure, "savefig", fail_save)
    with pytest.raises(OSError, match="simulated save"):
        generate_training_figures(report, tmp_path)
    assert set(plt.get_fignums()) == previous


def test_cli_reads_saved_report_without_fetching_source(report, tmp_path, monkeypatch):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    calls = []
    monkeypatch.setattr("src.visualization.eda_plots.generate_training_figures", lambda loaded, output: calls.append((loaded, output)) or [])
    main(["--report", str(report_path), "--output-dir", str(tmp_path / "figures")])
    assert calls == [(report, tmp_path / "figures")]


_STEMS = (
    "training_class_distribution", "training_caption_lengths",
    "training_caption_vocabulary", "training_caption_repetition",
)


def existing_bundle(path):
    path.mkdir(parents=True)
    for stem in _STEMS:
        for extension in ("png", "pdf"):
            name = f"{stem}.{extension}"
            (path / name).write_bytes(f"prior bytes: {name}".encode())
    (path / "unrelated.txt").write_bytes(b"Keep this unrelated user file")
    return {item.name: item.read_bytes() for item in path.iterdir()}


def bundle_bytes(path):
    return {item.name: item.read_bytes() for item in path.iterdir()}


@pytest.mark.parametrize(
    "case",
    [
        "legacy_schema", "missing_histogram", "missing_late_field", "negative_count",
        "boolean_count", "fractional_count", "nan_marker", "infinite_share",
        "duplicate_bins", "histogram_total", "weighted_word_total", "coverage_total",
        "token_total", "hapax_count", "image_denominator", "repetition_total",
        "late_example_count", "missing_population_count",
        "unexhausted_complete", "reference_mismatch", "false_match_flag",
        "false_verification_flag", "random_sample", "out_of_range_marker",
        "prefix_without_limit", "prefix_exceeding_limit",
        "example_across_extras", "example_within_extras",
    ],
)
def test_invalid_report_preserves_existing_bundle_and_creates_no_output(
    report, tmp_path, monkeypatch, case
):
    if case == "legacy_schema":
        report["schema_version"] = "1.0"
    elif case == "missing_histogram":
        del report["captions"]["word_length_distribution"]
    elif case == "missing_late_field":
        del report["captions"]["repetition"]["across_images"]["examples"]
    elif case in {"negative_count", "boolean_count", "fractional_count"}:
        report["dataset"]["class_distribution"][0]["count"] = {
            "negative_count": -1, "boolean_count": True, "fractional_count": 2.0,
        }[case]
    elif case == "nan_marker":
        report["captions"]["word_lengths"]["median"] = float("nan")
    elif case == "infinite_share":
        report["captions"]["hapax_token_share"] = float("inf")
    elif case == "duplicate_bins":
        report["captions"]["word_length_distribution"].append({"word_count": 2, "caption_count": 1})
    elif case == "histogram_total":
        report["captions"]["character_length_distribution"][0]["caption_count"] = 2
    elif case == "weighted_word_total":
        report["captions"]["word_length_distribution"][0]["word_count"] = 1
    elif case == "coverage_total":
        report["captions"]["per_image_nonempty_count_distribution"][0]["caption_count"] = 0
    elif case == "token_total":
        report["captions"]["token_frequencies"]["a"] = 4
    elif case == "hapax_count":
        report["captions"]["hapax_word_types"] = 2
    elif case == "image_denominator":
        report["captions"]["image_count"] = 4
    elif case == "repetition_total":
        report["captions"]["repetition"]["global_extra_occurrences"] = 3
    elif case == "late_example_count":
        report["captions"]["repetition"]["across_images"]["examples"][0]["image_count"] = 4
    elif case == "missing_population_count":
        del report["population"]["records_analyzed"]
    elif case == "unexhausted_complete":
        report["population"]["source_exhausted"] = False
    elif case == "reference_mismatch":
        report["population"]["reference_record_count"] = 4
    elif case == "false_match_flag":
        report["population"]["matches_reference_count"] = False
    elif case == "false_verification_flag":
        report["population"]["full_split_verified"] = False
    elif case == "random_sample":
        report["population"]["random_sample"] = True
    elif case == "out_of_range_marker":
        report["captions"]["word_lengths"].update(median=1000, p95=2000)
    elif case in {"prefix_without_limit", "prefix_exceeding_limit"}:
        report["population"].update(selection_method="source_order_prefix", full_split_verified=False)
        report["population"]["requested_limit"] = None if case == "prefix_without_limit" else 2
    elif case == "example_across_extras":
        report["captions"]["repetition"]["across_images"]["examples"][0]["image_count"] = 3
    elif case == "example_within_extras":
        report["captions"]["repetition"]["across_images"]["examples"][0]["occurrence_count"] = 4

    def forbidden_render(*args, **kwargs):
        raise AssertionError("Invalid reports must be rejected before rendering")

    monkeypatch.setattr(eda_plots, "create_class_distribution_figure", forbidden_render)
    output = tmp_path / "existing"
    before = existing_bundle(output)
    with pytest.raises(ValueError) as error:
        generate_training_figures(report, output)
    if case == "legacy_schema":
        assert "Regenerate" in str(error.value)
    assert bundle_bytes(output) == before
    absent = tmp_path / "new" / "figures"
    with pytest.raises(ValueError):
        generate_training_figures(report, absent)
    assert not absent.parent.exists()
    assert list(tmp_path.glob(".eda-figures-*")) == []


def stub_renderers(monkeypatch):
    for name in (
        "create_class_distribution_figure", "create_caption_length_figure",
        "create_vocabulary_figure", "create_caption_repetition_figure",
    ):
        monkeypatch.setattr(eda_plots, name, lambda report: plt.figure())


@pytest.mark.parametrize("failure", ["late_creator", "late_save"])
def test_later_render_failure_preserves_destination_and_cleans_staging(
    report, tmp_path, monkeypatch, failure
):
    output = tmp_path / "figures"
    before = existing_bundle(output)
    previous_figures = set(plt.get_fignums())
    written = []
    stub_renderers(monkeypatch)

    def fake_save(figure, destination, **kwargs):
        destination = Path(destination)
        assert destination.parent.name == "rendered"
        if failure == "late_save" and destination.name == "training_caption_vocabulary.pdf":
            raise OSError("simulated late save failure")
        destination.write_bytes(b"new rendered figure")
        written.append(destination.name)

    def failed_creator(report):
        plt.figure()  # Cleanup also covers a creator failing after allocation.
        raise RuntimeError("simulated late creator failure")

    monkeypatch.setattr(Figure, "savefig", fake_save)
    if failure == "late_creator":
        monkeypatch.setattr(eda_plots, "create_vocabulary_figure", failed_creator)
    with pytest.raises((OSError, RuntimeError), match="simulated late"):
        generate_training_figures(report, output)
    assert len(written) == (4 if failure == "late_creator" else 5)
    assert bundle_bytes(output) == before
    assert set(plt.get_fignums()) == previous_figures
    assert list(tmp_path.glob(".eda-figures-*")) == []


@pytest.mark.parametrize("prior_bundle", [True, False])
def test_mid_publication_failure_rolls_back_only_owned_files(
    report, tmp_path, monkeypatch, prior_bundle
):
    output = tmp_path / "figures" if prior_bundle else tmp_path / "new" / "figures"
    before = existing_bundle(output) if prior_bundle else None
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_bytes(b"unrelated parent file")
    previous_figures = set(plt.get_fignums())
    stub_renderers(monkeypatch)
    monkeypatch.setattr(Figure, "savefig", lambda figure, path, **kwargs: Path(path).write_bytes(b"new bytes"))
    real_replace = figure_export.os.replace
    publication_calls = []

    def fail_fourth_publication(source, destination):
        if Path(source).parent.name == "rendered":
            publication_calls.append(Path(destination).name)
            if len(publication_calls) == 4:
                raise OSError("simulated publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(figure_export.os, "replace", fail_fourth_publication)
    with pytest.raises(OSError, match="simulated publication"):
        generate_training_figures(report, output)
    assert len(publication_calls) == 4
    if prior_bundle:
        assert bundle_bytes(output) == before
    else:
        assert not output.parent.exists()
    assert unrelated.read_bytes() == b"unrelated parent file"
    assert set(plt.get_fignums()) == previous_figures
    assert list(tmp_path.glob(".eda-figures-*")) == []


def test_successful_publication_replaces_bundle_and_preserves_unrelated_files(
    report, tmp_path, monkeypatch
):
    output = tmp_path / "figures"
    previous = existing_bundle(output)
    previous_figures = set(plt.get_fignums())
    stub_renderers(monkeypatch)
    rendered = []

    def fake_save(figure, path, **kwargs):
        path = Path(path)
        assert path.parent.name == "rendered"
        path.write_bytes(f"new bytes: {path.name}".encode())
        rendered.append(path.name)

    monkeypatch.setattr(Figure, "savefig", fake_save)
    real_replace = figure_export.os.replace

    def publish_after_complete_render(source, destination):
        assert len(rendered) == 8
        return real_replace(source, destination)

    monkeypatch.setattr(figure_export.os, "replace", publish_after_complete_render)
    paths = generate_training_figures(report, output)
    assert paths == [output / f"{stem}.png" for stem in _STEMS]
    for stem in _STEMS:
        for extension in ("png", "pdf"):
            name = f"{stem}.{extension}"
            assert (output / name).read_bytes() != previous[name]
    assert (output / "unrelated.txt").read_bytes() == previous["unrelated.txt"]
    assert set(plt.get_fignums()) == previous_figures
    assert list(tmp_path.glob(".eda-figures-*")) == []


@pytest.mark.parametrize("scope", ["prefix", "reference_mismatch", "empty_scan"])
def test_valid_prefix_mismatch_and_empty_reports_can_publish(report, tmp_path, monkeypatch, scope):
    if scope == "prefix":
        report["population"].update(
            selection_method="source_order_prefix", requested_limit=5,
            full_split_verified=False,
        )
    else:
        report["population"].update(
            reference_record_count=4, matches_reference_count=False,
            full_split_verified=False,
        )
    if scope == "empty_scan":
        report["population"]["records_analyzed"] = 0
        report["dataset"]["total_records"] = 0
        for row in report["dataset"]["class_distribution"]:
            row["count"] = 0
        captions = report["captions"]
        captions.update(
            image_count=0, nonempty_caption_count=0, total_tokens=0,
            word_lengths={"median": None, "p95": None},
            character_lengths={"median": None, "p95": None},
            word_length_distribution=[], character_length_distribution=[],
            per_image_nonempty_count_distribution=[], token_frequencies={},
            vocabulary_size=0, hapax_word_types=0, hapax_token_share=None,
            repetition={
                "unique_nonempty_texts": 0, "global_extra_occurrences": 0,
                "within_images": {"repeated_distinct_texts": 0, "extra_occurrences": 0},
                "across_images": {"repeated_distinct_texts": 0, "extra_occurrences": 0, "examples": []},
            },
        )
    stub_renderers(monkeypatch)
    monkeypatch.setattr(Figure, "savefig", lambda figure, path, **kwargs: Path(path).write_bytes(b"figure bytes"))
    paths = generate_training_figures(report, tmp_path / "figures")
    assert len(paths) == 4
    assert all(path.exists() and path.with_suffix(".pdf").exists() for path in paths)
    assert list(tmp_path.glob(".eda-figures-*")) == []
