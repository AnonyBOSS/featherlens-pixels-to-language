from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from src.analysis.explore_caption_details import build_caption_details_report
from src.data.captions import parse_captions
from src.data.schemas import ImageRecord, SourceSpec
import src.visualization.caption_detail_plots as detail_plots
from src.visualization.caption_detail_plots import (
    create_caption_diversity_figure,
    create_caption_language_figure,
    create_class_caption_variation_figure,
    generate_caption_detail_figures,
)


NAMES = ("Red bird", "Blue bird", "Uncaptioned bird", "Absent bird")


def synthetic_source(*, empty=False):
    spec = SourceSpec("synthetic/captions", "train", "test-revision")
    records = []
    examples = () if empty else (
        (0, "red bird wing\nred bird wing"),
        (1, "blue beak\nleft wing"),
        (0, "!!!"),
        (2, None),
    )
    for source_row, (class_id, description) in enumerate(examples):
        identity = spec.image_id(source_row)
        image = ImageRecord(identity, spec.repo_id, spec.revision, "train", source_row, source_row, f"bird-{source_row}.jpg", class_id, NAMES[class_id])
        records.append((image, parse_captions(description, image_id=identity)))
    return SimpleNamespace(spec=spec, class_names=NAMES, dataset=records, iter_records=lambda: iter(records))


@pytest.fixture
def report():
    return build_caption_details_report(synthetic_source(), expected_records=4)


@pytest.fixture(autouse=True)
def close_new_figures():
    before = set(plt.get_fignums())
    yield
    for number in set(plt.get_fignums()) - before:
        plt.close(number)


def test_language_bars_use_saved_nonempty_caption_prevalence_and_top_term_labels(report):
    original = deepcopy(report)
    figure = create_caption_language_figure(report)
    by_category = {row["category"]: row for row in report["caption_details"]["term_prevalence"]}

    for axis, category in zip(figure.axes, ("color", "anatomy", "direction")):
        selected = sorted((row for row in by_category[category]["terms"] if row["caption_count"]), key=lambda row: (-row["caption_count"], row["term"]))[:10]
        assert [label.get_text() for label in axis.get_yticklabels()] == [row["term"] for row in selected]
        assert [bar.get_width() for bar in axis.patches] == pytest.approx([100 * row["caption_prevalence"] for row in selected])
        expected_upper = min(100, max(1, 1.25 * max((row["caption_prevalence"] * 100 for row in selected), default=0))) if category == "direction" else 100
        assert axis.get_xlim() == (0, expected_upper)
    colors = dict(zip([label.get_text() for label in figure.axes[0].get_yticklabels()], [bar.get_width() for bar in figure.axes[0].patches]))
    assert colors["red"] == pytest.approx(40)  # Two mentions / five nonempty captions.
    assert "5 nonempty captions" in figure._suptitle.get_text()
    assert "mentions do not establish visual truth" in figure._supxlabel.get_text()
    assert report == original


def test_diversity_counts_images_and_uses_saved_scatter_coordinates(report):
    figure = create_caption_diversity_figure(report)
    ttr, overlap, scatter = figure.axes

    assert sum(bar.get_height() for bar in ttr.patches) == 2
    assert sum(bar.get_height() for bar in overlap.patches) == 2
    assert list(map(tuple, scatter.collections[0].get_offsets())) == [(6, 0.5), (4, 1.0)]
    assert "Available images: 2" in ttr.get_title()
    assert "Image count" == overlap.get_ylabel()
    assert "2 pairs total" in figure._supxlabel.get_text()
    assert "not semantics" in figure._supxlabel.get_text()


def test_rare_direction_mention_is_visible_on_finer_percentage_scale():
    source = synthetic_source(empty=True)
    records = []
    for index in range(2501):
        identity = source.spec.image_id(index)
        image = ImageRecord(identity, source.spec.repo_id, source.spec.revision, "train", index, index, f"bird-{index}.jpg", 0, NAMES[0])
        text = "left bird" if index == 0 else "bird"
        records.append((image, parse_captions(text, image_id=identity)))
    source.dataset = records
    source.iter_records = lambda: iter(records)
    report = build_caption_details_report(source, expected_records=len(records))

    figure = create_caption_language_figure(report)
    direction = figure.axes[2]

    assert direction.get_xlim() == (0, 1)
    assert direction.patches[0].get_width() == pytest.approx(100 / 2501)
    assert any(text.get_text() == "0.040% (n=1)" for text in direction.texts)
    assert direction.xaxis.get_major_formatter().decimals == 2
    assert "finer percentage scale" in figure._supxlabel.get_text()
    assert "not directly comparable" in figure._supxlabel.get_text()


def test_class_rank_uses_equal_image_means_and_omits_unavailable_classes(report):
    figure = create_class_caption_variation_figure(report)

    assert list(figure.axes[0].lines[0].get_ydata()) == pytest.approx([2, 1.5])
    assert "2 / 4" in figure.axes[0].get_title()
    assert [label.get_text() for label in figure.axes[1].get_yticklabels()] == ["Blue bird", "Red bird"]
    assert [bar.get_width() for bar in figure.axes[2].patches] == pytest.approx([1.5, 2])
    assert "images weighted equally" in figure.axes[1].get_xlabel()
    assert "punctuation-only" in figure._supxlabel.get_text()


def test_prefix_titles_disclose_nonrandom_scope():
    report = build_caption_details_report(synthetic_source(), limit=2, expected_records=4)
    figure = create_caption_diversity_figure(report)

    assert "nonrandom source-order prefix" in figure._suptitle.get_text()
    assert "| complete split" not in figure._suptitle.get_text()
    assert "2 images" in figure._suptitle.get_text()


def test_empty_completed_report_has_no_invented_observations_or_percentages():
    report = build_caption_details_report(synthetic_source(empty=True), expected_records=1)
    language = create_caption_language_figure(report)
    diversity = create_caption_diversity_figure(report)
    classes = create_class_caption_variation_figure(report)

    assert all(not len(axis.patches) for axis in language.axes)
    assert all("undefined" in axis.get_title() for axis in language.axes)
    assert sum(bar.get_height() for bar in diversity.axes[0].patches) == 0
    assert not len(diversity.axes[2].collections[0].get_offsets())
    assert not len(classes.axes[0].lines[0].get_ydata())
    assert "reference count mismatch" in classes._suptitle.get_text()


def test_all_three_groups_use_existing_transactional_export(report, tmp_path, monkeypatch):
    captured = []

    def fake_export(figures, output_dir):
        captured.extend(figures)
        assert output_dir == tmp_path
        return [Path(output_dir) / f"{stem}.png" for stem, _ in figures]

    monkeypatch.setattr(detail_plots, "save_figure_bundle", fake_export)
    paths = generate_caption_detail_figures(report, tmp_path)

    assert [stem for stem, _ in captured] == ["training_caption_language", "training_caption_diversity", "training_class_caption_variation"]
    assert len(paths) == 3
    assert all(callable(create) for _, create in captured)


@pytest.mark.parametrize("invalid", ["schema", "test_source", "denominator", "lexicon"])
def test_invalid_report_is_rejected_before_render_or_existing_output_changes(report, tmp_path, monkeypatch, invalid):
    existing = tmp_path / "training_caption_language.png"
    existing.write_bytes(b"prior valid figure")
    if invalid == "schema":
        report["schema_version"] = "unsupported"
    elif invalid == "test_source":
        report["source"]["source_split"] = "test"
    elif invalid == "lexicon":
        report["caption_details"]["lexicon"]["categories"]["color"]["red"] = ["blue"]
    else:
        report["caption_details"]["coverage"]["nonempty_caption_count"] += 1
    monkeypatch.setattr(detail_plots, "save_figure_bundle", lambda *_: pytest.fail("Unexpected export"))
    monkeypatch.setattr(detail_plots, "_pyplot", lambda: pytest.fail("Unexpected rendering"))

    with pytest.raises(ValueError):
        generate_caption_detail_figures(report, tmp_path)

    assert existing.read_bytes() == b"prior valid figure"


def test_direct_figure_api_validates_before_rendering(report, monkeypatch):
    report["schema_version"] = "unsupported"
    monkeypatch.setattr(detail_plots, "_pyplot", lambda: pytest.fail("Unexpected rendering"))
    for create in (create_caption_language_figure, create_caption_diversity_figure, create_class_caption_variation_figure):
        with pytest.raises(ValueError):
            create(report)
