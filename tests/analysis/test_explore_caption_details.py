from copy import deepcopy
from dataclasses import replace
import json
import sys
from types import SimpleNamespace

import pytest

from src.analysis.explore_caption_details import (
    build_caption_details_report, main, validate_caption_details_report,
)
from src.data.captions import parse_captions
from src.data.schemas import ImageRecord, SourceSpec


def source(descriptions=("Red bird.\nBlue wing.", "Black tail.", None)):
    spec = SourceSpec("synthetic/cub", "train", "revision")
    names = ("Alpha", "Beta")
    records = []
    for index, description in enumerate(descriptions):
        image = ImageRecord(spec.image_id(index), spec.repo_id, spec.revision, "train", index,
                            index, f"bird-{index}.png", index % 2, names[index % 2])
        records.append((image, parse_captions(description, image_id=image.image_id)))
    return SimpleNamespace(spec=spec, class_names=names, dataset=records,
                           iter_records=lambda: iter(records))


def test_complete_builder_reuses_caption_records_and_preserves_source():
    data = source()
    before = deepcopy(data.dataset)
    report = build_caption_details_report(data, expected_records=3)

    assert report["population"]["full_split_verified"] is True
    assert report["caption_details"]["coverage"]["image_count"] == 3
    assert report["caption_details"]["coverage"]["nonempty_caption_count"] == 3
    assert report["caption_details"]["coverage"]["images_with_pairwise_overlap"] == 1
    assert data.dataset == before
    assert validate_caption_details_report(json.loads(json.dumps(report))) == report["caption_details"]


@pytest.mark.parametrize("limit", [1, 10])
def test_explicit_prefix_never_claims_full_split_even_if_it_reaches_the_end(limit):
    data = source()
    calls = []
    def records():
        for index, row in enumerate(data.dataset):
            calls.append(index)
            yield row
    data.iter_records = records
    report = build_caption_details_report(data, limit=limit, expected_records=3)

    assert calls == list(range(min(limit, 3)))
    assert report["population"]["records_analyzed"] == min(limit, 3)
    assert report["population"]["selection_method"] == "source_order_prefix"
    assert report["population"]["source_exhausted"] is False
    assert report["population"]["full_split_verified"] is False


def test_reference_mismatch_remains_visible_without_changing_measured_population():
    report = build_caption_details_report(source(), expected_records=4)
    assert report["population"]["records_analyzed"] == 3
    assert report["population"]["source_exhausted"] is True
    assert report["population"]["full_split_verified"] is False


def test_nontraining_source_is_rejected_before_iteration():
    data = source()
    data.spec = replace(data.spec, source_split="test")
    data.iter_records = lambda: pytest.fail("Held-out records must never be read")
    with pytest.raises(ValueError, match="train"):
        build_caption_details_report(data)


@pytest.mark.parametrize("options", [{"limit": 0}, {"limit": True}, {"limit": 1.5},
                                    {"expected_records": 0}, {"expected_records": True}])
def test_invalid_configuration_is_rejected_before_iteration(options):
    data = source()
    data.iter_records = lambda: pytest.fail("Invalid configuration must fail before reading")
    with pytest.raises(ValueError):
        build_caption_details_report(data, **options)


def test_interrupted_iterator_does_not_return_a_completed_report():
    data = source()
    def interrupted():
        yield data.dataset[0]
        raise OSError("Synthetic metadata interruption")
    data.iter_records = interrupted
    with pytest.raises(OSError, match="interruption"):
        build_caption_details_report(data)


def test_incomplete_iterator_cannot_certify_the_indexed_population():
    data = source()
    data.iter_records = lambda: iter(data.dataset[:2])
    with pytest.raises(ValueError, match="indexed"):
        build_caption_details_report(data, expected_records=2)


def test_mismatched_top_level_source_is_rejected_even_when_record_ids_are_valid():
    data = source()
    data.spec = replace(data.spec, revision="other-revision")
    with pytest.raises(ValueError, match="provenance"):
        build_caption_details_report(data)


@pytest.mark.parametrize("path,value", [
    (("schema_version",), "old-schema"),
    (("source", "source_split"), "test"),
    (("source", "revision"), "different"),
    (("class_names", 0), "Changed"),
    (("population", "indexed_source_size"), 4),
    (("population", "records_analyzed"), 2),
    (("population", "random_sample"), True),
    (("population", "full_split_verified"), False),
    (("caption_details", "images", 0, "image_id"), "wrong"),
    (("caption_details", "images", 0, "source_split"), "test"),
    (("caption_details", "images", 0, "class_id"), 99),
    (("caption_details", "images", 0, "mean_caption_words"), 100.),
    (("caption_details", "images", 0, "type_token_ratio"), float("nan")),
    (("caption_details", "images", 0, "pair_count"), 5),
    (("caption_details", "images", 0, "pairwise_jaccard_mean"), 2.),
    (("caption_details", "images", 0, "category_caption_counts", "color"), 0),
    (("caption_details", "coverage", "image_count"), 99),
    (("caption_details", "coverage", "pair_count"), 99),
    (("caption_details", "overall", "token_count", "mean"), 100.),
    (("caption_details", "classes", 0, "mean_caption_words", "mean"), 100.),
    (("caption_details", "term_prevalence", 0, "caption_count"), 99),
    (("caption_details", "term_prevalence", 0, "caption_prevalence"), -1.),
    (("caption_details", "lexicon", "version"), "unknown"),
    (("caption_details", "lexicon", "categories", "color", "red"), ["blue"]),
])
def test_inconsistent_saved_reports_are_rejected(path, value):
    report = build_caption_details_report(source(), expected_records=3)
    parent = report
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(ValueError):
        validate_caption_details_report(report)


def test_row_reordering_does_not_change_the_validated_population():
    report = build_caption_details_report(source(), expected_records=3)
    report["caption_details"]["images"].reverse()
    assert validate_caption_details_report(report)["coverage"]["image_count"] == 3


@pytest.mark.parametrize("arguments", [["--split", "test"], ["--limit", "0"], ["--expected-records", "0"]])
def test_invalid_cli_inputs_do_not_open_a_source(monkeypatch, arguments):
    monkeypatch.setitem(sys.modules, "src.data.source", SimpleNamespace(load_cub_split=lambda *_args, **_kwargs: pytest.fail("Unexpected source open")))
    with pytest.raises(SystemExit):
        main(arguments)


def test_cli_preserves_completed_report_when_optional_figure_export_fails(monkeypatch, tmp_path):
    import src.visualization.caption_detail_plots as plots
    data = source()
    monkeypatch.setitem(sys.modules, "src.data.source", SimpleNamespace(load_cub_split=lambda *_args, **_kwargs: data))
    def fail_export(*args, **kwargs):
        raise OSError("Synthetic export failure")
    monkeypatch.setattr(plots, "generate_caption_detail_figures", fail_export)
    report_path = tmp_path / "details.json"
    with pytest.raises(OSError, match="export failure"):
        main(["--expected-records", "3", "--output", str(report_path), "--figures-dir", str(tmp_path / "figures")])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert validate_caption_details_report(report)["coverage"]["image_count"] == 3
    assert not (tmp_path / "figures").exists()


def test_cli_invalid_population_does_not_replace_an_existing_report(monkeypatch, tmp_path):
    data = source()
    data.iter_records = lambda: iter(data.dataset[:1])
    monkeypatch.setitem(sys.modules, "src.data.source", SimpleNamespace(load_cub_split=lambda *_args, **_kwargs: data))
    report_path = tmp_path / "details.json"
    previous = b'{"preserved":"previous report"}'
    report_path.write_bytes(previous)
    with pytest.raises(ValueError, match="indexed"):
        main(["--output", str(report_path)])
    assert report_path.read_bytes() == previous
