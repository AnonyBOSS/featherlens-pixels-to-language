from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import src.analysis.explore as explore
from src.analysis.explore import build_exploration_report, build_training_report, main
from src.data.captions import parse_captions
from src.data.schemas import ImageRecord, SourceSpec


def synthetic_source():
    spec = SourceSpec("synthetic/cub", "train", "test-revision")
    rows = []
    for index, description in enumerate(("Red bird.\nSmall wings", "", "A blue bird")):
        image = ImageRecord(
            spec.image_id(index), spec.repo_id, spec.revision, "train", index,
            index, f"bird-{index}.jpg", index % 2, ("Red", "Blue")[index % 2],
        )
        rows.append((image, parse_captions(description, image_id=image.image_id)))
    return SimpleNamespace(spec=spec, class_names=("Red", "Blue"), iter_records=lambda: iter(rows))


def test_report_honors_limit_and_keeps_image_caption_denominators_separate():
    report = build_exploration_report(synthetic_source(), limit=2)
    assert report["dataset"]["total_records"] == 2
    assert report["captions"]["image_count"] == 2
    assert report["captions"]["caption_line_count"] == 2
    assert report["captions"]["images_without_caption_lines"] == 1
    assert report["source"]["revision"] == "test-revision"
    assert report["population"]["selection_method"] == "source_order_prefix"
    assert report["population"]["random_sample"] is False
    assert report["population"]["full_split_verified"] is False
    assert build_exploration_report(synthetic_source(), limit=2) == report


def test_fewer_available_records_does_not_invent_rows_or_full_audit_claim():
    report = build_exploration_report(synthetic_source(), limit=10)
    assert report["population"]["records_analyzed"] == 3
    assert report["population"]["full_split_verified"] is False


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_limit_is_rejected(limit):
    with pytest.raises(ValueError, match="positive integer"):
        build_exploration_report(synthetic_source(), limit=limit)


def test_command_rejects_nonpositive_limit_without_loading_data(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--limit", "0"])
    assert error.value.code == 2
    assert "must be positive" in capsys.readouterr().err


def test_training_report_exhausts_source_and_reconciles_measured_counts():
    source = synthetic_source()
    original_records = source.iter_records
    visited = []

    def tracked_records():
        for image, captions in original_records():
            visited.append(image.source_row)
            yield image, captions
        visited.append("exhausted")

    source.iter_records = tracked_records

    report = build_training_report(source, expected_records=3)

    assert visited == [0, 1, 2, "exhausted"]
    population = report["population"]
    assert population["selection_method"] == "complete_official_training_split"
    assert population["source_exhausted"] is True
    assert population["requested_limit"] is None
    assert population["records_analyzed"] == 3
    assert population["reference_record_count"] == 3
    assert population["matches_reference_count"] is True
    assert population["full_split_verified"] is True
    assert report["source"]["source_split"] == "train"
    assert report["source"]["revision"] == "test-revision"
    assert report["dataset"]["total_records"] == 3
    assert report["captions"]["image_count"] == 3
    assert [row["count"] for row in report["dataset"]["class_distribution"]] == [2, 1]
    assert report["captions"]["caption_line_count"] == 3
    assert report["captions"]["nonempty_caption_count"] == 3
    assert report["captions"]["images_without_caption_lines"] == 1
    assert sum(row["caption_count"] for row in report["captions"]["word_length_distribution"]) == 3
    assert sum(row["caption_count"] for row in report["captions"]["character_length_distribution"]) == 3
    assert sum(row["image_count"] for row in report["captions"]["per_image_nonempty_count_distribution"]) == 3
    assert sum(
        row["caption_count"] * row["image_count"]
        for row in report["captions"]["per_image_nonempty_count_distribution"]
    ) == 3


@pytest.mark.parametrize("reference_count", [2, 4])
def test_training_reference_count_neither_truncates_nor_fabricates_rows(reference_count):
    report = build_training_report(synthetic_source(), expected_records=reference_count)

    population = report["population"]
    assert population["source_exhausted"] is True
    assert population["records_analyzed"] == 3
    assert population["reference_record_count"] == reference_count
    assert population["matches_reference_count"] is False
    assert population["full_split_verified"] is False
    assert report["dataset"]["total_records"] == report["captions"]["image_count"] == 3


def test_training_default_reference_is_not_reported_as_observed_data():
    report = build_training_report(synthetic_source())

    assert report["population"]["reference_record_count"] == 5994
    assert report["population"]["records_analyzed"] == 3
    assert report["population"]["matches_reference_count"] is False
    assert report["population"]["full_split_verified"] is False


@pytest.mark.parametrize("builder", [build_exploration_report, build_training_report])
@pytest.mark.parametrize("source_split", ["test", "validation"])
def test_analysis_rejects_nontraining_source_before_consuming_generator(builder, source_split):
    source = synthetic_source()
    source.spec = SourceSpec("synthetic/cub-held-out", source_split, "test-revision")
    consumed = []

    def forbidden_records():
        consumed.append("test source consumed")
        yield next(synthetic_source().iter_records())

    source.iter_records = forbidden_records

    with pytest.raises(ValueError, match="train"):
        builder(source)

    assert consumed == []


def test_training_iterator_failure_propagates_instead_of_certifying_partial_scan():
    source = synthetic_source()
    first_record = next(source.iter_records())

    def interrupted_records():
        yield first_record
        raise RuntimeError("synthetic source interrupted")

    source.iter_records = interrupted_records

    with pytest.raises(RuntimeError, match="synthetic source interrupted"):
        build_training_report(source, expected_records=1)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--full-training", "--split", "test"],
        ["--split", "test"],
        ["--split", "test", "--limit", "2"],
        ["--full-training", "--limit", "2"],
        ["--full-training", "--limit", "200"],
    ],
)
def test_invalid_training_cli_combinations_fail_before_loading(monkeypatch, arguments):
    loader_calls = []

    def forbidden_loader(*args, **kwargs):
        loader_calls.append((args, kwargs))
        raise AssertionError("Invalid arguments must be rejected before source access")

    monkeypatch.setitem(sys.modules, "src.data.source", SimpleNamespace(load_cub_split=forbidden_loader))

    with pytest.raises(SystemExit) as error:
        main(arguments)

    assert error.value.code == 2
    assert loader_calls == []


def test_default_cli_keeps_prefix_limit_200_and_streaming_train_source(monkeypatch):
    source = synthetic_source()
    calls = []

    def fake_loader(split, *, streaming):
        calls.append(("loader", split, streaming))
        return source

    def fake_prefix_report(received_source, *, limit):
        assert received_source is source
        calls.append(("prefix", limit))
        return {"population": {"records_analyzed": 3}}

    def forbidden_training_report(*args, **kwargs):
        raise AssertionError("Default CLI mode must remain prefix exploration")

    monkeypatch.setitem(sys.modules, "src.data.source", SimpleNamespace(load_cub_split=fake_loader))
    monkeypatch.setattr(explore, "build_exploration_report", fake_prefix_report)
    monkeypatch.setattr(explore, "build_training_report", forbidden_training_report)
    monkeypatch.setattr(explore, "write_json_report", lambda report, output: Path(output))

    main([])

    assert calls == [("loader", "train", True), ("prefix", 200)]


def test_full_training_cli_dispatches_only_training_report(monkeypatch):
    source = synthetic_source()
    calls = []

    def fake_loader(split, *, streaming):
        calls.append(("loader", split, streaming))
        return source

    def fake_training_report(received_source, *, expected_records=5994):
        assert received_source is source
        calls.append(("training", expected_records))
        return {
            "population": {
                "records_analyzed": 3,
                "reference_record_count": expected_records,
                "matches_reference_count": False,
                "full_split_verified": False,
            }
        }

    def forbidden_prefix_report(*args, **kwargs):
        raise AssertionError("Full-training mode must not use a prefix limit")

    monkeypatch.setitem(sys.modules, "src.data.source", SimpleNamespace(load_cub_split=fake_loader))
    monkeypatch.setattr(explore, "build_training_report", fake_training_report)
    monkeypatch.setattr(explore, "build_exploration_report", forbidden_prefix_report)
    monkeypatch.setattr(explore, "write_json_report", lambda report, output: Path(output))

    main(["--full-training"])

    assert calls == [("loader", "train", True), ("training", 5994)]
