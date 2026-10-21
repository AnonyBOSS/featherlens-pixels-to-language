"""Synthetic end-to-end tests for the read-only CUB integrity audit."""

import json
from io import BytesIO
from types import SimpleNamespace

from PIL import Image

from src.data.integrity_audit import audit_sources
from src.data.schemas import SourceSpec


REVISION = "a" * 40
CLASS_NAMES = tuple(f"Bird {index}" for index in range(200))
TEN_CAPTIONS = "one\ntwo\nthree\nfour\nfive\nsix\nseven\neight\nnine\nten"


def png_bytes(color=(12, 34, 56)):
    buffer = BytesIO()
    Image.new("RGB", (3, 2), color).save(buffer, format="PNG")
    return buffer.getvalue()


def source(split, rows):
    return SimpleNamespace(
        spec=SourceSpec(f"synthetic/CUB_{split}", split, REVISION),
        dataset=rows,
        class_names=CLASS_NAMES,
    )


def row(image_bytes, **changes):
    record = {
        "image": {"bytes": image_bytes, "path": None},
        "description": TEN_CAPTIONS,
        "label": 0,
        "file_name": "bird.png",
        "id": 7,
    }
    record.update(changes)
    return record


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def run_audit(tmp_path, train_rows, test_rows, expected_counts):
    summary = audit_sources(
        {"train": source("train", train_rows), "test": source("test", test_rows)},
        tmp_path,
        expected_counts=expected_counts,
    )
    images = read_jsonl(tmp_path / "images.jsonl")
    captions = read_jsonl(tmp_path / "captions.jsonl")
    groups = read_jsonl(tmp_path / "exact_groups.jsonl")
    saved_summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved_summary["processed_counts"] == summary["processed_counts"]
    return summary, images, captions, groups


def test_bad_metadata_is_flagged_without_losing_later_source_rows(tmp_path):
    summary, images, _, _ = run_audit(
        tmp_path,
        [
            row(png_bytes(), label=999, file_name=42),
            row(png_bytes((1, 2, 3)), id=8),
        ],
        [],
        {"train": 2, "test": 0},
    )

    assert summary["processed_counts"] == {"train": 2, "test": 0}
    assert summary["coverage_complete"] is True
    assert [image["source_row"] for image in images] == [0, 1]
    assert images[0]["image_id"] == f"hf:synthetic/CUB_train@{REVISION}:train:0"
    assert images[0]["source_split"] == "train"
    assert {"metadata.invalid_label", "metadata.invalid_filename"} <= set(images[0]["error_flags"])
    assert not any(flag.startswith("metadata.") for flag in images[1]["error_flags"])


def test_blank_caption_keeps_original_index_and_is_not_usable(tmp_path):
    description = "one\ntwo\nthree\nfour\nfive\n\nseven\neight\nnine\nten"
    _, images, captions, _ = run_audit(
        tmp_path,
        [row(png_bytes(), description=description)],
        [],
        {"train": 1, "test": 0},
    )

    assert [caption["caption_index"] for caption in captions] == list(range(10))
    assert captions[5]["raw_caption"] == ""
    assert captions[5]["clean_caption"] == ""
    assert captions[5]["usable"] is False
    assert "caption.empty" in captions[5]["error_flags"]
    assert captions[4]["usable"] is True
    assert captions[6]["usable"] is True
    assert images[0]["caption_structural_valid"] is False


def test_non_string_description_is_reported_without_dropping_image(tmp_path):
    summary, images, captions, _ = run_audit(
        tmp_path,
        [row(png_bytes(), description=["not", "a", "string"])],
        [],
        {"train": 1, "test": 0},
    )

    assert summary["coverage_complete"] is True
    assert len(images) == 1
    assert images[0]["caption_structural_valid"] is False
    assert any(flag.startswith("caption.") for flag in images[0]["error_flags"])
    assert captions == []


def test_corrupt_image_is_a_decode_failure_and_later_row_is_audited(tmp_path):
    summary, images, _, _ = run_audit(
        tmp_path,
        [row(b"not an image"), row(png_bytes((99, 77, 55)))],
        [],
        {"train": 2, "test": 0},
    )

    assert summary["processed_counts"] == {"train": 2, "test": 0}
    assert summary["coverage_complete"] is True
    assert "image.decode_failed" in images[0]["error_flags"]
    assert images[0]["image_status"] != images[1]["image_status"]
    assert images[1]["pixel_sha256"] is not None


def test_exact_hash_group_reports_cross_split_leakage(tmp_path):
    encoded = png_bytes((200, 10, 50))
    _, images, _, groups = run_audit(
        tmp_path,
        [row(encoded)],
        [row(encoded)],
        {"train": 1, "test": 1},
    )

    assert len(images) == 2
    assert images[0]["encoded_sha256"] == images[1]["encoded_sha256"]
    assert images[0]["pixel_sha256"] == images[1]["pixel_sha256"]
    assert all(len(image["encoded_sha256"]) == 64 for image in images)
    assert all(len(image["pixel_sha256"]) == 64 for image in images)
    cross_split = [group for group in groups if group["crosses_official_split"]]
    assert len(cross_split) == 1
    group = cross_split[0]
    assert set(group["image_ids"]) == {image["image_id"] for image in images}
    assert group["hash_kind"] in {"encoded_sha256", "pixel_sha256", "mixed"}
    assert group["hash_evidence"]
    assert all(group["group_id"] in image["exact_group_ids"] for image in images)


def test_short_source_marks_coverage_incomplete(tmp_path):
    summary, images, _, _ = run_audit(
        tmp_path,
        [row(png_bytes())],
        [row(png_bytes((11, 22, 33)))],
        {"train": 2, "test": 1},
    )

    assert summary["expected_counts"] == {"train": 2, "test": 1}
    assert summary["processed_counts"] == {"train": 1, "test": 1}
    assert summary["coverage_complete"] is False
    assert len(images) == 2


def test_iterator_failure_preserves_partial_results_and_reports_fatal_error(tmp_path):
    class InterruptedRows:
        def __iter__(self):
            yield row(png_bytes())
            raise OSError("synthetic source interruption")

    summary, images, _, _ = run_audit(
        tmp_path,
        InterruptedRows(),
        [row(png_bytes((60, 80, 90)))],
        {"train": 2, "test": 1},
    )

    assert summary["processed_counts"] == {"train": 1, "test": 1}
    assert summary["coverage_complete"] is False
    assert summary["fatal_errors"]
    assert {image["source_split"] for image in images} == {"train", "test"}


def test_lossless_encodings_are_grouped_by_pixels_even_when_bytes_differ(tmp_path):
    color = (17, 85, 201)
    bmp_buffer = BytesIO()
    Image.new("RGB", (3, 2), color).save(bmp_buffer, format="BMP")
    _, images, _, groups = run_audit(
        tmp_path,
        [row(png_bytes(color)), row(bmp_buffer.getvalue(), file_name="bird.bmp")],
        [],
        {"train": 2, "test": 0},
    )

    assert images[0]["encoded_sha256"] != images[1]["encoded_sha256"]
    assert images[0]["pixel_sha256"] == images[1]["pixel_sha256"]
    assert len(groups) == 1
    assert groups[0]["hash_kind"] == "pixel_sha256"
    assert set(groups[0]["image_ids"]) == {image["image_id"] for image in images}
    assert groups[0]["crosses_official_split"] is False


def test_indexed_row_read_failure_keeps_placeholder_and_recovers_later_rows(tmp_path):
    class InterruptedIndexedRows:
        def __len__(self):
            return 3

        def __getitem__(self, index):
            if index == 1:
                raise OSError("synthetic row read failure")
            return row(png_bytes((index, 10, 20)), id=index)

    summary, images, _, _ = run_audit(
        tmp_path,
        InterruptedIndexedRows(),
        [],
        {"train": 3, "test": 0},
    )

    assert summary["processed_counts"] == {"train": 3, "test": 0}
    assert summary["row_read_errors"]["train"] == 1
    assert summary["coverage_complete"] is False
    assert [image["source_row"] for image in images] == [0, 1, 2]
    failed = images[1]
    assert failed["image_id"] == f"hf:synthetic/CUB_train@{REVISION}:train:1"
    assert failed["source_split"] == "train"
    assert "coverage.row_read_failed" in failed["error_flags"]
    assert "coverage.row_read_failed" not in images[2]["error_flags"]


def test_indexed_row_audit_failure_keeps_placeholder_and_continues(tmp_path):
    from collections.abc import Mapping

    class BrokenMapping(Mapping):
        def __init__(self, values):
            self.values = values

        def __getitem__(self, key):
            return self.values[key]

        def __iter__(self):
            return iter(self.values)

        def __len__(self):
            return len(self.values)

        def get(self, key, default=None):
            raise RuntimeError("synthetic mapping access failure")

    class IndexedRows:
        def __len__(self):
            return 3

        def __getitem__(self, index):
            values = row(png_bytes((index, 30, 40)), id=index)
            return BrokenMapping(values) if index == 1 else values

    summary, images, _, _ = run_audit(
        tmp_path,
        IndexedRows(),
        [],
        {"train": 3, "test": 0},
    )

    assert summary["processed_counts"] == {"train": 3, "test": 0}
    assert summary["coverage_complete"] is False
    assert [image["source_row"] for image in images] == [0, 1, 2]
    assert images[1]["image_id"] == f"hf:synthetic/CUB_train@{REVISION}:train:1"
    assert "coverage.row_audit_failed" in images[1]["error_flags"]
    assert "coverage.row_audit_failed" not in images[2]["error_flags"]


def test_manifest_write_failure_invalidates_prior_complete_summary(tmp_path, monkeypatch):
    import src.data.integrity_audit as integrity_audit

    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps({"coverage_complete": True, "run_status": "complete"}),
        encoding="utf-8",
    )

    def fail_manifest_write(path, records):
        raise OSError("synthetic manifest write failure")

    monkeypatch.setattr(integrity_audit, "_write_jsonl", fail_manifest_write)
    try:
        audit_sources(
            {"train": source("train", [row(png_bytes())]), "test": source("test", [])},
            tmp_path,
            expected_counts={"train": 1, "test": 0},
        )
    except OSError:
        pass

    if summary_path.exists():
        persisted = json.loads(summary_path.read_text(encoding="utf-8"))
        assert persisted.get("coverage_complete") is not True
        assert persisted.get("run_status") != "complete"


def test_length_lookup_failure_falls_back_to_readable_source_iterator(tmp_path):
    class RowsWithBrokenLength:
        def __len__(self):
            raise OSError("synthetic index metadata unavailable")

        def __iter__(self):
            yield row(png_bytes((1, 40, 80)), id=1)
            yield row(png_bytes((2, 50, 90)), id=2)

    summary, images, _, _ = run_audit(
        tmp_path,
        RowsWithBrokenLength(),
        [],
        {"train": 2, "test": 0},
    )

    assert summary["processed_counts"] == {"train": 2, "test": 0}
    assert summary["coverage_complete"] is True
    assert [image["source_row"] for image in images] == [0, 1]
