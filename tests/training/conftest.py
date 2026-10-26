"""Synthetic full-coverage audit and a real indexed pinned CubSource, offline."""
from collections import Counter
from io import BytesIO
import json

from PIL import Image
import pytest

from src.data.preparation import prepare_classification
from src.data.source import CubSource, _SOURCES


class SyntheticRows:
    def __init__(self, rows, pixels):
        self.rows, self.pixels, self.reads = rows, pixels, []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        self.reads.append(index)
        row = self.rows[index]
        return {"image": {"bytes": self.pixels}, "label": row["class_id"],
                "id": row["source_id"], "file_name": row["file_name"],
                "description": "A synthetic bird."}

    def select_columns(self, columns):
        return [{"description": "A synthetic bird.", "label": row["class_id"],
                 "id": row["source_id"], "file_name": row["file_name"]}
                for row in self.rows]


@pytest.fixture(scope="module")
def audited_prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("pretrained-preparation")
    audit = root / "audit"
    audit.mkdir()
    images = []
    counts = {"train": 5994, "test": 5794}
    for split in counts:
        sizes = [30] * 194 + [29] * 6 if split == "train" else [29] * 194 + [28] * 6
        labels = [label for label, count in enumerate(sizes) for _ in range(count)]
        spec = _SOURCES[split]
        for index, label in enumerate(labels):
            digest = f"{len(images) + 1:064x}"
            images.append({
                "image_id": spec.image_id(index), "source_repo": spec.repo_id,
                "source_revision": spec.revision, "source_split": split,
                "source_row": index, "source_id": index, "file_name": f"{split}_{index}.png",
                "class_id": label, "class_name": f"Bird {label}", "metadata_valid": True,
                "image_valid": True, "image_status": "ok", "image_error_code": None,
                "image_format": "PNG", "image_mode": "RGB", "width": 8, "height": 4,
                "encoded_sha256": digest, "pixel_sha256": digest,
                "caption_count": 10, "usable_caption_count": 10, "caption_structural_valid": True,
                "exact_group_ids": [], "error_flags": [], "warning_flags": [],
                "error_messages": {}, "audit_status": "valid",
            })
    summary = {
        "schema_version": 1, "run_status": "complete", "coverage_complete": True,
        "expected_counts": counts, "observed_counts": counts, "processed_counts": counts,
        "source_revisions": {split: spec.revision for split, spec in _SOURCES.items()},
        "row_read_errors": {"train": 0, "test": 0}, "row_audit_errors": {"train": 0, "test": 0},
        "fatal_errors": [], "pixel_hash_algorithm": "exif-rgb8-white:v1",
        "audit_status_counts": dict(Counter(row["audit_status"] for row in images)),
        "error_flag_counts": {}, "warning_flag_counts": {},
        "error_flag_counts_by_split": {split: {} for split in counts},
        "warning_flag_counts_by_split": {split: {} for split in counts},
        "exact_duplicate_groups": 0, "exact_duplicate_images": 0, "cross_split_exact_groups": 0,
    }
    (audit / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (audit / "images.jsonl").write_text("".join(json.dumps(row) + "\n" for row in images), encoding="utf-8")
    (audit / "exact_groups.jsonl").write_text("", encoding="utf-8")
    prepared = root / "prepared"
    prepare_classification(audit, prepared)
    image = Image.new("RGB", (8, 4), (255, 0, 0))
    image.paste((0, 0, 255), (4, 0, 8, 4))
    pixels = BytesIO()
    image.save(pixels, format="PNG")
    return prepared, [row for row in images if row["source_split"] == "train"], pixels.getvalue()


@pytest.fixture
def pinned_source(audited_prepared):
    _, rows, pixels = audited_prepared
    return CubSource(_SOURCES["train"], SyntheticRows(rows, pixels), tuple(f"Bird {i}" for i in range(200)))
