"""Pinned CUB source loading with metadata access that avoids image decoding."""

from collections.abc import Iterator, Mapping
from typing import Any

from datasets import ClassLabel, Image, Value, load_dataset

from .captions import parse_captions
from .schemas import CaptionRecord, ImageRecord, SourceSpec


_SOURCES = {
    "train": SourceSpec(
        "Multimodal-Fatima/CUB_train",
        "train",
        "bd3988995e6db193427255abdebd522afe24ce76",
    ),
    "test": SourceSpec(
        "Multimodal-Fatima/CUB_test",
        "test",
        "82447a42c5e05c7fe21cd36accfdf399148e8668",
    ),
}
_REQUIRED_FIELDS = ("image", "description", "label", "file_name", "id")
_METADATA_FIELDS = ("description", "label", "file_name", "id")


def validate_source_schema(features: Mapping[str, Any]) -> tuple[str, ...]:
    """Check the required CUB fields and return class names in label-ID order."""
    missing = [name for name in _REQUIRED_FIELDS if name not in features]
    if missing:
        raise ValueError(f"CUB source is missing required fields: {', '.join(missing)}")
    if not isinstance(features["image"], Image):
        raise ValueError("CUB image field must use the Hugging Face Image feature")
    for name, dtype in (("description", "string"), ("file_name", "string"), ("id", "int64")):
        feature = features[name]
        if not isinstance(feature, Value) or feature.dtype != dtype:
            raise ValueError(f"CUB {name} field must be Value({dtype!r})")
    label = features["label"]
    if not isinstance(label, ClassLabel):
        raise ValueError("CUB label field must be a ClassLabel")
    names = tuple(label.names)
    if len(names) != 200:
        raise ValueError(f"CUB label mapping must contain 200 classes, got {len(names)}")
    if len(set(names)) != len(names):
        raise ValueError("CUB class names must be unique")
    return names


class CubSource:
    """A pinned split, with raw image references and iterable metadata records."""

    def __init__(self, spec: SourceSpec, dataset: Any, class_names: tuple[str, ...]):
        self.spec = spec
        self.dataset = dataset
        self.class_names = class_names
        self._metadata = dataset.select_columns(list(_METADATA_FIELDS))

    def iter_records(self) -> Iterator[tuple[ImageRecord, tuple[CaptionRecord, ...]]]:
        """Yield source metadata and captions without decoding image pixels."""
        for source_row, row in enumerate(self._metadata):
            label = row["label"]
            if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(self.class_names):
                raise ValueError(f"Invalid label at {self.spec.source_split} source row {source_row}: {label!r}")
            image_id = self.spec.image_id(source_row)
            image = ImageRecord(
                image_id=image_id,
                source_repo=self.spec.repo_id,
                source_revision=self.spec.revision,
                source_split=self.spec.source_split,
                source_row=source_row,
                source_id=row["id"],
                file_name=row["file_name"],
                class_id=label,
                class_name=self.class_names[label],
            )
            yield image, parse_captions(row["description"], image_id=image_id)


def load_cub_split(source_split: str, *, streaming: bool = False) -> CubSource:
    """Load one official CUB split at its pinned revision.

    Full loading is the default for later random access. Set ``streaming=True``
    for a small inspection without downloading the entire split.
    """
    try:
        spec = _SOURCES[source_split]
    except KeyError as error:
        raise ValueError(f"Unknown source_split {source_split!r}; expected 'train' or 'test'") from error
    dataset = load_dataset(spec.repo_id, split=spec.source_split, revision=spec.revision, streaming=streaming)
    class_names = validate_source_schema(dataset.features)
    raw_dataset = dataset.cast_column("image", Image(decode=False))
    return CubSource(spec, raw_dataset, class_names)


def load_cub_sources(*, streaming: bool = False) -> dict[str, CubSource]:
    """Load both official splits and reject an inconsistent label-ID mapping."""
    train = load_cub_split("train", streaming=streaming)
    test = load_cub_split("test", streaming=streaming)
    if train.class_names != test.class_names:
        raise ValueError("CUB train/test class mapping differs")
    return {"train": train, "test": test}
