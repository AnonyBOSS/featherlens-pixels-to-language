"""CUB-200-2011 loading from the pinned Hugging Face copies, plus record types."""

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from datasets import ClassLabel, Image, Value, load_dataset


@dataclass(frozen=True)
class SourceSpec:
    """A Hugging Face dataset repo, split and revision."""

    repo_id: str
    source_split: str
    revision: str

    def image_id(self, source_row: int) -> str:
        """Stable id for a row that doesn't depend on its filename."""
        if source_row < 0:
            raise ValueError("source_row must be non-negative")
        return f"hf:{self.repo_id}@{self.revision}:{self.source_split}:{source_row}"


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    source_repo: str
    source_revision: str
    source_split: str
    source_row: int
    source_id: int | None
    file_name: str | None
    class_id: int
    class_name: str


@dataclass(frozen=True)
class CaptionRecord:
    image_id: str
    caption_index: int
    raw_caption: str
    clean_caption: str


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


def parse_captions(description: str | None, *, image_id: str) -> tuple[CaptionRecord, ...]:
    """Split a newline-separated description into captions.

    Every line is kept (including empty ones) so indices match the source;
    the clean copy only collapses whitespace.
    """
    if description is None:
        return ()
    if not isinstance(description, str):
        raise TypeError("description must be a string or None")
    lines = description.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines[-1] == "":
        lines.pop()  # trailing newline, not an extra caption
    return tuple(
        CaptionRecord(
            image_id=image_id,
            caption_index=index,
            raw_caption=line,
            clean_caption=re.sub(r"\s+", " ", line).strip(),
        )
        for index, line in enumerate(lines)
    )


def validate_source_schema(features: Mapping[str, Any]) -> tuple[str, ...]:
    """Check the expected CUB columns and return class names in label order."""
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
    """One split: the raw (undecoded) dataset plus metadata iteration."""

    def __init__(self, spec: SourceSpec, dataset: Any, class_names: tuple[str, ...]):
        self.spec = spec
        self.dataset = dataset
        self.class_names = class_names
        self._metadata = dataset.select_columns(list(_METADATA_FIELDS))

    def iter_records(self) -> Iterator[tuple[ImageRecord, tuple[CaptionRecord, ...]]]:
        """Yield (image record, captions) per row without decoding any pixels."""
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
    """Load the official 'train' or 'test' split at its pinned revision.

    Use ``streaming=True`` for a quick look without downloading everything.
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
    """Load both splits and check they use the same label mapping."""
    train = load_cub_split("train", streaming=streaming)
    test = load_cub_split("test", streaming=streaming)
    if train.class_names != test.class_names:
        raise ValueError("CUB train/test class mapping differs")
    return {"train": train, "test": test}
