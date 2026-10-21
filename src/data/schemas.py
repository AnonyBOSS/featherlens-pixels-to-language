from dataclasses import dataclass


@dataclass(frozen=True)
class SourceSpec:
    """A fixed Hugging Face source and its official split."""

    repo_id: str
    source_split: str
    revision: str

    def image_id(self, source_row: int) -> str:
        """Identify one row independently of its filename or source-provided id."""
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
