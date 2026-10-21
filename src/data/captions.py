"""Loss-minimizing parsing of newline-separated CUB descriptions."""

import re

from .schemas import CaptionRecord


def parse_captions(description: str | None, *, image_id: str) -> tuple[CaptionRecord, ...]:
    """Preserve each source line and index; only collapse whitespace in the cleaned copy."""
    if description is None:
        return ()
    if not isinstance(description, str):
        raise TypeError("description must be a string or None")
    lines = description.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines[-1] == "":
        lines.pop()  # A terminal newline is not an extra caption.
    return tuple(
        CaptionRecord(
            image_id=image_id,
            caption_index=index,
            raw_caption=line,
            clean_caption=re.sub(r"\s+", " ", line).strip(),
        )
        for index, line in enumerate(lines)
    )
