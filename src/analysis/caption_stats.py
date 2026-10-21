"""Descriptive caption statistics over the shared, loss-minimizing records.

Tokenization is an analysis-only view: apply Unicode NFC normalization and
casefolding, normalize NFC again, and replace left/right curly apostrophes with
straight apostrophes. A token contains Unicode alphabetic characters (``isalpha``)
with single internal apostrophes or hyphens (ASCII, U+2010, U+2011) retained when
immediately surrounded by letters. Numbers, underscores, other punctuation, and
whitespace are separators. Raw and cleaned source strings are never modified.

Lengths, vocabulary, and repetition exclude empty cleaned captions. A nonempty
punctuation-only caption has zero words and remains in the length denominators.
Character length counts Python Unicode code points in the cleaned source text.
Whitespace anomalies count lines whose raw text differs from whitespace-collapsed
and stripped raw text, including whitespace-only lines.

Repetition uses cleaned, CASE-SENSITIVE text, independently of tokenization.
Within-image extra occurrences sum ``count - 1`` for each image/text group.
Across-image extra occurrences sum ``distinct_image_count - 1`` for each text;
these count additional images, excluding repetitions already counted within an
image. Both scopes report the number of distinct repeated texts. Examples retain
at most five image IDs, with image_count showing the full denominator.
The within-image affected_image_count counts each image once even when several
different captions repeat in it. Global extras equal nonempty caption count minus
unique_nonempty_texts, and equal the sum of within-image and across-image extras.

Plot-ready word/character length distributions count nonempty cleaned captions;
the per-image nonempty-caption distribution includes images with zero captions.
All distributions are sorted by their integer measurement value.

Call once per source split when split-specific summaries are needed. This module
does not load data, clean captions, fit a model vocabulary, or choose thresholds.
"""

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
import unicodedata

from src.data.schemas import CaptionRecord, ImageRecord

from .reporting import summarize_numbers


_EXAMPLE_LIMIT = 5
_INTERNAL_JOINERS = {"'", "-", "\u2010", "\u2011"}


def tokenize_caption(text: str) -> tuple[str, ...]:
    """Return deterministic alphabetic word tokens without changing source text."""
    normalized = unicodedata.normalize("NFC", text).casefold()
    normalized = unicodedata.normalize("NFC", normalized)
    normalized = normalized.replace("\u2018", "'").replace("\u2019", "'")
    tokens: list[str] = []
    current: list[str] = []
    for index, character in enumerate(normalized):
        if character.isalpha():
            current.append(character)
        elif (
            character in _INTERNAL_JOINERS
            and current
            and current[-1].isalpha()
            and index + 1 < len(normalized)
            and normalized[index + 1].isalpha()
        ):
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tuple(tokens)


def summarize_captions(
    records: Iterable[tuple[ImageRecord, Sequence[CaptionRecord]]],
    *,
    expected_captions_per_image: int = 10,
    top_k: int = 20,
) -> dict:
    """Summarize captions with explicit image, line, caption, and token counts.

    Duplicate image inputs and captions linked to a different image raise
    ``ValueError`` to prevent inflated or misattributed statistics. Empty token
    shares are ``None`` rather than zero. The returned structure is JSON-safe.
    """
    if expected_captions_per_image < 0:
        raise ValueError("expected_captions_per_image must be non-negative")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")

    image_ids: set[str] = set()
    line_counts: list[int] = []
    nonempty_counts: list[int] = []
    character_lengths: list[int] = []
    word_lengths: list[int] = []
    token_counts: Counter[str] = Counter()
    occurrences: dict[str, Counter[str]] = defaultdict(Counter)
    examples: list[dict] = []
    whitespace_anomaly_count = 0

    for image, captions in records:
        if image.image_id in image_ids:
            raise ValueError(f"image_id must be unique: {image.image_id}")
        image_ids.add(image.image_id)
        line_counts.append(len(captions))
        nonempty_count = 0

        for caption in captions:
            if caption.image_id != image.image_id:
                raise ValueError("caption image_id must match its ImageRecord")
            if caption.raw_caption != " ".join(caption.raw_caption.split()):
                whitespace_anomaly_count += 1
            if not caption.clean_caption:
                continue

            nonempty_count += 1
            text = caption.clean_caption
            tokens = tokenize_caption(text)
            character_lengths.append(len(text))
            word_lengths.append(len(tokens))
            token_counts.update(tokens)
            occurrences[text][image.image_id] += 1
            examples.append(
                {
                    "image_id": image.image_id,
                    "caption_index": caption.caption_index,
                    "text": text,
                    "word_count": len(tokens),
                }
            )

        nonempty_counts.append(nonempty_count)

    total_tokens = sum(token_counts.values())
    hapax_word_types = sum(count == 1 for count in token_counts.values())
    ranked_tokens = sorted(token_counts.items(), key=lambda item: (-item[1], item[0]))

    def repeated_summary(*, across_images: bool) -> dict:
        repeated: list[tuple[str, Counter[str], int]] = []
        repeated_image_text_groups = 0
        affected_image_ids: set[str] = set()
        for text, counts_by_image in occurrences.items():
            if across_images:
                extra = len(counts_by_image) - 1
            else:
                extra = sum(count - 1 for count in counts_by_image.values())
                repeated_image_text_groups += sum(
                    count > 1 for count in counts_by_image.values()
                )
                affected_image_ids.update(
                    image_id for image_id, count in counts_by_image.items() if count > 1
                )
            if extra:
                repeated.append((text, counts_by_image, extra))
        repeated.sort(key=lambda item: (-item[2], item[0]))
        result = {
            "repeated_distinct_texts": len(repeated),
            "extra_occurrences": sum(extra for _, _, extra in repeated),
            "examples": [
                {
                    "text": text,
                    "occurrence_count": sum(counts_by_image.values()),
                    "image_count": len(counts_by_image),
                    "image_ids": sorted(counts_by_image)[:_EXAMPLE_LIMIT],
                    "extra_occurrences": extra,
                }
                for text, counts_by_image, extra in repeated[:_EXAMPLE_LIMIT]
            ],
        }
        if not across_images:
            result["repeated_image_text_groups"] = repeated_image_text_groups
            result["affected_image_count"] = len(affected_image_ids)
        return result

    shortest = sorted(
        examples,
        key=lambda row: (
            row["word_count"], len(row["text"]), row["image_id"], row["caption_index"]
        ),
    )[:_EXAMPLE_LIMIT]
    longest = sorted(
        examples,
        key=lambda row: (
            -row["word_count"], -len(row["text"]), row["image_id"], row["caption_index"]
        ),
    )[:_EXAMPLE_LIMIT]
    nonempty_caption_count = sum(nonempty_counts)
    caption_line_count = sum(line_counts)

    return {
        "image_count": len(image_ids),
        "caption_line_count": caption_line_count,
        "nonempty_caption_count": nonempty_caption_count,
        "empty_caption_count": caption_line_count - nonempty_caption_count,
        "images_without_caption_lines": line_counts.count(0),
        "images_without_nonempty_captions": nonempty_counts.count(0),
        "expected_captions_per_image": expected_captions_per_image,
        "per_image_line_counts": summarize_numbers(line_counts),
        "per_image_nonempty_counts": summarize_numbers(nonempty_counts),
        "line_count_mismatch_images": sum(
            count != expected_captions_per_image for count in line_counts
        ),
        "nonempty_count_mismatch_images": sum(
            count != expected_captions_per_image for count in nonempty_counts
        ),
        "character_lengths": summarize_numbers(character_lengths),
        "word_lengths": summarize_numbers(word_lengths),
        "word_length_distribution": [
            {"word_count": length, "caption_count": count}
            for length, count in sorted(Counter(word_lengths).items())
        ],
        "character_length_distribution": [
            {"character_count": length, "caption_count": count}
            for length, count in sorted(Counter(character_lengths).items())
        ],
        "per_image_nonempty_count_distribution": [
            {"caption_count": caption_count, "image_count": image_count}
            for caption_count, image_count in sorted(Counter(nonempty_counts).items())
        ],
        "zero_word_caption_count": word_lengths.count(0),
        "vocabulary_size": len(token_counts),
        "total_tokens": total_tokens,
        "token_frequencies": dict(sorted(token_counts.items())),
        "top_tokens": [
            {"token": token, "count": count} for token, count in ranked_tokens[:top_k]
        ],
        "hapax_word_types": hapax_word_types,
        "hapax_token_share": hapax_word_types / total_tokens if total_tokens else None,
        "whitespace_anomaly_count": whitespace_anomaly_count,
        "repetition": {
            "unique_nonempty_texts": len(occurrences),
            "global_extra_occurrences": nonempty_caption_count - len(occurrences),
            "within_images": repeated_summary(across_images=False),
            "across_images": repeated_summary(across_images=True),
        },
        "shortest_examples": shortest,
        "longest_examples": longest,
    }
