"""Training-caption lexical mentions and bounded within-image comparisons.

The existing tokenizer and source strings are unchanged. The matching view also
checks components separated by ASCII/U+2010/U+2011 hyphens against an explicit
lexicon. Mentions do not verify visible attributes, meaning, or caption quality.
TTR pools all tokens for one image; Jaccard compares unordered pairs of nonempty
word sets within that image. Stopwords and repeated captions remain included.
Per-image values receive equal weight in overall and class summaries.
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import asdict
from itertools import combinations
import re
from statistics import fmean
from typing import Any

from src.data.schemas import CaptionRecord, ImageRecord, SourceSpec

from .caption_stats import tokenize_caption
from .reporting import summarize_numbers


LEXICON_VERSION = "caption-terms-v1"
MATCHING_RULE = "whole-token-or-hyphen-component"
TERM_LEXICON = {
    "color": {
        "black": ["black", "blackish"],
        "white": ["white", "whitish"],
        "gray": ["gray", "grey", "grayish", "greyish"],
        "brown": ["brown", "brownish"],
        "red": ["red", "reddish"],
        "blue": ["blue", "bluish"],
        "green": ["green", "greenish"],
        "yellow": ["yellow", "yellowish"],
        "orange": ["orange"],
        "pink": ["pink", "pinkish"],
        "purple": ["purple", "purplish"],
        "buff": ["buff"],
        "olive": ["olive"],
    },
    "anatomy": {
        "beak": ["beak", "beaks", "bill", "bills"],
        "head": ["head", "heads", "headed"],
        "wing": ["wing", "wings", "winged"],
        "tail": ["tail", "tails", "tailed"],
        "breast": ["breast", "breasts", "breasted", "chest"],
        "belly": ["belly", "bellies", "underbelly"],
        "back": ["back", "backs"],
        "throat": ["throat", "throats", "throated"],
        "neck": ["neck", "necks", "necked"],
        "eye": ["eye", "eyes", "eyed"],
        "crown": ["crown", "crowns", "crowned"],
        "leg": ["leg", "legs", "legged"],
        "foot": ["foot", "feet"],
        "feather": ["feather", "feathers", "feathered", "plumage"],
    },
    "direction": {
        "left": ["left"],
        "right": ["right"],
        "above": ["above"],
        "below": ["below"],
        "front": ["front"],
        "behind": ["behind"],
        "upper": ["upper"],
        "lower": ["lower"],
    },
}
_HYPHENS = re.compile(r"[-\u2010\u2011]")
_EXAMPLE_LIMIT = 5
_METRICS = ("mean_caption_words", "type_token_ratio", "pairwise_jaccard_mean", "token_count", "vocabulary_size")


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path} must be a nonnegative integer")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a nonempty string")
    return value


def _keep_example(candidates: list, key: tuple, example: dict) -> None:
    """Keep only five deterministic candidates; never retain a pairwise corpus."""
    if len(candidates) < _EXAMPLE_LIMIT or key < candidates[-1][0]:
        candidates.append((key, example))
        candidates.sort(key=lambda candidate: candidate[0])
        del candidates[_EXAMPLE_LIMIT:]


def _mention_terms(tokens: tuple[str, ...]) -> dict[str, set[str]]:
    matching = set(tokens)
    for token in tokens:
        matching.update(_HYPHENS.split(token))
    return {
        category: {
            canonical for canonical, variants in terms.items()
            if matching.intersection(variants)
        }
        for category, terms in TERM_LEXICON.items()
    }


def summarize_caption_details(
    records: Iterable[tuple[ImageRecord, Sequence[CaptionRecord]]], *, class_names: Sequence[str]
) -> dict[str, Any]:
    """Describe linked captions without loading, cleaning, fitting, or modeling.

    Identities must be canonical for one training source. Duplicate image IDs,
    source rows, or caption indices and parent/class mismatches raise explicitly.
    Nonempty zero-word captions count toward mean caption length and mention
    denominators, but cannot form a token-bearing Jaccard pair. Undefined rates
    and per-image statistics are null, never invented zero measurements.
    """
    if isinstance(class_names, (str, bytes)):
        raise ValueError("class_names must be an ordered sequence of names")
    names = tuple(class_names)
    for index, name in enumerate(names):
        _text(name, f"class_names[{index}]")
    if len(set(names)) != len(names):
        raise ValueError("class_names must be unique")
    image_ids = set()
    source_rows = set()
    provenance = None
    images = []
    term_examples = {category: {term: [] for term in terms} for category, terms in TERM_LEXICON.items()}
    lowest_pairs, highest_pairs = [], []

    for record in records:
        try:
            image, captions = record
        except (TypeError, ValueError) as error:
            raise ValueError("Each record must contain one ImageRecord and its caption sequence") from error
        if not isinstance(image, ImageRecord):
            raise ValueError("Each record must use the shared ImageRecord type")
        position = _integer(image.source_row, "ImageRecord.source_row")
        repo = _text(image.source_repo, "ImageRecord.source_repo")
        revision = _text(image.source_revision, "ImageRecord.source_revision")
        if image.source_split != "train":
            raise ValueError("Caption details accept only the official training split")
        expected_id = SourceSpec(repo, image.source_split, revision).image_id(position)
        if image.image_id != expected_id:
            raise ValueError("ImageRecord.image_id must match canonical source provenance")
        if image.image_id in image_ids or position in source_rows:
            raise ValueError("ImageRecord image_id and source_row must each be unique")
        current_provenance = (repo, revision, image.source_split)
        if provenance is not None and current_provenance != provenance:
            raise ValueError("Caption records must come from one source repository/revision/split")
        provenance = current_provenance
        image_ids.add(image.image_id)
        source_rows.add(position)
        class_id = _integer(image.class_id, "ImageRecord.class_id")
        if class_id >= len(names) or image.class_name != names[class_id]:
            raise ValueError("ImageRecord class_id/class_name must match the ordered source mapping")
        if image.source_id is not None and (isinstance(image.source_id, bool) or not isinstance(image.source_id, int)):
            raise ValueError("ImageRecord.source_id must be an integer or null")
        if image.file_name is not None and not isinstance(image.file_name, str):
            raise ValueError("ImageRecord.file_name must be text or null")
        if isinstance(captions, (str, bytes)) or not isinstance(captions, Sequence):
            raise ValueError("Captions must be a sequence of shared CaptionRecord values")
        indices = set()
        token_captions = []
        word_counts = []
        total_tokens = 0
        vocabulary = set()
        category_counts = {category: 0 for category in TERM_LEXICON}
        term_counts = {category: Counter() for category in TERM_LEXICON}
        for caption in captions:
            if not isinstance(caption, CaptionRecord):
                raise ValueError("Captions must use the shared CaptionRecord type")
            index = _integer(caption.caption_index, "CaptionRecord.caption_index")
            if index in indices:
                raise ValueError("CaptionRecord.caption_index must be unique within its image")
            indices.add(index)
            if caption.image_id != image.image_id:
                raise ValueError("CaptionRecord.image_id must match its parent ImageRecord")
            if not isinstance(caption.raw_caption, str) or not isinstance(caption.clean_caption, str):
                raise ValueError("CaptionRecord raw_caption and clean_caption must be strings")
            if not caption.clean_caption:
                continue
            tokens = tokenize_caption(caption.clean_caption)
            word_counts.append(len(tokens))
            total_tokens += len(tokens)
            vocabulary.update(tokens)
            if tokens:
                token_captions.append((index, caption.clean_caption, frozenset(tokens)))
            for category, hits in _mention_terms(tokens).items():
                if hits:
                    category_counts[category] += 1
                term_counts[category].update(hits)
                for term in hits:
                    _keep_example(
                        term_examples[category][term], (position, index),
                        {"image_id": image.image_id, "caption_index": index, "text": caption.clean_caption},
                    )

        pair_values = []
        token_captions.sort(key=lambda row: row[0])
        for first, second in combinations(token_captions, 2):
            overlap = len(first[2].intersection(second[2])) / len(first[2].union(second[2]))
            pair_values.append(overlap)
            example = {
                "image_id": image.image_id,
                "caption_indices": [first[0], second[0]],
                "texts": [first[1], second[1]],
                "jaccard": overlap,
            }
            tie = (position, first[0], second[0])
            _keep_example(lowest_pairs, (overlap, *tie), example)
            _keep_example(highest_pairs, (-overlap, *tie), example)
        images.append({
            **asdict(image),
            "caption_line_count": len(captions),
            "nonempty_caption_count": len(word_counts),
            "token_bearing_caption_count": len(token_captions),
            "token_count": total_tokens,
            "vocabulary_size": len(vocabulary),
            "type_token_ratio": len(vocabulary) / total_tokens if total_tokens else None,
            "mean_caption_words": fmean(word_counts) if word_counts else None,
            "pair_count": len(pair_values),
            "pairwise_jaccard_mean": fmean(pair_values) if pair_values else None,
            "category_hits": [category for category, count in category_counts.items() if count],
            "term_hits": {category: sorted(counts) for category, counts in term_counts.items()},
            "term_caption_counts": {category: dict(sorted(counts.items())) for category, counts in term_counts.items()},
            "category_caption_counts": category_counts,
        })

    images.sort(key=lambda row: row["source_row"])
    image_count = len(images)
    nonempty_count = sum(row["nonempty_caption_count"] for row in images)
    captioned_images = sum(row["nonempty_caption_count"] > 0 for row in images)

    def rates(captions: int, matching_images: int) -> dict:
        return {
            "caption_count": captions,
            "image_count": matching_images,
            "caption_prevalence": captions / nonempty_count if nonempty_count else None,
            "image_prevalence_among_captioned": matching_images / captioned_images if captioned_images else None,
            "image_prevalence_all": matching_images / image_count if image_count else None,
        }

    prevalence = []
    for category, terms in TERM_LEXICON.items():
        prevalence.append({
            "category": category,
            **rates(
                sum(row["category_caption_counts"][category] for row in images),
                sum(category in row["category_hits"] for row in images),
            ),
            "terms": [
                {
                    "term": term,
                    **rates(
                        sum(row["term_caption_counts"][category].get(term, 0) for row in images),
                        sum(term in row["term_hits"][category] for row in images),
                    ),
                    "examples": [example for _, example in term_examples[category][term]],
                }
                for term in terms
            ],
        })
    classes = []
    for class_id, class_name in enumerate(names):
        selected = [row for row in images if row["class_id"] == class_id]
        classes.append({
            "class_id": class_id,
            "class_name": class_name,
            "image_count": len(selected),
            "images_with_nonempty_captions": sum(row["nonempty_caption_count"] > 0 for row in selected),
            "nonempty_caption_count": sum(row["nonempty_caption_count"] for row in selected),
            **{
                metric: summarize_numbers(row[metric] for row in selected if row[metric] is not None)
                for metric in ("mean_caption_words", "pairwise_jaccard_mean", "type_token_ratio")
            },
            "category_image_counts": {
                category: sum(category in row["category_hits"] for row in selected)
                for category in TERM_LEXICON
            },
        })
    return {
        "lexicon": {"version": LEXICON_VERSION, "matching_rule": MATCHING_RULE, "categories": deepcopy(TERM_LEXICON)},
        "coverage": {
            "image_count": image_count,
            "caption_line_count": sum(row["caption_line_count"] for row in images),
            "nonempty_caption_count": nonempty_count,
            "token_bearing_caption_count": sum(row["token_bearing_caption_count"] for row in images),
            "images_with_nonempty_captions": captioned_images,
            "images_with_token_bearing_captions": sum(row["token_bearing_caption_count"] > 0 for row in images),
            "images_with_pairwise_overlap": sum(row["pair_count"] > 0 for row in images),
            "pair_count": sum(row["pair_count"] for row in images),
        },
        "term_prevalence": prevalence,
        "images": images,
        "overall": {metric: summarize_numbers(row[metric] for row in images if row[metric] is not None) for metric in _METRICS},
        "classes": classes,
        "overlap_examples": {
            "lowest": [example for _, example in lowest_pairs],
            "highest": [example for _, example in highest_pairs],
        },
    }
