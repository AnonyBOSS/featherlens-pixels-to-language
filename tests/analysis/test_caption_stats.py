import copy
from dataclasses import replace
import json

import pytest

from src.analysis.caption_stats import summarize_captions, tokenize_caption
from src.data.captions import parse_captions
from src.data.schemas import ImageRecord


def image_and_captions(image_id, description):
    image = ImageRecord(
        image_id=image_id,
        source_repo="synthetic",
        source_revision="test",
        source_split="train",
        source_row=0,
        source_id=None,
        file_name=None,
        class_id=0,
        class_name="Synthetic bird",
    )
    return image, parse_captions(description, image_id=image_id)


def test_tokenization_handles_unicode_words_joiners_and_numeric_separators():
    text = "CAFÉ Cafe\u0301 bird’s blue-green R2D2 123 foo_bar 'wings' naïve\u2011bird"

    assert tokenize_caption(text) == (
        "café", "café", "bird's", "blue-green", "r", "d", "foo", "bar", "wings",
        "naïve\u2011bird",
    )
    assert tokenize_caption("!!! 123 ___ -- ''") == ()
    assert tokenize_caption("bird--like ‘crest’") == ("bird", "like", "crest")


def test_hand_counted_denominators_lengths_vocabulary_and_whitespace():
    records = [
        image_and_captions("image-a", "  Red  bird.\nRed bird.\n\n!!!"),
        image_and_captions("image-b", "Red bird.\nBlue bird"),
        image_and_captions("image-c", None),
    ]

    summary = summarize_captions(records, expected_captions_per_image=2, top_k=2)

    assert summary["image_count"] == 3
    assert summary["caption_line_count"] == 6
    assert summary["nonempty_caption_count"] == 5
    assert summary["empty_caption_count"] == 1
    assert summary["images_without_caption_lines"] == 1
    assert summary["images_without_nonempty_captions"] == 1
    assert summary["per_image_line_counts"]["mean"] == 2
    assert summary["per_image_nonempty_counts"]["mean"] == pytest.approx(5 / 3)
    assert summary["line_count_mismatch_images"] == 2
    assert summary["nonempty_count_mismatch_images"] == 2
    assert summary["character_lengths"]["count"] == 5
    assert summary["character_lengths"]["mean"] == pytest.approx(7.8)
    assert summary["word_lengths"]["mean"] == pytest.approx(1.6)
    assert summary["word_lengths"]["min"] == 0
    assert summary["zero_word_caption_count"] == 1
    assert summary["word_length_distribution"] == [
        {"word_count": 0, "caption_count": 1},
        {"word_count": 2, "caption_count": 4},
    ]
    assert summary["character_length_distribution"] == [
        {"character_count": 3, "caption_count": 1},
        {"character_count": 9, "caption_count": 4},
    ]
    assert summary["per_image_nonempty_count_distribution"] == [
        {"caption_count": 0, "image_count": 1},
        {"caption_count": 2, "image_count": 1},
        {"caption_count": 3, "image_count": 1},
    ]
    for distribution in ("word_length_distribution", "character_length_distribution"):
        assert sum(row["caption_count"] for row in summary[distribution]) == summary["nonempty_caption_count"]
    image_distribution = summary["per_image_nonempty_count_distribution"]
    assert sum(row["image_count"] for row in image_distribution) == summary["image_count"]
    assert sum(row["caption_count"] * row["image_count"] for row in image_distribution) == summary["nonempty_caption_count"]
    assert summary["total_tokens"] == 8
    assert sum(
        row["word_count"] * row["caption_count"]
        for row in summary["word_length_distribution"]
    ) == summary["total_tokens"]
    assert summary["vocabulary_size"] == 3
    assert summary["token_frequencies"] == {"bird": 4, "blue": 1, "red": 3}
    assert summary["top_tokens"] == [
        {"token": "bird", "count": 4}, {"token": "red", "count": 3}
    ]
    assert summary["hapax_word_types"] == 1
    assert summary["hapax_token_share"] == pytest.approx(1 / 8)
    assert summary["whitespace_anomaly_count"] == 1
    assert summary["shortest_examples"][0] == {
        "image_id": "image-a", "caption_index": 3, "text": "!!!", "word_count": 0
    }


def test_repetition_separates_within_image_from_across_image_occurrences():
    records = [
        image_and_captions("image-a", "Same\nSame\nOnly\nOnly\n\n"),
        image_and_captions("image-b", "Same\nOnly\nOther"),
        image_and_captions("image-c", "Same\nOnly\nonly"),
    ]

    summary = summarize_captions(records)
    within = summary["repetition"]["within_images"]
    across = summary["repetition"]["across_images"]

    assert within["repeated_distinct_texts"] == 2
    assert within["repeated_image_text_groups"] == 2
    assert within["affected_image_count"] == 1  # Two repeating texts in one image.
    assert within["extra_occurrences"] == 2
    assert across["repeated_distinct_texts"] == 2
    assert across["extra_occurrences"] == 4
    assert [example["text"] for example in across["examples"]] == ["Only", "Same"]
    assert across["examples"][0] == {
        "text": "Only",
        "occurrence_count": 4,
        "image_count": 3,
        "image_ids": ["image-a", "image-b", "image-c"],
        "extra_occurrences": 2,
    }
    assert within["extra_occurrences"] + across["extra_occurrences"] == 6
    assert summary["repetition"]["unique_nonempty_texts"] == 4
    assert summary["repetition"]["global_extra_occurrences"] == 6
    assert (
        summary["repetition"]["global_extra_occurrences"]
        == summary["nonempty_caption_count"] - summary["repetition"]["unique_nonempty_texts"]
        == within["extra_occurrences"] + across["extra_occurrences"]
    )


def test_missing_blank_and_punctuation_only_captions_remain_distinguishable():
    summary = summarize_captions(
        [
            image_and_captions("missing", None),
            image_and_captions("blank", " \t\n"),
            image_and_captions("punctuation", "!!!"),
        ]
    )

    assert summary["image_count"] == 3
    assert summary["caption_line_count"] == 2
    assert summary["empty_caption_count"] == 1
    assert summary["nonempty_caption_count"] == 1
    assert summary["images_without_caption_lines"] == 1
    assert summary["images_without_nonempty_captions"] == 2
    assert summary["word_lengths"]["count"] == 1
    assert summary["word_lengths"]["mean"] == 0
    assert summary["total_tokens"] == 0
    assert summary["hapax_token_share"] is None
    assert summary["whitespace_anomaly_count"] == 1
    assert summary["repetition"]["within_images"]["repeated_distinct_texts"] == 0
    assert summary["repetition"]["within_images"]["affected_image_count"] == 0


def test_empty_input_is_json_safe_and_has_no_invented_observations():
    summary = summarize_captions(iter(()), top_k=0)

    assert summary["image_count"] == 0
    assert summary["word_lengths"]["count"] == 0
    assert summary["word_lengths"]["mean"] is None
    assert summary["per_image_line_counts"]["min"] is None
    assert summary["line_count_mismatch_images"] == 0
    assert summary["top_tokens"] == []
    assert summary["shortest_examples"] == []
    assert summary["longest_examples"] == []
    assert summary["hapax_token_share"] is None
    assert summary["word_length_distribution"] == []
    assert summary["character_length_distribution"] == []
    assert summary["per_image_nonempty_count_distribution"] == []
    assert summary["repetition"]["unique_nonempty_texts"] == 0
    assert summary["repetition"]["global_extra_occurrences"] == 0
    assert summary["repetition"]["within_images"]["affected_image_count"] == 0
    json.dumps(summary, allow_nan=False)


def test_summary_is_order_independent_and_does_not_mutate_source_records():
    records = [
        image_and_captions("image-b", "  Bird’s  wing \nblue-green"),
        image_and_captions("image-a", "Bird’s wing\nBlue-green"),
    ]
    before = copy.deepcopy(records)

    summary = summarize_captions(iter(records))

    assert records == before
    assert summary == summarize_captions(reversed(records))
    assert summary["token_frequencies"] == {"bird's": 2, "blue-green": 2, "wing": 2}
    assert summary["top_tokens"] == [
        {"token": "bird's", "count": 2},
        {"token": "blue-green", "count": 2},
        {"token": "wing", "count": 2},
    ]
    assert summary["repetition"]["across_images"]["repeated_distinct_texts"] == 1
    json.dumps(summary, allow_nan=False)


def test_duplicate_images_and_mislinked_captions_are_rejected():
    record = image_and_captions("image-a", "Bird")
    with pytest.raises(ValueError, match="unique"):
        summarize_captions([record, record])

    image, captions = record
    mislinked = replace(captions[0], image_id="another-image")
    with pytest.raises(ValueError, match="match"):
        summarize_captions([(image, [mislinked])])


@pytest.mark.parametrize("kwargs", [{"top_k": -1}, {"expected_captions_per_image": -1}])
def test_negative_summary_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError, match="non-negative"):
        summarize_captions([], **kwargs)
