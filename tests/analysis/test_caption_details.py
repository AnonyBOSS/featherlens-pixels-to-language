"""Hand-counted lexical mentions and per-image overlap on shared records."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from src.analysis.caption_details import summarize_caption_details
from src.analysis.caption_stats import tokenize_caption
from src.data.captions import parse_captions
from src.data.schemas import CaptionRecord, ImageRecord, SourceSpec


SPEC = SourceSpec("synthetic/cub", "train", "a" * 40)
NAMES = ("Bird A", "Bird B", "Absent bird")


def record(position, description, *, class_id=0, spec=SPEC):
    image = ImageRecord(
        image_id=spec.image_id(position), source_repo=spec.repo_id,
        source_revision=spec.revision, source_split=spec.source_split,
        source_row=position, source_id=position, file_name=f"bird-{position}.jpg",
        class_id=class_id, class_name=NAMES[class_id],
    )
    return image, parse_captions(description, image_id=image.image_id)


def category(details, name):
    return next(item for item in details["term_prevalence"] if item["category"] == name)


def term(details, family, name):
    return next(item for item in category(details, family)["terms"] if item["term"] == name)


@pytest.fixture
def records():
    return [
        record(0, "Red red bird wings left\nBlue bird beak\n\n!!!"),
        record(1, "Red bird\nRed bird"),
        record(2, None, class_id=1),
        record(3, "???", class_id=1),
    ]


def test_hand_counted_coverage_and_mention_denominators(records):
    details = summarize_caption_details(records, class_names=NAMES)

    assert details["coverage"] == {
        "image_count": 4, "caption_line_count": 7, "nonempty_caption_count": 6,
        "token_bearing_caption_count": 4, "images_with_nonempty_captions": 3,
        "images_with_token_bearing_captions": 2, "images_with_pairwise_overlap": 2,
        "pair_count": 2,
    }
    red = term(details, "color", "red")
    assert red["caption_count"] == 3  # Four occurrences, three caption mentions.
    assert red["image_count"] == 2
    assert red["caption_prevalence"] == 0.5
    assert red["image_prevalence_among_captioned"] == pytest.approx(2 / 3)
    assert red["image_prevalence_all"] == 0.5
    assert red["examples"][0] == {
        "image_id": SPEC.image_id(0), "caption_index": 0,
        "text": "Red red bird wings left",
    }


def test_category_counts_are_caption_unions_and_categories_can_overlap():
    details = summarize_caption_details([record(0, "red blue wing beak left\nred red")], class_names=NAMES)
    colors = category(details, "color")
    assert colors["caption_count"] == 2
    assert term(details, "color", "red")["caption_count"] == 2
    assert term(details, "color", "blue")["caption_count"] == 1
    assert colors["caption_count"] != sum(item["caption_count"] for item in colors["terms"])
    row = details["images"][0]
    assert row["category_caption_counts"] == {"color": 2, "anatomy": 1, "direction": 1}
    assert row["term_caption_counts"]["color"] == {"blue": 1, "red": 2}
    assert row["term_hits"]["color"] == ["blue", "red"]
    assert row["category_hits"] == ["color", "anatomy", "direction"]
    assert sum(row["category_caption_counts"].values()) > row["nonempty_caption_count"]


def test_explicit_variants_and_unicode_hyphen_components_do_not_change_base_tokens():
    text = "Grey wings bills red-breasted blue\u2010green left\u2011facing"
    item = record(0, text)
    before = deepcopy(item)
    details = summarize_caption_details([item], class_names=NAMES)
    row = details["images"][0]

    assert details["lexicon"]["version"] == "caption-terms-v1"
    assert details["lexicon"]["matching_rule"] == "whole-token-or-hyphen-component"
    assert "grey" in details["lexicon"]["categories"]["color"]["gray"]
    assert row["term_hits"] == {
        "color": ["blue", "gray", "green", "red"],
        "anatomy": ["beak", "breast", "wing"], "direction": ["left"],
    }
    assert row["token_count"] == len(tokenize_caption(text)) == 6
    assert row["vocabulary_size"] == 6
    assert row["mean_caption_words"] == 6
    assert item == before


def test_whole_token_matching_avoids_substrings_and_does_not_resolve_negation():
    details = summarize_caption_details(
        [record(0, "hundred wingless billion leftover upright\nnot left")],
        class_names=NAMES,
    )
    row = details["images"][0]
    assert row["term_hits"]["color"] == row["term_hits"]["anatomy"] == []
    assert row["category_caption_counts"] == {"color": 0, "anatomy": 0, "direction": 1}
    assert term(details, "direction", "left")["caption_count"] == 1
    assert term(details, "direction", "right")["caption_count"] == 0


def test_pooled_ttr_mean_length_and_class_summaries_weight_images_equally(records):
    details = summarize_caption_details(records, class_names=NAMES)
    first, repeated, missing, punctuation = details["images"]

    assert first["token_count"] == 8
    assert first["vocabulary_size"] == 6
    assert first["type_token_ratio"] == 0.75
    assert first["mean_caption_words"] == pytest.approx(8 / 3)
    assert first["pairwise_jaccard_mean"] == pytest.approx(1 / 6)
    assert repeated["nonempty_caption_count"] == 2  # Repeated source text remains.
    assert repeated["type_token_ratio"] == 0.5
    assert repeated["pairwise_jaccard_mean"] == 1.0
    assert missing["mean_caption_words"] is None
    assert punctuation["mean_caption_words"] == 0.0
    assert punctuation["type_token_ratio"] is None
    classes = details["classes"]
    assert [item["class_name"] for item in classes] == list(NAMES)
    assert [item["image_count"] for item in classes] == [2, 2, 0]
    assert classes[0]["mean_caption_words"]["mean"] == pytest.approx(7 / 3)
    assert classes[0]["mean_caption_words"]["mean"] != pytest.approx(12 / 5)
    assert classes[0]["pairwise_jaccard_mean"]["mean"] == pytest.approx(7 / 12)
    assert classes[0]["type_token_ratio"]["mean"] == 0.625
    assert classes[1]["mean_caption_words"]["count"] == 1
    assert classes[1]["mean_caption_words"]["mean"] == 0.0
    assert classes[2]["mean_caption_words"]["mean"] is None
    assert classes[2]["category_image_counts"] == {"color": 0, "anatomy": 0, "direction": 0}
    assert details["overall"]["mean_caption_words"]["mean"] == pytest.approx(14 / 9)
    assert details["overall"]["token_count"]["mean"] == 3.0


@pytest.mark.parametrize(
    "description,expected",
    [("red\nblue", 0.0), ("red bird\nblue bird", 1 / 3), ("red bird\nbird red", 1.0)],
)
def test_jaccard_has_hand_computable_word_set_values(description, expected):
    details = summarize_caption_details([record(0, description)], class_names=NAMES)
    assert details["images"][0]["pair_count"] == 1
    assert details["images"][0]["pairwise_jaccard_mean"] == pytest.approx(expected)
    assert details["overlap_examples"]["lowest"][0]["jaccard"] == pytest.approx(expected)


def test_pairs_are_within_image_only_and_zero_word_captions_cannot_form_pairs():
    details = summarize_caption_details(
        [record(0, "same bird"), record(1, "same bird"), record(2, "!!!\n???")],
        class_names=NAMES,
    )
    assert details["coverage"]["nonempty_caption_count"] == 4
    assert details["coverage"]["token_bearing_caption_count"] == 2
    assert details["coverage"]["pair_count"] == 0
    assert details["coverage"]["images_with_pairwise_overlap"] == 0
    assert all(row["pairwise_jaccard_mean"] is None for row in details["images"])
    assert details["overlap_examples"] == {"lowest": [], "highest": []}


def test_pair_count_is_choose_two_without_saving_every_pair():
    details = summarize_caption_details([record(0, "\n".join(["red bird"] * 5))], class_names=NAMES)
    assert details["coverage"]["pair_count"] == 10
    assert len(details["overlap_examples"]["lowest"]) == 5
    assert len(details["overlap_examples"]["highest"]) == 5
    assert [pair["caption_indices"] for pair in details["overlap_examples"]["lowest"]] == [
        [0, 1], [0, 2], [0, 3], [0, 4], [1, 2],
    ]


def test_examples_are_bounded_deterministic_source_linked_and_inputs_immutable():
    records = [record(index, "red bird\nblue bird") for index in range(8)]
    before = deepcopy(records)
    details = summarize_caption_details(records, class_names=NAMES)
    reversed_records = [(image, tuple(reversed(captions))) for image, captions in reversed(records)]

    assert details == summarize_caption_details(reversed_records, class_names=NAMES)
    assert records == before
    assert [item["image_id"] for item in term(details, "color", "red")["examples"]] == [SPEC.image_id(i) for i in range(5)]
    assert [item["image_id"] for item in details["overlap_examples"]["lowest"]] == [SPEC.image_id(i) for i in range(5)]
    assert len(details["overlap_examples"]["highest"]) == 5
    assert all(pair["caption_indices"] == [0, 1] for pair in details["overlap_examples"]["highest"])
    json.dumps(details, allow_nan=False)
    details["lexicon"]["categories"]["color"]["red"].append("invented")
    assert "invented" not in summarize_caption_details([], class_names=NAMES)["lexicon"]["categories"]["color"]["red"]


def test_empty_and_punctuation_only_populations_have_explicit_null_denominators():
    empty = summarize_caption_details([], class_names=NAMES)
    assert empty["coverage"]["image_count"] == 0
    assert empty["images"] == []
    assert empty["overall"]["mean_caption_words"]["count"] == 0
    assert empty["overall"]["type_token_ratio"]["mean"] is None
    assert category(empty, "color")["caption_prevalence"] is None
    assert category(empty, "color")["image_prevalence_all"] is None
    assert category(empty, "color")["image_prevalence_among_captioned"] is None
    punctuation = summarize_caption_details([record(0, "!!!")], class_names=NAMES)
    assert punctuation["coverage"]["images_with_nonempty_captions"] == 1
    assert punctuation["coverage"]["images_with_token_bearing_captions"] == 0
    assert punctuation["images"][0]["mean_caption_words"] == 0.0
    assert punctuation["images"][0]["type_token_ratio"] is None
    assert category(punctuation, "color")["caption_prevalence"] == 0.0
    json.dumps(empty, allow_nan=False)
    json.dumps(punctuation, allow_nan=False)


def test_duplicate_images_caption_indices_and_mislinked_captions_are_rejected():
    item = record(0, "red bird\nblue bird")
    with pytest.raises(ValueError, match="unique"):
        summarize_caption_details([item, item], class_names=NAMES)
    image, captions = item
    with pytest.raises(ValueError, match="caption_index.*unique"):
        summarize_caption_details([(image, (captions[0], replace(captions[1], caption_index=0)))], class_names=NAMES)
    with pytest.raises(ValueError, match="parent"):
        summarize_caption_details([(image, (replace(captions[0], image_id=SPEC.image_id(1)),))], class_names=NAMES)


@pytest.mark.parametrize(
    "updates",
    [
        {"image_id": "noncanonical"}, {"source_row": True}, {"source_row": -1},
        {"class_id": True}, {"class_id": 99}, {"class_name": "Wrong bird"},
        {"source_repo": ""}, {"source_revision": ""},
        {"source_id": True}, {"file_name": 4},
    ],
)
def test_invalid_image_identity_class_and_provenance_values_are_rejected(updates):
    image, captions = record(0, "red bird")
    with pytest.raises(ValueError):
        summarize_caption_details([(replace(image, **updates), captions)], class_names=NAMES)


def test_mixed_revisions_and_held_out_records_are_rejected():
    other_revision = SourceSpec(SPEC.repo_id, "train", "b" * 40)
    with pytest.raises(ValueError, match="one source"):
        summarize_caption_details([record(0, "red"), record(1, "blue", spec=other_revision)], class_names=NAMES)
    with pytest.raises(ValueError, match="training"):
        summarize_caption_details([record(0, "red", spec=SourceSpec(SPEC.repo_id, "test", SPEC.revision))], class_names=NAMES)


@pytest.mark.parametrize("updates", [{"caption_index": True}, {"caption_index": -1}, {"raw_caption": None}, {"clean_caption": 1}])
def test_invalid_caption_indices_and_text_types_are_rejected(updates):
    image, captions = record(0, "red bird")
    with pytest.raises(ValueError):
        summarize_caption_details([(image, (replace(captions[0], **updates),))], class_names=NAMES)


@pytest.mark.parametrize("names", ["Bird", ("Bird", "Bird"), ("",)])
def test_invalid_class_mapping_is_rejected(names):
    with pytest.raises(ValueError):
        summarize_caption_details([], class_names=names)


def test_shared_record_types_are_required_without_reinterpreting_source_data():
    image, _ = record(0, "red")
    for malformed in [(("not an ImageRecord", ()),), ((image, "red"),), ((image, ("not a CaptionRecord",)),), ((image,),)]:
        with pytest.raises(ValueError):
            summarize_caption_details(malformed, class_names=NAMES)
