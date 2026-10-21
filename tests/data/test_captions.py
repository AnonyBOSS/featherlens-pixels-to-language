import pytest

from src.data.captions import parse_captions


def test_caption_parser_preserves_line_positions_and_raw_text():
    records = parse_captions(
        "  A  red\tbird.  \r\n\r\nSecond caption\n",
        image_id="image-1",
    )

    assert [record.caption_index for record in records] == [0, 1, 2]
    assert [record.raw_caption for record in records] == ["  A  red\tbird.  ", "", "Second caption"]
    assert [record.clean_caption for record in records] == ["A red bird.", "", "Second caption"]
    assert {record.image_id for record in records} == {"image-1"}


def test_missing_description_yields_no_captions_without_affecting_image_record():
    assert parse_captions(None, image_id="image-2") == ()


def test_non_string_description_is_rejected():
    with pytest.raises(TypeError, match="description"):
        parse_captions(["not", "a", "string"], image_id="image-3")


def test_vertical_tab_stays_within_caption_line():
    records = parse_captions("Red\vbrown bird\nSecond", image_id="image-4")

    assert [record.raw_caption for record in records] == ["Red\vbrown bird", "Second"]
    assert [record.clean_caption for record in records] == ["Red brown bird", "Second"]
