import json

import numpy as np
import pytest

from src.analysis.image_stats import measure_resolution, measure_rgb_array


def test_resolution_reports_nonsquare_native_dimensions_and_pixel_units():
    assert measure_resolution(4, 2) == {
        "width": 4,
        "height": 2,
        "pixel_count": 8,
        "megapixels": 0.000008,
        "aspect_ratio": 2.0,
    }
    json.dumps(measure_resolution(np.int64(4), np.int64(2)), allow_nan=False)


@pytest.mark.parametrize(
    "color,brightness,saturation",
    [
        ([0, 0, 0], 0.0, 0.0),
        ([255, 255, 255], 1.0, 0.0),
        ([255, 0, 0], 0.299, 1.0),
        ([0, 255, 0], 0.587, 1.0),
        ([0, 0, 255], 0.114, 1.0),
    ],
)
def test_constant_black_white_and_primary_colors_have_known_statistics(
    color, brightness, saturation
):
    rgb = np.array([[color]], dtype=np.uint8)

    result = measure_rgb_array(rgb)

    assert result["rgb_mean"] == pytest.approx([channel / 255 for channel in color])
    assert result["rgb_std"] == [0.0, 0.0, 0.0]
    assert result["brightness_mean"] == pytest.approx(brightness)
    assert result["contrast_std"] == 0.0
    assert result["saturation_mean"] == pytest.approx(saturation)
    assert result["pixel_count"] == 1
    json.dumps(result, allow_nan=False)


def test_two_tone_grayscale_uses_population_contrast_and_has_no_saturation():
    rgb = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)

    result = measure_rgb_array(rgb)

    assert result["rgb_mean"] == pytest.approx([0.5, 0.5, 0.5])
    assert result["rgb_std"] == pytest.approx([0.5, 0.5, 0.5])
    assert result["brightness_mean"] == pytest.approx(0.5)
    assert result["contrast_std"] == pytest.approx(0.5)
    assert result["saturation_mean"] == 0.0


def test_nonsquare_red_blue_image_measures_every_pixel_with_explicit_luma_weights():
    red = [255, 0, 0]
    blue = [0, 0, 255]
    rgb = np.array([[red, red, blue], [blue, red, blue]], dtype=np.uint8)

    result = measure_rgb_array(rgb)

    assert result["width"] == 3
    assert result["height"] == 2
    assert result["pixel_count"] == 6
    assert result["megapixels"] == pytest.approx(0.000006)
    assert result["aspect_ratio"] == 1.5
    assert result["rgb_mean"] == pytest.approx([0.5, 0.0, 0.5])
    assert result["rgb_std"] == pytest.approx([0.5, 0.0, 0.5])
    assert result["brightness_mean"] == pytest.approx((0.299 + 0.114) / 2)
    assert result["contrast_std"] == pytest.approx((0.299 - 0.114) / 2)
    assert result["saturation_mean"] == 1.0


def test_uint8_and_normalized_float_views_agree_without_mutating_inputs():
    encoded = np.array(
        [[[0, 120, 255], [20, 30, 40]], [[255, 255, 255], [0, 0, 0]]],
        dtype=np.uint8,
    )
    normalized = encoded.astype(np.float64) / 255
    encoded_before = encoded.copy()
    normalized_before = normalized.copy()

    encoded_result = measure_rgb_array(encoded)
    normalized_result = measure_rgb_array(normalized)

    for key, value in encoded_result.items():
        assert value == pytest.approx(normalized_result[key])
    np.testing.assert_array_equal(encoded, encoded_before)
    np.testing.assert_array_equal(normalized, normalized_before)
    json.dumps(normalized_result, allow_nan=False)


@pytest.mark.parametrize(
    "rgb",
    [
        np.array([[[np.nan, 0, 0]]]),
        np.array([[[np.inf, 0, 0]]]),
        np.array([[[-0.01, 0, 0]]]),
        np.array([[[1.01, 0, 0]]]),
    ],
)
def test_nonfinite_and_out_of_range_float_values_are_rejected(rgb):
    with pytest.raises(ValueError, match="finite|\\[0, 1\\]"):
        measure_rgb_array(rgb)


@pytest.mark.parametrize(
    "rgb",
    [
        np.zeros((2, 2)),
        np.zeros((2, 2, 4)),
        np.zeros((1, 2, 2, 3)),
        np.empty((0, 2, 3)),
        np.empty((2, 0, 3)),
    ],
)
def test_wrong_shape_and_empty_images_are_rejected(rgb):
    with pytest.raises(ValueError, match="shape"):
        measure_rgb_array(rgb)


@pytest.mark.parametrize("dtype", [np.uint16, np.int32, np.bool_, np.complex128, object])
def test_ambiguous_integer_depths_and_unsupported_dtypes_are_rejected(dtype):
    with pytest.raises(TypeError, match="dtype"):
        measure_rgb_array(np.zeros((1, 1, 3), dtype=dtype))


@pytest.mark.parametrize("width,height", [(0, 1), (1, 0), (-1, 2)])
def test_nonpositive_dimensions_are_rejected(width, height):
    with pytest.raises(ValueError, match="positive"):
        measure_resolution(width, height)


@pytest.mark.parametrize("width,height", [(True, 1), (1, False), (1.5, 2), (2, "3")])
def test_noninteger_dimensions_are_rejected(width, height):
    with pytest.raises(TypeError, match="integers"):
        measure_resolution(width, height)
