"""Pure per-image measurements on caller-supplied native-resolution RGB arrays.

All pixels are measured without resizing. The caller owns decoding, orientation,
alpha handling, and color conversion; this module does not infer those policies.
RGB statistics use encoded channel values in [0, 1], with population standard
deviation (ddof=0). Brightness is the mean of Y = .299 R + .587 G + .114 B on
encoded RGB, a luma proxy rather than linear-light physical luminance. Contrast is
the population standard deviation of that same Y. Saturation is the mean HSV S,
(max(R,G,B) - min(R,G,B)) / max(R,G,B), defined as zero for black pixels.

Each result describes one image. Equal-image weighting of later aggregates is a
caller policy and does not produce a pooled, globally pixel-weighted estimate.
No quality thresholds, validity decisions, normalization parameters, or source
image transformations are fitted or applied here. Inputs are never modified.
"""

from numbers import Integral

import numpy as np


def measure_resolution(width: int, height: int) -> dict:
    """Return JSON-safe native dimensions; aspect ratio is width / height."""
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, Integral)
        or not isinstance(height, Integral)
    ):
        raise TypeError("width and height must be integers")
    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    width = int(width)
    height = int(height)
    pixel_count = width * height
    return {
        "width": width,
        "height": height,
        "pixel_count": pixel_count,
        "megapixels": pixel_count / 1_000_000,
        "aspect_ratio": width / height,
    }


def measure_rgb_array(rgb: np.ndarray) -> dict:
    """Measure a nonempty H x W x 3 uint8 or floating [0, 1] RGB array.

    uint8 is interpreted as encoded RGB in [0, 255] and divided by 255.
    Floating values must already be finite and in [0, 1]. Other integer bit
    depths and nonnumeric dtypes are rejected rather than implicitly rescaled.
    Returned channel arrays are ordinary lists, and all scalar values are Python
    ints/floats suitable for strict JSON serialization.
    """
    if not isinstance(rgb, np.ndarray):
        raise TypeError("rgb must be a NumPy array")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] < 1 or rgb.shape[1] < 1:
        raise ValueError("rgb must have nonempty shape H x W x 3")
    if rgb.dtype == np.uint8:
        values = rgb.astype(np.float64) / 255.0
    elif np.issubdtype(rgb.dtype, np.floating):
        values = rgb.astype(np.float64, copy=False)
    else:
        raise TypeError("rgb dtype must be uint8 or floating point")
    if not np.isfinite(values).all():
        raise ValueError("rgb values must be finite")
    if (values < 0).any() or (values > 1).any():
        raise ValueError("floating rgb values must be in [0, 1]")

    height, width = values.shape[:2]
    luma = 0.299 * values[..., 0] + 0.587 * values[..., 1] + 0.114 * values[..., 2]
    maximum = values.max(axis=2)
    minimum = values.min(axis=2)
    saturation = np.divide(
        maximum - minimum,
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 0,
    )
    return {
        **measure_resolution(width, height),
        "rgb_mean": values.mean(axis=(0, 1)).tolist(),
        "rgb_std": values.std(axis=(0, 1), ddof=0).tolist(),
        "brightness_mean": float(luma.mean()),
        "contrast_std": float(luma.std(ddof=0)),
        "saturation_mean": float(saturation.mean()),
    }
