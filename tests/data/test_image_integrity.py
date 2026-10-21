"""Focused tests for one-image integrity inspection."""

from hashlib import sha256
from io import BytesIO

import fsspec
from PIL import Image


def encoded(image: Image.Image, image_format: str, **save_kwargs) -> bytes:
    output = BytesIO()
    image.save(output, format=image_format, **save_kwargs)
    return output.getvalue()


def test_lossless_encodings_share_pixel_hash_and_bytes_take_precedence():
    from src.data.image_integrity import inspect_image

    image = Image.new("RGB", (2, 1))
    image.putdata([(255, 0, 0), (0, 0, 255)])
    png = encoded(image, "PNG")
    bmp = encoded(image, "BMP")

    first = inspect_image({"bytes": png, "path": "does-not-exist.png"})
    second = inspect_image({"bytes": bmp, "path": None})

    assert first.status == second.status == "ok"
    assert first.encoded_sha256 == sha256(png).hexdigest()
    assert second.encoded_sha256 == sha256(bmp).hexdigest()
    assert first.encoded_sha256 != second.encoded_sha256
    assert first.pixel_sha256 == second.pixel_sha256
    assert (first.width, first.height, first.mode, first.format) == (2, 1, "RGB", "PNG")
    assert (second.width, second.height, second.mode, second.format) == (2, 1, "RGB", "BMP")
    assert first.error_code is None and first.error_message is None


def test_corrupt_bytes_report_decode_failure_and_preserve_encoded_hash():
    from src.data.image_integrity import inspect_image

    bad_bytes = b"not an image"
    result = inspect_image({"bytes": bad_bytes, "path": None})

    assert result.status == "decode_error"
    assert result.encoded_sha256 == sha256(bad_bytes).hexdigest()
    assert result.pixel_sha256 is None
    assert result.error_code == "decode_failed"
    assert result.error_message


def test_missing_path_reports_fetch_failure_without_hash(tmp_path):
    from src.data.image_integrity import inspect_image

    result = inspect_image({"bytes": None, "path": str(tmp_path / "missing.png")})

    assert result.status == "fetch_error"
    assert result.encoded_sha256 is None
    assert result.pixel_sha256 is None
    assert result.error_code == "fetch_failed"


def test_local_and_fsspec_paths_are_read_without_retaining_image_bytes(tmp_path):
    from src.data.image_integrity import inspect_image

    payload = encoded(Image.new("RGB", (1, 1), "green"), "PNG")
    local_path = tmp_path / "bird.png"
    local_path.write_bytes(payload)
    with fsspec.open("memory://featherlens-stage2/bird.png", "wb") as target:
        target.write(payload)

    local = inspect_image({"bytes": None, "path": str(local_path)})
    remote = inspect_image({"bytes": None, "path": "memory://featherlens-stage2/bird.png"})

    assert local.status == remote.status == "ok"
    assert local.encoded_sha256 == remote.encoded_sha256 == sha256(payload).hexdigest()
    assert local.pixel_sha256 == remote.pixel_sha256


def test_exif_orientation_is_applied_before_pixel_hash():
    from src.data.image_integrity import inspect_image

    source = Image.new("RGB", (2, 1))
    source.putdata([(255, 0, 0), (0, 0, 255)])
    exif = Image.Exif()
    exif[274] = 6  # rotate 90 degrees clockwise for display
    oriented_payload = encoded(source, "PNG", exif=exif)
    visually_oriented = source.transpose(Image.Transpose.ROTATE_270)
    equivalent_payload = encoded(visually_oriented, "PNG")

    oriented = inspect_image({"bytes": oriented_payload, "path": None})
    equivalent = inspect_image({"bytes": equivalent_payload, "path": None})

    assert oriented.status == equivalent.status == "ok"
    assert (oriented.width, oriented.height) == (1, 2)
    assert oriented.pixel_sha256 == equivalent.pixel_sha256
    assert oriented.encoded_sha256 != equivalent.encoded_sha256


def test_transparent_pixels_are_composited_on_white_before_hashing():
    from src.data.image_integrity import inspect_image

    red_hidden = encoded(Image.new("RGBA", (1, 1), (255, 0, 0, 0)), "PNG")
    blue_hidden = encoded(Image.new("RGBA", (1, 1), (0, 0, 255, 0)), "PNG")
    white = encoded(Image.new("RGB", (1, 1), (255, 255, 255)), "PNG")

    red = inspect_image({"bytes": red_hidden, "path": None})
    blue = inspect_image({"bytes": blue_hidden, "path": None})
    opaque = inspect_image({"bytes": white, "path": None})

    assert red.status == blue.status == opaque.status == "ok"
    assert red.pixel_sha256 == blue.pixel_sha256 == opaque.pixel_sha256
    assert red.encoded_sha256 != blue.encoded_sha256


def test_missing_reference_and_invalid_bytes_type_are_fetch_errors():
    from src.data.image_integrity import inspect_image

    missing = inspect_image({"bytes": None, "path": None})
    malformed = inspect_image({"bytes": "raw text", "path": None})

    assert (missing.status, missing.error_code) == ("fetch_error", "missing_reference")
    assert (malformed.status, malformed.error_code) == ("fetch_error", "invalid_reference")


def test_pillow_decompression_bomb_limit_remains_enabled(monkeypatch):
    from src.data.image_integrity import inspect_image

    payload = encoded(Image.new("RGB", (4, 4), "white"), "PNG")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)

    result = inspect_image({"bytes": payload, "path": None})

    assert result.status == "decode_error"
    assert result.error_code == "decompression_bomb"
    assert result.encoded_sha256 == sha256(payload).hexdigest()
