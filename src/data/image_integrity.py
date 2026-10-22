"""Fetch, fully decode, and fingerprint one source image without changing it.

Pixel hash v1 is SHA-256 over a fixed domain tag, two unsigned 64-bit
big-endian dimensions, and row-major RGB8 pixels after EXIF orientation.
Images with transparency are composited over opaque white before hashing.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from struct import pack
from typing import Any, Literal

from PIL import Image, ImageOps, UnidentifiedImageError

from .image_io import ImageFetchError, _encoded_image, _normalize_rgb

_PIXEL_STRIP_ROWS = 64
_PIXEL_HASH_DOMAIN = b"featherlens:pixel:exif-rgb8-white:v1\0"


@dataclass(frozen=True)
class ImageInspection:
    """A single image check; ``mode``/``format`` describe the encoded image.

    Width and height describe the orientation-corrected pixels. Failed fields
    remain ``None``; an encoded hash can still be present after decode failure.
    """

    status: Literal["ok", "fetch_error", "decode_error"]
    encoded_sha256: str | None = None
    pixel_sha256: str | None = None
    width: int | None = None
    height: int | None = None
    mode: str | None = None
    format: str | None = None
    error_code: str | None = None
    error_message: str | None = None


def _fetch_error(code: str, message: str) -> ImageInspection:
    return ImageInspection(status="fetch_error", error_code=code, error_message=message)


def _pixel_hash(rgb: Image.Image) -> str:
    digest = sha256()
    digest.update(_PIXEL_HASH_DOMAIN)
    digest.update(pack(">QQ", rgb.width, rgb.height))
    for top in range(0, rgb.height, _PIXEL_STRIP_ROWS):
        bottom = min(top + _PIXEL_STRIP_ROWS, rgb.height)
        with rgb.crop((0, top, rgb.width, bottom)) as strip:
            digest.update(strip.tobytes())
    return digest.hexdigest()


def inspect_image(reference: Mapping[str, Any] | None) -> ImageInspection:
    """Inspect a Hugging Face ``Image(decode=False)`` bytes/path reference.

    Embedded bytes take precedence over a path. Path reads use fsspec, which
    supports local paths and ``hf://`` references, and spool encoded data in
    bounded chunks. Pillow performs a full ``load()``; its decompression-bomb
    limit remains enabled. No source image is modified.
    """
    digest = sha256()
    try:
        with _encoded_image(reference, digest=digest) as spool:
            return _inspect_encoded(spool, digest.hexdigest())
    except ImageFetchError as error:
        return _fetch_error(error.code, str(error))


def _inspect_encoded(spool, encoded_sha256: str) -> ImageInspection:
    try:
        with Image.open(spool) as source_image:
            source_image.load()
            source_mode = source_image.mode
            source_format = source_image.format
            with ImageOps.exif_transpose(source_image) as oriented:
                with _normalize_rgb(oriented) as rgb:
                    pixel_sha256 = _pixel_hash(rgb)
                    width, height = rgb.size
    except Image.DecompressionBombError as error:
        return ImageInspection(
            status="decode_error", encoded_sha256=encoded_sha256,
            error_code="decompression_bomb", error_message=f"{type(error).__name__}: {error}",
        )
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, EOFError) as error:
        return ImageInspection(
            status="decode_error", encoded_sha256=encoded_sha256,
            error_code="decode_failed", error_message=f"{type(error).__name__}: {error}",
        )

    return ImageInspection(
        status="ok",
        encoded_sha256=encoded_sha256,
        pixel_sha256=pixel_sha256,
        width=width,
        height=height,
        mode=source_mode,
        format=source_format,
    )
