"""Fetch, fully decode, and fingerprint one source image without changing it.

Pixel hash v1 is SHA-256 over a fixed domain tag, two unsigned 64-bit
big-endian dimensions, and row-major RGB8 pixels after EXIF orientation.
Images with transparency are composited over opaque white before hashing.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from os import PathLike, fspath
from struct import pack
from tempfile import SpooledTemporaryFile
from typing import Any, Literal

import fsspec
from PIL import Image, ImageOps, UnidentifiedImageError


_CHUNK_BYTES = 1024 * 1024
_SPOOL_MEMORY_BYTES = 8 * 1024 * 1024
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


def _normalize_rgb(image: Image.Image) -> Image.Image:
    """Return RGB8 display pixels, filling transparent areas with white."""
    if "A" not in image.getbands() and "transparency" not in image.info:
        return image.convert("RGB")
    with image.convert("RGBA") as rgba:
        with rgba.getchannel("A") as alpha:
            rgb = Image.new("RGB", rgba.size, (255, 255, 255))
            rgb.paste(rgba, mask=alpha)
            return rgb


def inspect_image(reference: Mapping[str, Any] | None) -> ImageInspection:
    """Inspect a Hugging Face ``Image(decode=False)`` bytes/path reference.

    Embedded bytes take precedence over a path. Path reads use fsspec, which
    supports local paths and ``hf://`` references, and spool encoded data in
    bounded chunks. Pillow performs a full ``load()``; its decompression-bomb
    limit remains enabled. No source image is modified.
    """
    if not isinstance(reference, Mapping):
        return _fetch_error("invalid_reference", "Image reference must be a bytes/path mapping")
    content = reference.get("bytes")
    path = reference.get("path")
    if content is not None and not isinstance(content, (bytes, bytearray, memoryview)):
        return _fetch_error("invalid_reference", "Image bytes must be bytes-like")
    if content is None:
        if path is None or path == "":
            return _fetch_error("missing_reference", "Image reference contains neither bytes nor a path")
        if not isinstance(path, (str, PathLike)):
            return _fetch_error("invalid_reference", "Image path must be text or a path-like object")

    digest = sha256()
    with SpooledTemporaryFile(max_size=_SPOOL_MEMORY_BYTES, mode="w+b") as spool:
        try:
            if content is not None:
                digest.update(content)
                spool.write(content)
            else:
                with fsspec.open(fspath(path), "rb") as source:
                    while chunk := source.read(_CHUNK_BYTES):
                        digest.update(chunk)
                        spool.write(chunk)
        except Exception as error:
            return _fetch_error("fetch_failed", f"{type(error).__name__}: {error}")

        encoded_sha256 = digest.hexdigest()
        spool.seek(0)
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
