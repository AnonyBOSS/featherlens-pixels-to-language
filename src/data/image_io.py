"""Bounded source access and the shared EXIF/RGB8/white-alpha pixel policy."""
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from os import PathLike, fspath
from tempfile import SpooledTemporaryFile
from typing import Any

import fsspec
from PIL import Image, ImageOps, UnidentifiedImageError


class ImageFetchError(ValueError):
    """A source reference could not be fetched, before image decoding."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@contextmanager
def _encoded_image(reference: Mapping[str, Any] | None, *, digest=None):
    if not isinstance(reference, Mapping):
        raise ImageFetchError('invalid_reference', 'Image reference must be a bytes/path mapping')
    content, path = reference.get('bytes'), reference.get('path')
    if content is not None and not isinstance(content, (bytes, bytearray, memoryview)):
        raise ImageFetchError('invalid_reference', 'Image bytes must be bytes-like')
    if content is None:
        if path is None or path == '':
            raise ImageFetchError('missing_reference', 'Image reference contains neither bytes nor a path')
        if not isinstance(path, (str, PathLike)):
            raise ImageFetchError('invalid_reference', 'Image path must be text or a path-like object')
    with SpooledTemporaryFile(max_size=8 * 1024 * 1024, mode='w+b') as spool:
        try:
            if content is not None:
                if digest is not None:
                    digest.update(content)
                spool.write(content)
            else:
                with fsspec.open(fspath(path), 'rb') as source:
                    while chunk := source.read(1024 * 1024):
                        if digest is not None:
                            digest.update(chunk)
                        spool.write(chunk)
        except Exception as error:
            raise ImageFetchError('fetch_failed', f'{type(error).__name__}: {error}') from error
        spool.seek(0)
        yield spool


def _normalize_rgb(image: Image.Image) -> Image.Image:
    """Return independent RGB8 display pixels, filling transparency with white."""
    if 'A' not in image.getbands() and 'transparency' not in image.info:
        return image.convert('RGB')
    with image.convert('RGBA') as rgba:
        with rgba.getchannel('A') as alpha:
            rgb = Image.new('RGB', rgba.size, (255, 255, 255))
            rgb.paste(rgba, mask=alpha)
            return rgb


def load_rgb(reference: Mapping[str, Any]) -> Image.Image:
    """Fully decode one HF Image(decode=False) reference; caller closes result.

    Embedded bytes precede paths. EXIF orientation is applied, transparency is
    composited on white, and pixels become RGB8 without resize or ICC conversion.
    Source assets are unchanged. Fetch and decoder exceptions remain explicit;
    no hashing or decoded-image cache is performed during training access.
    """
    return _decode_rgb(reference)


@dataclass(frozen=True)
class SourceImageView:
    """One unfiltered source row and its native display pixels, if decodable.

    Caller closes image when present. metadata_errors are independent of the
    image status; this view neither cleans rows nor decides task eligibility.
    Sizes are (width, height). Original facts survive RGB/EXIF conversion.
    """

    image_id: str
    source_repo: str
    source_revision: str
    source_split: str
    source_row: int
    status: str
    source_id: int | None = None
    file_name: str | None = None
    class_id: int | None = None
    class_name: str | None = None
    image: Image.Image | None = None
    original_size: tuple[int, int] | None = None
    oriented_size: tuple[int, int] | None = None
    original_mode: str | None = None
    image_format: str | None = None
    exif_orientation: int | None = None
    has_transparency: bool | None = None
    icc_profile_present: bool | None = None
    metadata_errors: tuple[str, ...] = ()
    error_code: str | None = None
    error_message: str | None = None


def _decode_rgb(reference, facts=None) -> Image.Image:
    with _encoded_image(reference) as encoded:
        with Image.open(encoded) as image:
            if facts is not None:
                facts.update(
                    original_size=image.size, original_mode=image.mode,
                    image_format=image.format,
                    has_transparency=bool({'A', 'a'} & set(image.getbands())) or 'transparency' in image.info,
                    icc_profile_present='icc_profile' in image.info,
                )
                facts['exif_orientation'] = image.getexif().get(274)
            image.load()
            with ImageOps.exif_transpose(image) as oriented:
                if facts is not None:
                    facts['oriented_size'] = oriented.size
                return _normalize_rgb(oriented)


def read_source_image(source: Any, source_row: int) -> SourceImageView:
    """Read one raw indexed CubSource row, returning recoverable row failures.

    Use load_cub_split('train') for pinned training access. No transforms,
    resizing, hashes, filtering, audit dependency, or Torch imports occur.
    Invalid requested indices/streaming inputs raise ValueError; actual row,
    fetch, and decode failures return a stable image_id and explicit reason.
    """
    try:
        size = len(source.dataset)
        if not callable(getattr(source.dataset, '__getitem__', None)):
            raise TypeError
    except (TypeError, AttributeError) as error:
        raise ValueError('Image access requires an indexed, nonstreaming source') from error
    if isinstance(source_row, bool) or not isinstance(source_row, int) or not 0 <= source_row < size:
        raise ValueError('source_row must be an integer within the indexed source')
    spec = source.spec
    identity = dict(
        image_id=spec.image_id(source_row), source_repo=spec.repo_id,
        source_revision=spec.revision, source_split=spec.source_split,
        source_row=source_row,
    )
    try:
        row = source.dataset[source_row]
    except Exception as error:
        return SourceImageView(**identity, status='row_error', error_code='source.row_read_failed', error_message=f'{type(error).__name__}: {error}')
    if not isinstance(row, Mapping):
        return SourceImageView(**identity, status='row_error', error_code='metadata.invalid_row', metadata_errors=('metadata.invalid_row',), error_message='Source row must be a mapping')
    errors = []
    for source_key, field, invalid_code, missing_code in (
        ('id', 'source_id', 'metadata.invalid_source_id', 'metadata.missing_id'),
        ('file_name', 'file_name', 'metadata.invalid_filename', 'metadata.missing_file_name'),
        ('label', 'class_id', 'metadata.invalid_label', 'metadata.missing_label'),
    ):
        value = row.get(source_key)
        if source_key == 'file_name':
            valid = isinstance(value, str) and bool(value.strip())
        else:
            valid = isinstance(value, int) and not isinstance(value, bool)
            if source_key == 'label':
                valid = valid and 0 <= value < len(source.class_names)
        identity[field] = value if valid else None
        if not valid:
            errors.append(missing_code if source_key not in row else invalid_code)
    identity['class_name'] = source.class_names[identity['class_id']] if identity['class_id'] is not None else None
    identity['metadata_errors'] = tuple(errors)
    facts = {}
    try:
        image = _decode_rgb(row.get('image', {}), facts)
    except ImageFetchError as error:
        return SourceImageView(**identity, **facts, status='fetch_error', error_code=error.code, error_message=str(error))
    except Image.DecompressionBombError as error:
        return SourceImageView(**identity, **facts, status='decode_error', error_code='decompression_bomb', error_message=f'{type(error).__name__}: {error}')
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, EOFError) as error:
        return SourceImageView(**identity, **facts, status='decode_error', error_code='decode_failed', error_message=f'{type(error).__name__}: {error}')
    return SourceImageView(**identity, **facts, status='ok', image=image)
