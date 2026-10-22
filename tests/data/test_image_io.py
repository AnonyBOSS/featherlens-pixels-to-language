"""Shared decoding must preserve the auditor's display-pixel policy."""
from io import BytesIO
import importlib
import pytest
from PIL import Image


def encoded(image, format='PNG', **kwargs):
    out=BytesIO(); image.save(out, format=format, **kwargs); return out.getvalue()


def load(reference):
    return importlib.import_module('src.data.image_io').load_rgb(reference)


def test_bytes_precede_path_and_return_owned_pixels(tmp_path):
    data=encoded(Image.new('RGB',(9,7),(1,2,3)))
    result=load({'bytes':data,'path':str(tmp_path/'missing')})
    assert result.mode=='RGB' and result.size==(9,7)
    assert result.getpixel((0,0))==(1,2,3)
    result.close()


def test_local_path_matches_bytes(tmp_path):
    data=encoded(Image.new('L',(5,3),71)); path=tmp_path/'gray.png'; path.write_bytes(data)
    with load({'path':path}) as a, load({'bytes':data}) as b:
        assert a.tobytes()==b.tobytes() and a.getpixel((0,0))==(71,71,71)


@pytest.mark.parametrize('reference', [None, {}, {'bytes':'wrong'}, {'path':12}, {'bytes':b'corrupted'}])
def test_bad_references_and_corruption_fail(reference):
    with pytest.raises((ValueError,OSError)):
        load(reference)


def test_missing_path_fails(tmp_path):
    with pytest.raises((ValueError,OSError)):
        load({'path':str(tmp_path/'missing')})


def test_orientation_and_transparency_match_audit():
    from src.data.image_integrity import inspect_image, _pixel_hash
    image=Image.new('RGB',(6,4)); image.putpixel((0,0),(255,0,0))
    exif=Image.Exif(); exif[274]=6
    reference={'bytes':encoded(image,exif=exif)}
    with load(reference) as rgb:
        assert rgb.size==(4,6)
        assert _pixel_hash(rgb)==inspect_image(reference).pixel_sha256
    for mode in ('RGBA','LA','P'):
        image=Image.new(mode,(3,2))
        if mode=='P': image.info['transparency']=0
        reference={'bytes':encoded(image)}
        with load(reference) as rgb:
            assert rgb.mode=='RGB' and rgb.getpixel((0,0))==(255,255,255)
            assert _pixel_hash(rgb)==inspect_image(reference).pixel_sha256


def test_truncated_pixels_fail_even_when_image_header_is_readable():
    from hashlib import sha256
    from src.data.image_integrity import inspect_image

    payload = encoded(Image.new('RGB', (32, 16), (10, 20, 30)), format='BMP')[:-100]
    # Header inspection alone would miss the missing pixel data.
    with Image.open(BytesIO(payload)) as header:
        assert header.size == (32, 16)
    with pytest.raises(OSError):
        load({'bytes': payload})
    inspection = inspect_image({'bytes': payload})
    assert inspection.status == 'decode_error'
    assert inspection.encoded_sha256 == sha256(payload).hexdigest()
    assert inspection.pixel_sha256 is None
