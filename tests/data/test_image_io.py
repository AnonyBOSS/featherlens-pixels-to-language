from io import BytesIO
import importlib
import pytest
from PIL import Image
from hashlib import sha256
import fsspec
from types import SimpleNamespace
import numpy as np
from src.data.source import _SOURCES


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
    from src.data.image_io import inspect_image, _pixel_hash
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
    from src.data.image_io import inspect_image

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


def test_lossless_encodings_share_pixel_hash_and_bytes_take_precedence():
    from src.data.image_io import inspect_image

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
    from src.data.image_io import inspect_image

    bad_bytes = b"not an image"
    result = inspect_image({"bytes": bad_bytes, "path": None})

    assert result.status == "decode_error"
    assert result.encoded_sha256 == sha256(bad_bytes).hexdigest()
    assert result.pixel_sha256 is None
    assert result.error_code == "decode_failed"
    assert result.error_message


def test_missing_path_reports_fetch_failure_without_hash(tmp_path):
    from src.data.image_io import inspect_image

    result = inspect_image({"bytes": None, "path": str(tmp_path / "missing.png")})

    assert result.status == "fetch_error"
    assert result.encoded_sha256 is None
    assert result.pixel_sha256 is None
    assert result.error_code == "fetch_failed"


def test_local_and_fsspec_paths_are_read_without_retaining_image_bytes(tmp_path):
    from src.data.image_io import inspect_image

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
    from src.data.image_io import inspect_image

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
    from src.data.image_io import inspect_image

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
    from src.data.image_io import inspect_image

    missing = inspect_image({"bytes": None, "path": None})
    malformed = inspect_image({"bytes": "raw text", "path": None})

    assert (missing.status, missing.error_code) == ("fetch_error", "missing_reference")
    assert (malformed.status, malformed.error_code) == ("fetch_error", "invalid_reference")


def test_pillow_decompression_bomb_limit_remains_enabled(monkeypatch):
    from src.data.image_io import inspect_image

    payload = encoded(Image.new("RGB", (4, 4), "white"), "PNG")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)

    result = inspect_image({"bytes": payload, "path": None})

    assert result.status == "decode_error"
    assert result.error_code == "decompression_bomb"
    assert result.encoded_sha256 == sha256(payload).hexdigest()


def payload(image, **kwargs):
    stream=BytesIO(); image.save(stream,format='PNG',**kwargs); return stream.getvalue()


def source(row=None):
    row=row if row is not None else {'id':1,'file_name':'bird.png','label':3,'image':{'bytes':payload(Image.new('RGB',(40,20),(10,20,30)))}}
    return SimpleNamespace(spec=_SOURCES['train'],class_names=tuple(f'Bird {i}' for i in range(200)),dataset=[row])


def read(value,index=0):
    return importlib.import_module('src.data.image_io').read_source_image(value,index)


def test_pixels_are_linked_to_original_identity_and_remain_native():
    data=source(); view=read(data)
    assert view.image_id==data.spec.image_id(0) and view.source_row==0
    assert (view.source_repo,view.source_revision,view.source_split)==(data.spec.repo_id,data.spec.revision,'train')
    assert (view.class_id,view.class_name,view.source_id,view.file_name)==(3,'Bird 3',1,'bird.png')
    assert view.status=='ok' and view.error_code is None and view.metadata_errors==()
    assert view.original_size==view.oriented_size==(40,20) and view.original_mode=='RGB'
    assert view.image_format=='PNG' and view.has_transparency is False and view.icc_profile_present is False
    with view.image as image:
        pixels=np.asarray(image)
        assert image.mode=='RGB' and image.size==(40,20)
        assert pixels.dtype==np.uint8 and pixels.shape==(20,40,3)
        assert tuple(pixels[0,0])==(10,20,30)


def test_exif_dimensions_and_original_facts_survive_conversion():
    exif=Image.Exif(); exif[274]=6
    image=Image.new('RGBA',(6,4),(255,0,0,0))
    data=source(); data.dataset[0]['image']={'bytes':payload(image,exif=exif,icc_profile=b'test-profile-presence')}
    view=read(data)
    assert view.status=='ok' and view.original_size==(6,4) and view.oriented_size==(4,6)
    assert view.exif_orientation==6 and view.original_mode=='RGBA'
    assert view.has_transparency and view.icc_profile_present
    with view.image as rgb:
        assert rgb.getpixel((0,0))==(255,255,255)


@pytest.mark.parametrize('mode',['L','P'])
def test_opaque_mode_conversion_keeps_original_mode(mode):
    data=source(); image=Image.new(mode,(7,5))
    data.dataset[0]['image']={'bytes':payload(image)}
    view=read(data)
    assert view.status=='ok' and view.original_mode==mode and not view.has_transparency
    with view.image as rgb: assert rgb.mode=='RGB' and rgb.size==(7,5)


@pytest.mark.parametrize('change,reason', [({'label':True},'metadata.invalid_label'),({'label':200},'metadata.invalid_label'),({'id':False},'metadata.invalid_source_id'),({'file_name':''},'metadata.invalid_filename')])
def test_bad_metadata_does_not_erase_decodable_pixels(change,reason):
    data=source(); data.dataset[0].update(change)
    view=read(data)
    assert view.status=='ok' and reason in view.metadata_errors
    assert view.image is not None
    view.image.close()


@pytest.mark.parametrize('reference,status,code',[({},'fetch_error','missing_reference'),({'path':'missing-image-file.png'},'fetch_error','fetch_failed'),({'bytes':b'broken'},'decode_error','decode_failed')])
def test_failures_keep_stable_identity(reference,status,code):
    data=source(); data.dataset[0]['image']=reference
    view=read(data)
    assert view.image_id==data.spec.image_id(0) and view.source_row==0 and view.class_id==3
    assert view.status==status and view.error_code==code and view.error_message
    assert view.image is None


def test_row_failure_is_recoverable_with_stable_id():
    class BrokenRows:
        def __len__(self): return 1
        def __getitem__(self,index): raise OSError('row unavailable')
    data=source(); data.dataset=BrokenRows()
    view=read(data)
    assert view.status=='row_error' and view.error_code=='source.row_read_failed'
    assert view.image_id==data.spec.image_id(0) and view.image is None


def test_nonmapping_row_is_reported():
    data=source('bad row')
    view=read(data)
    assert view.status=='row_error' and view.error_code=='metadata.invalid_row'


@pytest.mark.parametrize('index',[True,-1,1,1.5])
def test_invalid_requested_rows_raise_instead_of_inventing_identity(index):
    with pytest.raises(ValueError): read(source(),index)


def test_streaming_source_is_rejected():
    data=source(); data.dataset=iter(data.dataset)
    with pytest.raises(ValueError,match='indexed|streaming'): read(data)


def test_legacy_rgb_loader_returns_identical_display_pixels():
    module=importlib.import_module('src.data.image_io'); data=source()
    view=read(data)
    with module.load_rgb(data.dataset[0]['image']) as rgb, view.image as native:
        assert rgb.size==native.size and rgb.tobytes()==native.tobytes()


def test_truncated_image_keeps_readable_header_facts():
    stream=BytesIO(); Image.new('RGB',(32,16),(10,20,30)).save(stream,format='BMP')
    data=source(); data.dataset[0]['image']={'bytes':stream.getvalue()[:-100]}
    view=read(data)
    assert view.status=='decode_error' and view.error_code=='decode_failed'
    assert view.original_size==(32,16) and view.original_mode=='RGB' and view.image_format=='BMP'
    assert view.oriented_size is None and view.image is None and view.metadata_errors==()
