"""Native RGB access for source-level EDA; no Torch or audit dependency."""
from io import BytesIO
from types import SimpleNamespace
import importlib

import numpy as np
import pytest
from PIL import Image
from src.data.source import _SOURCES


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
