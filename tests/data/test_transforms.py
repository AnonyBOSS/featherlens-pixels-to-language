"""Explicit classification geometry, normalization, and evaluation determinism."""
import importlib
import pytest
from PIL import Image


def factory(**kwargs):
    return importlib.import_module('src.data.transforms').classification_transform(**kwargs)


def test_default_tensor_and_eval_determinism():
    transform=factory(training=False)
    import torch
    image=Image.new('RGB',(300,400),(255,128,0))
    first=transform(image)
    assert first.shape==(3,224,224) and first.dtype==torch.float32
    assert torch.isfinite(first).all() and torch.equal(first,transform(image))
    expected=(torch.tensor([1.,128/255,0])-torch.tensor([.485,.456,.406]))/torch.tensor([.229,.224,.225])
    assert torch.allclose(first[:,112,112],expected,atol=1e-6)


def test_normalization_is_optional_or_custom():
    image=Image.new('RGB',(40,40),(255,128,0))
    plain=factory(training=False,image_size=16,normalization=None)(image)
    custom=factory(training=False,image_size=16,normalization=((.5,.5,.5),(.5,.25,1.)))(image)
    import torch
    assert plain.shape==(3,16,16) and plain.min()==0 and plain.max()==1
    assert torch.allclose(custom[:,0,0],(plain[:,0,0]-.5)/torch.tensor([.5,.25,1.]))


@pytest.mark.parametrize('kwargs',[
    {'image_size':True},{'image_size':0},{'image_size':2.5},{'training':1},
    {'normalization':((0,0,0),(1,0,1))}, {'normalization':((0,0),(1,1))},
    {'normalization':((0,0,float('nan')),(1,1,1))},
])
def test_invalid_transform_settings_fail(kwargs):
    options={'training':False}; options.update(kwargs)
    with pytest.raises(ValueError): factory(**options)


def test_training_geometry_is_seeded_and_varies():
    train=factory(training=True,image_size=32,normalization=None,geometry='crop')
    evaluate=factory(training=False,image_size=32,normalization=None)
    import torch
    image=Image.new('RGB',(80,50)); image.putdata([(i%256,(i//80)%256,0) for i in range(4000)])
    torch.manual_seed(42); a=train(image)
    torch.manual_seed(42); assert torch.equal(a,train(image))
    values=[train(image) for _ in range(6)]
    assert any(not torch.equal(a,value) for value in values)
    assert torch.equal(evaluate(image),evaluate(image))


def test_transform_output_remains_float32_with_nondefault_torch_dtype():
    transform=factory(training=False,image_size=16)
    import torch
    previous=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        assert transform(Image.new('RGB',(40,50))).dtype==torch.float32
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize('size,content_size', [((100,200),(112,224)),((200,100),(224,112)),((224,224),(224,224)),((1,1024),(1,224)),((1024,1),(224,1))])
def test_full_image_fit_preserves_all_edges_and_centered_white_padding(size, content_size):
    image=Image.new('RGB',size,(0,0,0))
    result=factory(training=False,normalization=None)(image)
    import torch
    nonwhite=(result!=1).any(dim=0)
    y,x=torch.where(nonwhite)
    width,height=content_size
    assert (int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1)==((224-width)//2,(224-height)//2,(224-width)//2+width,(224-height)//2+height)
    assert int(nonwhite.sum())==width*height


def test_fit_retains_opposite_edge_features_and_train_only_flip():
    image=Image.new('RGB',(400,100),(0,0,0))
    image.paste((255,0,0),(0,0,50,100)); image.paste((0,0,255),(350,0,400,100))
    evaluate=factory(training=False,image_size=32,normalization=None)
    train=factory(training=True,image_size=32,normalization=None)
    import torch
    expected=evaluate(image)
    assert expected[0,16,0]>.9 and expected[2,16,-1]>.9
    torch.manual_seed(42)
    values=[train(image) for _ in range(10)]
    assert all(torch.equal(value,expected) or torch.equal(value,expected.flip(-1)) for value in values)
    assert any(torch.equal(value,expected.flip(-1)) for value in values)
    assert torch.equal(expected,evaluate(image))


def test_geometry_is_explicit_and_rejects_unknown_profile():
    with pytest.raises(ValueError): factory(training=False,geometry='unknown')
    crop=factory(training=False,geometry='crop',normalization=None)
    assert crop(Image.new('RGB',(500,300))).shape==(3,224,224)
