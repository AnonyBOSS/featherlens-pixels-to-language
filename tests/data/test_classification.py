"""Lazy indexed samples, stable labels, and reproducible loader membership."""
from io import BytesIO
import importlib
import random
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from src.data.source import _SOURCES


class CountingRows:
    def __init__(self, rows): self.rows=rows; self.reads=[]
    def __len__(self): return len(self.rows)
    def __getitem__(self,index): self.reads.append(index); return self.rows[index]


def fixture_data(split='train'):
    image=Image.new('RGB',(40,50)); image.putdata([(x*5,y*4,(x+y)*2) for y in range(50) for x in range(40)])
    stream=BytesIO(); image.save(stream,format='PNG')
    spec=_SOURCES[split]; names=tuple(f'Bird {i}' for i in range(200))
    raw=[{'image':{'bytes':stream.getvalue()},'label':i%2,'id':i,'file_name':f'bird{i}.png'} for i in range(8)]
    rows=[dict(image_id=spec.image_id(i),source_repo=spec.repo_id,source_revision=spec.revision,source_split=split,source_row=i,source_id=i,file_name=f'bird{i}.png',class_id=i%2,class_name=names[i%2],eligible=True,partition='test' if split=='test' else 'val' if i>=6 else 'train') for i in range(8)]
    return SimpleNamespace(spec=spec,class_names=names,dataset=CountingRows(raw)),rows


def build(source,rows,partition='train'):
    module=importlib.import_module('src.data.classification')
    transform=importlib.import_module('src.data.transforms').classification_transform(training=partition=='train',image_size=16,normalization=None)
    return module.CubClassificationDataset(source,rows,partition=partition,transform=transform)


def test_constructor_is_lazy_and_sample_contract_is_exact():
    source,rows=fixture_data(); dataset=build(source,rows)
    assert source.dataset.reads==[] and len(dataset)==6
    sample=dataset[2]
    assert source.dataset.reads==[2] and set(sample)=={'image','label','image_id'}
    assert sample['label']==0 and sample['image_id']==rows[2]['image_id']
    assert sample['image'].shape==(3,16,16)


@pytest.mark.parametrize('mutation',['pin','id','row','label','class_name','duplicate','partition','streaming'])
def test_bad_manifest_source_contract_is_rejected(mutation):
    source,rows=fixture_data()
    if mutation=='pin': rows[0]['source_revision']='wrong'
    elif mutation=='id': rows[0]['image_id']='wrong'
    elif mutation=='row': rows[0]['source_row']=True
    elif mutation=='label': rows[0]['class_id']=True
    elif mutation=='class_name': rows[0]['class_name']='wrong'
    elif mutation=='duplicate': rows.append(dict(rows[0]))
    elif mutation=='partition':
        with pytest.raises(ValueError): build(source,rows,'oops')
        return
    elif mutation=='streaming': source.dataset=iter(source.dataset.rows)
    with pytest.raises(ValueError): build(source,rows)


def test_wrong_official_source_and_ineligible_test_fail_closed():
    source,rows=fixture_data('test')
    with pytest.raises(ValueError): build(source,rows,'train')
    rows[3]['eligible']=False
    with pytest.raises(ValueError,match='test|eligible'): build(source,rows,'test')


@pytest.mark.parametrize('failure',['image','label','id','file_name'])
def test_changed_source_failure_has_identity_context(failure):
    source,rows=fixture_data(); dataset=build(source,rows)
    raw=source.dataset.rows[0]
    raw[failure]={'bytes':b'corrupt'} if failure=='image' else 'wrong'
    with pytest.raises(RuntimeError) as caught: dataset[0]
    assert rows[0]['image_id'] in str(caught.value) and 'row 0' in str(caught.value)


def test_loader_collation_and_independent_sampler_seeds():
    source,rows=fixture_data(); datasets={'train':build(source,rows),'val':build(source,rows,'val')}
    module=importlib.import_module('src.data.classification')
    import torch
    random.seed(9); np.random.seed(9); torch.manual_seed(9)
    states=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    a=module.make_classification_loaders(datasets,batch_size=4,seed=42)
    assert random.getstate()==states[0] and np.array_equal(np.random.get_state()[1],states[1][1]) and torch.equal(torch.get_rng_state(),states[2])
    list(a['val'])
    ids_a=[identifier for batch in a['train'] for identifier in batch['image_id']]
    b=module.make_classification_loaders(datasets,batch_size=4,seed=42)
    ids_b=[identifier for batch in b['train'] for identifier in batch['image_id']]
    assert ids_a==ids_b and len(ids_a)==6 and len(set(ids_a))==6
    batch=next(iter(b['val']))
    assert batch['image'].dtype==torch.float32 and batch['label'].dtype==torch.int64
    assert batch['image'].shape==(2,3,16,16)
    assert batch['image_id']==[rows[i]['image_id'] for i in (6,7)]


def test_two_worker_batches_are_reproducible():
    source,rows=fixture_data(); datasets={'train':build(source,rows)}
    module=importlib.import_module('src.data.classification')
    import torch
    a=list(module.make_classification_loaders(datasets,batch_size=3,num_workers=2,seed=7)['train'])
    b=list(module.make_classification_loaders(datasets,batch_size=3,num_workers=2,seed=7)['train'])
    assert [batch['image_id'] for batch in a]==[batch['image_id'] for batch in b]
    assert all(torch.equal(x['image'],y['image']) for x,y in zip(a,b))
