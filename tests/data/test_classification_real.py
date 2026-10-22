"""Opt-in cached-source integration; normal synthetic tests never download CUB."""
import os
from pathlib import Path

import pytest


@pytest.mark.skipif(not os.environ.get('FEATHERLENS_PREPARED_DIR'), reason='Set FEATHERLENS_PREPARED_DIR to opt into cached CUB verification')
def test_prepared_cub_train_validation_and_loss():
    import torch
    from src.data.preparation import load_classification_manifest
    from src.data.source import load_cub_split
    from src.data.classification import CubClassificationDataset, make_classification_loaders
    from src.data.transforms import classification_transform
    rows=load_classification_manifest(Path(os.environ['FEATHERLENS_PREPARED_DIR']))
    source=load_cub_split('train')
    datasets={partition:CubClassificationDataset(source,rows,partition=partition,transform=classification_transform(training=partition=='train')) for partition in ('train','val')}
    assert sum(len(dataset) for dataset in datasets.values())==sum(row['source_split']=='train' and row['eligible'] for row in rows)
    loaders=make_classification_loaders(datasets,batch_size=4)
    torch.manual_seed(42)
    batch=next(iter(loaders['train']))
    assert batch['image'].shape==(4,3,224,224) and batch['label'].dtype==torch.int64
    model=torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),torch.nn.Linear(3,200))
    loss=torch.nn.functional.cross_entropy(model(batch['image']),batch['label'])
    loss.backward()
    assert torch.isfinite(loss) and model[-1].weight.grad is not None
    first=next(iter(loaders['val']))
    repeat=next(iter(loaders['val']))
    assert first['image_id']==repeat['image_id'] and torch.equal(first['image'],repeat['image'])
