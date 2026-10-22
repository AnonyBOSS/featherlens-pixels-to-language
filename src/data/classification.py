"""Lazy classification samples and deliberately small DataLoader interfaces."""
from collections.abc import Callable, Mapping, Sequence
import random
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from .image_io import load_rgb
from .source import CubSource, _SOURCES


class CubClassificationDataset(Dataset):
    """Indexed access to one prepared partition of a pinned CubSource.

    Pass rows from load_classification_manifest; a bare sequence cannot certify
    the preparation summary. Construction inspects manifest metadata only.
    Runtime source errors raise with stable identity; samples are never skipped.
    """

    def __init__(
        self, source: CubSource, manifest: Sequence[Mapping[str, Any]], *,
        partition: str, transform: Callable[[Image.Image], torch.Tensor],
    ):
        if partition not in ('train', 'val', 'test'):
            raise ValueError('partition must be train, val, or test')
        official = 'test' if partition == 'test' else 'train'
        if source.spec != _SOURCES[official]:
            raise ValueError('Partition requires the matching pinned official source')
        if not callable(transform):
            raise ValueError('A tensor-producing transform is required')
        try:
            source_size = len(source.dataset)
            if not callable(getattr(source.dataset, '__getitem__', None)):
                raise TypeError
        except (TypeError, AttributeError) as error:
            raise ValueError('Classification requires indexed nonstreaming source data') from error
        if len(source.class_names) != 200 or len(set(source.class_names)) != 200:
            raise ValueError('Expected the original ordered 200-class mapping')
        selected = []
        identifiers = set()
        for row in manifest:
            if row.get('partition') != partition:
                continue
            index, label = row.get('source_row'), row.get('class_id')
            if (isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < source_size
                    or row.get('source_split') != official
                    or row.get('source_repo') != source.spec.repo_id
                    or row.get('source_revision') != source.spec.revision
                    or row.get('image_id') != source.spec.image_id(index)):
                raise ValueError('Manifest identity disagrees with the indexed pinned source')
            if row.get('image_id') in identifiers:
                raise ValueError('Duplicate manifest membership')
            identifiers.add(row['image_id'])
            if row.get('eligible') is not True:
                raise ValueError(f'{partition} contains ineligible rows; review policy before evaluation')
            if (isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < 200
                    or row.get('class_name') != source.class_names[label]):
                raise ValueError('Manifest class mapping disagrees with the source')
            selected.append(dict(row))
        if not selected:
            raise ValueError(f'No prepared rows in partition {partition}')
        self.source, self.partition, self.transform = source, partition, transform
        self.rows = tuple(sorted(selected, key=lambda row: row['source_row']))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        try:
            raw = self.source.dataset[row['source_row']]
            label = raw['label']
            if isinstance(label, bool) or not isinstance(label, int) or label != row['class_id']:
                raise ValueError('Source label changed since preparation')
            if raw['id'] != row['source_id'] or raw['file_name'] != row['file_name']:
                raise ValueError('Source metadata changed since preparation')
            with load_rgb(raw['image']) as rgb:
                tensor = self.transform(rgb)
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3 or tensor.shape[0] != 3:
                raise ValueError('Transform must return a three-channel CHW tensor')
            return {'image': tensor, 'label': row['class_id'], 'image_id': row['image_id']}
        except Exception as error:
            raise RuntimeError(f"Classification read failed for {row['image_id']} at source row {row['source_row']}: {error}") from error


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_classification_loaders(
    datasets: Mapping[str, Dataset], *, batch_size: int = 32, num_workers: int = 0,
    seed: int = 42, pin_memory: bool = False,
) -> dict[str, DataLoader]:
    """Batch explicitly supplied partitions; shuffle only train, drop nothing.

    Independent generators seed sampler order and worker startup. With zero
    workers, callers seed global Python/NumPy/torch once for augmentation.
    This factory never reseeds global RNGs or opens additional source splits.
    """
    for name, value, minimum in (('batch_size', batch_size, 1), ('num_workers', num_workers, 0), ('seed', seed, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if seed >= 2**32 or not isinstance(pin_memory, bool):
        raise ValueError('seed must be below 2**32 and pin_memory must be bool')
    offsets = {'train': 0, 'val': 1, 'test': 2}
    if not datasets or any(partition not in offsets for partition in datasets):
        raise ValueError('Supply one or more named train/val/test datasets')
    loaders = {}
    for partition, dataset in datasets.items():
        if getattr(dataset, 'partition', partition) != partition:
            raise ValueError('Dataset partition disagrees with loader name')
        generator = torch.Generator().manual_seed(seed + offsets[partition])
        loaders[partition] = DataLoader(
            dataset, batch_size=batch_size, shuffle=partition == 'train',
            drop_last=False, num_workers=num_workers, pin_memory=pin_memory,
            generator=generator, worker_init_fn=_seed_worker,
        )
    return loaders
