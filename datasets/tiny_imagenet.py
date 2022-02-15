"""Tiny ImageNet contains 200 classes and 64px images.

Download from:
http://cs231n.stanford.edu/tiny-imagenet-200.zip

Expects directory structure:
    wnids.txt
    train/{class}/{example}.{ext}
    val/val_annotations.txt
    val/images/{example}.{ext}
    test/images/{example}.{ext}
"""

import csv
import os
from typing import Dict, Callable, Iterable, List, Optional, Tuple

import torchvision.datasets

default_loader = torchvision.datasets.folder.default_loader


def TinyImageNet(root: str, split: str, **kwargs):
    """Creates a dataset for one split of Tiny ImageNet.

    Args:
        root: Path to dataset.
        split: Either 'train' or 'val'.
    """
    if split == 'train':
        return FromFolder(root, split, **kwargs)
    else:
        return FromList(root, split, **kwargs)


class FromFolder(torchvision.datasets.ImageFolder):

    def __init__(self, root: str, split: str, **kwargs):
        self.wnids = load_wnids(os.path.join(root, 'wnids.txt'))
        # Parent constructor calls find_classes().
        super().__init__(os.path.join(root, split), **kwargs)

    def find_classes(self, _) -> Tuple[List[str], Dict[str, int]]:
        wnid_to_index = {x: i for i, x in enumerate(self.wnids)}
        return self.wnids, wnid_to_index


class FromList(torchvision.datasets.VisionDataset):

    def __init__(self, root: str, split: str,
                 loader: Callable = default_loader,
                 **kwargs):
        # Parent constructor sets root and handles transforms.
        super().__init__(root, **kwargs)
        self.split = split
        self.loader = loader
        self.classes = load_wnids(os.path.join(root, 'wnids.txt'))
        self.class_to_idx = {x: i for i, x in enumerate(self.classes)}
        self.samples = list(self.load_samples())

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def load_samples(self) -> Iterable[Tuple[str, int]]:
        split_dir = os.path.join(self.root, self.split)
        with open(os.path.join(split_dir, f'{self.split}_annotations.txt')) as f:
            records = list(csv.reader(f, delimiter='\t'))
        for row in records:
            path = os.path.join(split_dir, 'images', row[0])
            target = self.class_to_idx[row[1]]
            yield path, target


def load_wnids(path: str) -> List[str]:
    # Ignore directory and take classes from file in root.
    with open(path) as f:
        wnids = f.read().splitlines()
    if len(wnids) != 200:
        raise ValueError('expect 200 classes', len(wnids))
    return wnids
