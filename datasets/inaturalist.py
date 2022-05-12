"""iNaturalist datasets using annotations from JSON files.

The torchvision INaturalist class does not support the train/val split.
Furthermore, it may give a different order to the labels (to confirm).
(The hierarchy uses dataset['category']['id'] from the JSON file.)
"""

import io
import json
import os
from typing import BinaryIO, Callable

import PIL
import torchvision

default_loader = torchvision.datasets.folder.default_loader


def INaturalist2017(root: str, split: str, **kwargs):
    return INaturalist(root=root, json_file=f'{split}2017.json', **kwargs)


def INaturalist2018(root: str, split: str, **kwargs):
    return INaturalist(root=root, json_file=f'{split}2018.json', **kwargs)


def INaturalist2019(root: str, split: str, **kwargs):
    return INaturalist(root=root, json_file=f'{split}2019.json', **kwargs)


def INaturalist2021(root: str, split: str, **kwargs):
    # Note: 2021 edition does not include year in JSON filename.
    return INaturalist(root=root, json_file=f'{split}.json', **kwargs)


class INaturalist(torchvision.datasets.VisionDataset):
    """Loads annotations for train and val sets from JSON file.

    Expects directory structure:
        {root}/{json_file}.json
        {root}/{image['file_name']} for each image in the JSON file
    """

    def __init__(self,
                 root: str,
                 json_file: str,
                 loader: Callable = default_loader,
                 use_mem: bool = False,
                 **kwargs):
        # Parent constructor sets root and handles transforms.
        super().__init__(root, **kwargs)
        self.loader = loader
        self.use_mem = use_mem

        with open(os.path.join(root, json_file)) as f:
            dataset = json.load(f)
        image_to_fname = {im['id']: im['file_name'] for im in dataset['images']}
        # Note: The image filename is relative to root dir, not absolute.
        self.samples = [(image_to_fname[ann['image_id']], ann['category_id'])
                        for ann in dataset['annotations']]
        self.targets = [target for _, target in self.samples]

        if self.use_mem:
            self.images = [_load_bytes(fname) for fname, _ in self.samples]

    def __getitem__(self, index: int):
        fname, target = self.samples[index]
        if self.use_mem:
            sample = _pil_load(io.BytesIO(self.images[index]))
        else:
            sample = self.loader(os.path.join(self.root, fname))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)


def _load_bytes(fname: str) -> bytes:
    with open(fname, 'rb') as f:
        return f.read()


def _pil_load(f: BinaryIO) -> PIL.Image.Image:
    return PIL.Image.open(f).convert('RGB')
