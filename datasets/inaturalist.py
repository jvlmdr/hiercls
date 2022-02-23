"""Use annotations from JSON files.

The torchvision INaturalist class does not load the train/val split.
Furthermore, it may give a different order to the labels (to confirm).
"""

import json
import os
from typing import Callable

import torchvision

default_loader = torchvision.datasets.folder.default_loader


class INaturalist2018(torchvision.datasets.VisionDataset):
    """Loads annotations for train and val sets from JSON file.

    Expects directory structure:
        train2018.json
        val2018.json
        train_val2018/{biological_class}/{category_id}/{filename}.jpg
    """

    def __init__(self, root: str, split: str,
                 loader: Callable = default_loader,
                 **kwargs):
        # Parent constructor sets root and handles transforms.
        super().__init__(root, **kwargs)
        self.loader = loader

        with open(os.path.join(root, f'{split}2018.json')) as f:
            dataset = json.load(f)
        image_to_fname = {im['id']: im['file_name'] for im in dataset['images']}
        self.samples = [(image_to_fname[ann['image_id']], ann['category_id'])
                        for ann in dataset['annotations']]

    def __getitem__(self, index: int):
        fname, target = self.samples[index]
        sample = self.loader(os.path.join(self.root, fname))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)
