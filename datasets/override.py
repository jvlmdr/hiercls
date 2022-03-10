from typing import Sequence

from torch.utils import data
import torch


class OverrideTargets(data.Dataset):
    r"""Override the labels of a given dataset.

    Args:
        dataset (Dataset): The whole Dataset
        targets (sequence): The new labels.
    """

    def __init__(self, dataset: data.Dataset, targets: torch.Tensor):
        if len(dataset) != len(targets):
            raise ValueError('expect same length')
        self.dataset = dataset
        self.targets = targets

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]
        y = self.targets[idx]
        return x, y

    def __len__(self):
        return len(self.dataset)
