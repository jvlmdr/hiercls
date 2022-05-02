import os
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.utils.data


class NpzDataset(torch.utils.data.Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Loads tensors from an npz file."""

    def __init__(self, root: str, split: str, transform: Optional[Callable] = None):
        fname = os.path.join(root, split + '.npz')
        data = np.load(fname)
        features = torch.from_numpy(data['features']).type(torch.get_default_dtype())
        labels = torch.from_numpy(data['labels'])
        self.targets = labels  # For sampling, etc.
        self.tensor_dataset = torch.utils.data.TensorDataset(features, labels)
        self.transform = transform

    def __getitem__(self, index):
        sample, target = self.tensor_dataset[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.tensor_dataset)
