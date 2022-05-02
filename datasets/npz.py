import os
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.utils.data


class NpzDataset(torch.utils.data.Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Loads tensors from an npz file."""

    def __init__(self, root: str, split: str, transform: Optional[Callable] = None):
        data = np.load(os.path.join(root, split + '.npz'))
        mean = np.load(os.path.join(root, 'mean.npy')).astype(np.float32)
        features = data['features'].astype(np.float32)
        labels = data['labels']
        features -= np.expand_dims(mean, 0)

        features = torch.from_numpy(features)
        labels = torch.from_numpy(labels)
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
