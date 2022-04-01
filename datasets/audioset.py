import glob
import gzip
import os
from typing import Dict, Callable, Iterable, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from torch.utils.data import Dataset


NUM_TFRECORD_FILES = {
    'unbal_train': 4096,
    'bal_train': 4070,
    'eval': 4062,
}

# From research/audioset/vggish/vggish_params.py
# in https://github.com/tensorflow/models/
QUANTIZE_MIN_VAL = -2.0
QUANTIZE_MAX_VAL = +2.0


class AudioSet(Dataset):
    """Feature embeddings for AudioSet.

    Loads entire dataset into memory.
    """

    def __init__(
            self,
            root: str,
            split: str,
            label_order: Optional[List[str]] = None,
            label_override_file: Optional[str] = None,
            reduction: str = 'mean',
            ):
        if reduction == 'none':
            reduce_fn = None
        elif reduction == 'mean':
            reduce_fn = np.mean
        elif reduction == 'max':
            reduce_fn = np.max
        else:
            raise ValueError('unknown reduction', reduction)

        pattern = os.path.join(root, split, '*.tfrecord')
        tfrecord_files = glob.glob(pattern)
        if len(tfrecord_files) != NUM_TFRECORD_FILES[split]:
            raise ValueError('wrong number of files', len(tfrecord_files), NUM_TFRECORD_FILES[split])
        tf_dataset = tf.data.TFRecordDataset(tfrecord_files)

        if label_override_file:
            # Construct list of integer labels. Use -1 if example is to be excluded.
            label_to_index = {x: i for i, x in enumerate(label_order)}
            with gzip.open(label_override_file, 'rt') as f:
                label_override = [
                    label_to_index[label_name] if label_name else -1
                    for label_name in (line.strip() for line in f)
                ]
        else:
            label_override = None

        self.features = []
        self.labels = []
        for i, serialized_example in enumerate(tf_dataset):
            if label_override and not label_override[i] >= 0:
                continue
            x, y = _parse(serialized_example.numpy())
            if reduce_fn is not None:
                x = reduce_fn(x, axis=0)
            if label_override:
                y = label_override[i]
            self.features.append(x)
            self.labels.append(y)

    def __getitem__(self, i: int) -> Tuple[np.ndarray, List[int]]:
        return self.features[i], self.labels[i]

    def __len__(self):
        return len(self.labels)


class AudioSetOriginal(Dataset):
    """Feature embeddings for AudioSet.

    Loads entire dataset into memory.
    """

    def __init__(
            self,
            root: str,
            split: str,
            reduction: str = 'mean',
            ):
        if reduction == 'none':
            reduce_fn = None
        elif reduction == 'mean':
            reduce_fn = np.mean
        elif reduction == 'max':
            reduce_fn = np.max
        else:
            raise ValueError('unknown reduction', reduction)

        pattern = os.path.join(root, split, '*.tfrecord')
        tfrecord_files = glob.glob(pattern)
        if len(tfrecord_files) != NUM_TFRECORD_FILES[split]:
            raise ValueError('wrong number of files', len(tfrecord_files), NUM_TFRECORD_FILES[split])
        tf_dataset = tf.data.TFRecordDataset(tfrecord_files)

        self.features = []
        self.labels = []
        for i, serialized_example in enumerate(tf_dataset):
            x, y = _parse(serialized_example.numpy())
            if reduce_fn is not None:
                x = reduce_fn(x, axis=0)
            self.features.append(x)
            self.labels.append(y)

    def __getitem__(self, i: int) -> Tuple[np.ndarray, List[int]]:
        return self.features[i], self.labels[i]

    def __len__(self):
        return len(self.labels)


def _parse(serialized: bytes) -> Tuple[np.ndarray, List[int]]:
    example = tf.train.SequenceExample.FromString(serialized)
    labels = list(example.context.feature['labels'].int64_list.value)
    features = np.asarray([
        _decode_features(feature.bytes_list.value[0])
        for feature in example.feature_lists.feature_list['audio_embedding'].feature
    ])
    features = _dequantize(features)
    return features, labels


def _decode_features(raw_bytes):
    # return np.frombuffer(raw_bytes, dtype=np.uint8)
    return tf.io.decode_raw(raw_bytes, tf.uint8).numpy()


def _dequantize(quantized):
    quantized = quantized.astype(np.float32)
    return quantized / 255 * (QUANTIZE_MAX_VAL - QUANTIZE_MIN_VAL) + QUANTIZE_MIN_VAL
