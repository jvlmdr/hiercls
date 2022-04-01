import csv
import glob
import itertools
from multiprocessing import reduction
import pathlib
from typing import Dict, Callable, Iterable, List, Optional, Tuple

from absl import app
from absl import flags
import networkx as nx
import numpy as np
import tensorflow as tf
import tqdm

flags.DEFINE_string('hierarchy_file', None, 'File that defines new hierarchy.', required=True)
flags.DEFINE_string('labels_file', None, 'File that defines original labels.', required=True)
flags.DEFINE_string('src_dir', None, 'Dir that contains dataset in tfrecord format.', required=True)
flags.DEFINE_string('dst_dir', None, 'Dir to which to write dataset.', required=True)
flags.DEFINE_list('splits', ['bal_train', 'unbal_train', 'eval'], 'Which splits to do.')
flags.DEFINE_string('reduction', 'mean', 'How to reduce features over time.')

FLAGS = flags.FLAGS

NUM_TFRECORD_FILES = {
    'unbal_train': 4096,
    'bal_train': 4070,
    'eval': 4062,
}

# From research/audioset/vggish/vggish_params.py
# in https://github.com/tensorflow/models/
QUANTIZE_MIN_VAL = -2.0
QUANTIZE_MAX_VAL = +2.0

REDUCE_FNS = {
    'none': None,
    'mean': np.mean,
    'max': np.max,
}


def main(_):
    src_dir = pathlib.Path(FLAGS.src_dir)
    dst_dir = pathlib.Path(FLAGS.dst_dir)

    # Load mapping from label index to tag.
    with open(FLAGS.labels_file) as f:
        reader = csv.reader(f)
        next(reader)
        # index,mid,display_name
        old_label_order = [row[1] for row in reader]

    # Load tree.
    with open(FLAGS.hierarchy_file) as f:
        reader = csv.reader(f)
        edges = list(reader)
    g = nx.DiGraph()
    g.add_edges_from(edges)
    root, = [x for x in g if g.in_degree[x] == 0]
    _assert_is_tree(g, root)

    reduce_fn = REDUCE_FNS[FLAGS.reduction]
    label_transform_fn = _make_label_transform_fn(g, old_label_order)

    for split in FLAGS.splits:
        tfrecord_files = list((src_dir / split).glob('*.tfrecord'))
        assert len(tfrecord_files) == NUM_TFRECORD_FILES[split]

        features = []
        labels = []
        indexes = []
        n = 0
        exclude_multiple = 0
        exclude_none = 0

        # Process shards one at a time to give progress meter.
        # tf_dataset = tf.data.TFRecordDataset(tfrecord_files)
        file_sizes = [f.stat().st_size for f in tfrecord_files]
        with tqdm.tqdm(desc=split, total=sum(file_sizes), unit='B', unit_scale=True) as pbar:
            for tfrecord_file, file_size in zip(tfrecord_files, file_sizes):
                tf_dataset = tf.data.TFRecordDataset(tfrecord_file)
                for serialized_example in tf_dataset:
                    x, ys = _parse(serialized_example.numpy())
                    ys = label_transform_fn(ys)
                    if len(ys) != 1:
                        if len(ys) > 1:
                            exclude_multiple += 1
                        else:
                            exclude_none += 1
                    else:
                        y, = ys
                        if reduce_fn is not None:
                            x = reduce_fn(x, axis=0)
                        features.append(x)
                        labels.append(y)
                        indexes.append(n)
                    n += 1
                pbar.update(file_size)

        print('summary for {}: keep {} of {}; no label {}, multiple labels {}'.format(
            split, len(features), n, exclude_none, exclude_multiple))

        dst_dir.mkdir(parents=True, exist_ok=True)
        with open(dst_dir / (split + '.npy'), 'wb') as f:
            np.savez(f, features=np.array(features), labels=np.array(labels), indexes=np.array(indexes))


def _make_label_transform_fn(g, old_label_order) -> Callable[[List[int]], List[int]]:
    ancestors = {v: nx.ancestors(g, v) for v in g}
    new_label_order = [u for u in old_label_order if u in g]
    label_to_index = {u: i for i, u in enumerate(new_label_order)}

    def transform(labels: List[int]) -> List[int]:
        tags = [old_label_order[n] for n in labels]
        # Keep only labels that are present in new graph.
        tags = [u for u in tags if u in g]
        # Exclude any labels that are ancestors of other labels.
        tags = [u for u in tags if not any(u in ancestors[v] for v in tags)]
        return [label_to_index[u] for u in tags]

    return transform


def _assert_is_tree(g: nx.DiGraph, root):
    assert g.number_of_edges() == g.number_of_nodes() - 1
    assert g.in_degree[root] == 0
    assert all([x == root or g.in_degree[x] == 1 for x in g])


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


if __name__ == '__main__':
    app.run(main)
