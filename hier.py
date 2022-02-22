import collections
import csv
import itertools
from typing import Callable, Dict, List, Sequence, TextIO, Tuple

import numpy as np


class Hierarchy:
    """Hierarchy of nodes 0, ..., n-1."""

    def __init__(self, parents):
        n = len(parents)
        assert np.all(parents[1:] < np.arange(1, n))
        self._parents = parents

    def num_nodes(self) -> int:
        return len(self._parents)

    def edges(self) -> List[Tuple[int, int]]:
        return list(zip(self._parents[1:], itertools.count(1)))

    def parents(self) -> np.ndarray:
        return np.array(self._parents)

    def children(self) -> Dict[int, np.ndarray]:
        result = collections.defaultdict(list)
        for i, j in self.edges():
            result[i].append(j)
        return {k: np.array(v, dtype=int) for k, v in result.items()}

    def num_children(self) -> np.ndarray:
        n = len(self._parents)
        unique, counts = np.unique(self._parents[1:], return_counts=True)
        result = np.zeros([n], dtype=int)
        result[unique] = counts
        return result

    def leaf_mask(self) -> np.ndarray:
        return self.num_children() == 0

    def leaf_subset(self) -> np.ndarray:
        index, = self.leaf_mask().nonzero()
        return index

    def internal_subset(self) -> np.ndarray:
        index, = np.logical_not(self.leaf_mask()).nonzero()
        return index

    def num_leaf_nodes(self) -> int:
        return np.count_nonzero(self.leaf_mask())

    def num_internal_nodes(self) -> int:
        return np.count_nonzero(np.logical_not(self.leaf_mask()))

    def depths(self) -> np.ndarray:
        n = len(self._parents)
        d = np.zeros([n], dtype=int)
        for i, j in self.edges():
            assert i < j, 'require edges in topological order'
            d[j] = d[i] + 1
        return d

    def num_leaf_descendants(self) -> np.ndarray:
        n = len(self._parents)
        c = self.leaf_mask().astype(int)
        for i, j in reversed(self.edges()):
            assert i < j, 'require edges in topological order'
            c[i] += c[j]
        return c

    # def accumulate_ancestors_inplace(self, func: Callable, values: MutableSequence):
    #     # Start from root and move down.
    #     for i, j in self.edges():
    #         values[j] = func(values[i], values[j])

    # def accumulate_descendants_inplace(self, func: Callable, values: MutableSequence):
    #     # Start from leaves and move up.
    #     for i, j in reversed(self.edges()):
    #         values[i] = func(values[i], values[j])

    def accumulate_ancestors(self, func: Callable, values: Sequence) -> List:
        # Start from root and move down.
        n = len(self._parents)
        partials = [None] * n
        partials[0] = values[0]
        for i, j in self.edges():
            partials[j] = func(partials[i], values[j])
        return partials

    def accumulate_descendants(self, func: Callable, values: Sequence) -> List:
        # Start from leaves and move up.
        n = len(self._parents)
        partials = [None] * n
        for i in range(n):
            partials[i] = values[i]
        for i, j in reversed(self.edges()):
            partials[i] = func(partials[i], partials[j])
        return partials

    def ancestor_mask(self, strict=False) -> np.ndarray:
        n = len(self._parents)
        # If path exists i, ..., j then i < j and is_ancestor[i, j] is True.
        # Work with is_descendant instead to use consecutive blocks of memory.
        # Note that is_ancestor[i, j] == is_descendant[j, i].
        is_descendant = np.zeros([n, n], dtype=bool)
        if not strict:
            is_descendant[0, 0] = 1
        for i, j in self.edges():
            # Node i is parent of node j.
            assert i < j, 'require edges in topological order'
            is_descendant[j, :] = is_descendant[i, :]
            if strict:
                is_descendant[j, i] = 1
            else:
                is_descendant[j, j] = 1
        is_ancestor = is_descendant.T
        return is_ancestor

    def paths(
            self,
            exclude_root: bool = False,
            exclude_self: bool = False,
            ) -> List[np.ndarray]:
        is_descendant = self.ancestor_mask(strict=exclude_self).T
        if exclude_root:
            paths = [np.flatnonzero(mask) + 1 for mask in is_descendant[:, 1:]]
        else:
            paths = [np.flatnonzero(mask) for mask in is_descendant]
        return paths

    def paths_padded(self, value=-1, **kwargs) -> np.ndarray:
        n = self.num_nodes()
        paths = self.paths(**kwargs)
        path_lens = list(map(len, paths))
        max_len = max(path_lens)
        padded = np.full((n, max_len), value, dtype=int)
        row_index = np.concatenate([np.full(n, i) for i, n in enumerate(path_lens)])
        col_index = np.concatenate([np.arange(n) for n in path_lens])
        padded[row_index, col_index] = np.concatenate(paths)
        return padded


def make_hierarchy_from_edges(
        pairs : List[Tuple[str, str]],
        ) -> Tuple[Hierarchy, List[str]]:
    num_edges = len(pairs)
    num_nodes = num_edges + 1
    # Data structures to populate from list of pairs.
    parents = np.full([num_nodes], -1, dtype=int)
    names = [None] * num_nodes
    name_to_index = {}
    # Set name of root from first pair.
    root, _ = pairs[0]
    names[0] = root
    name_to_index[root] = 0
    for r, (u, v) in enumerate(pairs):
        if v in name_to_index:
            raise ValueError('has multiple parents', v)
        i = name_to_index[u]
        j = r + 1
        parents[j] = i
        names[j] = v
        name_to_index[v] = j
    return Hierarchy(parents), names


def load_edges(f: TextIO, delimiter=',') -> List[Tuple[str, str]]:
    """Load from file containing (parent, node) pairs."""
    reader = csv.reader(f)
    pairs = []
    for row in reader:
        if not row:
            continue
        if len(row) != 2:
            raise ValueError('invalid row', row)
        pairs.append(tuple(row))
    return pairs


def uniform_leaf(tree: Hierarchy) -> np.ndarray:
    """Returns a uniform distribution over leaf nodes."""
    is_ancestor = tree.ancestor_mask(strict=False)
    is_leaf = tree.leaf_mask()
    num_leaf_descendants = is_ancestor[:, is_leaf].sum(axis=1)
    return (1. / is_leaf.sum()) * num_leaf_descendants


def uniform_cond(tree: Hierarchy) -> np.ndarray:
    """Returns a uniform distribution over child nodes at each conditional."""
    node_to_num_children = {k: len(v) for k, v in tree.children().items()}
    num_children = np.asarray([node_to_num_children.get(x, 0) for x in range(tree.num_nodes())])
    parent_index = tree.parents()
    # Root node has likelihood 1 and no parent.
    log_cond_p = np.concatenate([[0.], -np.log(num_children[parent_index[1:]])])
    is_ancestor = tree.ancestor_mask(strict=False)
    log_p = np.dot(is_ancestor.T, log_cond_p)
    return np.exp(log_p)


def lca_depth(tree: Hierarchy, inds_a: np.ndarray, inds_b: np.ndarray) -> np.ndarray:
    """Returns the depth of the LCA node.

    Supports multi-dimensional index arrays.
    """
    paths = tree.paths_padded(exclude_root=True)
    paths_a = paths[inds_a]
    paths_b = paths[inds_b]
    return np.count_nonzero(
        ((paths_a == paths_b) & (paths_a >= 0) & (paths_b >= 0)),
        axis=-1)


def lca(tree: Hierarchy, inds_a: np.ndarray, inds_b: np.ndarray) -> np.ndarray:
    """Returns the index of the LCA node.

    Supports multi-dimensional index arrays.
    For example, to obtain an exhaustive table:
        n = tree.num_nodes()
        lca(tree, np.arange(n)[:, np.newaxis], np.arange(n)[np.newaxis, :])
    """
    paths = tree.paths_padded(exclude_root=False)
    paths_a = paths[inds_a]
    paths_b = paths[inds_b]
    num_common = np.count_nonzero(
        ((paths_a == paths_b) & (paths_a >= 0) & (paths_b >= 0)),
        axis=-1)
    return paths[inds_a, num_common - 1]


class FindLCA:

    def __init__(self, tree: Hierarchy):
        self.paths = tree.paths_padded(exclude_root=False)

    def __call__(self, inds_a: np.ndarray, inds_b: np.ndarray) -> np.ndarray:
        paths = self.paths
        paths_a = paths[inds_a]
        paths_b = paths[inds_b]
        num_common = np.count_nonzero(
            ((paths_a == paths_b) & (paths_a >= 0) & (paths_b >= 0)),
            axis=-1)
        return paths[inds_a, num_common - 1]


def truncate_given_lca(gt: np.ndarray, pr: np.ndarray, lca: np.ndarray) -> np.ndarray:
    """Truncates the prediction if a descendant of the ground-truth."""
    return np.where(gt == lca, gt, pr)
