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
        self.parents = parents

    def num_nodes(self) -> int:
        return len(self.parents)

    def edges(self) -> List[Tuple[int, int]]:
        return list(zip(self.parents[1:], itertools.count(1)))

    def children(self) -> Dict[int, np.ndarray]:
        result = collections.defaultdict(list)
        for i, j in self.edges():
            result[i].append(j)
        return {k: np.array(v, dtype=int) for k, v in result.items()}

    def num_children(self) -> np.ndarray:
        n = len(self.parents)
        unique, counts = np.unique(self.parents[1:], return_counts=True)
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

    def depths(self) -> np.ndarray:
        n = len(self.parents)
        d = np.zeros([n], dtype=int)
        for i, j in self.edges():
            assert i < j, 'require edges in topological order'
            d[j] = d[i] + 1
        return d

    def num_leaf_descendants(self) -> np.ndarray:
        n = len(self.parents)
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

    def accumulate_ancestors(self, func: Callable, values: Sequence) -> list:
        # Start from root and move down.
        n = len(self.parents)
        partials = [None] * n
        partials[0] = values[0]
        for i, j in self.edges():
            partials[j] = func(partials[i], values[j])
        return partials

    def accumulate_descendants(self, func: Callable, values: Sequence) -> list:
        # Start from leaves and move up.
        n = len(self.parents)
        partials = [None] * n
        for i in range(n):
            partials[i] = values[i]
        for i, j in reversed(self.edges()):
            partials[i] = func(partials[i], partials[j])
        return partials

    def ancestor_mask(self, strict=False) -> np.ndarray:
        n = len(self.parents)
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
