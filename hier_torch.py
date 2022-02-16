from functools import partial
from typing import Optional

import numpy as np
import torch
from torch import nn

import hier


def hier_softmax_nll(
        tree: hier.Hierarchy,
        scores: torch.Tensor,
        target: torch.Tensor) -> torch.Tensor:
    log_prob = hier_log_softmax(tree, scores, dim=-1)
    nll = -torch.gather(log_prob, -1, target)
    return torch.mean(nll)


def hier_softmax_nll_with_leaf(
        tree: hier.Hierarchy,
        scores: torch.Tensor,
        target: torch.Tensor) -> torch.Tensor:
    leaf_subset = torch.from_numpy(tree.leaf_subset()).to(scores.device)
    log_prob = hier_log_softmax(tree, scores, dim=-1)
    leaf_log_prob = torch.index_select(log_prob, -1, leaf_subset)
    nll = -torch.gather(leaf_log_prob, -1, target.unsqueeze(-1)).squeeze(-1)
    return torch.mean(nll)


# class HierSoftmaxNLL(nn.Module):
#
#     def __init__(self, tree):
#         super().__init__()
#         self.tree = tree
#
#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         """Target is node index."""
#         pass


def hier_log_softmax(
        tree: hier.Hierarchy,
        scores: torch.Tensor,
        dim: int = -1) -> torch.Tensor:
    internal_nodes = tree.internal_subset()
    node_to_children = tree.children()
    cond_children = [node_to_children[x] for x in internal_nodes]
    cond_sizes = [len(x) for x in cond_children]
    cond_scores = scores.split(cond_sizes, dim=dim)
    cond_log_softmax = [x.log_softmax(dim=dim) for x in cond_scores]
    shape = list(scores.shape)
    shape[dim] = tree.num_nodes()
    log_cond_prob = torch.zeros(shape, device=scores.device).index_add(
        dim,
        torch.from_numpy(np.concatenate(cond_children)).to(scores.device),
        torch.cat(cond_log_softmax, dim=dim))
    log_prob = sum_ancestors(tree, log_cond_prob, dim=dim, strict=False)
    return log_prob


class HierLogSoftmax(nn.Module):
    """Avoids re-computation in hier_log_softmax()."""

    def __init__(self, tree: hier.Hierarchy):
        super().__init__()
        internal_nodes = tree.internal_subset()
        node_to_children = tree.children()
        cond_children = [node_to_children[x] for x in internal_nodes]
        cond_sizes = [len(x) for x in cond_children]
        cat_cond_children = torch.from_numpy(np.concatenate(cond_children))

        self.cond_sizes = cond_sizes
        self.num_nodes = tree.num_nodes()
        # self.cat_cond_children = cat_cond_children
        self.register_buffer('cat_cond_children', cat_cond_children)
        self.cat_cond_children: Optional[torch.Tensor]
        self.sum_ancestors = SumAncestors(tree, strict=False)

    def forward(self, scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
        device = scores.device
        cond_scores = scores.split(self.cond_sizes, dim=dim)
        cond_log_softmax = [x.log_softmax(dim=dim) for x in cond_scores]
        shape = list(scores.shape)
        shape[dim] = self.num_nodes
        log_cond_prob = torch.zeros(shape, device=device).index_add(
            dim, self.cat_cond_children, torch.cat(cond_log_softmax, dim=dim))
        log_prob = self.sum_ancestors(log_cond_prob, dim=dim)
        return log_prob


def leaf_embed(
        tree: hier.Hierarchy,
        values: torch.Tensor,
        dim: int = -1) -> torch.Tensor:
    """Embeds leaf values in a tree. Internal nodes are set to zero."""
    shape = list(values.shape)
    shape[dim] = tree.num_nodes()
    node_values = torch.zeros(shape, device=values.device)
    return leaf_add(tree, node_values, values, dim=dim)


def leaf_add(
        tree: hier.Hierarchy,
        node_values: torch.Tensor,
        leaf_values: torch.Tensor,
        dim: int = -1) -> torch.Tensor:
    """Adds values for leaf nodes to values for all nodes."""
    device = node_values.device
    leaf_index = torch.from_numpy(tree.leaf_subset())
    return node_values.index_add(dim, leaf_index.to(device), leaf_values)


def sum_leaf_descendants(
        tree: hier.Hierarchy,
        values: torch.Tensor,
        dim: int = -1) -> torch.Tensor:
    """Computes sum over leaf descendants for each node."""
    # The value is_ancestor[i, j] is true if i is an ancestor of j.
    is_ancestor = tree.ancestor_mask()
    leaf_is_descendant = is_ancestor[:, tree.leaf_mask()].T
    matrix = torch.from_numpy(leaf_is_descendant)
    matrix = matrix.to(device=values.device, dtype=torch.get_default_dtype())
    # TODO: Re-order dimensions to make this work with dim != -1.
    assert dim == -1 or dim == values.ndim - 1
    return torch.tensordot(values, matrix, dims=1)


def sum_descendants(
        tree: hier.Hierarchy,
        values: torch.Tensor,
        dim: int = -1,
        strict: bool = False) -> torch.Tensor:
    """Computes sum over all descendants for each node."""
    # The value is_ancestor[i, j] is true if i is an ancestor of j.
    is_ancestor = tree.ancestor_mask(strict=strict)
    matrix = (torch.from_numpy(is_ancestor.T)
              .to(device=values.device, dtype=torch.get_default_dtype()))
    # TODO: Re-order dimensions to make this work with dim != -1.
    assert dim == -1 or dim == values.ndim - 1
    return torch.tensordot(values, matrix, dims=1)


def sum_ancestors(
        tree: hier.Hierarchy,
        values: torch.Tensor,
        dim: int = -1,
        strict: bool = False) -> torch.Tensor:
    """Computes sum over ancestors of each node."""
    # The value is_ancestor[i, j] is true if i is an ancestor of j.
    is_ancestor = tree.ancestor_mask(strict=strict)
    matrix = (torch.from_numpy(is_ancestor)
              .to(device=values.device, dtype=torch.get_default_dtype()))
    # TODO: Re-order dimensions to make this work with dim != -1.
    assert dim == -1 or dim == values.ndim - 1
    return torch.tensordot(values, matrix, dims=1)


def sum_leaf_ancestors(
        tree: hier.Hierarchy,
        values: torch.Tensor,
        dim: int = -1) -> torch.Tensor:
    """Computes sum over ancestors of each leaf."""
    # The value is_ancestor[i, j] is true if i is an ancestor of j.
    is_ancestor = tree.ancestor_mask()
    is_ancestor_for_leaf = is_ancestor[:, tree.leaf_mask()]
    matrix = (torch.from_numpy(is_ancestor_for_leaf)
              .to(device=values.device, dtype=torch.get_default_dtype()))
    # TODO: Re-order dimensions to make this work with dim != -1.
    assert dim == -1 or dim == values.ndim - 1
    return torch.tensordot(values, matrix, dims=1)



class Sum(nn.Module):
    """Avoids re-computation in sum_xxx()."""

    def __init__(
            self,
            tree: hier.Hierarchy,
            leaf: bool,
            transpose: bool,
            strict: bool = False):
        super().__init__()

        # The value is_ancestor[i, j] is true if i is an ancestor of j.
        matrix = tree.ancestor_mask(strict=strict)
        if leaf:
            matrix = matrix[:, tree.leaf_mask()]
        if transpose:
            matrix = matrix.T
        matrix = torch.from_numpy(matrix).type(torch.get_default_dtype())

        # self.matrix = matrix
        self.register_buffer('matrix', matrix)
        self.matrix: Optional[torch.Tensor]

    def forward(self, values: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # TODO: Re-order dimensions to make this work with dim != -1.
        assert dim == -1 or dim == values.ndim - 1
        return torch.tensordot(values, self.matrix, dims=1)


SumAncestors = partial(Sum, leaf=False, transpose=False)
SumLeafAncestors = partial(Sum, leaf=False, transpose=True)
SumDescendants = partial(Sum, leaf=False, transpose=True)
SumLeafDescendants = partial(Sum, leaf=True, transpose=True)
