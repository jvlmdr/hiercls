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
    device = scores.device
    leaf_subset = torch.from_numpy(tree.leaf_subset()).to(device)
    # TODO: Could make this faster by only computing likelihood for targets.
    log_prob = hier_log_softmax(tree, scores, dim=-1)
    leaf_log_prob = torch.index_select(log_prob, -1, leaf_subset)
    nll = -torch.gather(leaf_log_prob, -1, target.unsqueeze(-1)).squeeze(-1)
    return torch.mean(nll)


class HierSoftmaxNLL(nn.Module):
    """Avoids recomputation in hier_softmax_nll."""

    def __init__(self, tree: hier.Hierarchy, with_leaf_targets: bool = False):
        super().__init__()
        if with_leaf_targets:
            self.label_order = torch.from_numpy(tree.leaf_subset())
        else:
            self.label_order = None
        self.hier_log_softmax = HierLogSoftmax(tree)

    def _apply(self, fn):
        super()._apply(fn)
        # Do not apply fn to indices because it might convert dtype.
        self.hier_log_softmax = self.hier_log_softmax._apply(fn)
        return self

    @property
    def device(self) -> torch.device:
        return self.hier_log_softmax.device

    def forward(self, scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # TODO: Could make this faster by only computing likelihood for targets.
        device = self.device
        log_prob = self.hier_log_softmax(scores, dim=-1)
        if self.label_order is not None:
            log_prob = torch.index_select(log_prob, -1, self.label_order.to(device))
        nll = -torch.gather(log_prob, -1, target.unsqueeze(-1)).squeeze(-1)
        return torch.mean(nll)


def hier_log_softmax(
        tree: hier.Hierarchy,
        scores: torch.Tensor,
        dim: int = -1) -> torch.Tensor:
    # Split scores into softmax for each internal node over its children.
    # Convert from [s[0], s[1], ..., s[n-1]]
    # to [[s[0], ..., s[k-1], -inf, -inf, ...],
    #     ...
    #     [..., s[n-1], -inf, -inf, ...]].
    # Use index_copy with flat_index, then reshape and compute log_softmax.
    # Then re-flatten and use index_select with flat_index.
    # This is faster than using torch.split() and map(log_softmax, ...).
    # Finally, take sum over ancestor conditionals to obtain likelihoods.
    assert dim == -1 or dim == scores.ndim - 1
    num_nodes = tree.num_nodes()
    num_internal = tree.num_internal_nodes()
    node_to_children = tree.children()
    cond_children = [node_to_children[x] for x in tree.internal_subset()]
    cond_num_children = list(map(len, cond_children))
    max_num_children = max(cond_num_children)
    row_index = np.concatenate([np.full(n, i) for i, n in enumerate(cond_num_children)])
    col_index = np.concatenate([np.arange(n) for n in cond_num_children])
    flat_index = row_index * max_num_children + col_index
    child_index = np.concatenate(cond_children)
    sum_ancestors_fn = SumAncestors(tree, exclude_root=False)

    device = scores.device
    flat_index = torch.from_numpy(flat_index).to(device)
    child_index = torch.from_numpy(child_index).to(device)
    sum_ancestors_fn = sum_ancestors_fn.to(device)
    input_shape = list(scores.shape)
    flat_shape = [*input_shape[:-1], num_internal * max_num_children]
    # Pad with -inf for log_softmax.
    # flat[..., flat_index] = scores
    flat = torch.full(flat_shape, -torch.inf, device=device).index_copy_(
        -1, flat_index, scores)
    split_shape = [*input_shape[:-1], num_internal, max_num_children]
    child_scores = flat.reshape(split_shape)
    child_log_p = torch.log_softmax(child_scores, dim=-1)
    child_log_p = child_log_p.reshape(flat_shape)
    output_shape = [*input_shape[:-1], num_nodes]
    # log_cond_p[..., child_index] = child_log_p[..., flat_index]
    log_cond_p = torch.zeros(output_shape, device=device).index_copy_(
        -1, child_index, child_log_p.index_select(-1, flat_index))
    return sum_ancestors_fn(log_cond_p, dim=-1)


class HierLogSoftmax(nn.Module):
    """Avoids re-computation in hier_log_softmax()."""

    def __init__(self, tree: hier.Hierarchy):
        super().__init__()
        num_nodes = tree.num_nodes()
        num_internal = tree.num_internal_nodes()
        node_to_children = tree.children()
        cond_children = [node_to_children[x] for x in tree.internal_subset()]
        cond_num_children = list(map(len, cond_children))
        max_num_children = max(cond_num_children)
        row_index = np.concatenate([np.full(n, i) for i, n in enumerate(cond_num_children)])
        col_index = np.concatenate([np.arange(n) for n in cond_num_children])
        flat_index = torch.from_numpy(row_index * max_num_children + col_index)
        child_index = torch.from_numpy(np.concatenate(cond_children))
        sum_ancestors_fn = SumAncestors(tree, exclude_root=False)

        self.num_nodes = num_nodes
        self.num_internal = num_internal
        self.max_num_children = max_num_children
        self.flat_index = flat_index
        self.child_index = child_index
        self.sum_ancestors_fn = sum_ancestors_fn

    def _apply(self, fn):
        super()._apply(fn)
        # Do not apply fn to indices because it might convert dtype.
        self.sum_ancestors_fn = self.sum_ancestors_fn._apply(fn)
        return self

    @property
    def device(self) -> torch.device:
        return self.sum_ancestors_fn.device

    def forward(self, scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim == -1 or dim == scores.ndim - 1
        device = self.device
        input_shape = list(scores.shape)
        flat_shape = [*input_shape[:-1], self.num_internal * self.max_num_children]
        # Pad with -inf for log_softmax.
        flat_index = self.flat_index.to(device)
        # flat[..., flat_index] = scores
        flat = torch.full(flat_shape, -torch.inf, device=device).index_copy_(
            -1, flat_index, scores)
        split_shape = [*input_shape[:-1], self.num_internal, self.max_num_children]
        child_scores = flat.reshape(split_shape)
        child_log_p = torch.log_softmax(child_scores, dim=-1)
        child_log_p = child_log_p.reshape(flat_shape)
        output_shape = [*input_shape[:-1], self.num_nodes]
        child_index = self.child_index.to(device)
        # log_cond_p[..., child_index] = child_log_p[..., flat_index]
        log_cond_p = torch.zeros(output_shape, device=device).index_copy_(
            -1, child_index, child_log_p.index_select(-1, flat_index))
        return self.sum_ancestors_fn(log_cond_p, dim=-1)


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
    leaf_index = torch.from_numpy(tree.leaf_subset()).to(device)
    return node_values.index_add(dim, leaf_index, leaf_values)


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
            transpose: bool,
            leaf_only: bool = False,
            exclude_root: bool = False,
            strict: bool = False):
        super().__init__()
        # The value matrix[i, j] is true if i is an ancestor of j.
        # Take transpose for sum over descendants.
        matrix = tree.ancestor_mask(strict=strict)
        if leaf_only:
            matrix = matrix[:, tree.leaf_mask()]
        if exclude_root:
            matrix = matrix[1:, :]
        if transpose:
            matrix = matrix.T
        matrix = torch.from_numpy(matrix).type(torch.get_default_dtype())
        self.matrix = matrix

    def _apply(self, fn):
        super()._apply(fn)
        self.matrix = fn(self.matrix)
        return self

    @property
    def device(self) -> torch.device:
        return self.matrix.device

    def forward(self, values: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # TODO: Re-order dimensions to make this work with dim != -1.
        assert dim == -1 or dim == values.ndim - 1
        return torch.tensordot(values, self.matrix, dims=1)


SumAncestors = partial(Sum, transpose=False, leaf_only=False)
SumLeafAncestors = partial(Sum, transpose=True, leaf_only=True)
SumDescendants = partial(Sum, transpose=True, leaf_only=False)
SumLeafDescendants = partial(Sum, transpose=True, leaf_only=True)


class MultiLabelNLL(nn.Module):

    def __init__(self, tree: hier.Hierarchy, with_leaf_targets: bool = False):
        super().__init__()
        # The boolean array binary_labels[i, :] indicates whether
        # each node is an ancestor of node i (including i itself).
        # The root node is excluded since it is always positive.
        binary_labels = tree.ancestor_mask(strict=False).T[:, 1:]
        if with_leaf_targets:
            binary_labels = binary_labels[tree.leaf_subset(), :]

        dtype = torch.get_default_dtype()
        self.binary_labels = torch.from_numpy(binary_labels).type(dtype)
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def _apply(self, fn):
        super()._apply(fn)
        self.binary_labels = fn(self.binary_labels)
        self.bce_loss = self.bce_loss._apply(fn)
        return self

    @property
    def device(self) -> torch.device:
        return self.binary_labels.device

    def forward(self, scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce_loss(scores,
                             torch.index_select(self.binary_labels, 0, target))


def multilabel_likelihood(tree: hier.Hierarchy, scores: torch.Tensor, dim=-1):
    return torch.exp(multilabel_log_likelihood(tree, scores, dim=dim))


def multilabel_log_likelihood(tree: hier.Hierarchy, scores: torch.Tensor, dim=-1):
    assert scores.shape[dim] == tree.num_nodes() - 1
    assert dim == -1
    shape = list(scores.shape)
    shape[-1] = 1
    zero = torch.zeros(shape, dtype=scores.dtype, device=scores.device)
    return torch.cat([zero, nn.functional.logsigmoid(scores)], dim=-1)
