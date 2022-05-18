from functools import partial
from multiprocessing import reduction
from typing import Callable, List, Optional, Sequence, Tuple

from absl import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision

import hier
import metrics


def flat_log_softmax(
        tree: hier.Hierarchy,
        scores: torch.Tensor,
        dim: int = -1) -> torch.Tensor:
    assert dim in (-1, scores.ndim - 1)
    logp_leaf = F.log_softmax(scores, dim=-1)
    # The value is_ancestor[i, j] is true if node i is an ancestor of node j.
    is_ancestor = tree.ancestor_mask(strict=False)
    is_leaf = tree.leaf_mask()
    # The value is_ancestor_leaf[i, k] is true if node i is an ancestor of leaf k.
    is_ancestor_leaf = is_ancestor[:, is_leaf]
    # Obtain logp for all leaf descendants, -inf for other nodes.
    is_ancestor_leaf = torch.from_numpy(is_ancestor_leaf).to(scores.device)
    logp_descendants = torch.where(
        is_ancestor_leaf,
        logp_leaf.unsqueeze(-2),
        torch.tensor(-torch.inf, device=scores.device))
    return torch.logsumexp(logp_descendants, dim=-1)


class FlatLogSoftmax(nn.Module):

    def __init__(self, tree):
        super().__init__()
        # The value is_ancestor[i, j] is true if node i is an ancestor of node j.
        is_ancestor = tree.ancestor_mask(strict=False)
        # The value is_ancestor_leaf[i, k] is true if node i is an ancestor of leaf k.
        is_ancestor_leaf = is_ancestor[:, tree.leaf_mask()]

        # TODO: May be important to avoid copying this to the device.
        # However, overriding _apply() will change the dtype.
        self.is_ancestor_leaf = torch.from_numpy(is_ancestor_leaf)

    def _apply(self, fn):
        super()._apply(fn)
        self.is_ancestor_leaf = fn(self.is_ancestor_leaf)
        return self

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        logp_leaf = F.log_softmax(scores, dim=-1)
        # Obtain logp for leaf descendants, -inf for other nodes.
        # TODO: This is hacky. Would prefer not to mutate object here.
        logp_descendants = torch.where(
            self.is_ancestor_leaf,
            logp_leaf.unsqueeze(-2),
            torch.tensor(-torch.inf, device=scores.device))
        return torch.logsumexp(logp_descendants, dim=-1)


class FlatSoftmaxNLL(nn.Module):
    """Like cross_entropy() but supports internal labels."""

    def __init__(self, tree, with_leaf_targets: bool = False, reduction: str = 'mean'):
        super().__init__()
        assert reduction in ('mean', 'none', None)
        if with_leaf_targets:
            raise ValueError('use F.cross_entropy() instead!')
        # The value is_ancestor[i, j] is true if node i is an ancestor of node j.
        is_ancestor = tree.ancestor_mask(strict=False)
        leaf_masks = is_ancestor[:, tree.leaf_mask()]
        self.leaf_masks = torch.from_numpy(leaf_masks)
        self.reduction = reduction

    def _apply(self, fn):
        super()._apply(fn)
        self.leaf_masks = fn(self.leaf_masks)
        return self

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logp_leaf = F.log_softmax(scores, dim=-1)
        # Obtain logp for leaf descendants, -inf for other nodes.
        label_leaf_mask = self.leaf_masks[labels, :]
        inf = torch.tensor(torch.inf, device=scores.device)
        logp_descendants = torch.where(label_leaf_mask, logp_leaf, -inf)
        logp_label = torch.logsumexp(logp_descendants, dim=-1)
        loss = -logp_label
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss


def flat_bertinetto_hxe(
        tree: hier.Hierarchy,
        scores: torch.Tensor,
        labels: torch.Tensor,
        alpha: float = 0.0,
        dim: int = -1) -> torch.Tensor:
    """The target is an index of a node in the tree."""
    device = scores.device
    # Take log-sum-exp of leaf descendants.
    parent = torch.from_numpy(tree.parents(root_loop=True)).to(device)
    parent_depth = torch.from_numpy(tree.depths() - 1).to(device)
    weight = torch.exp(-alpha * parent_depth)
    leaf_subset = torch.from_numpy(tree.leaf_subset()).to(device)
    scores_full = (
        torch.full((*scores.shape[:-1], tree.num_nodes()), -torch.inf, dtype=torch.float32)
        .index_copy(-1, leaf_subset, scores))
    logsumexp = descendant_logsumexp(tree, scores_full)
    log_cond_p = logsumexp - logsumexp[..., parent]
    # Take weighted sum over ancestors.
    # Weight each conditional likelihood by exp(-alpha * parent_depth).
    # Note that log_cond_p of root is always zero.
    assert dim in (-1, scores.ndim - 1)
    weighted_cond_nll = weight * -log_cond_p
    weighted_nll = sum_ancestors(tree, weighted_cond_nll, dim=dim, strict=False)
    assert labels.ndim == scores.ndim - 1
    label_nll = torch.gather(weighted_nll, dim, labels.unsqueeze(-1)).squeeze(-1)
    return torch.mean(label_nll)


class FlatBertinettoHXE(nn.Module):

    def __init__(self, tree: hier.Hierarchy, alpha: float, with_leaf_targets: bool, reduction: str = 'mean'):
        super().__init__()
        assert reduction in ('mean', 'none', None)
        paths = tree.paths_padded(method='constant', pad_value=-1, exclude_root=False)
        is_ancestor = tree.ancestor_mask(strict=False)
        leaf_masks = is_ancestor[:, tree.leaf_mask()]
        if with_leaf_targets:
            label_order = torch.from_numpy(tree.leaf_subset())
            paths = paths[label_order, :]

        self.alpha = alpha
        self.reduction = reduction
        self.max_depth = np.max(tree.depths())
        self.paths = torch.from_numpy(paths)
        self.leaf_masks = torch.from_numpy(leaf_masks)

    def _apply(self, fn):
        super()._apply(fn)
        self.paths = fn(self.paths)
        self.leaf_masks = fn(self.leaf_masks)
        return self

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        assert labels.ndim == scores.ndim - 1
        device = scores.device

        # TODO: Can use for loop if this uses too much memory (n d with d = log n).
        # Get leaf-descendant mask for all ancestors.
        label_path = self.paths[labels]
        path_valid = (label_path >= 0)
        label_ancestor_leaf_masks = torch.logical_and(
            path_valid.unsqueeze(-1),
            self.leaf_masks[torch.where(path_valid, label_path, 0), :])

        inf = torch.tensor(torch.inf, device=device)
        lse_ancestor = torch.logsumexp(
            torch.where(label_ancestor_leaf_masks, scores.unsqueeze(-2), -inf), dim=-1)
        # lse(parent) - lse(child)
        cond_nll = -torch.diff(lse_ancestor, dim=-1)
        cond_nll = torch.where(path_valid[:, 1:], cond_nll, torch.tensor(0.0, device=device))
        weight = torch.exp(-self.alpha * torch.arange(self.max_depth, device=device))
        weighted_nll = cond_nll * weight
        loss = torch.sum(weighted_nll, dim=-1)

        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss


def hier_softmax_nll(
        tree: hier.Hierarchy,
        scores: torch.Tensor,
        labels: torch.Tensor) -> torch.Tensor:
    log_prob = hier_log_softmax(tree, scores, dim=-1)
    assert labels.ndim == scores.ndim - 1
    nll = -torch.gather(log_prob, -1, labels.unsqueeze(-1)).squeeze(-1)
    return torch.mean(nll)


def hier_softmax_nll_with_leaf(
        tree: hier.Hierarchy,
        scores: torch.Tensor,
        labels: torch.Tensor) -> torch.Tensor:
    device = scores.device
    leaf_subset = torch.from_numpy(tree.leaf_subset()).to(device)
    # TODO: Could make this faster by only computing likelihood for targets.
    log_prob = hier_log_softmax(tree, scores, dim=-1)
    leaf_log_prob = torch.index_select(log_prob, -1, leaf_subset)
    assert labels.ndim == scores.ndim - 1
    nll = -torch.gather(leaf_log_prob, -1, labels.unsqueeze(-1)).squeeze(-1)
    return torch.mean(nll)


class HierSoftmaxNLL(nn.Module):
    """Avoids recomputation in hier_softmax_nll."""

    def __init__(
            self,
            tree: hier.Hierarchy,
            with_leaf_targets: bool = False):
        super().__init__()
        if with_leaf_targets:
            self.label_order = torch.from_numpy(tree.leaf_subset())
        else:
            self.label_order = None
        self.hier_log_softmax = HierLogSoftmax(tree)

    def _apply(self, fn):
        super()._apply(fn)
        self.label_order = _apply_to_maybe(fn, self.label_order)
        self.hier_log_softmax = self.hier_log_softmax._apply(fn)
        return self

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # TODO: Could make this faster by only computing likelihood for targets.
        log_prob = self.hier_log_softmax(scores, dim=-1)
        if self.label_order is not None:
            log_prob = torch.index_select(log_prob, -1, self.label_order)
        nll = -torch.gather(log_prob, -1, labels.unsqueeze(-1)).squeeze(-1)
        return torch.mean(nll)


class HierSoftmaxCrossEntropy(nn.Module):
    """Supports integer label targets or distribution targets."""

    def __init__(
            self,
            tree: hier.Hierarchy,
            with_leaf_targets: bool = False,
            label_smoothing: float = 0.0,
            node_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.label_smoothing = label_smoothing
        if with_leaf_targets:
            self.label_order = torch.from_numpy(tree.leaf_subset())
            self.num_labels = len(self.label_order)
        else:
            self.label_order = None
            self.num_labels = tree.num_nodes()
        self.hier_cond_log_softmax = HierCondLogSoftmax(tree)
        self.sum_label_descendants = SumDescendants(tree, subset=self.label_order)
        self.prior = torch.from_numpy(hier.uniform_leaf(tree))
        self.node_weight = node_weight

    def _apply(self, fn):
        super()._apply(fn)
        self.label_order = _apply_to_maybe(fn, self.label_order)
        self.hier_cond_log_softmax = self.hier_cond_log_softmax._apply(fn)
        self.sum_label_descendants = self.sum_label_descendants._apply(fn)
        self.prior = fn(self.prior)
        self.node_weight = _apply_to_maybe(fn, self.node_weight)
        return self

    def forward(self, scores: torch.Tensor, labels: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert labels.ndim in [scores.ndim, scores.ndim - 1]
        assert dim in (-1, scores.ndim - 1)
        # Convert labels to one-hot if they are not.
        if labels.ndim < scores.ndim:
            labels = F.one_hot(labels, self.num_labels)
        labels = labels.type(torch.get_default_dtype())
        q = self.sum_label_descendants(labels)
        if self.label_smoothing:
            q = (1 - self.label_smoothing) * q + self.label_smoothing * self.prior
        log_cond_p = self.hier_cond_log_softmax(scores, dim=-1)
        xent = q * -log_cond_p
        if self.node_weight is not None:
            xent = xent * self.node_weight
        xent = torch.sum(xent, dim=-1)
        return torch.mean(xent)


def hier_cond_log_softmax(
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
    assert dim == -1 or dim == scores.ndim - 1
    num_nodes = tree.num_nodes()
    num_internal = tree.num_internal_nodes()
    node_to_children = tree.children()
    cond_children = [node_to_children[x] for x in tree.internal_subset()]
    cond_num_children = list(map(len, cond_children))
    max_num_children = max(cond_num_children)
    # TODO: Use _split_and_pad?
    row_index = np.concatenate([np.full(n, i) for i, n in enumerate(cond_num_children)])
    col_index = np.concatenate([np.arange(n) for n in cond_num_children])
    flat_index = row_index * max_num_children + col_index
    child_index = np.concatenate(cond_children)

    device = scores.device
    flat_index = torch.from_numpy(flat_index).to(device)
    child_index = torch.from_numpy(child_index).to(device)
    input_shape = list(scores.shape)
    flat_shape = [*input_shape[:-1], num_internal * max_num_children]
    # Pad with -inf for log_softmax.
    # flat[..., flat_index] = scores
    flat = torch.full(flat_shape, -torch.inf, device=device).index_copy(
        -1, flat_index, scores)
    split_shape = [*input_shape[:-1], num_internal, max_num_children]
    child_scores = flat.reshape(split_shape)
    child_log_p = F.log_softmax(child_scores, dim=-1)
    child_log_p = child_log_p.reshape(flat_shape)
    output_shape = [*input_shape[:-1], num_nodes]
    # log_cond_p[..., child_index] = child_log_p[..., flat_index]
    log_cond_p = torch.zeros(output_shape, device=device).index_copy(
        -1, child_index, child_log_p.index_select(-1, flat_index))
    return log_cond_p


def hier_log_softmax(
        tree: hier.Hierarchy,
        scores: torch.Tensor,
        dim: int = -1) -> torch.Tensor:
    # Finally, take sum over ancestor conditionals to obtain likelihoods.
    assert dim in (-1, scores.ndim - 1)
    log_cond_p = hier_cond_log_softmax(tree, scores, dim=dim)
    # TODO: Use functional form here?
    device = scores.device
    sum_ancestors_fn = SumAncestors(tree, exclude_root=False).to(device)
    return sum_ancestors_fn(log_cond_p, dim=-1)


class HierCondLogSoftmax(nn.Module):
    """Returns log-likelihood of each node given its parent."""

    def __init__(self, tree: hier.Hierarchy):
        super().__init__()
        num_nodes = tree.num_nodes()
        num_internal = tree.num_internal_nodes()
        node_to_children = tree.children()
        cond_children = [node_to_children[x] for x in tree.internal_subset()]
        cond_num_children = list(map(len, cond_children))
        max_num_children = max(cond_num_children)
        # TODO: Use _split_and_pad?
        row_index = np.concatenate([np.full(n, i) for i, n in enumerate(cond_num_children)])
        col_index = np.concatenate([np.arange(n) for n in cond_num_children])
        flat_index = torch.from_numpy(row_index * max_num_children + col_index)
        child_index = torch.from_numpy(np.concatenate(cond_children))

        self.num_nodes = num_nodes
        self.num_internal = num_internal
        self.max_num_children = max_num_children
        self.flat_index = flat_index
        self.child_index = child_index

    def _apply(self, fn):
        super()._apply(fn)
        self.flat_index = fn(self.flat_index)
        self.child_index = fn(self.child_index)
        return self

    def forward(self, scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim in (-1, scores.ndim - 1)
        device = scores.device
        input_shape = list(scores.shape)
        flat_shape = [*input_shape[:-1], self.num_internal * self.max_num_children]
        # Pad with -inf for log_softmax.
        # flat[..., flat_index] = scores
        flat = torch.full(flat_shape, -torch.inf, device=device).index_copy(
            -1, self.flat_index, scores)
        split_shape = [*input_shape[:-1], self.num_internal, self.max_num_children]
        child_scores = flat.reshape(split_shape)
        child_log_p = F.log_softmax(child_scores, dim=-1)
        child_log_p = child_log_p.reshape(flat_shape)
        output_shape = [*input_shape[:-1], self.num_nodes]
        # log_cond_p[..., child_index] = child_log_p[..., flat_index]
        log_cond_p = torch.zeros(output_shape, device=device).index_copy(
            -1, self.child_index, child_log_p.index_select(-1, self.flat_index))
        return log_cond_p


class HierLogSoftmax(nn.Module):
    """Avoids re-computation in hier_log_softmax()."""

    def __init__(self, tree: hier.Hierarchy):
        super().__init__()
        self.cond_log_softmax = HierCondLogSoftmax(tree)
        self.sum_ancestors_fn = SumAncestors(tree, exclude_root=False)

    def _apply(self, fn):
        super()._apply(fn)
        self.cond_log_softmax = self.cond_log_softmax._apply(fn)
        self.sum_ancestors_fn = self.sum_ancestors_fn._apply(fn)
        return self

    def forward(self, scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
        log_cond_p = self.cond_log_softmax(scores, dim=dim)
        return self.sum_ancestors_fn(log_cond_p, dim=dim)


# class HierSoftmaxNLLWithInactive(nn.Module):
#     """Hierarchical softmax with loss for inactive nodes."""
#
#     def __init__(
#             self,
#             tree: hier.Hierarchy,
#             with_leaf_targets: bool = False):
#         super().__init__()
#         if with_leaf_targets:
#             self.label_order = torch.from_numpy(tree.leaf_subset())
#         else:
#             self.label_order = None
#         self.hier_log_softmax = HierLogSoftmax(tree)
#
#     def _apply(self, fn):
#         super()._apply(fn)
#         # Do not apply fn to indices because it might convert dtype.
#         self.label_order = _apply_to_maybe(fn, self.label_order)
#         self.hier_log_softmax = self.hier_log_softmax._apply(fn)
#         return self
#
#     def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
#         # TODO: Could make this faster by only computing likelihood for targets.
#         device = scores.device
#
#         # Log-sum-exp for each internal node.
#         self.logsumexp = ()
#
#         log_prob = self.hier_log_softmax(scores, dim=-1)
#         if self.label_order is not None:
#             log_prob = torch.index_select(log_prob, -1, self.label_order.to(device))
#         nll = -torch.gather(log_prob, -1, labels.unsqueeze(-1)).squeeze(-1)
#         return torch.mean(nll)


def leaf_put(
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
    assert dim in (-1, values.ndim - 1)
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
    assert dim in (-1, values.ndim - 1)
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
    assert dim in (-1, values.ndim - 1)
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
    assert dim in (-1, values.ndim - 1)
    return torch.tensordot(values, matrix, dims=1)



class Sum(nn.Module):
    """Avoids re-computation in sum_xxx()."""

    def __init__(
            self,
            tree: hier.Hierarchy,
            transpose: bool,
            subset: Optional[np.ndarray] = None,
            # leaf_only: bool = False,
            exclude_root: bool = False,
            strict: bool = False):
        super().__init__()
        # The value matrix[i, j] is true if i is an ancestor of j.
        # Take transpose for sum over descendants.
        matrix = tree.ancestor_mask(strict=strict)
        if subset is not None:
            matrix = matrix[:, subset]
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

    def forward(self, values: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # TODO: Re-order dimensions to make this work with dim != -1.
        assert dim in (-1, values.ndim - 1)
        return torch.tensordot(values, self.matrix, dims=1)


SumAncestors = partial(Sum, transpose=False)
SumDescendants = partial(Sum, transpose=True)
# SumLeafAncestors = partial(Sum, transpose=True, leaf_only=True)
# SumLeafDescendants = partial(Sum, transpose=True, leaf_only=True)


def SumLeafAncestors(
        tree: hier.Hierarchy,
        **kwargs):
    return SumAncestors(tree, subset=tree.leaf_mask(), **kwargs)


def SumLeafDescendants(
        tree: hier.Hierarchy,
        **kwargs):
    return SumDescendants(tree, subset=tree.leaf_mask(), **kwargs)


class MultiLabelNLL(nn.Module):

    def __init__(
            self,
            tree: hier.Hierarchy,
            with_leaf_targets: bool = False,
            include_root: bool = False,
            node_weight: Optional[torch.Tensor] = None):
        super().__init__()
        # The boolean array binary_targets[i, :] indicates whether
        # each node is an ancestor of node i (including i itself).
        # The root node is excluded since it is always positive.
        binary_targets = tree.ancestor_mask(strict=False).T
        if not include_root:
            binary_targets = binary_targets[:, 1:]
        if with_leaf_targets:
            binary_targets = binary_targets[tree.leaf_subset(), :]

        if node_weight is not None:
            if not include_root:
                node_weight = node_weight[:, 1:]

        dtype = torch.get_default_dtype()
        self.binary_targets = torch.from_numpy(binary_targets).type(dtype)
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.node_weight = _apply_to_maybe(torch.from_numpy, node_weight)

    def _apply(self, fn):
        super()._apply(fn)
        self.binary_targets = fn(self.binary_targets)
        self.bce_loss = self.bce_loss._apply(fn)
        self.node_weight = _apply_to_maybe(fn, self.node_weight)
        return self

    def forward(self, scores: torch.Tensor, labels: torch.Tensor, dim: int = -1) -> torch.Tensor:
        targets = torch.index_select(self.binary_targets, 0, labels)
        node_loss = self.bce_loss(scores, targets)
        # Reduce over classes.
        if self.node_weight is not None:
            node_loss = node_loss * self.node_weight
        loss = torch.sum(node_loss, dim=dim)
        # Take mean over examples.
        return torch.mean(loss)


class MultiLabelFocalLoss(nn.Module):

    def __init__(
            self,
            tree: hier.Hierarchy,
            alpha: float,
            gamma: float,
            with_leaf_targets: bool = False,
            include_root: bool = False,
            weighting_strategy: str = 'none'):
        """
        It may be useful to have both alpha and weighting_strategy.
        With alpha = 0.5, node_weight = 1, have huge initial loss with most for negative.
        With alpha = 0.9, node_weight = 1, have log(n) nodes with most weight per example.
        With alpha = 0.5, node_weight = 1/size, have equal weight for all nodes on average.
        With alpha = 0.9, node_weight = 1/size, equal weight is given to nodes at all levels.
        """
        super().__init__()
        # The boolean array binary_targets[i, :] indicates whether
        # each node is an ancestor of node i (including i itself).
        # The root node is excluded since it is always positive.
        binary_targets = tree.ancestor_mask(strict=False).T
        if not include_root:
            binary_targets = binary_targets[:, 1:]
        if with_leaf_targets:
            binary_targets = binary_targets[tree.leaf_subset(), :]

        node_weight = make_loss_weights(weighting_strategy, tree, exclude_root=True)

        dtype = torch.get_default_dtype()
        self.binary_targets = torch.from_numpy(binary_targets).type(dtype)
        self.alpha = alpha
        self.gamma = gamma
        self.node_weight = _apply_to_maybe(torch.from_numpy, node_weight)

    def _apply(self, fn):
        super()._apply(fn)
        self.binary_targets = fn(self.binary_targets)
        self.node_weight = _apply_to_maybe(fn, self.node_weight)
        return self

    def forward(self, scores: torch.Tensor, labels: torch.Tensor, dim: int = -1) -> torch.Tensor:
        targets = torch.index_select(self.binary_targets, 0, labels)
        binary_loss = torchvision.ops.sigmoid_focal_loss(
            scores, targets, alpha=self.alpha, gamma=self.gamma, reduction='none')
        # Reduce over classes.
        if self.node_weight is not None:
            binary_loss = binary_loss * self.node_weight
        loss = torch.sum(binary_loss, dim=dim)
        # Take mean over examples.
        return torch.mean(loss)


def multilabel_log_likelihood(
        scores: torch.Tensor,
        dim: int = -1,
        insert_root: bool = False,
        replace_root: bool = False) -> torch.Tensor:
    assert not (insert_root and replace_root)
    assert dim in (-1, scores.ndim - 1)
    device = scores.device
    logp = F.logsigmoid(scores)
    if insert_root:
        zero = torch.zeros((*scores.shape[:-1], 1), device=device)
        logp = torch.cat([zero, logp], dim=-1)
    elif replace_root:
        zero = torch.zeros((*scores.shape[:-1], 1), device=device)
        logp = logp.index_copy(-1, torch.tensor([0], device=device), zero)
    return logp


class HierLogSigmoid(nn.Module):

    def __init__(self, tree: hier.Hierarchy):
        super().__init__()
        self.log_sigmoid = torch.nn.LogSigmoid()
        self.sum_ancestors = SumAncestors(tree, exclude_root=True)

    def _apply(self, fn):
        super()._apply(fn)
        self.log_sigmoid = self.log_sigmoid._apply(fn)
        self.sum_ancestors = self.sum_ancestors._apply(fn)
        return self

    def forward(self, scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
        log_cond = self.log_sigmoid(scores)
        return self.sum_ancestors(log_cond, dim=dim)


# class HierSigmoidNLLLoss(nn.Module):
#
#     def __init__(self, tree: hier.Hierarchy, with_leaf_targets: bool = False):
#         super().__init__()
#         self.log_sigmoid = torch.nn.LogSigmoid()
#         subset = tree.leaf_mask() if with_leaf_targets else None
#         self.sum_ancestors = SumAncestors(tree, exclude_root=True, strict=True)
#         self.parents = torch.from_numpy(tree.parents()[1:])
#
#     def _apply(self, fn):
#         super()._apply(fn)
#         self.log_sigmoid = self.log_sigmoid._apply(fn)
#         self.sum_ancestors = self.sum_ancestors._apply(fn)
#         return self
#
#     def forward(self, scores: torch.Tensor, labels: torch.Tensor, dim: int = -1) -> torch.Tensor:
#         logp_node_given_parent = self.log_sigmoid(scores)
#         logp_not_node_given_parent = self.log_sigmoid(-scores)
#         logp_parent = self.sum_ancestors(logp_node_given_parent, dim=dim)
#         logp_node = logp_parent + logp_node_given_parent  # TODO: Insert root node.
#         logp_parent_and_not_node = logp_node[self.parents] + logp_not_node_given_parent
#         # Need logsumexp_ancestors!
#         logp_not_node = self.sum_ancestors(torch.exp(logp_parent_and_not_node), dim=dim)


def hier_softmax_cross_entropy(
        tree: hier.Hierarchy,
        scores: torch.Tensor,
        q: torch.Tensor,
        dim: int = -1) -> torch.Tensor:
    """The target distribution q should be hierarchical."""
    # Get conditional likelihoods for each node given its parent.
    log_cond_p = hier_cond_log_softmax(tree, scores, dim=dim)
    xent = torch.sum(q * -log_cond_p, dim=dim)
    return torch.mean(xent)


def bertinetto_hxe(
        tree: hier.Hierarchy,
        scores: torch.Tensor,
        labels: torch.Tensor,
        alpha: float = 0.0,
        dim: int = -1) -> torch.Tensor:
    """The target is an index of a node in the tree."""
    device = scores.device
    # Get conditional likelihoods for each node given its parent.
    log_cond_p = hier_cond_log_softmax(tree, scores, dim=dim)
    # Take weighted sum over ancestors.
    # Weight each conditional likelihood by exp(-alpha * parent_depth).
    # Note that log_cond_p of root is always zero.
    parent_depth = torch.from_numpy(tree.depths() - 1)
    weight = torch.exp(-alpha * parent_depth).to(device)
    assert dim in (-1, scores.ndim - 1)
    weighted_cond_nll = -weight * log_cond_p
    weighted_nll = sum_ancestors(tree, weighted_cond_nll, dim=dim, strict=False)
    assert labels.ndim == scores.ndim - 1
    label_nll = torch.gather(weighted_nll, dim, labels.unsqueeze(-1)).squeeze(-1)
    return torch.mean(label_nll)


class BertinettoHXE(nn.Module):
    """Avoids re-computation in bertinetto_hxe()."""

    def __init__(self,
                 tree: hier.Hierarchy,
                 alpha: float = 0.,
                 with_leaf_targets: bool = False):
        super().__init__()
        self.hier_cond_log_softmax_fn = HierCondLogSoftmax(tree)
        parent_depth = torch.from_numpy(tree.depths() - 1)
        self.weight = torch.exp(-alpha * parent_depth)
        if with_leaf_targets:
            self.label_order = torch.from_numpy(tree.leaf_subset())
        else:
            self.label_order = None
        self.sum_ancestors_fn = SumAncestors(tree, strict=False)

    def _apply(self, fn):
        super()._apply(fn)
        self.weight = fn(self.weight)
        self.hier_cond_log_softmax_fn = self.hier_cond_log_softmax_fn._apply(fn)
        self.label_order = _apply_to_maybe(fn, self.label_order)
        self.sum_ancestors_fn = self.sum_ancestors_fn._apply(fn)
        return self

    def forward(self,
                scores: torch.Tensor,
                labels: torch.Tensor,
                dim: int = -1) -> torch.Tensor:
        log_cond_p = self.hier_cond_log_softmax_fn(scores, dim=dim)
        assert dim in (-1, scores.ndim - 1)
        # Take weighted sum over ancestors.
        # Weight each conditional likelihood by exp(-alpha * parent_depth).
        # Note that log_cond_p of root is always zero.
        weighted_cond_nll = -self.weight * log_cond_p
        weighted_nll = self.sum_ancestors_fn(weighted_cond_nll, dim=dim)
        assert labels.ndim == scores.ndim - 1
        if self.label_order is not None:
            weighted_nll = torch.index_select(weighted_nll, dim, self.label_order)
        label_nll = torch.gather(weighted_nll, dim, labels.unsqueeze(-1)).squeeze(-1)
        return torch.mean(label_nll)


def levelwise_softmax_nll(
        tree: hier.Hierarchy,
        scores: torch.Tensor,
        labels: torch.Tensor,
        with_leaf_targets: bool) -> torch.Tensor:
    level_nodes = hier.level_nodes(tree, extend=True)
    level_sizes = tuple(map(len, level_nodes))
    num_levels = len(level_nodes)
    # Construct map from node to index within softmax at each level.
    node_to_level_target = np.full([num_levels, tree.num_nodes()], -1, dtype=int)
    for i in range(num_levels):
        node_to_level_target[i, level_nodes[i]] = np.arange(level_sizes[i])

    # Get label set.
    if with_leaf_targets:
        label_to_node = tree.leaf_subset()
    else:
        label_to_node = np.arange(tree.num_nodes())
    # Find mapping from label to index within softmax (of ancestor) at each level.
    paths = tree.paths_padded(method='self', exclude_root=True)
    label_to_level_target = np.full([num_levels, len(label_to_node)], -1, dtype=int)
    for i in range(num_levels):
        label_to_level_target[i, :] = node_to_level_target[i, paths[label_to_node, i]]
        # Every label should correspond to a valid target.
        # Note that this will fail for non-leaf targets.
        # TODO: Implement no loss for descendants of label? (for non-leaf targets)
        assert np.all(label_to_level_target[i, :] >= 0)
        assert np.all(label_to_level_target[i, :] < level_sizes[i])

    label_to_level_target = torch.from_numpy(label_to_level_target).to(scores.device)

    level_scores = torch.split(scores, level_sizes, dim=-1)
    level_targets = label_to_level_target[:, labels].unbind(0)
    # Dense, padded implementation:
    # level_scores = _split_and_pad(scores, level_sizes, -torch.inf, dim=-1)
    # level_logp = F.log_softmax(level_scores, dim=-1)
    # level_nll = -_gather_2d(level_logp, j=level_targets)
    level_nll = torch.stack(
        [F.cross_entropy(x, y, reduction='none') for x, y in zip(level_scores, level_targets)],
        dim=-1)
    mean_nll = torch.sum(level_nll, dim=-1)
    return torch.mean(mean_nll)


class LevelwiseSoftmaxNLL(nn.Module):

    def __init__(self, tree: hier.Hierarchy, with_leaf_targets: bool = False):
        super().__init__()
        level_nodes = hier.level_nodes(tree, extend=True)
        level_sizes = tuple(map(len, level_nodes))
        num_levels = len(level_nodes)
        # Construct map from node to index within softmax at each level.
        node_to_level_target = np.full([num_levels, tree.num_nodes()], -1, dtype=int)
        for i in range(num_levels):
            node_to_level_target[i, level_nodes[i]] = np.arange(level_sizes[i])

        # Get label set.
        if with_leaf_targets:
            label_to_node = tree.leaf_subset()
        else:
            label_to_node = np.arange(tree.num_nodes())
        # Find mapping from label to index within softmax (of ancestor) at each level.
        paths = tree.paths_padded(method='self', exclude_root=True)
        label_to_level_target = np.full([num_levels, len(label_to_node)], -1, dtype=int)
        for i in range(num_levels):
            label_to_level_target[i, :] = node_to_level_target[i, paths[label_to_node, i]]
            # Every label should correspond to a valid target.
            # Note that this will fail for non-leaf targets.
            # TODO: Implement no loss for descendants of label? (for non-leaf targets)
            assert np.all(label_to_level_target[i, :] >= 0)
            assert np.all(label_to_level_target[i, :] < level_sizes[i])

        self.level_sizes = level_sizes
        self.label_to_level_target = torch.from_numpy(label_to_level_target)

    def _apply(self, fn):
        super()._apply(fn)
        self.label_to_level_target = fn(self.label_to_level_target)
        return self

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        level_scores = torch.split(scores, self.level_sizes, dim=-1)
        level_targets = self.label_to_level_target[:, labels].unbind(0)
        # Dense, padded implementation:
        # level_scores = _split_and_pad(scores, level_sizes, -torch.inf, dim=-1)
        # level_logp = F.log_softmax(level_scores, dim=-1)
        # level_nll = -_gather_2d(level_logp, j=level_targets)
        level_nll = torch.stack(
            [F.cross_entropy(x, y, reduction='none') for x, y in zip(level_scores, level_targets)],
            dim=-1)
        mean_nll = torch.sum(level_nll, dim=-1)
        return torch.mean(mean_nll)


def levelwise_log_softmax(tree: hier.Hierarchy, scores: torch.Tensor) -> torch.Tensor:
    batch_dims = tuple(scores.shape[:-1])
    num_nodes = tree.num_nodes()
    level_nodes = hier.level_nodes(tree, extend=True)
    level_sizes = tuple(map(len, level_nodes))

    level_scores = torch.split(scores, level_sizes, dim=-1)
    level_logp = [F.log_softmax(x, dim=-1) for x in level_scores]
    # Take probability of each node at its level.
    # Note that node may appear at multiple levels.
    # Perform in reverse order such that shallow overwrites deep.
    logp = torch.full((*batch_dims, num_nodes), -torch.inf, dtype=scores.dtype, device=scores.device)
    for i in reversed(range(len(level_nodes))):
        logp = logp.index_copy(-1, level_nodes[i], level_logp[i])
    logp = logp.index_fill(-1, torch.zeros((), dtype=int, device=scores.device), 0.)
    return logp


class LevelwiseLogSoftmax(nn.Module):

    def __init__(self, tree: hier.Hierarchy):
        super().__init__()
        num_nodes = tree.num_nodes()
        level_nodes = hier.level_nodes(tree, extend=True)
        level_sizes = tuple(map(len, level_nodes))

        self.num_nodes = num_nodes
        self.level_sizes = level_sizes
        self.level_nodes = list(map(torch.from_numpy, level_nodes))

    def _apply(self, fn):
        super()._apply(fn)
        self.level_nodes = list(map(fn, self.level_nodes))
        return self

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        num_levels = len(self.level_nodes)
        batch_dims = tuple(scores.shape[:-1])
        level_scores = torch.split(scores, self.level_sizes, dim=-1)
        level_logp = [F.log_softmax(x, dim=-1) for x in level_scores]
        # Take probability of each node at its level.
        # Note that node may appear at multiple levels.
        # Perform in reverse order such that shallow overwrites deep.
        logp = torch.full((*batch_dims, self.num_nodes), -torch.inf, dtype=scores.dtype, device=scores.device)
        for i in reversed(range(num_levels)):
            logp = logp.index_copy(-1, self.level_nodes[i], level_logp[i])
        logp = logp.index_fill(-1, torch.zeros((), dtype=int, device=scores.device), 0.)
        return logp


# def descendant_max(tree: hier.Hierarchy, x: torch.Tensor) -> torch.Tensor:
#     # TODO: Avoid high memory usage.
#     # Maybe this could be done with a (parent[i], children[i, j]) array per depth,
#     # where children is padded.
#     is_ancestor = torch.from_numpy(tree.ancestor_mask()).to(device=x.device)
#     neg_inf = torch.full((), -torch.inf, dtype=x.dtype, device=x.device)
#     descendant_values = torch.where(is_ancestor, x.unsqueeze(-1), neg_inf)
#     max_value, _ = torch.max(descendant_values, dim=-1)
#     return max_value


def descendant_max(tree: hier.Hierarchy, x: torch.Tensor) -> torch.Tensor:
    """Finds the max over all descendants of each node."""
    level_parents, level_children = hier.level_successors_padded(tree, method='last')
    level_parents = list(map(torch.from_numpy, level_parents))
    level_children = list(map(torch.from_numpy, level_children))
    num_levels = len(level_parents)
    out = x
    # Work up from the maximum depth, taking max of node and its children.
    for i in reversed(range(num_levels)):
        # child_values = _index_select(out, -1, level_children[i])
        child_values = out[..., level_children[i]]
        max_child_value, _ = torch.max(child_values, dim=-1)
        out = out.index_copy(
            -1, level_parents[i],
            torch.maximum(out[..., level_parents[i]], max_child_value))
    return out


class DescendantMax(nn.Module):
    """Finds the max over all descendants of each node."""

    def __init__(self, tree: hier.Hierarchy):
        super().__init__()
        level_parents, level_children = hier.level_successors_padded(tree, method='last')
        self.level_parents = list(map(torch.from_numpy, level_parents))
        self.level_children = list(map(torch.from_numpy, level_children))

    def _apply(self, fn):
        super()._apply(fn)
        self.level_parents = list(map(fn, self.level_parents))
        self.level_children = list(map(fn, self.level_children))
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_levels = len(self.level_parents)
        out = x
        # Work up from the maximum depth, taking max of node and its children.
        for i in reversed(range(num_levels)):
            # child_values = _index_select(out, -1, level_children[i])
            child_values = out[..., self.level_children[i]]
            max_child_value, _ = torch.max(child_values, dim=-1)
            out = out.index_copy(
                -1, self.level_parents[i],
                torch.maximum(out[..., self.level_parents[i]], max_child_value))
        return out


def descendant_logsumexp(tree: hier.Hierarchy, x: torch.Tensor) -> torch.Tensor:
    """Finds the max over all descendants of each node."""
    level_parents, level_children = hier.level_successors_padded(tree, 'constant', -1)
    level_parents = list(map(torch.from_numpy, level_parents))
    level_children = list(map(torch.from_numpy, level_children))
    num_levels = len(level_parents)
    out = x
    neg_inf = torch.tensor(-torch.inf, dtype=torch.float32)
    # Work up from the maximum depth, taking max of node and its children.
    for i in reversed(range(num_levels)):
        # child_values = _index_select(out, -1, level_children[i])
        valid = level_children[i] >= 0
        child_values = out[..., torch.where(valid, level_children[i], 0)]
        max_child_value = torch.logsumexp(torch.where(valid, child_values, neg_inf), dim=-1)
        # TODO: Add option for leaf nodes only.
        out = out.index_copy(
            -1, level_parents[i],
            _logsumexp2(out[..., level_parents[i]], max_child_value))
    return out


class DescendantLogSumExp(nn.Module):
    """Finds the logsumexp over all descendants of each node."""

    # TODO: This is the same as MaxCutLogSumExp with max_reduction='logsumexp'.

    def __init__(self, tree: hier.Hierarchy):
        super().__init__()
        level_parents, level_children = hier.level_successors_padded(tree, 'constant', -1)
        self.level_parents = list(map(torch.from_numpy, level_parents))
        self.level_children = list(map(torch.from_numpy, level_children))

    def _apply(self, fn):
        super()._apply(fn)
        self.level_parents = list(map(fn, self.level_parents))
        self.level_children = list(map(fn, self.level_children))
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_levels = len(self.level_parents)
        neg_inf = torch.tensor(-torch.inf, dtype=torch.float32, device=x.device)
        out = x
        # Work up from the maximum depth, taking max of node and its children.
        for i in reversed(range(num_levels)):
            valid = self.level_children[i] >= 0
            # child_values = out[..., self.level_children[i]]
            # max_child_value = torch.logsumexp(child_values, dim=-1)
            child_values = out[..., torch.where(valid, self.level_children[i], 0)]
            max_child_value = torch.logsumexp(torch.where(valid, child_values, neg_inf), dim=-1)
            out = out.index_copy(
                -1, self.level_parents[i],
                _logsumexp2(out[..., self.level_parents[i]], max_child_value))
        return out


def _index_select(x: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    """Like Tensor.index_select, but supports multi-dimensional index.

    Note: When dim is -1, can use Tensor[..., index] instead.
    """
    index_shape = tuple(index.shape)
    input_shape = tuple(x.shape)
    flat_index = index.reshape(-1)
    flat_out = x.index_select(dim, flat_index)
    out_shape = input_shape[:dim] + index_shape + input_shape[dim:][1:]
    out = flat_out.reshape(*out_shape)
    return out


def _split_and_pad(x: torch.Tensor, sizes: List[int], pad_value, dim: int = -1) -> torch.Tensor:
    assert dim in (-1, x.ndim - 1)
    assert sum(sizes) == x.shape[-1]
    max_size = max(sizes)
    row_index = np.concatenate([np.full(n, i) for i, n in enumerate(sizes)])
    col_index = np.concatenate([np.arange(n) for n in sizes])
    flat_index = row_index * max_size + col_index

    input_shape = tuple(x.shape)
    flat_shape = (*input_shape[:-1], len(sizes) * max_size)
    split_shape = (*input_shape[:-1], len(sizes), max_size)
    return (
        torch.full(flat_shape, pad_value, device=x.device)
        .index_copy(-1, flat_index, x)
        .reshape(split_shape))


def _gather_2d(
        x: torch.Tensor,
        i: Optional[np.ndarray] = None,
        j: Optional[np.ndarray] = None):
    """Performs gather on the last two dimensions."""
    assert x.ndim >= 2
    input_shape = tuple(x.shape)
    m, n = input_shape[-2:]
    i = np.arange(m) if i is None else i
    j = np.arange(j) if j is None else j
    assert np.all(0 <= i)
    assert np.all(i < m)
    assert np.all(0 <= j)
    assert np.all(j < n)
    flat_shape = (*input_shape[:-2], m * n)
    return x.reshape(x, flat_shape).gather(-1, i * n + j)


def max_cut_logsumexp(
        tree: hier.Hierarchy,
        scores: torch.Tensor,
        max_reduction: str = 'max') -> torch.Tensor:
    """Finds f(node) = max(theta[node], logsumexp(f(children))."""
    max_reduce_fn = get_reduce_fn(max_reduction)
    parents, children = hier.level_successors(tree)
    num_levels = len(parents)
    num_parents = [len(parents[d]) for d in range(num_levels)]
    num_children = [[len(ch) for ch in children[d]] for d in range(num_levels)]
    max_num_children = [max(num_children[d]) for d in range(num_levels)]
    flat_inputs_index = [None] * num_levels
    for d in range(num_levels):
        i, j = ragged_to_sparse_index(num_children[d])
        flat_inputs_index[d] = i * max_num_children[d] + j
    concat_children = [np.concatenate(children[d]) for d in range(num_levels)]

    parents = list(map(torch.from_numpy, parents))
    flat_inputs_index = list(map(torch.from_numpy, flat_inputs_index))
    concat_children = list(map(torch.from_numpy, concat_children))

    fn = partial(torch.Tensor.to, device=scores.device)
    parents = list(map(fn, parents))
    flat_inputs_index = list(map(fn, flat_inputs_index))
    concat_children = list(map(fn, concat_children))

    batch_dims = tuple(scores.shape[:-1])
    out = scores
    for d in reversed(range(num_levels)):
        flat_inputs = torch.full(
            (*batch_dims, num_parents[d] * max_num_children[d]),
            -torch.inf, device=scores.device)
        flat_inputs = flat_inputs.index_copy(
            -1, flat_inputs_index[d], out[..., concat_children[d]])
        inputs = flat_inputs.reshape(*batch_dims, num_parents[d], max_num_children[d])
        logsumexp = torch.logsumexp(inputs, dim=-1)
        out = out.index_copy(
            -1, parents[d], max_reduce_fn(out[..., parents[d]], logsumexp))

    return out


def get_reduce_fn(reduction: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if reduction == 'max':
        return torch.maximum
    elif reduction == 'logsumexp':
        return _logsumexp2
    else:
        raise ValueError('unknown max reduction', reduction)


class MaxCutLogSumExp(nn.Module):

    def __init__(self, tree: hier.Hierarchy, max_reduction='max'):
        super().__init__()
        parents, children = hier.level_successors(tree)
        num_levels = len(parents)
        num_parents = [len(parents[d]) for d in range(num_levels)]
        num_children = [[len(ch) for ch in children[d]] for d in range(num_levels)]
        max_num_children = [max(num_children[d]) for d in range(num_levels)]
        flat_inputs_index = [None] * num_levels
        for d in range(num_levels):
            i, j = ragged_to_sparse_index(num_children[d])
            flat_inputs_index[d] = i * max_num_children[d] + j
        concat_children = [np.concatenate(children[d]) for d in range(num_levels)]

        self.num_parents = num_parents
        # self.num_children = num_children
        self.max_num_children = max_num_children
        self.parents = list(map(torch.from_numpy, parents))
        self.flat_inputs_index = list(map(torch.from_numpy, flat_inputs_index))
        self.concat_children = list(map(torch.from_numpy, concat_children))
        self.max_reduce_fn = get_reduce_fn(max_reduction)

    def _apply(self, fn):
        super()._apply(fn)
        self.parents = list(map(fn, self.parents))
        self.flat_inputs_index = list(map(fn, self.flat_inputs_index))
        self.concat_children = list(map(fn, self.concat_children))
        return self

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """Finds the max over all descendants of each node."""
        num_levels = len(self.parents)
        batch_dims = tuple(scores.shape[:-1])
        out = scores
        for d in reversed(range(num_levels)):
            flat_inputs = torch.full(
                (*batch_dims, self.num_parents[d] * self.max_num_children[d]),
                -torch.inf, device=scores.device)
            flat_inputs = flat_inputs.index_copy(
                -1, self.flat_inputs_index[d], out[..., self.concat_children[d]])
            inputs = flat_inputs.reshape(*batch_dims, self.num_parents[d], self.max_num_children[d])
            logsumexp = torch.logsumexp(inputs, dim=-1)
            out = out.index_copy(
                -1, self.parents[d], self.max_reduce_fn(out[..., self.parents[d]], logsumexp))
        return out


def max_cut_log_softmax(
        tree: hier.Hierarchy,
        scores: torch.Tensor,
        max_reduction: str = 'max') -> torch.Tensor:
    """Finds the max over all descendants of each node."""
    log_z = max_cut_logsumexp(tree, scores, max_reduction=max_reduction)
    return log_z - log_z[..., [0]]


class DescendantLogSoftmax(nn.Module):

    def __init__(self, tree: hier.Hierarchy, max_reduction: str = 'logsumexp'):
        super().__init__()
        self.max_cut_logsumexp = MaxCutLogSumExp(tree, max_reduction=max_reduction)

    def _apply(self, fn):
        super()._apply(fn)
        self.max_cut_logsumexp = self.max_cut_logsumexp._apply(fn)
        return self

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        log_z = self.max_cut_logsumexp(scores)
        return log_z - log_z[..., [0]]


def max_cut_softmax_loss(
        tree: hier.Hierarchy,
        scores: torch.Tensor,
        labels: torch.Tensor,
        with_leaf_targets: bool,
        max_reduction: str = 'max') -> torch.Tensor:
    label_paths = tree.paths_padded()
    label_depths = tree.depths()
    if with_leaf_targets:
        label_order = tree.leaf_subset()
        label_paths = label_paths[label_order, :]
        label_depths = label_depths[label_order]

    label_paths = torch.from_numpy(label_paths).to(scores.device)
    label_depths = torch.from_numpy(label_depths).to(scores.device)

    # Get energy for each node.
    log_energy = max_cut_logsumexp(tree, scores, max_reduction=max_reduction)
    # Get log partition function (root energy).
    log_z = log_energy[..., 0]
    depth = label_depths[labels]
    ancestors = label_paths[labels]
    ancestors_valid = torch.where(ancestors >= 0, ancestors, 0)
    zero = torch.zeros((), device=scores.device)
    sum_ancestor_scores = torch.sum(
        torch.where(ancestors >= 0, scores.gather(-1, ancestors_valid), zero),
        dim=-1)
    num_ancestors = depth + 1
    # loss = torch.sqrt(num_ancestors) * (log_z - sum_ancestor_scores / num_ancestors)
    loss = num_ancestors * log_z - sum_ancestor_scores
    return torch.mean(loss)


class DescendantSoftmaxLoss(nn.Module):

    def __init__(
            self,
            tree: hier.Hierarchy,
            with_leaf_targets: bool,
            max_reduction: str = 'logsumexp',
            node_weight: Optional[torch.Tensor] = None,
            focal_power: Optional[float] = None,
            reduction: str = 'mean'):
        super().__init__()
        assert reduction in ('mean', 'none', None)
        label_paths = tree.paths_padded()
        label_depths = tree.depths()
        if with_leaf_targets:
            label_order = tree.leaf_subset()
            label_paths = label_paths[label_order, :]
            label_depths = label_depths[label_order]
        self.label_paths = torch.from_numpy(label_paths)
        self.label_depths = torch.from_numpy(label_depths)
        self.max_cut_logsumexp = MaxCutLogSumExp(tree, max_reduction=max_reduction)
        self.node_weight = node_weight
        self.focal_power = focal_power
        self.reduction = reduction

    def _apply(self, fn):
        super()._apply(fn)
        self.label_paths = fn(self.label_paths)
        self.label_depths = fn(self.label_depths)
        self.max_cut_logsumexp = self.max_cut_logsumexp._apply(fn)
        self.node_weight = _apply_to_maybe(fn, self.node_weight)
        return self

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Get energy for each node.
        log_energy = self.max_cut_logsumexp(scores)
        # Get log partition function (root energy).
        log_z = log_energy[..., 0]
        ancestors = self.label_paths[labels]
        # depth = self.label_depths[labels]
        mask = (ancestors >= 0)
        ancestors = torch.where(mask, ancestors, 0)
        ancestor_scores = scores.gather(-1, ancestors)
        loss = log_z.unsqueeze(-1) - ancestor_scores
        if self.focal_power:
            loss = loss * (1 - torch.exp(-loss)) ** self.focal_power
        if self.node_weight is not None:
            loss = loss * self.node_weight[ancestors]
        loss = torch.where(mask, loss, torch.tensor(0., device=scores.device))
        loss = torch.sum(loss, dim=-1)
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss


def max_cut_softmax_complement_loss(
        tree: hier.Hierarchy,
        scores: torch.Tensor,
        labels: torch.Tensor,
        with_leaf_targets: bool,
        max_reduction: str = 'max') -> torch.Tensor:
    device = scores.device

    node_siblings = hier.siblings_padded(tree, method='constant', constant_value=-1)
    label_paths = tree.paths_padded()
    if with_leaf_targets:
        label_order = tree.leaf_subset()
        label_paths = label_paths[label_order, :]

    label_paths = torch.from_numpy(label_paths).to(device)
    node_siblings = torch.from_numpy(node_siblings).to(device)

    # Get ancestors of the label.
    ancestors = label_paths[labels]
    ancestor_mask = (ancestors >= 0)
    ancestors = torch.where(ancestor_mask, ancestors, 0)
    # Get energy of siblings of each ancestor of the label.
    # siblings has shape (batch, depth, sibling)
    siblings = node_siblings[ancestors]
    sibling_mask = (siblings >= 0)
    siblings = torch.where(sibling_mask, siblings, 0)

    # Get energy of all nodes.
    log_energy = max_cut_logsumexp(tree, scores, max_reduction=max_reduction)
    *batch_dims, m, n = siblings.shape
    sibling_energy = log_energy.gather(-1, siblings.reshape(*batch_dims, -1)).reshape(*batch_dims, m, n)
    ancestor_logsumexp = torch.logsumexp(
        torch.where(sibling_mask, sibling_energy, torch.tensor(-torch.inf, device=device)),
        dim=-1)

    # Get logsumexp of complement.
    sum_ancestor_logsumexp = torch.logcumsumexp(ancestor_logsumexp, axis=-1)
    ancestor_scores = scores.gather(-1, ancestors)
    log_z = _logsumexp2(ancestor_scores, sum_ancestor_logsumexp)
    ancestor_loss = log_z - ancestor_scores
    loss = torch.sum(torch.where(ancestor_mask, ancestor_loss, torch.tensor(0., device=device)), -1)
    return torch.mean(loss)


class DescendantSoftmaxCousinLoss(nn.Module):

    def __init__(
            self,
            tree: hier.Hierarchy,
            with_leaf_targets: bool,
            max_reduction: str = 'logsumexp',
            node_weight: Optional[torch.Tensor] = None,
            reduction: str = 'mean'):
        super().__init__()
        assert reduction in ('mean', 'none', None)
        node_siblings = hier.siblings_padded(tree, method='constant', constant_value=-1)
        label_paths = tree.paths_padded()
        if with_leaf_targets:
            label_order = tree.leaf_subset()
            label_paths = label_paths[label_order, :]

        self.node_siblings = torch.from_numpy(node_siblings)
        self.label_paths = torch.from_numpy(label_paths)
        self.max_cut_logsumexp = MaxCutLogSumExp(tree, max_reduction=max_reduction)
        self.node_weight = node_weight
        self.reduction = reduction

    def _apply(self, fn):
        super()._apply(fn)
        self.node_siblings = fn(self.node_siblings)
        self.label_paths = fn(self.label_paths)
        self.max_cut_logsumexp = self.max_cut_logsumexp._apply(fn)
        self.node_weight = _apply_to_maybe(fn, self.node_weight)
        return self

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = scores.device
        # Get ancestors of the label.
        ancestors = self.label_paths[labels]
        ancestor_mask = (ancestors >= 0)
        ancestors = torch.where(ancestor_mask, ancestors, 0)
        # Get energy of siblings of each ancestor of the label.
        # siblings has shape (batch, depth, sibling)
        siblings = self.node_siblings[ancestors]
        sibling_mask = (siblings >= 0)
        siblings = torch.where(sibling_mask, siblings, 0)

        # Get energy of all nodes.
        log_energy = self.max_cut_logsumexp(scores)
        *batch_dims, m, n = siblings.shape
        sibling_energy = log_energy.gather(-1, siblings.reshape(*batch_dims, -1)).reshape(*batch_dims, m, n)
        ancestor_logsumexp = torch.logsumexp(
            torch.where(sibling_mask, sibling_energy, torch.tensor(-torch.inf, device=device)),
            dim=-1)

        # Get logsumexp of complement.
        sum_ancestor_logsumexp = torch.logcumsumexp(ancestor_logsumexp, axis=-1)
        ancestor_scores = scores.gather(-1, ancestors)
        log_z = _logsumexp2(ancestor_scores, sum_ancestor_logsumexp)
        ancestor_loss = log_z - ancestor_scores
        if self.node_weight is not None:
            ancestor_loss = ancestor_loss * self.node_weight[ancestors]
        loss = torch.sum(torch.where(ancestor_mask, ancestor_loss, torch.tensor(0., device=device)), -1)

        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss


def _logsumexp2(a, b):
    # return torch.log(torch.exp(a) + torch.exp(b))
    z = torch.maximum(a, b)
    return z + torch.log(torch.exp(a - z) + torch.exp(b - z))


def ragged_to_sparse_index(row_lens: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    row = [np.full(row_len, i) for i, row_len in enumerate(row_lens)]
    col = [np.arange(row_len) for row_len in row_lens]
    concat_row = np.concatenate(row)
    concat_col = np.concatenate(col)
    return concat_row, concat_col


class SoftMarginLoss(nn.Module):

    def __init__(
            self, tree: hier.Hierarchy,
            with_leaf_targets: bool,
            hardness: str = 'soft',
            margin: str = 'depth_dist',
            tau: float = 1.0):
        super().__init__()
        if hardness not in ('soft', 'hard'):
            raise ValueError('unknown hardness', hardness)
        n = tree.num_nodes()
        label_order = tree.leaf_subset() if with_leaf_targets else np.arange(n)

        # Construct array label_margin[gt_label, pr_node].
        if margin in ('edge_dist', 'depth_dist'):
            # label_margin = metrics.edge_dist(tree, label_order[:, None], np.arange(n)[None, :])
            depth = tree.depths()
            margin_arr = metrics.LCAMetric(tree, depth).dist(label_order[:, None], np.arange(n))
        elif margin == 'incorrect':
            is_ancestor = tree.ancestor_mask()
            is_correct = is_ancestor[:, label_order].T
            margin_arr = 1 - is_correct
        elif margin == 'info_dist':
            # TODO: Does natural log make most sense here?
            info = np.log(tree.num_leaf_nodes() / tree.num_leaf_descendants())
            margin_arr = metrics.LCAMetric(tree, info).dist(label_order[:, None], np.arange(n))
        elif margin == 'depth_deficient':
            depth = tree.depths()
            margin_arr = metrics.LCAMetric(tree, depth).deficient(label_order[:, None], np.arange(n))
        elif margin == 'log_depth_f1_error':
            depth = tree.depths()
            margin_arr = np.log(1 - metrics.LCAMetric(tree, depth).f1(label_order[:, None], np.arange(n)))
        else:
            raise ValueError('unknown margin', margin)

        # correct_margins = margin_arr[np.arange(len(label_order)), label_order]
        # if not np.all(correct_margins == 0):
        #     raise ValueError('margin with self is not zero', correct_margins)

        self.hardness = hardness
        self.tau = tau
        self.label_order = torch.from_numpy(label_order)
        self.margin = torch.from_numpy(margin_arr)

    def _apply(self, fn):
        super()._apply(fn)
        self.label_order = fn(self.label_order)
        self.margin = fn(self.margin)
        return self

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_node = labels if self.label_order is None else self.label_order[labels]
        label_score = scores.gather(-1, label_node.unsqueeze(-1)).squeeze(-1)
        label_margin = self.margin[labels, :]
        if self.hardness == 'soft':
            loss = -label_score + torch.logsumexp(scores + self.tau * label_margin, axis=-1)
        elif self.hardness == 'hard':
            # loss = -label_score + torch.max(torch.relu(scores + self.tau * label_margin), axis=-1)[0]
            loss = torch.relu(torch.max(scores - label_score.unsqueeze(-1) + self.tau * label_margin, axis=-1)[0])
        else:
            assert False
        return torch.mean(loss)


def _apply_to_maybe(fn: Callable, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return fn(x) if x is not None else None


def subtract_children(tree: hier.Hierarchy, p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    assert dim in (-1, p.ndim - 1)
    parent = tree.parents()
    parent = torch.from_numpy(parent).to(p.device)
    # Subtract each child from its parent.
    return p.index_add(-1, parent[1:], p[..., 1:], alpha=-1)


class SubtractChildren(nn.Module):

    def __init__(self, tree: hier.Hierarchy):
        super().__init__()
        self.parent = torch.from_numpy(tree.parents())

    def _apply(self, fn):
        super()._apply(fn)
        self.parent = fn(self.parent)
        return self

    def forward(self, p: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim in (-1, p.ndim - 1)
        # Subtract each child from its parent.
        return p.index_add(-1, self.parent[1:], p[..., 1:], alpha=-1)


class ConditionalMultiLabelLogSigmoid(nn.Module):
    """Brust and Denzler"""

    def __init__(self, tree: hier.Hierarchy):
        super().__init__()
        self.sum_ancestors_fn = SumAncestors(tree, exclude_root=True)

    def _apply(self, fn):
        super()._apply(fn)
        self.sum_ancestors_fn._apply(fn)
        return self

    def forward(self, scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim in (-1, scores.ndim - 1)
        log_cond_prob = F.logsigmoid(scores)
        log_prob = self.sum_ancestors_fn(log_cond_prob)
        return log_prob


class ConditionalMultiLabelLoss(nn.Module):
    """Brust and Denzler

    Positive examples are ancestors.
    Negative examples are other children of ancestors.

    Assume that labels are leaf nodes for now.
    """

    def __init__(self, tree: hier.Hierarchy, with_leaf_targets: bool = True):
        super().__init__()
        if not with_leaf_targets:
            raise NotImplementedError

        label_order = tree.leaf_subset()
        is_ancestor = tree.ancestor_mask()
        parent = tree.parents(root_loop=True)
        # is_example[gt, node] = parent[node] `is_ancestor` label_order[gt]
        is_example = is_ancestor[parent, :][:, label_order].T
        # is_pos[gt, node] = node `is_ancestor` label_order[gt]
        is_pos = is_ancestor[:, label_order].T

        dtype = torch.get_default_dtype()
        self.is_example = torch.from_numpy(is_example).type(dtype)
        self.is_pos = torch.from_numpy(is_pos).type(dtype)

    def _apply(self, fn):
        super()._apply(fn)
        self.is_example = fn(self.is_example)
        self.is_pos = fn(self.is_pos)
        return self

    def forward(self, scores: torch.Tensor, labels: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim in (-1, scores.ndim - 1)
        is_example = self.is_example[labels, 1:]  # Exclude root node.
        is_pos = self.is_pos[labels, 1:]
        bce_loss = is_example * (is_pos * -F.logsigmoid(scores) + (1 - is_pos) * -F.logsigmoid(-scores))
        loss = torch.sum(bce_loss, dim=-1)
        return torch.mean(loss)


class MultiLabelLossWithAncestorSum(nn.Module):
    """
    Salakhutdinov et al. 2011, "Learning to Share Visual Appearance for Multiclass Object Detection"
    """

    def __init__(self, tree: hier.Hierarchy, multilabel_loss: Callable):
        super().__init__()
        self.sum_ancestors = SumAncestors(tree, exclude_root=False)
        self.multilabel_loss = multilabel_loss

    def _apply(self, fn):
        super()._apply(fn)
        self.sum_ancestors._apply(fn)
        return self

    def forward(self, scores: torch.Tensor, labels: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim in (-1, scores.ndim - 1)
        sum_scores = self.sum_ancestors(scores, dim=dim)
        # # Exclude score for root when passing to loss.
        # return self.multilabel_loss(sum_scores[:, 1:], labels)
        return self.multilabel_loss(sum_scores, labels)


# class LeafLossWithAncestorSum(nn.Module):
#
#     def __init__(self, tree: hier.Hierarchy, leaf_loss: Callable):
#         super().__init__()
#         self.leaf_nodes = torch.from_numpy(tree.leaf_subset())
#         self.sum_leaf_ancestors = SumLeafAncestors(tree, exclude_root=False)
#         self.leaf_loss = leaf_loss
#
#     def _apply(self, fn):
#         super()._apply(fn)
#         self.leaf_nodes = fn(self.leaf_nodes)
#         self.sum_leaf_ancestors._apply(fn)
#         return self
#
#     def forward(self, scores: torch.Tensor, labels: torch.Tensor, dim: int = -1) -> torch.Tensor:
#         assert dim in (-1, scores.ndim - 1)
#         sum_scores = self.sum_leaf_ancestors(scores, dim=dim)
#         # Exclude score for root when passing to loss.
#         return self.leaf_loss(sum_scores, labels)


class RandomCut(nn.Module):

    def __init__(self, tree: hier.Hierarchy, cut_prob: float, permit_root_cut: bool = False):
        super().__init__()
        self.num_nodes = tree.num_nodes()
        self.cut_prob = cut_prob
        self.permit_root_cut = permit_root_cut
        self.sum_ancestors = SumAncestors(tree)
        self.node_parent = torch.from_numpy(tree.parents(root_loop=True))

    def _apply(self, fn):
        super()._apply(fn)
        self.sum_ancestors._apply(fn)
        self.node_parent = fn(self.node_parent)
        return self

    def forward(self, batch_shape: Sequence[int]) -> torch.Tensor:
        device = self.node_parent.device
        # Random Bernoulli for every node.
        drop = torch.bernoulli(torch.full((*batch_shape, self.num_nodes), self.cut_prob))
        drop = drop.to(device=device)
        if not self.permit_root_cut:
            drop[..., 0] = 0
        # Check whether to keep all ancestors of each node (drop zero ancestors).
        subtree_mask = (self.sum_ancestors(drop, dim=-1) == 0)
        # Dilate to keep nodes whose parents belong to subtree.
        subtree_mask = subtree_mask[..., self.node_parent]
        subtree_mask[..., 0] = True  # Root is always kept.
        # Count number of children (do not count parent as child of itself).
        num_children = torch.zeros((*batch_shape, self.num_nodes), device=device)
        num_children = num_children.index_add(-1, self.node_parent[1:], subtree_mask[..., 1:].float())
        # Find boundary of subtree.
        boundary = subtree_mask & (num_children == 0)
        return boundary


class RandomCutLoss(nn.Module):

    def __init__(
            self,
            tree: hier.Hierarchy,
            cut_prob: float,
            permit_root_cut: bool = False,
            with_leaf_targets: bool = True):
        super().__init__()
        is_ancestor = tree.ancestor_mask(strict=False)
        # label_to_targets[gt, pr] = 1 iff pr `is_ancestor` gt
        label_to_targets = is_ancestor.T
        if with_leaf_targets:
            label_order = tree.leaf_subset()
            label_to_targets = label_to_targets[label_order]
        else:
            # Need to use FlatSoftmaxNLL?
            raise NotImplementedError

        self.random_cut_fn = RandomCut(tree, cut_prob=cut_prob, permit_root_cut=permit_root_cut)
        self.label_to_targets = torch.from_numpy(label_to_targets)

    def _apply(self, fn):
        super()._apply(fn)
        self.random_cut_fn._apply(fn)
        self.label_to_targets = fn(self.label_to_targets)
        return self

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cut = self.random_cut_fn(scores.shape[:-1])
        targets = self.label_to_targets[labels, :]
        # No loss for root node.
        cut = cut[..., 1:]
        targets = targets[..., 1:]
        # Obtain targets in cut subset.
        cut_targets = (cut & targets)
        assert torch.all(torch.sum(cut_targets, dim=-1) == 1)
        # loss = F.cross_entropy(cut_scores, cut_targets, reduction='none')
        neg_inf = torch.tensor(-torch.inf, device=scores.device)
        zero = torch.tensor(0.0, device=scores.device)
        pos_score = torch.sum(torch.where(cut_targets, scores, zero), dim=-1)
        loss = -pos_score + torch.logsumexp(torch.where(cut, scores, neg_inf), dim=-1)
        return torch.mean(loss)


class RandomCutLossWithAncestorSum(nn.Module):

    def __init__(
            self,
            tree: hier.Hierarchy,
            cut_prob: float,
            permit_root_cut: bool = False,
            with_leaf_targets: bool = True):
        super().__init__()
        # Exclude root! No loss for it.
        self.sum_ancestors = SumAncestors(tree, exclude_root=True)
        self.random_cut_loss_fn = RandomCutLoss(
            tree, cut_prob, permit_root_cut=permit_root_cut,
            with_leaf_targets=with_leaf_targets)

    def _apply(self, fn):
        super()._apply(fn)
        self.sum_ancestors._apply(fn)
        self.random_cut_loss_fn._apply(fn)
        return self

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        sum_scores = self.sum_ancestors(scores, dim=-1)
        sum_scores = sum_scores[..., 1:]
        return self.random_cut_loss_fn(sum_scores, labels)


class SoftmaxNLLWithAncestorSum(nn.Module):

    def __init__(
            self,
            tree: hier.Hierarchy,
            with_leaf_targets: bool = True,
            reduction: str = 'mean'):
        super().__init__()
        # Exclude root!
        self.sum_ancestors = SumLeafAncestors(tree, exclude_root=True)
        if not with_leaf_targets:
            self.softmax_nll = FlatSoftmaxNLL(
                tree, with_leaf_targets=with_leaf_targets, reduction=reduction)
        else:
            self.softmax_nll = nn.CrossEntropyLoss(reduction=reduction)

    def _apply(self, fn):
        super()._apply(fn)
        self.sum_ancestors._apply(fn)
        self.softmax_nll._apply(fn)
        return self

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        sum_scores = self.sum_ancestors(scores, dim=-1)
        return self.softmax_nll(sum_scores, labels)


def make_loss_weights(strategy: str, tree: hier.Hierarchy, exclude_root: bool = False) -> Optional[np.ndarray]:
    if strategy == 'none' or not strategy:
        return None
    if strategy == 'parent':  # Legacy name.
        strategy = 'inv_parent'
    node_size = tree.num_leaf_descendants()
    parent_size = node_size[tree.parents(root_loop=True)]
    if exclude_root:
        node_size = node_size[1:]
        parent_size = parent_size[1:]
    if strategy == 'inv':
        return node_size ** -1
    elif strategy == 'inv_sqrt':
        return node_size ** -0.5
    elif strategy == 'inv_parent':
        return parent_size ** -1
    elif strategy == 'inv_sqrt_parent':
        return parent_size ** -0.5
    else:
        raise ValueError('unknown weighting strategy', strategy)
