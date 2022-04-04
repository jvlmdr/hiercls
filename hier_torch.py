from functools import partial
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import hier


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
        self.is_ancestor_leaf = fn(self.is_ancestor_leaf).bool()  # Preserve dtype.
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


class FlatLogSoftmaxNLL(nn.Module):

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
        self.is_ancestor_leaf = fn(self.is_ancestor_leaf).bool()  # Preserve dtype.
        return self

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logp_leaf = F.log_softmax(scores, dim=-1)
        # Obtain logp for leaf descendants, -inf for other nodes.
        logp_descendants = torch.where(
            self.is_ancestor_leaf,
            logp_leaf.unsqueeze(-2),
            torch.tensor(-torch.inf, device=scores.device))
        logp = torch.logsumexp(logp_descendants, dim=-1)
        nll = -torch.gather(logp, -1, labels.unsqueeze(-1))
        return torch.mean(nll)


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
        # Do not apply fn to indices because it might convert dtype.
        self.hier_log_softmax = self.hier_log_softmax._apply(fn)
        return self

    @property
    def device(self) -> torch.device:
        return self.hier_log_softmax.device

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # TODO: Could make this faster by only computing likelihood for targets.
        device = self.device
        log_prob = self.hier_log_softmax(scores, dim=-1)
        if self.label_order is not None:
            log_prob = torch.index_select(log_prob, -1, self.label_order.to(device))
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
        # Do not apply fn to indices because it might convert dtype.
        self.hier_cond_log_softmax = self.hier_cond_log_softmax._apply(fn)
        self.sum_label_descendants = self.sum_label_descendants._apply(fn)
        self.prior = fn(self.prior)
        self.node_weight = fn(self.node_weight) if self.node_weight is not None else None
        return self

    def forward(self, scores: torch.Tensor, labels: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert labels.ndim in [scores.ndim, scores.ndim - 1]
        assert dim in (-1, scores.ndim - 1)
        device = scores.device
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
            xent *= self.node_weight
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
    flat = torch.full(flat_shape, -torch.inf, device=device).index_copy_(
        -1, flat_index, scores)
    split_shape = [*input_shape[:-1], num_internal, max_num_children]
    child_scores = flat.reshape(split_shape)
    child_log_p = F.log_softmax(child_scores, dim=-1)
    child_log_p = child_log_p.reshape(flat_shape)
    output_shape = [*input_shape[:-1], num_nodes]
    # log_cond_p[..., child_index] = child_log_p[..., flat_index]
    log_cond_p = torch.zeros(output_shape, device=device).index_copy_(
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
        row_index = np.concatenate([np.full(n, i) for i, n in enumerate(cond_num_children)])
        col_index = np.concatenate([np.arange(n) for n in cond_num_children])
        flat_index = torch.from_numpy(row_index * max_num_children + col_index)
        child_index = torch.from_numpy(np.concatenate(cond_children))

        self.num_nodes = num_nodes
        self.num_internal = num_internal
        self.max_num_children = max_num_children
        self.flat_index = flat_index
        self.child_index = child_index

    # def _apply(self, fn):
    #     super()._apply(fn)
    #     # Do not apply fn to indices because it might convert dtype.
    #     return self

    def forward(self, scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim in (-1, scores.ndim - 1)
        device = scores.device
        input_shape = list(scores.shape)
        flat_shape = [*input_shape[:-1], self.num_internal * self.max_num_children]
        # Pad with -inf for log_softmax.
        flat_index = self.flat_index.to(device)
        # flat[..., flat_index] = scores
        flat = torch.full(flat_shape, -torch.inf, device=device).index_copy_(
            -1, flat_index, scores)
        split_shape = [*input_shape[:-1], self.num_internal, self.max_num_children]
        child_scores = flat.reshape(split_shape)
        child_log_p = F.log_softmax(child_scores, dim=-1)
        child_log_p = child_log_p.reshape(flat_shape)
        output_shape = [*input_shape[:-1], self.num_nodes]
        child_index = self.child_index.to(device)
        # log_cond_p[..., child_index] = child_log_p[..., flat_index]
        log_cond_p = torch.zeros(output_shape, device=device).index_copy_(
            -1, child_index, child_log_p.index_select(-1, flat_index))
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

    @property
    def device(self) -> torch.device:
        return self.sum_ancestors_fn.device

    def forward(self, scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
        log_cond_p = self.cond_log_softmax(scores, dim=dim)
        return self.sum_ancestors_fn(log_cond_p, dim=dim)


class HierSoftmaxNLLWithInactive(nn.Module):
    """Hierarchical softmax with loss for inactive nodes."""

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
        # Do not apply fn to indices because it might convert dtype.
        self.hier_log_softmax = self.hier_log_softmax._apply(fn)
        return self

    @property
    def device(self) -> torch.device:
        return self.hier_log_softmax.device

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # TODO: Could make this faster by only computing likelihood for targets.
        device = self.device

        # Log-sum-exp for each internal node.
        self.logsumexp = {}

        log_prob = self.hier_log_softmax(scores, dim=-1)
        if self.label_order is not None:
            log_prob = torch.index_select(log_prob, -1, self.label_order.to(device))
        nll = -torch.gather(log_prob, -1, labels.unsqueeze(-1)).squeeze(-1)
        return torch.mean(nll)


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

    @property
    def device(self) -> torch.device:
        return self.matrix.device

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
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def _apply(self, fn):
        super()._apply(fn)
        self.binary_labels = fn(self.binary_labels)
        self.bce_loss = self.bce_loss._apply(fn)
        return self

    @property
    def device(self) -> torch.device:
        return self.binary_labels.device

    def forward(self, scores: torch.Tensor, labels: torch.Tensor, dim: int = -1) -> torch.Tensor:
        targets = torch.index_select(self.binary_labels, 0, labels)
        # Reduce over classes.
        loss = torch.sum(self.bce_loss(scores, targets), dim=dim)
        # Take mean over examples.
        return torch.mean(loss)


def multilabel_likelihood(tree: hier.Hierarchy, scores: torch.Tensor, dim=-1):
    return torch.exp(multilabel_log_likelihood(tree, scores, dim=dim))


def multilabel_log_likelihood(tree: hier.Hierarchy, scores: torch.Tensor, dim=-1):
    assert scores.shape[dim] == tree.num_nodes() - 1
    assert dim in (-1, scores.ndim - 1)
    shape = list(scores.shape)
    shape[-1] = 1
    zero = torch.zeros(shape, dtype=scores.dtype, device=scores.device)
    return torch.cat([zero, F.logsigmoid(scores)], dim=-1)


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
        self.sum_ancestors_fn = self.sum_ancestors_fn._apply(fn)
        return self

    @property
    def device(self) -> torch.device:
        return self.weight.device

    def forward(self,
                scores: torch.Tensor,
                labels: torch.Tensor,
                dim: int = -1) -> torch.Tensor:
        device = self.device
        log_cond_p = self.hier_cond_log_softmax_fn(scores, dim=dim)
        assert dim in (-1, scores.ndim - 1)
        # Take weighted sum over ancestors.
        # Weight each conditional likelihood by exp(-alpha * parent_depth).
        # Note that log_cond_p of root is always zero.
        weighted_cond_nll = -self.weight * log_cond_p
        weighted_nll = self.sum_ancestors_fn(weighted_cond_nll, dim=dim)
        assert labels.ndim == scores.ndim - 1
        if self.label_order is not None:
            weighted_nll = torch.index_select(
                weighted_nll, dim, self.label_order.to(device))
        label_nll = torch.gather(weighted_nll, dim, labels.unsqueeze(-1)).squeeze(-1)
        return torch.mean(label_nll)
