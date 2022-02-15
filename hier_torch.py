import numpy as np
import torch
from torch import nn

import hier


def hier_log_softmax(
        tree: hier.Hierarchy,
        scores: torch.Tensor,
        dim: int = -1):
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


def leaf_embed(
        tree: hier.Hierarchy,
        values: torch.Tensor,
        dim: int = -1):
    """Embeds leaf values in a tree. Internal nodes are set to zero."""
    shape = list(values.shape)
    shape[dim] = tree.num_nodes()
    node_values = torch.zeros(shape, device=values.device)
    return leaf_add(tree, node_values, values, dim=dim)


def leaf_add(
        tree: hier.Hierarchy,
        node_values: torch.Tensor,
        leaf_values: torch.Tensor,
        dim: int = -1):
    """Adds values for leaf nodes to values for all nodes."""
    device = node_values.device
    leaf_index = torch.from_numpy(tree.leaf_subset())
    return node_values.index_add(dim, leaf_index.to(device), leaf_values)


def sum_leaf_descendants(
        tree: hier.Hierarchy,
        values: torch.Tensor,
        dim: int = -1):
    """Computes sum over leaf descendants for each node."""
    # The value is_ancestor[i, j] is true if i is an ancestor of j.
    is_ancestor = tree.ancestor_mask()
    leaf_is_descendant = is_ancestor[:, tree.leaf_mask()].T
    matrix = torch.from_numpy(leaf_is_descendant)
    matrix = matrix.to(device=values.device, dtype=values.dtype)
    # TODO: Re-order dimensions to make this work with dim != -1.
    assert dim == -1 or dim == values.ndim - 1
    return torch.tensordot(values, matrix, dims=1)


def sum_descendants(
        tree: hier.Hierarchy,
        values: torch.Tensor,
        dim: int = -1,
        strict: bool = False):
    """Computes sum over all descendants for each node."""
    # The value is_ancestor[i, j] is true if i is an ancestor of j.
    is_ancestor = tree.ancestor_mask(strict=strict)
    matrix = (torch.from_numpy(is_ancestor.T)
              .to(device=values.device, dtype=values.dtype))
    # TODO: Re-order dimensions to make this work with dim != -1.
    assert dim == -1 or dim == values.ndim - 1
    return torch.tensordot(values, matrix, dims=1)


def sum_ancestors(
        tree: hier.Hierarchy,
        values: torch.Tensor,
        dim: int = -1,
        strict: bool = False):
    """Computes sum over ancestors of each node."""
    # The value is_ancestor[i, j] is true if i is an ancestor of j.
    is_ancestor = tree.ancestor_mask(strict=strict)
    matrix = (torch.from_numpy(is_ancestor)
              .to(device=values.device, dtype=values.dtype))
    # TODO: Re-order dimensions to make this work with dim != -1.
    assert dim == -1 or dim == values.ndim - 1
    return torch.tensordot(values, matrix, dims=1)


def sum_leaf_ancestors(
        tree: hier.Hierarchy,
        values: torch.Tensor,
        dim: int = -1):
    """Computes sum over ancestors of each leaf."""
    # The value is_ancestor[i, j] is true if i is an ancestor of j.
    is_ancestor = tree.ancestor_mask()
    is_ancestor_for_leaf = is_ancestor[:, tree.leaf_mask()]
    matrix = (torch.from_numpy(is_ancestor_for_leaf)
              .to(device=values.device, dtype=values.dtype))
    # TODO: Re-order dimensions to make this work with dim != -1.
    assert dim == -1 or dim == values.ndim - 1
    return torch.tensordot(values, matrix, dims=1)
