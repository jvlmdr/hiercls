import operator
from typing import List, Sequence, Tuple

from jax import nn
from jax import numpy as jnp
import numpy as np

import hier


class HierSoftmax:

    def __init__(self, tree: hier.Hierarchy, log_alpha=0.):
        self.tree = tree
        self.log_alpha = log_alpha

    def dim(self):
        # Equal to num_edges, or num_nodes - 1.
        return sum(self.tree.num_children())

    def internal_log_softmax(
            self, scores, axis=-1,
            ) -> Tuple[Sequence[int], Sequence[List[int]], Sequence]:
        """Returns node, children, log_p for softmax at each internal node."""
        assert axis == -1
        internal_nodes, = np.logical_not(self.tree.leaf_mask()).nonzero()
        node_to_children = self.tree.children()
        node_children = [node_to_children[x] for x in internal_nodes]
        node_degree = [len(x) for x in node_children]
        node_scores = jnp.split(scores, np.cumsum(node_degree)[:-1], axis=axis)
        node_log_cond = [nn.log_softmax(x, axis=axis) for x in node_scores]
        return internal_nodes, node_children, node_log_cond

    def log_conditional(self, scores, axis=-1):
        """Returns log likelihood for each node given its parent."""
        num_nodes = self.tree.num_nodes()
        internal_nodes, softmax_children, softmax_log_cond = (
            self.internal_log_softmax(scores, axis=axis))
        # Re-assemble the conditional likelihoods into one array.
        # TODO: Could use .at[].set() to avoid unstack and re-stack?
        # However, this makes the operations serial.
        # Could stack and then re-order all?
        #   child_node = np.concatenate(softmax_children)
        #   child_log_cond = jnp.concatenate(softmax_log_cond, axis=axis)
        #   log_cond = jnp.zeros(scores.shape, dtype=scores.dtype)
        #   log_cond = log_cond.at[child_node].set(child_log_cond)
        # No way to specify axis of at[]?
        # Maybe this should all be done with vmap instead? Including sum.
        shape = scores.shape[:axis] + scores.shape[axis:][1:]
        log_cond = [jnp.zeros(shape, dtype=scores.dtype)] * num_nodes
        for i in range(len(internal_nodes)):
            nodes = softmax_children[i]
            values = list(jnp.moveaxis(softmax_log_cond[i], axis, 0))
            for node, value in zip(nodes, values):
                log_cond[node] = value
        # TODO: Avoid stack and unstack in non-dense sum?
        return jnp.stack(log_cond, axis=axis)

    def sum_ancestors(self, log_cond, axis=-1, dense=True):
        if dense:
            assert axis == -1
            # Obtain sum using matrix-vector product.
            # Will it be inefficient to copy this array to another device?
            is_ancestor = self.tree.ancestor_mask(strict=False).astype(log_cond.dtype)
            # We have is_ancestor[i, j] = 1 iff i is an ancestor of j (or equal).
            # Therefore log_prob[j] = sum_{i} log_cond[i] is_ancestor[i, j]
            # gives the total over all ancestors of node j.
            log_prob = jnp.dot(log_cond, is_ancestor)  # Requires axis is -1.
        else:
            # Obtain sum by traversing tree.
            log_cond = list(jnp.moveaxis(log_cond, axis, 0))
            log_prob = self.tree.accumulate_ancestors(operator.add, log_cond)
            log_prob = jnp.stack(log_prob, axis=axis)
        return log_prob

    def log_likelihood(self, scores, axis=-1, dense=True):
        log_cond = self.log_conditional(scores, axis=axis)
        log_prob = self.sum_ancestors(log_cond, axis=axis, dense=dense)
        return log_prob

    def likelihood(self, scores, axis=-1, dense=True):
        return jnp.exp(self.log_likelihood(scores, axis=axis, dense=dense))

    def loss_hard_leaf(self, scores, labels, axis=-1, dense=True):
        """Computes loss given hard (integer) label in leaf set."""
        if not self.log_alpha:
            # Simply use negative log-likelihood of leaf.
            nll = -self.log_likelihood(scores, axis=axis)
            is_leaf = self.tree.leaf_mask()
            leaf_nll = jnp.compress(is_leaf, nll, axis=axis)
            return np.take(leaf_nll, labels, axis=axis)

        # TODO: Implement as many individual cross-entropy losses?
        # However, this requires some messing around to get the supervision.

        # Weight the log-conditionals by their depth.
        assert axis == -1
        is_leaf = self.tree.leaf_mask()
        parent_depths = self.tree.depths() - 1
        log_cond = self.log_conditional(scores, axis=axis)
        # Broadcast requires that axis is -1.
        # Weight by alpha^(depth of parent).
        cond_nll = np.exp(self.log_alpha * parent_depths) * -log_cond
        # Take sum of log-conditionals over ancestors.
        nll = self.sum_ancestors(cond_nll, axis=axis, dense=dense)
        leaf_nll = jnp.compress(is_leaf, nll, axis=axis)
        return np.take(leaf_nll, labels, axis=axis)


class FlatSoftmax:

    def __init__(self, tree: hier.Hierarchy):
        self.tree = tree

    def dim(self):
        return self.tree.num_leaf_nodes()

    def likelihood(self, scores, axis=-1, dense=True):
        """Returns likelihood of each node in the hierarchy."""
        assert axis == -1
        shape = scores.shape[:axis] + scores.shape[axis:][1:]
        is_leaf = self.tree.leaf_mask()
        leaf_nodes, = np.nonzero(is_leaf)
        leaf_log_prob = nn.log_softmax(scores, axis=axis)
        # TODO: Should use log-sum-exp trick for log_likelihood?
        leaf_prob = jnp.exp(leaf_log_prob)
        # Simply take sum over leaf likelihoods to obtain 
        if dense:
            is_descendant = self.tree.ancestor_mask(strict=False).T
            leaf_is_descendant = is_descendant[is_leaf, :]
            prob = jnp.dot(leaf_prob, leaf_is_descendant)
        else:
            # Obtain sum by traversing tree.
            # Caution: Copies of same array (not mutated).
            # "Exclusive" likelihood is for this node but not its children.
            num_nodes = self.tree.num_nodes()
            excl_prob = [jnp.zeros(shape, dtype=scores.dtype)] * num_nodes
            for node, value in zip(leaf_nodes, leaf_prob):
                excl_prob[node] = value
            prob = self.tree.accumulate_descendants(operator.add, excl_prob)
            prob = jnp.stack(prob, axis=axis)
        return prob

    def log_likelihood(self, scores, axis=-1, dense=True):
        """Returns log-likelihood of each node in the hierarchy."""
        # TODO: Use log-sum-exp trick here? For gradients?
        return jnp.log(self.likelihood(scores, axis=axis, dense=dense))

    def loss_with_hard_leaf_label(self, scores, labels, axis=-1):
        """Standard cross-entropy loss for exclusive labels."""
        assert axis == -1
        log_prob = nn.log_softmax(scores, axis=axis)
        return jnp.take(-log_prob, labels, axis=axis)


class MultiSigmoid:
    """No guarantee that parent is at least sum of children."""

    def __init__(self, tree: hier.Hierarchy):
        self.tree = tree

    def dim(self):
        return self.tree.num_nodes() - 1  # Exclude root node.

    def loss_with_hard_leaf_label(self, scores, labels, axis=-1):
        # Get is_ancestor label for leaf nodes.
        is_leaf = self.tree.leaf_mask()
        is_ancestor = self.tree.ancestor_mask(strict=False)
        # binary_labels_for_leaf has shape [num_leaf_nodes, num_nodes - 1]
        binary_labels_for_leaf = is_ancestor[1:, is_leaf].T
        # scores has shape [Ni..., num_nodes - 1, Nj...]
        # labels has shape [Ni..., Nj...]
        # q has shape [Ni..., num_nodes - 1, Nj...]
        q = jnp.moveaxis(binary_labels_for_leaf[labels], -1, axis)
        # Require that axis is -1.
        loss = q * nn.log_sigmoid(scores) + (1 - q) * nn.log_sigmoid(-scores)
        # TODO: Add per-class weighting?
        return np.mean(loss, axis=axis)

    def log_likelihood(self, scores, axis=-1):
        # Insert zero for root label.
        shape = scores.shape[:axis] + scores.shape[axis:][1:]
        zero = jnp.expand_dims(jnp.zeros(shape, dtype=scores.dtype), axis=axis)
        log_prob = jnp.concatenate([zero, nn.log_sigmoid(scores)], axis=axis)
        return log_prob

    def likelihood(self, scores, axis=-1):
        return jnp.exp(self.log_likelihood(scores, axis=axis))