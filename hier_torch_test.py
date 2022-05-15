import unittest

from pytest import mark
import numpy as np
import torch
import torch.nn.functional as F

import hier
import hier_torch

DEVICES = ['cpu', 'cuda']


def _basic_tree():
    #      0
    #    /   \
    #   1     2
    #  /|\   /|\
    # 3 4 5 6 7 8
    return hier.Hierarchy(np.asarray([-1, 0, 0, 1, 1, 1, 2, 2, 2]))


@mark.parametrize('device', DEVICES)
def test_HierSoftMaxCrossEntropy_equal_hier_softmax_nll(device):
    tree = _basic_tree()
    batch_size = 16
    n = tree.num_nodes()
    scores = torch.testing.make_tensor((batch_size, n - 1), device, torch.float32)
    labels = torch.testing.make_tensor((batch_size,), device, torch.int64, low=0, high=n)
    expected = hier_torch.hier_softmax_nll(tree, scores, labels)
    actual = (
        hier_torch.HierSoftmaxCrossEntropy(tree, with_leaf_targets=False).to(device=device)
    )(scores, labels)
    torch.testing.assert_close(actual, expected, check_device=True, atol=1e-4, rtol=1e-4)


class TestSubtractChildren(unittest.TestCase):

    def test_basic(self):
        tree = _basic_tree()
        p = torch.tensor([1.0, 0.2, 0.6, 0.1, 0.1, 0.0, 0.2, 0.1, 0.0])
        expected = torch.tensor([0.2, 0.0, 0.3, 0.1, 0.1, 0.0, 0.2, 0.1, 0.0])
        actual = hier_torch.SubtractChildren(tree)(p)
        torch.testing.assert_allclose(actual, expected)

    def test_sum_descendants_random(self):
        tree = _basic_tree()
        batch_size = 16
        n = tree.num_nodes()
        scores = torch.randn((batch_size, n))
        expected = F.softmax(scores, dim=-1)
        p = hier_torch.SumDescendants(tree)(expected, dim=-1)
        actual = hier_torch.SubtractChildren(tree)(p, dim=-1)
        torch.testing.assert_allclose(actual, expected)


class TestFlatSoftmaxNLL(unittest.TestCase):

    def test_random(self):
        tree = _basic_tree()
        batch_size = 16
        labels = torch.randint(0, tree.num_nodes(), (batch_size,))
        scores = torch.randn((batch_size, tree.num_leaf_nodes()))
        actual = hier_torch.FlatSoftmaxNLL(tree, reduction='none')(scores, labels)
        p = hier_torch.SumLeafDescendants(tree)(F.softmax(scores, dim=-1), dim=-1)
        p_label = torch.gather(p, -1, labels.unsqueeze(-1)).squeeze(-1)
        expected = -torch.log(p_label)
        torch.testing.assert_allclose(actual, expected)

    def test_equal_log_softmax_random(self):
        tree = _basic_tree()
        batch_size = 16
        leaf_labels = torch.randint(0, tree.num_leaf_nodes(), (batch_size,))
        labels = torch.from_numpy(tree.leaf_subset())[leaf_labels]
        scores = torch.randn((batch_size, tree.num_leaf_nodes()))
        actual = hier_torch.FlatSoftmaxNLL(tree, reduction='none')(scores, labels)
        expected = F.cross_entropy(scores, leaf_labels, reduction='none')
        torch.testing.assert_allclose(actual, expected)


@mark.parametrize('device', DEVICES)
def test_FlatBertinettoHXE_equal_log_softmax_with_alpha_zero_random(device):
    tree = _basic_tree()
    batch_size = 16
    leaf_labels = torch.testing.make_tensor(
        (batch_size,), device, torch.int64, low=0, high=tree.num_leaf_nodes())
    scores = torch.testing.make_tensor(
        (batch_size, tree.num_leaf_nodes()), device, torch.float32)

    actual = (
        hier_torch.FlatBertinettoHXE(tree, alpha=0.0, with_leaf_targets=True, reduction='none')
        .to(device=device)
    )(scores, leaf_labels)
    expected = F.cross_entropy(scores, leaf_labels, reduction='none').to(device=device)
    torch.testing.assert_close(actual, expected, check_device=True, atol=1e-4, rtol=1e-4)


@mark.parametrize('device', DEVICES)
def test_FlatBertinettoHXE_less_log_softmax_with_alpha_positive_random(device):
    tree = _basic_tree()
    batch_size = 16
    leaf_labels = torch.testing.make_tensor(
        (batch_size,), device, torch.int64, low=0, high=tree.num_leaf_nodes())
    scores = torch.testing.make_tensor(
        (batch_size, tree.num_leaf_nodes()), device, torch.float32)

    hxe_loss = (
        hier_torch.FlatBertinettoHXE(tree, alpha=0.1, with_leaf_targets=True, reduction='none')
        .to(device=device)
    )(scores, leaf_labels)
    ce_loss = F.cross_entropy(scores, leaf_labels, reduction='none').to(device=device)
    assert torch.all(hxe_loss <= ce_loss)


@mark.parametrize('device', DEVICES)
def test_FlatBertinettoHXE_equal_flat_softmax_nll_with_non_leaf_random(device):
    tree = _basic_tree()
    batch_size = 16
    labels = torch.testing.make_tensor(
        (batch_size,), device, torch.int64, low=0, high=tree.num_nodes())
    scores = torch.testing.make_tensor(
        (batch_size, tree.num_leaf_nodes()), device, torch.float32)
    actual = (
        hier_torch.FlatBertinettoHXE(tree, alpha=0.0, with_leaf_targets=False, reduction='none')
        .to(device=device)
    )(scores, labels)
    expected = (
        hier_torch.FlatSoftmaxNLL(tree, with_leaf_targets=False, reduction='none')
        .to(device=device)
    )(scores, labels)
    torch.testing.assert_close(actual, expected, check_device=True, atol=1e-4, rtol=1e-4)


class TestDescendantLogSoftmax(unittest.TestCase):

    def test_random(self):
        tree = _basic_tree()
        batch_size = 16
        n = tree.num_nodes()
        scores = torch.randn((batch_size, n))
        log_softmax_fn = hier_torch.DescendantLogSoftmax(tree, max_reduction='logsumexp')
        logp = log_softmax_fn(scores)
        self.assertTrue(torch.all(logp <= 0))
        self.assertTrue(torch.allclose(logp[..., 0], torch.tensor(0.0)))
        self.assertTrue(torch.all(logp <= logp[..., tree.parents(root_loop=True)]))


class TestDescendantSoftmaxLoss(unittest.TestCase):

    def test_random(self):
        tree = _basic_tree()
        batch_size = 16
        n = tree.num_nodes()
        scores = torch.randn((batch_size, n))
        labels = torch.randint(1, n, (batch_size,))

        # Take the sum of losses over all ancestors of the label.
        node_loss = torch.logsumexp(scores, dim=-1, keepdim=True) - scores
        is_ancestor = tree.ancestor_mask()
        is_label_ancestor = is_ancestor.T[labels]
        is_label_ancestor = torch.from_numpy(is_label_ancestor)
        zero = torch.tensor(0.0)
        loss = torch.sum(torch.where(is_label_ancestor, node_loss, zero), dim=-1)

        loss_fn = hier_torch.DescendantSoftmaxLoss(
            tree, with_leaf_targets=False, max_reduction='logsumexp', reduction='none')
        actual = loss_fn(scores, labels)
        torch.testing.assert_allclose(actual, loss)


class TestDescendantSoftmaxCousinLoss(unittest.TestCase):

    def test_random(self):
        tree = _basic_tree()
        batch_size = 16
        n = tree.num_nodes()
        scores = torch.randn((batch_size, n))
        labels = torch.randint(1, n, (batch_size,))

        # For each node, obtain the set of negative nodes and itself.
        is_ancestor = tree.ancestor_mask()
        is_cousin = ~(is_ancestor | is_ancestor.T)
        subset = is_cousin | np.eye(n, dtype=bool)
        subset = torch.from_numpy(subset)
        inf = torch.tensor(torch.inf)
        node_logsumexp = torch.logsumexp(torch.where(subset, scores.unsqueeze(-2), -inf), dim=-1)
        node_loss = node_logsumexp - scores

        # Take the sum of this over all ancestors of the label.
        is_label_ancestor = is_ancestor.T[labels]
        is_label_ancestor = torch.from_numpy(is_label_ancestor)
        zero = torch.tensor(0.0)
        loss = torch.sum(torch.where(is_label_ancestor, node_loss, zero), dim=-1)

        loss_fn = hier_torch.DescendantSoftmaxCousinLoss(
            tree, with_leaf_targets=False, max_reduction='logsumexp', reduction='none')
        actual = loss_fn(scores, labels)
        torch.testing.assert_allclose(actual, loss)
