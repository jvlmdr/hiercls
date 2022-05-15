from pytest import mark
import numpy as np
import torch
import torch.nn.functional as F

import hier
import hier_torch


def _basic_tree():
    #      0
    #    /   \
    #   1     2
    #  /|\   /|\
    # 3 4 5 6 7 8
    return hier.Hierarchy(np.asarray([-1, 0, 0, 1, 1, 1, 2, 2, 2]))


@mark.parametrize('device,tol', [('cpu', None), ('cuda', 1e-4)])
def test_HierSoftMaxCrossEntropy_equal_hier_softmax_nll(device, tol):
    tree = _basic_tree()
    batch_size = 16
    n = tree.num_nodes()
    scores = torch.testing.make_tensor((batch_size, n - 1), device, torch.float32)
    labels = torch.testing.make_tensor((batch_size,), device, torch.int64, low=0, high=n)
    expected = hier_torch.hier_softmax_nll(tree, scores, labels)
    actual = (
        hier_torch.HierSoftmaxCrossEntropy(tree, with_leaf_targets=False).to(device=device)
    )(scores, labels)
    torch.testing.assert_close(actual, expected, check_device=True, atol=tol, rtol=tol)


@mark.parametrize('device,tol', [('cpu', None), ('cuda', 1e-4)])
def test_SubtractChildren_basic(device, tol):
    tree = _basic_tree()
    p = torch.tensor([1.0, 0.2, 0.6, 0.1, 0.1, 0.0, 0.2, 0.1, 0.0], device=device)
    expected = torch.tensor([0.2, 0.0, 0.3, 0.1, 0.1, 0.0, 0.2, 0.1, 0.0], device=device)
    actual = hier_torch.SubtractChildren(tree).to(device=device)(p)
    torch.testing.assert_close(actual, expected, check_device=True, atol=tol, rtol=tol)


@mark.parametrize('device,tol', [('cpu', None), ('cuda', 1e-3)])
def test_SubtractChildren_sum_descendants_random(device, tol):
    tree = _basic_tree()
    batch_size = 16
    n = tree.num_nodes()
    scores = torch.testing.make_tensor((batch_size, n), device, torch.float32)
    expected = F.softmax(scores, dim=-1)
    p = hier_torch.SumDescendants(tree).to(device=device)(expected, dim=-1)
    actual = hier_torch.SubtractChildren(tree).to(device=device)(p, dim=-1)
    torch.testing.assert_close(actual, expected, check_device=True, atol=tol, rtol=tol)


@mark.parametrize('device,tol', [('cpu', None), ('cuda', 1e-3)])
def test_FlatSoftmaxNLL_random(device, tol):
    tree = _basic_tree()
    batch_size = 16
    labels = torch.testing.make_tensor(
        (batch_size,), device, torch.int64, low=0, high=tree.num_nodes())
    scores = torch.testing.make_tensor(
        (batch_size, tree.num_leaf_nodes()), device, torch.float32)
    actual = hier_torch.FlatSoftmaxNLL(tree, reduction='none').to(device=device)(scores, labels)
    p = hier_torch.SumLeafDescendants(tree).to(device=device)(
        F.softmax(scores, dim=-1), dim=-1)
    p_label = torch.gather(p, -1, labels.unsqueeze(-1)).squeeze(-1)
    expected = -torch.log(p_label)
    torch.testing.assert_close(actual, expected, check_device=True, atol=tol, rtol=tol)


@mark.parametrize('device,tol', [('cpu', None), ('cuda', 1e-3)])
def test_FlatSoftmaxNLL_equal_log_softmax_random(device, tol):
    tree = _basic_tree()
    batch_size = 16
    leaf_labels = torch.testing.make_tensor(
        (batch_size,), device, torch.int64, low=0, high=tree.num_leaf_nodes())
    labels = torch.from_numpy(tree.leaf_subset())[leaf_labels]
    scores = torch.testing.make_tensor(
        (batch_size, tree.num_leaf_nodes()), device, torch.float32)
    actual = hier_torch.FlatSoftmaxNLL(tree, reduction='none').to(device=device)(scores, labels)
    expected = F.cross_entropy(scores, leaf_labels, reduction='none')
    torch.testing.assert_close(actual, expected, check_device=True, atol=tol, rtol=tol)


@mark.parametrize('device,tol', [('cpu', None), ('cuda', 1e-4)])
def test_FlatBertinettoHXE_equal_log_softmax_with_alpha_zero_random(device, tol):
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
    torch.testing.assert_close(actual, expected, check_device=True, atol=tol, rtol=tol)


@mark.parametrize('device,tol', [('cpu', 0), ('cuda', 1e-4)])
def test_FlatBertinettoHXE_less_log_softmax_with_alpha_positive_random(device, tol):
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
    # all(hxe_loss <= ce_loss)
    torch.testing.assert_close(
        torch.maximum(hxe_loss, ce_loss), ce_loss,
        check_device=True, atol=tol, rtol=tol)


@mark.parametrize('device,tol', [('cpu', None), ('cuda', 1e-4)])
def test_FlatBertinettoHXE_equal_flat_softmax_nll_with_non_leaf_random(device, tol):
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
    torch.testing.assert_close(actual, expected, check_device=True, atol=tol, rtol=tol)


@mark.parametrize('device', ['cpu', 'cuda'])
def test_DescendantLogSoftmax_random(device):
    tree = _basic_tree()
    batch_size = 16
    n = tree.num_nodes()
    scores = torch.testing.make_tensor((batch_size, n), device, torch.float32)
    logp = (
        hier_torch.DescendantLogSoftmax(tree, max_reduction='logsumexp').to(device=device)
    )(scores)
    assert torch.all(logp <= 0)
    assert torch.allclose(logp[..., 0], torch.tensor(0.0))
    assert torch.all(logp <= logp[..., tree.parents(root_loop=True)])


@mark.parametrize('device,tol', [('cpu', None), ('cuda', 1e-4)])
def test_DescendantSoftmaxLoss_random(device, tol):
    tree = _basic_tree()
    batch_size = 16
    n = tree.num_nodes()
    scores = torch.testing.make_tensor((batch_size, n), device, torch.float32)
    labels = torch.testing.make_tensor((batch_size,), device, torch.int64, low=1, high=n)

    actual = (
        hier_torch.DescendantSoftmaxLoss(
            tree, with_leaf_targets=False, max_reduction='logsumexp', reduction='none')
        .to(device=device)
    )(scores, labels)

    # Take the sum of losses over all ancestors of the label.
    node_loss = torch.logsumexp(scores, dim=-1, keepdim=True) - scores
    is_ancestor = torch.from_numpy(tree.ancestor_mask()).to(device=device)
    is_label_ancestor = is_ancestor.T[labels]
    zero = torch.tensor(0.0, device=device)
    expected = torch.sum(torch.where(is_label_ancestor, node_loss, zero), dim=-1)

    torch.testing.assert_close(actual, expected, check_device=True, atol=tol, rtol=tol)


@mark.parametrize('device,tol', [('cpu', None), ('cuda', 1e-4)])
def test_TestDescendantSoftmaxCousinLoss_random(device, tol):
    tree = _basic_tree()
    batch_size = 16
    n = tree.num_nodes()
    scores = torch.testing.make_tensor((batch_size, n), device, torch.float32)
    labels = torch.testing.make_tensor((batch_size,), device, torch.int64, low=1, high=n)

    # For each node, obtain the set of negative nodes and itself.
    is_ancestor = torch.from_numpy(tree.ancestor_mask()).to(device=device)
    is_cousin = ~(is_ancestor | is_ancestor.T)
    subset = is_cousin | torch.eye(n, dtype=bool, device=device)
    inf = torch.tensor(torch.inf, device=device)
    node_logsumexp = torch.logsumexp(torch.where(subset, scores.unsqueeze(-2), -inf), dim=-1)
    node_loss = node_logsumexp - scores

    # Take the sum of this over all ancestors of the label.
    is_label_ancestor = is_ancestor.T[labels]
    zero = torch.tensor(0.0, device=device)
    expected = torch.sum(torch.where(is_label_ancestor, node_loss, zero), dim=-1)

    actual = (
        hier_torch.DescendantSoftmaxCousinLoss( 
            tree, with_leaf_targets=False, max_reduction='logsumexp', reduction='none')
        .to(device=device)
    )(scores, labels)
    torch.testing.assert_close(actual, expected, check_device=True, atol=tol, rtol=tol)
