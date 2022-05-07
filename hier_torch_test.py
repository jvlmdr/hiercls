import unittest

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

class TestHierSoftmaxCrossEntropy(unittest.TestCase):

    def test_equal_hier_softmax_nll(self):
        tree = _basic_tree()
        batch_size = 16
        n = tree.num_nodes()
        scores = torch.randn((batch_size, n - 1))
        labels = torch.randint(n, size=(batch_size,))
        expected = hier_torch.hier_softmax_nll(tree, scores, labels)
        actual = hier_torch.HierSoftmaxCrossEntropy(tree, with_leaf_targets=False)(scores, labels)
        torch.testing.assert_allclose(actual, expected)


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
