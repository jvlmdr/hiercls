import unittest

import numpy as np
import torch

import hier
import hier_torch


class TestHierSoftmaxCrossEntropy(unittest.TestCase):

    def test_equal_hier_softmax_nll(self):
        # 0: [1, 2]
        # 1: [3, 4, 5]
        # 2: [6, 7, 8]
        tree = hier.Hierarchy(np.asarray([-1, 0, 0, 1, 1, 1, 2, 2, 2]))
        batch_size = 16
        num_nodes = 9
        scores = torch.randn((batch_size, num_nodes - 1))
        labels = torch.randint(num_nodes, size=(batch_size,))
        expected = hier_torch.hier_softmax_nll(tree, scores, labels)
        actual = hier_torch.HierSoftmaxCrossEntropy(tree, with_leaf_targets=False)(scores, labels)
        torch.testing.assert_allclose(actual, expected)
