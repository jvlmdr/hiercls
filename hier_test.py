import unittest

import numpy as np

import hier


class TestFindProjection(unittest.TestCase):

    def test_basic(self):
        # 0: [1, 2]
        # 1: [3, 4, 5]
        # 2: [6, 7, 8]
        tree = hier.Hierarchy(np.asarray([-1, 0, 0, 1, 1, 1, 2, 2, 2]))
        node_subset = np.asarray([0, 2, 6, 8])
        project = hier.find_projection(tree, node_subset)
        np.testing.assert_equal(project >= 0, True)
        np.testing.assert_equal(project < len(node_subset), True)
        expect = np.asarray([0, 0, 2, 0, 0, 0, 6, 2, 8])
        np.testing.assert_equal(expect, node_subset[project])

    def test_is_ancestor(self):
        tree = hier.Hierarchy(np.asarray([-1, 0, 0, 1, 1, 1, 2, 2, 2]))
        node_subset = np.asarray([0, 2, 6, 8])
        project = hier.find_projection(tree, node_subset)
        # Check that projection is an ancestor.
        is_ancestor = tree.ancestor_mask()
        n = tree.num_nodes()
        np.testing.assert_array_equal(is_ancestor[node_subset[project], np.arange(n)], True)
