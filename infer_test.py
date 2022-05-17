import unittest

import numpy as np

import hier
import infer


class TestPareto(unittest.TestCase):

    def test_basic(self):
        actual = infer.pareto_optimal_predictions(
            np.array([  1,   2,   2,   4,   4,   4,   8,   8]),
            np.array([1.0, 0.4, 0.6, 0.1, 0.3, 0.2, 0.1, 0.2]))
        expected = np.array([0, 2, 4, 7], dtype=int)
        np.testing.assert_equal(actual, expected)

    def test_raises_for_non_unique(self):
        with self.assertRaises(ValueError):
            infer.pareto_optimal_predictions(
                np.array([  1,   2,   2,   4,   4,   4,   8,   8]),
                np.array([1.0, 0.4, 0.6, 0.1, 0.3, 0.2, 0.1, 0.1]),
                require_unique=True)

    def test_does_not_raise_for_non_unique_but_irrelevant(self):
        infer.pareto_optimal_predictions(
            np.array([  1,   2,   2,   4,   4,   4,   8]),
            np.array([1.0, 0.4, 0.6, 0.2, 0.3, 0.2, 0.1]),
            require_unique=True)


class TestPlurality(unittest.TestCase):

    def test_basic(self):
        tree = _basic_tree()
        p = np.asarray([1.0, 0.8, 0.2, 0.4, 0.2, 0.2, 0.0, 0.0, 0.2])
        actual = infer.plurality_threshold(tree, p)
        expected = 0.2
        self.assertEqual(actual, expected)

    def test_2d(self):
        tree = _basic_tree()
        p = np.asarray([[1.0, 0.8, 0.2, 0.4, 0.2, 0.2, 0.0, 0.0, 0.2],
                        [1.0, 0.7, 0.3, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]])
        actual = infer.plurality_threshold(tree, p)
        expected = np.asarray([0.2, 0.3])
        np.testing.assert_allclose(actual, expected)


def _basic_tree():
    #      0
    #    /   \
    #   1     2
    #  /|\   /|\
    # 3 4 5 6 7 8
    return hier.Hierarchy(np.asarray([-1, 0, 0, 1, 1, 1, 2, 2, 2]))
