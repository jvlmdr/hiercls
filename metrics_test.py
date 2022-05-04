from multiprocessing.sharedctypes import Value
import unittest

import numpy as np

import metrics


class TestOperatingCurve(unittest.TestCase):

    def test_basic(self):
        actual_scores, actual_metrics = metrics.operating_curve(
            [np.array([1.0, 0.9, 0.2]), np.array([1.0, 0.8, 0.6, 0.1])],
            {'metric1': [np.array([1.0, 2.0, 4.0]), np.array([8.0, 16.0, 32.0, 64.0])],
             'metric2': [np.array([1.0, 0.9, 0.2]), np.array([1.0, 0.8, 0.6, 0.1])]})
        desired_scores = np.array([0.9, 0.8, 0.6, 0.2, 0.1])
        desired_metrics = {
            'metric1': np.array([9.0, 10.0, 18.0, 34.0, 36.0, 68.0]),
            'metric2': np.array([2.0, 1.9, 1.7, 1.5, 0.8, 0.3]),
        }
        self.assertEqual(set(actual_metrics.keys()), set(('metric1', 'metric2')))
        np.testing.assert_allclose(actual_scores, desired_scores)
        np.testing.assert_allclose(actual_metrics['metric1'], desired_metrics['metric1'])
        np.testing.assert_allclose(actual_metrics['metric2'], desired_metrics['metric2'])

    def test_repeated(self):
        actual_scores, actual_metrics = metrics.operating_curve(
            [np.array([1.0, 0.9, 0.2]), np.array([1.0, 0.9, 0.6, 0.2])],
            {'metric1': [np.array([1.0, 2.0, 4.0]), np.array([8.0, 16.0, 32.0, 64.0])]})
        desired_scores = np.array([0.9, 0.6, 0.2])
        desired_metrics = {'metric1': np.array([9.0, 18.0, 34.0, 68.0])}
        np.testing.assert_allclose(actual_scores, desired_scores)
        np.testing.assert_allclose(actual_metrics['metric1'], desired_metrics['metric1'])


class TestParetoIntegrate(unittest.TestCase):

    def test_basic(self):
        a = np.array([0.1, 0.2, 0.8])
        b = np.array([0.9, 0.5, 0.3])
        # 0.1 * 0.9 + 0.1 * 0.5 + 0.6 * 0.3 = 0.09 + 0.05 + 0.18 = 0.32
        desired = 0.32
        self.assertAlmostEqual(metrics.pareto_integrate(a, b), desired)
        self.assertAlmostEqual(metrics.pareto_integrate(b, a), desired)

    def test_irrelevant(self):
        a = np.array([0.05, 0.1, 0.15, 0.2, 0.7, 0.8])
        b = np.array([0.8, 0.9, 0.5, 0.5, 0.1, 0.3])
        # 0.1 * 0.9 + 0.1 * 0.5 + 0.6 * 0.3 = 0.09 + 0.05 + 0.18 = 0.32
        desired = 0.32
        self.assertAlmostEqual(metrics.pareto_integrate(a, b), desired)
        self.assertAlmostEqual(metrics.pareto_integrate(b, a), desired)

    def test_commutative_random(self):
        n = 100
        rng = np.random.RandomState(0)
        a = rng.uniform(size=(n,))
        b = rng.uniform(size=(n,))
        self.assertAlmostEqual(
            metrics.pareto_integrate(b, a),
            metrics.pareto_integrate(a, b))

    def test_bounds_random(self):
        n = 100
        rng = np.random.RandomState(0)
        a = rng.uniform(size=(n,))
        b = rng.uniform(size=(n,))
        result = metrics.pareto_integrate(a, b)
        self.assertGreater(result, 0)
        self.assertLess(result, 1)

    def test_monotonic_random(self):
        n = 100
        m = n // 2
        rng = np.random.RandomState(0)
        a = rng.uniform(size=(n,))
        b = rng.uniform(size=(n,))
        self.assertLessEqual(
            metrics.pareto_integrate(a[:m], b[:m]),
            metrics.pareto_integrate(a, b))


class TestParetoIntercept(unittest.TestCase):

    def test_basic(self):
        a = np.array([0.1, 0.2, 0.8])
        b = np.array([0.9, 0.5, 0.3])
        q = np.array([-1.0, 0.0, 0.05, 0.1, 0.15, 0.2, 0.5, 0.7, 0.8, 0.9, 1.0, 2.0])
        desired = np.array([0.9, 0.9, 0.9, 0.9, 0.5, 0.5, 0.3, 0.3, 0.3, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(metrics.pareto_intercept(a, b, q), desired)

    def test_irrelevant(self):
        a = np.array([0.05, 0.1, 0.15, 0.2, 0.7, 0.8])
        b = np.array([0.8, 0.9, 0.5, 0.5, 0.1, 0.3])
        q = np.array([-1.0, 0.0, 0.05, 0.1, 0.15, 0.2, 0.5, 0.7, 0.8, 0.9, 1.0, 2.0])
        desired = np.array([0.9, 0.9, 0.9, 0.9, 0.5, 0.5, 0.3, 0.3, 0.3, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(metrics.pareto_intercept(a, b, q), desired)

    def test_2d(self):
        a = np.array([0.1, 0.2, 0.8])
        b = np.array([0.9, 0.5, 0.3])
        q = np.array([[0.09, 0.11], [0.79, 0.81]])
        desired = np.array([[0.9, 0.5], [0.3, 0.0]])
        np.testing.assert_array_equal(metrics.pareto_intercept(a, b, q), desired)
