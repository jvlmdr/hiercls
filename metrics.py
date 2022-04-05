from typing import Dict, List, Optional, Tuple

import numpy as np

import hier


# TODO: Be careful using "excess" metric when gt is not leaf node.
# Truncate predictions at gt? Revisit!
def truncate(
        tree: hier.Hierarchy,
        gt: np.ndarray,
        pr: np.ndarray,
        lca: Optional[np.ndarray] = None,
        ) -> np.ndarray:
    if lca is None:
        lca = hier.lca(tree, gt, pr)
    raise NotImplementedError


def edge_dist(tree: hier.Hierarchy, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
    depth_gt = tree.depths()[gt]
    depth_pr = tree.depths()[pr]
    depth_lca = hier.lca_depth(tree, gt, pr)
    return (depth_gt - depth_lca) + (depth_pr - depth_lca)


def correct(tree: hier.Hierarchy, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
    depths = tree.depths()
    depth_gt = depths[gt]
    depth_pr = depths[pr]
    depth_lca = hier.lca_depth(tree, gt, pr)
    # Correct if gt is below pr or pr is below gt.
    # If this is the case, lca == gt or lca == pr.
    return (depth_lca == depth_gt) | (depth_lca == depth_pr)


def value_at_lca(
        value: np.ndarray,
        tree: hier.Hierarchy,
        gt: np.ndarray,
        pr: np.ndarray) -> np.ndarray:
    return value[hier.lca(tree, gt, pr)]


def excess(
        value: np.ndarray,
        tree: hier.Hierarchy,
        gt: np.ndarray,
        pr: np.ndarray) -> np.ndarray:
    lca = hier.lca(tree, gt, pr)
    return value[pr] - value[lca]


def deficient(
        value: np.ndarray,
        tree: hier.Hierarchy,
        gt: np.ndarray,
        pr: np.ndarray) -> np.ndarray:
    lca = hier.lca(tree, gt, pr)
    return value[gt] - value[lca]


def excess_depth(tree: hier.Hierarchy, *args) -> np.ndarray:
    return excess(tree.depths(), tree, *args)


def deficient_depth(tree: hier.Hierarchy, *args) -> np.ndarray:
    return deficient(tree.depths(), tree, *args)


def excess_info(tree: hier.Hierarchy, *args) -> np.ndarray:
    info = -np.log2(hier.uniform_leaf(tree))
    return excess(info, tree, *args)


def deficient_info(tree: hier.Hierarchy, *args) -> np.ndarray:
    info = -np.log2(hier.uniform_leaf(tree))
    return deficient(info, tree, *args)


class LCAMetric:

    def __init__(self, tree: hier.Hierarchy, value: np.ndarray):
        self.value = value
        self.find_lca = hier.FindLCA(tree)

    def value_at_lca(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        lca = self.find_lca(gt, pr)
        return self.value[lca]

    def value_at_gt(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        # TODO: Avoid broadcasting of unused array?
        gt, _ = np.broadcast_arrays(gt, pr)
        return self.value[gt]

    def value_at_pr(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        # TODO: Avoid broadcasting of unused array?
        _, pr = np.broadcast_arrays(gt, pr)
        return self.value[pr]

    def deficient(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        lca = self.find_lca(gt, pr)
        return self.value[gt] - self.value[lca]

    def excess(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        lca = self.find_lca(gt, pr)
        return self.value[pr] - self.value[lca]

    def dist(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        lca = self.find_lca(gt, pr)
        excess = self.value[pr] - self.value[lca]
        deficient = self.value[gt] - self.value[lca]
        return excess + deficient

    def recall(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        lca = self.find_lca(gt, pr)
        gt_value = self.value[gt]
        lca_value = self.value[lca]
        with np.errstate(invalid='ignore'):
            return np.where((lca_value == 0) & (gt_value == 0), 1.0, lca_value / gt_value)

    def precision(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        lca = self.find_lca(gt, pr)
        pr_value = self.value[pr]
        lca_value = self.value[lca]
        with np.errstate(invalid='ignore'):
            return np.where((lca_value == 0) & (pr_value == 0), 1.0, lca_value / pr_value)

    def f1(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        lca = self.find_lca(gt, pr)
        gt_value = self.value[gt]
        pr_value = self.value[pr]
        lca_value = self.value[lca]
        with np.errstate(invalid='ignore'):
            r = np.where((lca_value == 0) & (gt_value == 0), 1.0, lca_value / gt_value)
            p = np.where((lca_value == 0) & (pr_value == 0), 1.0, lca_value / pr_value)
        with np.errstate(divide='ignore'):
            return 2 / (1/r + 1/p)


def UniformLeafInfoMetric(tree: hier.Hierarchy) -> LCAMetric:
    info = -np.log2(hier.uniform_leaf(tree))
    return LCAMetric(tree, info)


def DepthMetric(tree: hier.Hierarchy) -> LCAMetric:
    return LCAMetric(tree, tree.depths())


class IsCorrect:

    def __init__(self, tree: hier.Hierarchy):
        self.find_lca = hier.FindLCA(tree)
        self.depths = tree.depths()

    def __call__(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        lca = self.find_lca(gt, pr)
        depth_gt = self.depths[gt]
        depth_pr = self.depths[pr]
        depth_lca = self.depths[lca]
        # Correct if gt is below pr or pr is below gt.
        # If this is the case, lca == gt or lca == pr.
        return (depth_lca == depth_gt) | (depth_lca == depth_pr)


def operating_curve(
        example_scores: List[np.ndarray],
        example_metrics: Dict[str, List[np.ndarray]],
        ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Obtains operating curve for set of metrics.

    For each field in example_metrics, `example_scores[i]` and `example_metrics[field][i]`
    are arrays of the same length, ordered descending by score.
    This can be obtained using `infer.prediction_sequence()`.
    """
    # Obtain order of scores.
    # Note: Could do a merge sort here, since each array is already sorted.
    step_scores = np.concatenate([seq[1:] for seq in example_scores])
    step_order = np.argsort(-step_scores)
    step_scores = step_scores[step_order]
    step_totals = {}
    for field, example_values in example_metrics.items():
        # Convert to float since np.diff() treats bools as mod 2 arithmetic.
        example_values = [seq.astype(float) for seq in example_values]
        total_init = np.sum([seq[0] for seq in example_values])
        total_deltas = np.concatenate([np.diff(seq) for seq in example_values])[step_order]
        step_totals[field] = total_init + _cumsum_with_zero(total_deltas)
    return step_scores, step_totals


def _cumsum_with_zero(x: np.ndarray, axis: int = 0) -> np.ndarray:
    ndim = x.ndim
    pad_width = [(0, 0)] * ndim
    pad_width[axis] = (1, 0)
    return np.cumsum(np.pad(x, pad_width, 'constant'), axis=axis)


# def auc():
#     raise NotImplementedError
