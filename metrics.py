from typing import Optional

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
    info = -np.log(hier.uniform_leaf(tree))
    return excess(info, tree, *args)


def deficient_info(tree: hier.Hierarchy, *args) -> np.ndarray:
    info = -np.log(hier.uniform_leaf(tree))
    return deficient(info, tree, *args)


class LCAMetric:

    def __init__(self, tree: hier.Hierarchy, value: np.ndarray):
        self.value = value
        self.find_lca = hier.FindLCA(tree)

    def excess(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        lca = self.find_lca(gt, pr)
        return self.value[pr] - self.value[lca]

    def deficient(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        lca = self.find_lca(gt, pr)
        return self.value[gt] - self.value[lca]

    def dist(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        lca = self.find_lca(gt, pr)
        excess = self.value[pr] - self.value[lca]
        deficient = self.value[gt] - self.value[lca]
        return excess + deficient


def InfoMetric(tree: hier.Hierarchy) -> LCAMetric:
    info = -np.log(hier.uniform_leaf(tree))
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
