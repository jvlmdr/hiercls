from typing import Any, Callable, Optional

            
def tree_map(fn: Callable, tree: Any, is_leaf: Optional[Callable] = None) -> Any:
    """Like jax.tree_util.tree_map()."""
    def helper(x: Any) -> Any:
        if is_leaf is not None and is_leaf(x):
            return fn(x)
        elif isinstance(x, dict):
            return dict(zip(x.keys(), map(helper, x.values())))
        elif isinstance(x, list):
            return list(map(helper, x))
        elif isinstance(x, tuple):
            return tuple(map(helper, x))
        elif is_leaf is None:
            return fn(x)
        else:
            raise ValueError('not a leaf, not a known container', type(x))

    return helper(tree)
