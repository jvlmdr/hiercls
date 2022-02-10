from typing import Hashable, Iterable, List, Sequence

import networkx as nx


def dfs_edges_with_order(g: nx.DiGraph, order: Sequence[Hashable]) -> List[Hashable]:
    visited = set()
    edges = []

    def visit(node):
        if node in visited:
            return
        visited.add(node)
        if not g.in_degree[node]:
            return
        parents = list(g.predecessors(node))
        if len(parents) > 1:
            raise ValueError('multiple parents', node, parents)
        parent, = parents
        visit(parent)
        edges.append((parent, node))
    
    for leaf in order:
        visit(leaf)

    return edges


def unique_in_order(elems: Iterable[Hashable]) -> Iterable[Hashable]:
    seen = set()
    for x in elems:
        if x in seen:
            continue
        yield x
        seen.add(x)
