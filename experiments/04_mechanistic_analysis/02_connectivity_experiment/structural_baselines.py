#!/usr/bin/env python3
"""
Structural connectivity algorithms and baseline graph expanders.

Provides:
1. build_fk_adj: Construct undirected table-level FK adjacency graph from DDL CSV.
2. shortest_path_union: Pairwise shortest FK paths between seed tables.
3. steiner_tree_union: Minimum Steiner tree approximation connecting seed tables.
4. rank_expansion & pad_with_dense: Budget-aligned candidate ranking up to K.
5. is_induced_connected: Exact check if target tables are connected in the induced subgraph.
"""

from collections import deque
import re
from typing import Dict, Iterable, List, Set, Tuple, Any
import pandas as pd


def strict_name(value: Any) -> str:
    """Standard table normalization: db_name.table_name, lowercase, normalized separators."""
    name = str(value).split("(", 1)[0].strip().lower()
    name = re.sub(r"[.\s]+", "_", name)
    return name.strip("_")


def build_fk_adj(schema_csv_path: str) -> Dict[str, Set[str]]:
    """Build table-level undirected foreign key adjacency graph per database."""
    df = pd.read_csv(schema_csv_path)
    adj: Dict[str, Set[str]] = {}

    for _, row in df.iterrows():
        db = str(row.get("db_name", "")).strip()
        tbl = str(row.get("table_name", "")).strip()
        t1 = strict_name(f"{db}.{tbl}")
        adj.setdefault(t1, set())

        fk = str(row.get("foreign_key_ref", ""))
        if "." in fk and pd.notna(row.get("column_name")):
            ref_tbl = fk.split(".")[0].strip()
            t2 = strict_name(f"{db}.{ref_tbl}")
            adj.setdefault(t2, set())
            if t1 != t2:
                adj[t1].add(t2)
                adj[t2].add(t1)

    return adj


def bfs_from(adj: Dict[str, Set[str]], source: str) -> Tuple[Dict[str, int], Dict[str, str]]:
    """Unweighted BFS on the undirected FK graph.

    Returns (dist, parent) where parent[source] is None.
    """
    dist = {source: 0}
    parent = {source: None}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in adj.get(u, set()):
            if v not in dist:
                dist[v] = dist[u] + 1
                parent[v] = u
                queue.append(v)
    return dist, parent


def reconstruct_path(parent: Dict[str, str], target: str) -> List[str]:
    """Reconstruct path from source to target using BFS parent map."""
    path = [target]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    return path[::-1]


def multi_source_bfs(adj: Dict[str, Set[str]], sources: Iterable[str]) -> Dict[str, int]:
    """BFS distance from the nearest source."""
    dist = {s: 0 for s in sources}
    queue = deque(dist.keys())
    while queue:
        u = queue.popleft()
        for v in adj.get(u, set()):
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def shortest_path_union(
    adj: Dict[str, Set[str]],
    seeds: List[str],
) -> Set[str]:
    """Intermediate tables on pairwise shortest paths between all seed pairs."""
    unique_seeds = list(dict.fromkeys(seeds))
    bfs_cache = {s: bfs_from(adj, s) for s in unique_seeds}
    intermediates: Set[str] = set()

    for i, a in enumerate(unique_seeds):
        for b in unique_seeds[i + 1:]:
            _, parent_b = bfs_cache[b]
            if a in parent_b:  # reachable in the same DB component
                for node in reconstruct_path(parent_b, a):
                    intermediates.add(node)

    return intermediates - set(unique_seeds)


def _mst_edges_on_terminals(
    dist_between: Dict[Tuple[str, str], int],
    terminals: List[str],
) -> List[Tuple[str, str]]:
    """Prim's MST over the terminal distance graph."""
    if len(terminals) < 2:
        return []
    in_tree = {terminals[0]}
    edges: List[Tuple[str, str]] = []
    remaining = set(terminals[1:])
    while remaining:
        best = None
        best_cost = None
        for u in in_tree:
            for v in remaining:
                cost = dist_between.get((u, v))
                if cost is not None and (best_cost is None or cost < best_cost):
                    best, best_cost = (u, v), cost
        if best is None:  # Disconnected terminal across components
            root = min(remaining)
            in_tree.add(root)
            remaining.discard(root)
            continue
        u, v = best
        edges.append(best)
        in_tree.add(v)
        remaining.discard(v)
    return edges


def steiner_tree_union(
    adj: Dict[str, Set[str]],
    seeds: List[str],
) -> Set[str]:
    """Approximate minimum Steiner tree connecting the seed tables.

    Standard 2-approximation:
    1. Compute pairwise BFS distance between seed terminals.
    2. Build MST over the metric closure.
    3. Union the shortest paths corresponding to MST edges.
    4. Iteratively prune non-terminal leaves.
    """
    unique_seeds = list(dict.fromkeys(seeds))
    if len(unique_seeds) < 2:
        return set()

    bfs_cache = {s: bfs_from(adj, s) for s in unique_seeds}
    dist_between: Dict[Tuple[str, str], int] = {}
    for i, a in enumerate(unique_seeds):
        for b in unique_seeds[i + 1:]:
            dist_a, _ = bfs_cache[a]
            if b in dist_a:
                dist_between[(a, b)] = dist_a[b]
                dist_between[(b, a)] = dist_a[b]

    nodes: Set[str] = set(unique_seeds)
    for u, v in _mst_edges_on_terminals(dist_between, unique_seeds):
        _, parent_v = bfs_cache[v]
        if u in parent_v:
            nodes.update(reconstruct_path(parent_v, u))

    # Prune non-terminal leaves iteratively
    terminal_set = set(unique_seeds)
    changed = True
    while changed:
        changed = False
        degree: Dict[str, int] = {n: 0 for n in nodes}
        for n in nodes:
            for nb in adj.get(n, set()):
                if nb in nodes:
                    degree[n] += 1
        for n in list(nodes):
            if n not in terminal_set and degree.get(n, 0) <= 1:
                nodes.discard(n)
                changed = True

    return nodes - terminal_set


def rank_expansion(
    seeds: List[str],
    intermediates: Set[str],
    retrieval_order: List[str],
    adj: Dict[str, Set[str]],
) -> List[str]:
    """Rank seeds first in dense order, then intermediates sorted by graph proximity."""
    unique_seeds = list(dict.fromkeys(seeds))
    if not intermediates:
        return list(unique_seeds)

    dist_seed = multi_source_bfs(adj, unique_seeds)
    # Order rank mapping for tie-breaking
    rank_map = {t: i for i, t in enumerate(retrieval_order)}

    ordered = sorted(
        intermediates,
        key=lambda t: (
            dist_seed.get(t, float("inf")),
            rank_map.get(t, 999999),
            t,
        ),
    )
    return unique_seeds + ordered


def shortest_path_to_set(
    adj: Dict[str, Set[str]],
    source: str,
    target_set: Set[str],
) -> List[str]:
    """Find shortest path from source to any node in target_set."""
    if source in target_set:
        return [source]
    dist = {source: 0}
    parent = {source: None}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        if u in target_set:
            return reconstruct_path(parent, u)
        for v in adj.get(u, set()):
            if v not in dist:
                dist[v] = dist[u] + 1
                parent[v] = u
                queue.append(v)
    return []


def sequential_shortest_path_expansion(
    seeds: List[str],
    adj: Dict[str, Set[str]],
    dense_order: List[str],
    k: int = 15,
) -> List[str]:
    """Sequentially expand seeds according to original dense ranking.

    As each seed is processed, intermediate tables on shortest paths between
    it and preceding seeds are appended in order immediately.
    """
    output: List[str] = []
    seen: Set[str] = set()
    bfs_cache: Dict[str, Tuple[Dict[str, int], Dict[str, str]]] = {}

    for i, s in enumerate(seeds):
        if s not in seen:
            output.append(s)
            seen.add(s)
        if s not in bfs_cache:
            bfs_cache[s] = bfs_from(adj, s)
        _, parent_s = bfs_cache[s]

        # Connect with all preceding seeds in priority order
        for j in range(i):
            pj = seeds[j]
            if pj in parent_s:
                path = reconstruct_path(parent_s, pj)
                for inter in path[1:-1]:
                    if inter not in seen:
                        output.append(inter)
                        seen.add(inter)

    # Pad with remaining dense retrieval ranking up to budget K
    for name in dense_order:
        if len(output) >= k:
            break
        if name not in seen:
            output.append(name)
            seen.add(name)

    return output[:k]


def sequential_steiner_tree_expansion(
    seeds: List[str],
    adj: Dict[str, Set[str]],
    dense_order: List[str],
    k: int = 15,
) -> List[str]:
    """Sequentially expand seeds via incremental Steiner tree (Takahashi-Matsuyama).

    Maintains the current tree/forest. For each seed in order, connects it to the
    nearest node in the current tree via shortest path, appending intermediate tables.
    """
    output: List[str] = []
    seen: Set[str] = set()
    tree: Set[str] = set()

    for i, s in enumerate(seeds):
        if s not in seen:
            output.append(s)
            seen.add(s)

        if not tree:
            tree.add(s)
        else:
            path = shortest_path_to_set(adj, s, tree)
            if path:
                for node in path:
                    tree.add(node)
                    if node not in seen:
                        output.append(node)
                        seen.add(node)
            else:
                tree.add(s)

    # Pad with remaining dense retrieval ranking up to budget K
    for name in dense_order:
        if len(output) >= k:
            break
        if name not in seen:
            output.append(name)
            seen.add(name)

    return output[:k]


def pad_with_dense(ranked: List[str], dense_order: List[str], k: int = 15) -> List[str]:
    """Pad ranked expansion candidates with dense retrieval ranking up to budget K."""
    output = list(dict.fromkeys(ranked))
    for name in dense_order:
        if len(output) >= k:
            break
        if name not in output:
            output.append(name)
    return output[:k]


def is_induced_connected(
    pred_set: Set[str],
    gold_targets: Set[str],
    adj: Dict[str, Set[str]],
) -> bool:
    """Are all gold_targets completely in pred_set AND in one connected component of pred_set's induced subgraph?"""
    if not gold_targets:
        return True
    # If any gold target is missing from the prediction set, cannot be connected
    if not (gold_targets <= pred_set):
        return False
    if len(gold_targets) == 1:
        return True

    # BFS within pred_set subgraph starting from an arbitrary gold table
    start = next(iter(gold_targets))
    seen = {start}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in adj.get(u, set()):
            if v in pred_set and v not in seen:
                seen.add(v)
                queue.append(v)

    # Check whether all gold tables were reached
    return gold_targets <= seen
