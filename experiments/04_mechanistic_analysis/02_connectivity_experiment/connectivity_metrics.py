#!/usr/bin/env python3
"""
Evaluation metrics for Join-Path Coverage (PathCov@K) and Induced Subgraph Connectivity (Conn@K).

Objective definitions (without subjective bridge table labels):
* PathCov@K: Fraction of queries where ALL gold tables referenced in the query
             are covered in the Top-K prediction (G_i ⊆ S_i^K).
* Conn@K: Fraction of queries where ALL gold tables are retrieved AND form
          a single connected component in the Top-K induced FK subgraph.
* Table Rec.@K: Macro-average recall over gold tables.
* Precision@K: Macro-average precision over Top-K candidate tables.
* F1@K: Macro-average harmonic mean of precision and recall.
"""

from typing import Dict, List, Sequence, Set
from structural_baselines import is_induced_connected, strict_name


def evaluate_query_connectivity(
    pred_tables: Sequence[str],
    gold_tables: Sequence[str],
    adj: Dict[str, Set[str]],
    k: int = 15,
) -> Dict[str, float]:
    """Compute retrieval, path coverage, and connectivity for a single query."""
    pred_list = [strict_name(t) for t in pred_tables[:k]]
    pred_set = set(pred_list)
    gold_set = {strict_name(t) for t in gold_tables}

    if not gold_set:
        return {
            "table_rec": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "path_coverage": 0.0,
            "connectivity": 0.0,
        }

    hits = len(gold_set & pred_set)
    rec = hits / len(gold_set)
    prec = hits / len(pred_list) if pred_list else 0.0
    f1 = (2 * rec * prec / (rec + prec)) if (rec + prec) > 0 else 0.0
    path_cov = 1.0 if (gold_set <= pred_set) else 0.0
    conn = 1.0 if is_induced_connected(pred_set, gold_set, adj) else 0.0

    return {
        "table_rec": rec,
        "table_cr": path_cov,
        "conn": conn,
        "path_coverage": path_cov,
        "connectivity": conn,
        "precision": prec,
        "f1": f1,
    }


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Calculate macro-average over query metric dictionaries."""
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    n = len(metrics_list)
    return {k: sum(m[k] for m in metrics_list) / n for k in keys}
