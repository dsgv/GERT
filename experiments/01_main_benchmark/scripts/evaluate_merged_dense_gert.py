#!/usr/bin/env python3
"""
Evaluate merged Dense and GERT predictions from a single JSON file.
Each JSON file contains query items with both:
  - 'pred_dense' (dense retrieval output, w/o PPR)
  - 'pred_gert'  (GERT graph-enhanced reranking output, GERT Full)

Computes:
  1. Standard metrics at K=15: Table Rec., Perfect Recall (Table CR), Precision, F1
  2. Multi-table analysis breakdown (1 Table, 2 Tables, 3 Tables, 4+ Tables)
  3. In-depth recovery metrics: Recovery@15, RescueRate@15, DamageRate@15
"""

import json
import os
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Set, Tuple
import pandas as pd


def strict_name(value: Any) -> str:
    """Normalize table name strictly."""
    name = str(value).split("(", 1)[0].strip().lower()
    name = re.sub(r"[.\s]+", "_", name)
    return name.strip("_")


def evaluate_dataset_file(json_path: Path, eval_k: int = 15) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Accumulators for Dense and GERT
    dense_recs, dense_prs, dense_precs, dense_f1s = [], [], [], []
    gert_recs, gert_prs, gert_precs, gert_f1s = [], [], [], []

    grouped_dense = {b: [0, 0] for b in ("1", "2", "3", "4+")}
    grouped_gert = {b: [0, 0] for b in ("1", "2", "3", "4+")}

    # Recovery & Rescue counters
    total_omitted = 0
    recovered_omitted = 0
    dense_fail_queries = 0
    rescued_queries = 0
    dense_succ_queries = 0
    damaged_queries = 0

    for item in data:
        # Extract gold tables
        gold_raw = item.get("gold_schema", item.get("schema_link", []))
        if isinstance(gold_raw, dict):
            gold_tables = {strict_name(t) for t in gold_raw.keys()}
        elif isinstance(gold_raw, list):
            gold_tables = {strict_name(t) for t in gold_raw}
        else:
            gold_tables = set()

        if not gold_tables:
            continue

        num_tables = len(gold_tables)
        b_key = str(num_tables) if num_tables in (1, 2, 3) else "4+"

        # Extract predictions for Dense and GERT
        pred_dense_raw = item.get("pred_dense") or item.get("dense") or []
        pred_gert_raw = item.get("pred_schema") or item.get("pred_gert") or item.get("gert") or item.get("pred_schema_link") or []

        d_k = [strict_name(t) for t in pred_dense_raw[:eval_k]]
        s_k = [strict_name(t) for t in pred_gert_raw[:eval_k]]

        set_d = set(d_k)
        set_s = set(s_k)

        # 1. Dense Metrics
        hits_d = len(gold_tables & set_d)
        rec_d = hits_d / len(gold_tables)
        prec_d = hits_d / len(d_k) if d_k else 0.0
        f1_d = 2 * rec_d * prec_d / (rec_d + prec_d) if (rec_d + prec_d) else 0.0
        pr_d = 1.0 if gold_tables <= set_d else 0.0

        dense_recs.append(rec_d)
        dense_prs.append(pr_d)
        dense_precs.append(prec_d)
        dense_f1s.append(f1_d)

        grouped_dense[b_key][0] += int(pr_d)
        grouped_dense[b_key][1] += 1

        # 2. GERT Metrics
        hits_s = len(gold_tables & set_s)
        rec_s = hits_s / len(gold_tables)
        prec_s = hits_s / len(s_k) if s_k else 0.0
        f1_s = 2 * rec_s * prec_s / (rec_s + prec_s) if (rec_s + prec_s) else 0.0
        pr_s = 1.0 if gold_tables <= set_s else 0.0

        gert_recs.append(rec_s)
        gert_prs.append(pr_s)
        gert_precs.append(prec_s)
        gert_f1s.append(f1_s)

        grouped_gert[b_key][0] += int(pr_s)
        grouped_gert[b_key][1] += 1

        # 3. Recovery & Rescue metrics
        omitted = gold_tables - set_d
        total_omitted += len(omitted)
        recovered_omitted += len(omitted & set_s)

        if pr_d == 0.0:
            dense_fail_queries += 1
            if pr_s == 1.0:
                rescued_queries += 1
        else:
            dense_succ_queries += 1
            if pr_s == 0.0:
                damaged_queries += 1

    return {
        "dataset": json_path.stem,
        "total_queries": len(dense_recs),
        "dense": {
            "Table Rec.": mean(dense_recs),
            "Table CR": mean(dense_prs),
            "Precision": mean(dense_precs),
            "F1": mean(dense_f1s),
            "Grouped Table CR": {
                b: (grouped_dense[b][0] / grouped_dense[b][1] if grouped_dense[b][1] else 0.0)
                for b in ("1", "2", "3", "4+")
            },
            "Grouped Counts": {b: grouped_dense[b][1] for b in ("1", "2", "3", "4+")},
        },
        "gert": {
            "Table Rec.": mean(gert_recs),
            "Table CR": mean(gert_prs),
            "Precision": mean(gert_precs),
            "F1": mean(gert_f1s),
            "Grouped Table CR": {
                b: (grouped_gert[b][0] / grouped_gert[b][1] if grouped_gert[b][1] else 0.0)
                for b in ("1", "2", "3", "4+")
            },
        },
        "recovery": {
            "total_omitted_tables": total_omitted,
            "recovered_tables": recovered_omitted,
            "Table Recovery": recovered_omitted / total_omitted if total_omitted else 0.0,
            "dense_fail_queries": dense_fail_queries,
            "rescued_queries": rescued_queries,
            "Query Gain": rescued_queries / dense_fail_queries if dense_fail_queries else 0.0,
            "dense_succ_queries": dense_succ_queries,
            "damaged_queries": damaged_queries,
            "Query Loss": damaged_queries / dense_succ_queries if dense_succ_queries else 0.0,
        },
    }


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]
    candidate_dirs = [
        script_dir / "output" / "merged_dense_gert",
        repo_root / "experiments" / "04_mechanistic_analysis" / "02_connectivity_experiment" / "data_inputs",
        repo_root / "output" / "merged_dense_gert",
    ]
    base_dir = next((p for p in candidate_dirs if p.exists() and (p / "SpiderUnion.json").exists()), candidate_dirs[0])
    datasets = ["SpiderUnion", "BirdUnion", "SynLink"]

    print("=" * 90)
    print("Merged JSON metric report (K=15)")
    print(f"Source directory: {base_dir}")
    print("=" * 90)

    summary_rows = []
    grouped_rows = []

    for ds in datasets:
        file_path = base_dir / f"{ds}.json"
        if not file_path.exists():
            print(f"[Warning] File not found: {file_path}")
            continue

        res = evaluate_dataset_file(file_path, eval_k=15)
        d = res["dense"]
        g = res["gert"]
        r = res["recovery"]

        summary_rows.append({
            "Dataset": ds,
            "Dense Table CR": f"{d['Table CR'] * 100:.2f}%",
            "GERT Table CR": f"{g['Table CR'] * 100:.2f}%",
            "ΔCR": f"{(g['Table CR'] - d['Table CR']) * 100:+.2f}%",
            "Table Recovery": f"{r['Table Recovery'] * 100:.2f}%",
            "Query Gain": f"{r['Query Gain'] * 100:.2f}%",
            "Query Loss": f"{r['Query Loss'] * 100:.2f}%",
        })

        for b in ("1", "2", "3", "4+"):
            grouped_rows.append({
                "Dataset": ds,
                "Gold Tables": b,
                "N": d["Grouped Counts"][b],
                "Dense Table CR": f"{d['Grouped Table CR'][b] * 100:.2f}%",
                "GERT Table CR": f"{g['Grouped Table CR'][b] * 100:.2f}%",
                "ΔCR": f"{(g['Grouped Table CR'][b] - d['Grouped Table CR'][b]) * 100:+.2f}%",
            })

    print("\n### 1. Main Dense vs. GERT Comparison (Table 2 & Tables 6, 7)")
    print(pd.DataFrame(summary_rows).to_markdown(index=False))

    print("\n### 2. Multi-Table Table CR Stratification (Figure 2)")
    print(pd.DataFrame(grouped_rows).to_markdown(index=False))

    print("\n" + "=" * 90)
    print("Metric recomputation completed; compare these values with the paper tables before reporting.")
    print("=" * 90)


if __name__ == "__main__":
    main()
