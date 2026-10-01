#!/usr/bin/env python3
"""
Join-Path Coverage (PathCov) and Induced Subgraph Connectivity (Conn) Experiment.

Methods Compared:
1. SingleDPR: Pure dense retrieval (BGE-M3), no structural constraint.
2. Shortest Path: Top-m dense seeds + pairwise shortest FK paths union between seeds, padded to K.
3. Steiner Tree: Top-m dense seeds + 2-approx minimum Steiner tree on FK graph, padded to K.
4. GERT (Ours): Global dense ranking + rank-decayed PPR continuous structural diffusion on FK graph.

Evaluated on:
- Multi-table query subset (at least two gold tables)
- Full dataset
- Table-count stratification (1, 2, 3, 4+ Tables)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from structural_baselines import (
    build_fk_adj,
    pad_with_dense,
    rank_expansion,
    sequential_shortest_path_expansion,
    sequential_steiner_tree_expansion,
    shortest_path_union,
    steiner_tree_union,
    strict_name,
)
from connectivity_metrics import aggregate_metrics, evaluate_query_connectivity


def _find_data_file(name: str) -> Path:
    candidates = [
        SCRIPT_DIR / "data_inputs" / f"{name}.json",
        REPO_ROOT / "experiments" / "04_mechanistic_analysis" / "02_connectivity_experiment" / "data_inputs" / f"{name}.json",
        REPO_ROOT / "output" / "merged_dense_gert" / f"{name}.json",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]


def _find_schema_csv(rel_path: str) -> Path:
    candidates = [
        REPO_ROOT / "data" / rel_path,
        REPO_ROOT / "schema_link" / "dataset_v2" / rel_path,
        REPO_ROOT / "dataset_v2" / rel_path,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]


DATASETS = {
    "SpiderUnion": {
        "data_file": _find_data_file("SpiderUnion"),
        "schema_csv": _find_schema_csv("spider/spider_union_schema_FK.csv"),
    },
    "BirdUnion": {
        "data_file": _find_data_file("BirdUnion"),
        "schema_csv": _find_schema_csv("bird/bird_union_schema_FK.csv"),
    },
    "SynLink": {
        "data_file": _find_data_file("SynLink"),
        "schema_csv": _find_schema_csv("SynLink/SynSQL_schema_csv_300.csv"),
    },
}

METHODS = ["SingleDPR", "Shortest Path", "Steiner Tree", "GERT"]


def extract_gold_tables(raw_schema: Any) -> List[str]:
    """Extract list of gold table names from gold_schema dict or list."""
    if isinstance(raw_schema, dict):
        return [strict_name(t) for t in raw_schema.keys()]
    elif isinstance(raw_schema, list):
        return [strict_name(t) for t in raw_schema]
    return []


def run_dataset_experiment(
    ds_name: str,
    data_path: Path,
    schema_path: Path,
    eval_k: int = 15,
    seed_budget: int = 20,
    save_pred: bool = True,
) -> Dict[str, Any]:
    print(f"\n{'='*75}")
    print(f"PROCESSING: {ds_name} (K={eval_k}, Unified Seed Budget m={seed_budget})")
    print(f"{'='*75}")

    print(f"[1/3] Loading foreign key adjacency graph from {schema_path.name}...")
    adj = build_fk_adj(str(schema_path))
    num_nodes = len(adj)
    num_edges = sum(len(neighbors) for neighbors in adj.values()) // 2
    print(f"      FK Graph: {num_nodes} tables, {num_edges} foreign key edges.")

    print(f"[2/3] Loading query data from {data_path.name}...")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = {m: [] for m in METHODS}
    metrics_overall = {m: [] for m in METHODS}
    metrics_multitable = {m: [] for m in METHODS}
    metrics_by_count = {m: {b: [] for b in ("1", "2", "3", "4+")} for m in METHODS}

    multi_table_count = 0

    print(f"[3/3] Generating predictions & computing PathCov / Conn for {len(data)} queries...")
    for item in tqdm(data, desc=f"Evaluating {ds_name}"):
        gold_tables = extract_gold_tables(item.get("gold_schema", item.get("schema_link", [])))
        if not gold_tables:
            continue

        num_gold = len(gold_tables)
        is_multi = num_gold >= 2
        if is_multi:
            multi_table_count += 1
        b_key = str(num_gold) if num_gold in (1, 2, 3) else "4+"

        pred_dense = [strict_name(t) for t in item["pred_dense"]]
        pred_gert = [strict_name(t) for t in item["pred_gert"]]

        # 1. SingleDPR (Pure dense retrieval)
        dpr_top = pred_dense[:eval_k]

        # 2. Shortest Path (Sequential pairwise expansion on top-20 seeds)
        sp_seeds = pred_dense[:seed_budget]
        sp_top = sequential_shortest_path_expansion(sp_seeds, adj, pred_dense, k=eval_k)

        # 3. Steiner Tree (Sequential incremental tree expansion on top-20 seeds)
        st_seeds = pred_dense[:seed_budget]
        st_top = sequential_steiner_tree_expansion(st_seeds, adj, pred_dense, k=eval_k)

        # 4. GERT (Ours: continuous rank-decayed PPR diffusion)
        gert_top = pred_gert[:eval_k]

        method_preds = {
            "SingleDPR": dpr_top,
            "Shortest Path": sp_top,
            "Steiner Tree": st_top,
            "GERT": gert_top,
        }

        for m in METHODS:
            pred_k = method_preds[m]
            predictions[m].append({
                "question_id": item.get("question_id"),
                "db_id": item.get("db_id"),
                "question": item.get("question"),
                "gold_tables": gold_tables,
                "pred_tables": pred_k,
            })

            q_metrics = evaluate_query_connectivity(pred_k, gold_tables, adj, k=eval_k)
            metrics_overall[m].append(q_metrics)
            metrics_by_count[m][b_key].append(q_metrics)
            if is_multi:
                metrics_multitable[m].append(q_metrics)

    # Aggregations
    agg_overall = {m: aggregate_metrics(metrics_overall[m]) for m in METHODS}
    agg_multitable = {m: aggregate_metrics(metrics_multitable[m]) for m in METHODS}
    agg_stratified = {
        m: {b: aggregate_metrics(metrics_by_count[m][b]) for b in ("1", "2", "3", "4+")}
        for m in METHODS
    }

    # Save predictions
    if save_pred:
        pred_dir = SCRIPT_DIR / "predictions" / ds_name
        pred_dir.mkdir(parents=True, exist_ok=True)
        for m in METHODS:
            m_slug = m.lower().replace(" ", "_")
            out_file = pred_dir / f"{m_slug}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(predictions[m], f, indent=2, ensure_ascii=False)

    return {
        "dataset": ds_name,
        "total_queries": len(data),
        "multi_table_queries": multi_table_count,
        "single_table_queries": len(data) - multi_table_count,
        "overall": agg_overall,
        "multi_table": agg_multitable,
        "stratified": agg_stratified,
    }


def format_markdown_table(all_results: Dict[str, Any], subset_type: str = "multi_table") -> str:
    """Format results into a publication-quality Markdown table."""
    title = "Multi-table queries (at least two gold tables)" if subset_type == "multi_table" else "All queries"
    lines = [
        f"### {title}",
        "",
        "| Dataset | Method | Path coverage (PathCov@15) | Induced connectivity (Conn@15) | Table recall | Precision | F1 |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for ds_name, res in all_results.items():
        sample_count = res["multi_table_queries"] if subset_type == "multi_table" else res["total_queries"]
        ds_label = f"**{ds_name}**<br>*(N={sample_count})*"
        metrics_dict = res[subset_type]

        for i, m in enumerate(METHODS):
            m_res = metrics_dict[m]
            prefix = ds_label if i == 0 else ""

            def cell(key: str) -> str:
                value = m_res[key]
                formatted = f"{value * 100:.2f}%"
                return f"**{formatted}**" if value == max(metrics_dict[other][key] for other in METHODS) else formatted

            pc = cell("path_coverage")
            conn = cell("connectivity")
            rec = cell("table_rec")
            prec = cell("precision")
            f1 = cell("f1")

            lines.append(f"| {prefix} | {m} | {pc} | {conn} | {rec} | {prec} | {f1} |")

    return "\n".join(lines)


def format_latex_table(all_results: Dict[str, Any]) -> str:
    """Format multi-table comparison into IEEE conference LaTeX code."""
    latex = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \caption{Path coverage and induced connectivity on queries with at least two gold tables (\%, $K=15$).}",
        r"  \label{tab:connectivity-pathcov}",
        r"  \vspace{-0.8em}",
        r"  \begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llccccc}",
        r"    \toprule",
        r"    Dataset & Retrieval method & Path coverage (PathCov) & Induced connectivity (Conn) & Table recall & Precision & F1 \\",
        r"    \midrule",
    ]

    for ds_name, res in all_results.items():
        m_dict = res["multi_table"]
        for index, method in enumerate(METHODS):
            def cell(key: str) -> str:
                value = m_dict[method][key]
                formatted = f"{value * 100:.2f}"
                return rf"\textbf{{{formatted}}}" if value == max(m_dict[other][key] for other in METHODS) else formatted

            label = rf"\multirow{{4}}{{*}}{{{ds_name}}}" if index == 0 else ""
            values = " & ".join(cell(key) for key in ("path_coverage", "connectivity", "table_rec", "precision", "f1"))
            latex.append(f"    {label} & {method} & {values} \\\\")
        latex.append(r"    \midrule" if ds_name != "SynLink" else r"    \bottomrule")

    latex.extend([
        r"  \end{tabular*}",
        r"\end{table*}",
    ])
    return "\n".join(latex)


def main():
    parser = argparse.ArgumentParser(description="Run connectivity and path coverage experiment.")
    parser.add_argument("--k", type=int, default=15, help="Candidate budget K (default: 15)")
    parser.add_argument("--seed-budget", type=int, default=20, help="Seed budget m for structural baselines (default: 20, unified with GERT k_s=20)")
    args = parser.parse_args()

    results_dir = SCRIPT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for ds_name, cfg in DATASETS.items():
        res = run_dataset_experiment(
            ds_name=ds_name,
            data_path=cfg["data_file"],
            schema_path=cfg["schema_csv"],
            eval_k=args.k,
            seed_budget=args.seed_budget,
            save_pred=True,
        )
        all_results[ds_name] = res

    # 1. Save JSON metrics
    with open(results_dir / "connectivity_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 2. Format Markdown tables
    md_multitable = format_markdown_table(all_results, subset_type="multi_table")
    with open(results_dir / "connectivity_table_multitable.md", "w", encoding="utf-8") as f:
        f.write(md_multitable)

    md_overall = format_markdown_table(all_results, subset_type="overall")
    with open(results_dir / "connectivity_table_overall.md", "w", encoding="utf-8") as f:
        f.write(md_overall)

    # 3. Format LaTeX table
    tex_code = format_latex_table(all_results)
    with open(results_dir / "connectivity_table.tex", "w", encoding="utf-8") as f:
        f.write(tex_code)

    print("\n" + "=" * 90)
    print("EXPERIMENT COMPLETED! SUMMARY ON MULTI-TABLE SUBSET (>= 2 Tables):")
    print("=" * 90)
    print(md_multitable)
    print("\n" + "=" * 90)
    print("SUMMARY ON OVERALL FULL DATASET:")
    print("=" * 90)
    print(md_overall)
    print(f"\nAll artifacts saved to: {results_dir}")


if __name__ == "__main__":
    main()

