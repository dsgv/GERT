#!/usr/bin/env python3
"""
Evaluate Table 3 from the paper: Component Ablation Study across SpiderUnion, BirdUnion, and SynLink (K=15).

Variants:
1. GERT (Full)
2. w/o Graph Propagation (beta=0, dense semantic retrieval only)
3. w/o Semantic Score (beta=1, stationary PPR ranking only)
4. w/o Seed Decay (uniform seed weights v_i = 1/|T_seed|)
"""

import json
import os
import re
import sys
from pathlib import Path
from statistics import mean
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]


def strict_name(value) -> str:
    name = str(value).split("(", 1)[0].strip().lower()
    name = re.sub(r"[.\s]+", "_", name)
    return name.strip("_")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(gold_rows, pred_rows, k: int = 15):
    recalls, complete, precisions, f1s = [], [], [], []
    for gold, pred_row in zip(gold_rows, pred_rows):
        gold_tables = {strict_name(t) for t in gold.get("gold_schema", gold.get("schema_link", []))}
        preds = pred_row.get("pred_schema", pred_row.get("pred_schema_link", []))
        if isinstance(preds, dict) and "k20" in preds:
            preds = preds["k20"]
        pred_tables = {strict_name(t) for t in preds[:k]}

        hits = len(gold_tables & pred_tables)
        rec = hits / len(gold_tables) if gold_tables else 0.0
        prec = hits / len(pred_tables) if pred_tables else 0.0
        f1 = (2 * rec * prec / (rec + prec)) if (rec + prec) > 0 else 0.0
        is_comp = float(gold_tables <= pred_tables)

        recalls.append(rec)
        complete.append(is_comp)
        precisions.append(prec)
        f1s.append(f1)

    return {
        "Table Recall (%)": mean(recalls) * 100,
        "Table CR (%)": mean(complete) * 100,
        "Precision (%)": mean(precisions) * 100,
        "F1-Score (%)": mean(f1s) * 100,
    }


def to_markdown_str(df) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join([":---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def main():
    gold_files = {
        "SpiderUnion": REPO_ROOT / "data" / "spider" / "murre_spider_dev.json",
        "BirdUnion": REPO_ROOT / "data" / "bird" / "murre_bird_dev.json",
        "SynLink": REPO_ROOT / "data" / "SynLink" / "Formal_moderate_1k_gd.json",
    }

    variants = [
        ("GERT (Full)", "GERT_Full.json"),
        ("w/o Graph Propagation", "w_o_PPR.json"),
        ("w/o Semantic Score", "w_o_Vector.json"),
        ("w/o Seed Decay", "w_o_SeedDecay.json"),
    ]

    res_dir = SCRIPT_DIR / "archive_results"
    rows = []

    for ds_name, gold_path in gold_files.items():
        if not gold_path.exists():
            print(f"Warning: Gold file not found: {gold_path}")
            continue
        gold = load_json(gold_path)
        sub_dir = res_dir / f"{ds_name.lower().replace('union', '')}_results"

        for label, fname in variants:
            pred_file = sub_dir / f"{ds_name}_{fname}"
            if pred_file.exists():
                pred = load_json(pred_file)
                cur_gold = gold[:len(pred)]
                metrics = evaluate(cur_gold, pred, k=15)
                rows.append({
                    "Dataset": ds_name,
                    "Method": label,
                    "Table Recall (%)": f"{metrics['Table Recall (%)']:.2f}",
                    "Table CR (%)": f"{metrics['Table CR (%)']:.2f}",
                    "Precision (%)": f"{metrics['Precision (%)']:.2f}",
                    "F1-Score (%)": f"{metrics['F1-Score (%)']:.2f}",
                })
            else:
                rows.append({
                    "Dataset": ds_name,
                    "Method": label,
                    "Table Recall (%)": "N/A",
                    "Table CR (%)": "N/A",
                    "Precision (%)": "N/A",
                    "F1-Score (%)": "N/A",
                })

    df = pd.DataFrame(rows)
    print("\n=======================================================")
    print("Table 3: Ablation Study Results across Benchmarks (K=15)")
    print("=======================================================\n")
    print(to_markdown_str(df))


if __name__ == "__main__":
    main()
