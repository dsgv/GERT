#!/usr/bin/env python3
"""Recompute the paper's downstream table from the archived three-run metrics.

The DIN-SQL native-selector runs and GERT runs come from distinct archived
series; their exact source paths are defined below and checked at runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1] / "runs_and_results"


def load_series(pattern: str, condition: str, ex_key: str, token_key: str):
    rows = []
    for run in (1, 2, 3):
        path = ROOT / pattern.format(run=run)
        metric = json.loads(path.read_text(encoding="utf-8"))["metrics"][condition]
        if metric["n"] != 100:
            raise ValueError(f"Expected 100 queries in {path}; got {metric['n']}")
        reduction = metric.get("context_reduction", metric.get("schema_token_reduction_vs_full"))
        if reduction is None:
            raise KeyError(f"No context reduction in {path}: {condition}")
        rows.append((metric[ex_key] * 100, reduction * 100, metric[token_key]))
    return rows


SERIES = [
    ("Zero-shot", "Full Schema", "e2e_generation/core_baseline_runs/run{run}/results/downstream_metrics.json", "full", "execution_accuracy", "avg_total_tokens"),
    ("Zero-shot", "DR (SGPT)", "e2e_generation/core_baseline_runs/run{run}/results/downstream_metrics.json", "dense", "execution_accuracy", "avg_total_tokens"),
    ("Zero-shot", "CRED-SQL", "e2e_generation/added_baseline_runs/run{run}/results/downstream_metrics.json", "cred_sql", "execution_accuracy", "avg_total_tokens"),
    ("Zero-shot", "MURRE", "e2e_generation/core_baseline_runs/run{run}/results/downstream_metrics.json", "murre", "execution_accuracy", "avg_total_tokens"),
    ("Zero-shot", "CRUSH4SQL", "e2e_generation/added_baseline_runs/run{run}/results/downstream_metrics.json", "crush4sql", "execution_accuracy", "avg_total_tokens"),
    ("Zero-shot", "LinkAlign", "e2e_generation/added_baseline_runs/run{run}/results/downstream_metrics.json", "linkalign", "execution_accuracy", "avg_total_tokens"),
    ("Zero-shot", "CORE-T", "e2e_generation/added_baseline_runs/run{run}/results/downstream_metrics.json", "core_t", "execution_accuracy", "avg_total_tokens"),
    ("Zero-shot", "GERT", "e2e_generation/core_baseline_runs/run{run}/results/downstream_metrics.json", "gert", "execution_accuracy", "avg_total_tokens"),
    ("MAC-SQL", "Native Selector", "macsql_runs/macsql_selector_runs/run{run}/results_mac_only/metrics.json", "mac_selector", "execution_accuracy", "avg_pipeline_tokens"),
    ("MAC-SQL", "GERT", "macsql_runs/macsql_selector_runs/run{run}/results_gert_only/metrics.json", "gert_selector", "execution_accuracy", "avg_pipeline_tokens"),
    ("DIN-SQL", "Native Linker", "dinsql_runs/dinsql_selector_runs/run{run}/results_din_only/metrics.json", "din_linker", "bird_execution_accuracy", "avg_pipeline_tokens"),
    ("DIN-SQL", "GERT", "dinsql_runs/dinsql_runs/run{run}/results/metrics.json", "gert_selector", "bird_execution_accuracy", "avg_pipeline_tokens"),
]


def main() -> None:
    print("| Pipeline | Schema input / linker | EX (%) | Context reduction (%) | Tokens/query |")
    print("| --- | --- | ---: | ---: | ---: |")
    for pipeline, label, pattern, condition, ex_key, token_key in SERIES:
        values = load_series(pattern, condition, ex_key, token_key)
        ex = [row[0] for row in values]
        reduction = [row[1] for row in values]
        tokens = [row[2] for row in values]
        print(
            f"| {pipeline} | {label} | {mean(ex):.1f} ± {stdev(ex):.1f} | "
            f"{mean(reduction):.2f} | {mean(tokens):,.1f} ± {stdev(tokens):,.1f} |"
        )


if __name__ == "__main__":
    main()
