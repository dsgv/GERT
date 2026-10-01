"""Aggregate three zero-shot Text-to-SQL runs for the main four conditions."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs_and_results" / "e2e_generation"
CONDITIONS = ("full", "dense", "murre", "gert")
LABELS = {
    "full": "Full Schema (no routing)",
    "dense": "SingleDPR (SGPT)",
    "murre": "MURRE",
    "gert": "GERT (ours)",
}
METRICS = (
    "execution_accuracy",
    "context_reduction",
    "avg_total_tokens",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(report: dict[str, Any], condition: str, metric: str) -> float:
    values = report["metrics"][condition]
    if metric == "context_reduction":
        return float(values.get(metric, values.get("schema_token_reduction_vs_full")))
    return float(values[metric])


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root", type=Path, default=RUNS / "core_baseline_runs"
    )
    args = parser.parse_args()
    runs_root = args.runs_root.resolve()

    reports = []
    report_paths = []
    for round_number in range(1, 4):
        path = runs_root / f"run{round_number}" / "results" / "downstream_metrics.json"
        report_paths.append(path)
        reports.append(read_json(path))

    aggregate: dict[str, Any] = {}
    for condition in CONDITIONS:
        values: dict[str, Any] = {}
        for metric in METRICS:
            values[metric] = summarize(
                [metric_value(report, condition, metric) for report in reports]
            )
        aggregate[condition] = values

    output_dir = runs_root / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "round_count": 3,
        "conditions": list(CONDITIONS),
        "result_files": [str(path) for path in report_paths],
        "std_definition": "sample standard deviation (n-1)",
        "token_definition": "API-reported downstream prompt + completion tokens per query",
        "metrics": aggregate,
    }
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Main zero-shot conditions: three-round mean ± sample std",
        "",
        "Total Tokens are API-reported downstream prompt plus completion tokens per query.",
        "",
        "| Schema Input | EX | Context Red. | Tokens/Query |",
        "|---|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        m = aggregate[condition]
        ex = m["execution_accuracy"]
        reduction = m["context_reduction"]
        total = m["avg_total_tokens"]
        lines.append(
            f"| {LABELS[condition]} | {ex['mean']:.3f} ± {ex['sample_std']:.3f} | "
            f"{100 * reduction['mean']:.2f}% ± {100 * reduction['sample_std']:.2f}% | "
            f"{total['mean']:.1f} ± {total['sample_std']:.1f} |"
        )
    (output_dir / "main_table_mean_std.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Aggregate written to {output_dir}")


if __name__ == "__main__":
    main()
