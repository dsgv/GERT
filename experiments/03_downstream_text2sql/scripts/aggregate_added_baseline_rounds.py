"""Aggregate three Text-to-SQL runs for the four newly added baselines."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs_and_results" / "e2e_generation"
CONDITIONS = ("crush4sql", "cred_sql", "core_t", "linkalign")
LABELS = {
    "crush4sql": "Crush4SQL",
    "cred_sql": "CRED-SQL",
    "core_t": "CORE-T",
    "linkalign": "LinkAlign",
    "gert": "GERT (fixed reference; not rerun)",
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


def summary(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root", type=Path, default=RUNS / "added_baseline_runs"
    )
    parser.add_argument(
        "--gert-report", type=Path, default=RUNS / "results" / "downstream_metrics.json"
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
        condition_summary = {}
        for metric in METRICS:
            values = [metric_value(report, condition, metric) for report in reports]
            condition_summary[metric] = summary(values)
        aggregate[condition] = condition_summary

    gert_report = read_json(args.gert_report.resolve())
    gert = {metric: metric_value(gert_report, "gert", metric) for metric in METRICS}

    output_dir = runs_root / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "round_count": 3,
        "result_files": [str(path) for path in report_paths],
        "std_definition": "sample standard deviation (n-1)",
        "metrics": aggregate,
        "gert_fixed_reference": {
            "source": str(args.gert_report.resolve()),
            "note": "GERT was not rerun; these are the existing single-run values.",
            "metrics": gert,
        },
    }
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Added baselines: three-round mean ± sample std",
        "",
        "GERT is shown only as the existing fixed single-run reference and was not rerun.",
        "",
        "| Schema Input | EX | Context Red. | Tokens/Query |",
        "|---|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        values = aggregate[condition]
        ex = values["execution_accuracy"]
        reduction = values["context_reduction"]
        tokens = values["avg_total_tokens"]
        lines.append(
            f"| {LABELS[condition]} | {ex['mean']:.3f} ± {ex['sample_std']:.3f} | "
            f"{100 * reduction['mean']:.2f}% ± {100 * reduction['sample_std']:.2f}% | "
            f"{tokens['mean']:.1f} ± {tokens['sample_std']:.1f} |"
        )
    lines.append(
        f"| {LABELS['gert']} | {gert['execution_accuracy']:.3f} | "
        f"{100 * gert['context_reduction']:.2f}% | {gert['avg_total_tokens']:.1f} |"
    )
    (output_dir / "main_table_mean_std.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Aggregate written to {output_dir}")


if __name__ == "__main__":
    main()
