#!/usr/bin/env python3
"""Calculate overall schema-routing metrics from one result JSON or a directory.

Examples:
    python calculate_overall_metrics_from_path.py results_file/CRED-SQL/bird_v2_predictions_exp2.json
    python calculate_overall_metrics_from_path.py results_file/CRED-SQL
    python calculate_overall_metrics_from_path.py results_file/CRUSH4SQL --output crush4sql_overall.md

For a directory, every compatible JSON file is treated as one run. Files are
grouped automatically into SpiderUnion, BirdUnion, and SynLink by matching their
questions to the project gold files. Multiple runs are reported as mean ± sample
standard deviation (ddof=1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any

DATASETS = {
    "Spider": "dataset_v2/spider/murre_spider_dev.json",
    "Bird": "dataset_v2/bird/murre_bird_dev.json",
    "SynLink": "dataset_v2/SynLink/Formal_moderate_1k_gd.json",
}
DATASET_LABELS = {
    "Spider": "SpiderUnion",
    "Bird": "BirdUnion",
    "SynLink": "SynLink",
}
METRICS = ("Table Rec.", "Table CR", "Precision", "F1")


def strict_name(value: Any) -> str:
    """Apply the project's strict table-name normalization."""

    name = str(value).split("(", 1)[0].strip().lower()
    name = re.sub(r"[.\s]+", "_", name)
    return name.strip("_")


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def prediction_list(row: dict[str, Any], k: int) -> list[Any]:
    if "pred_schema" in row:
        prediction = row["pred_schema"]
        if isinstance(prediction, dict):
            if "k20" not in prediction:
                raise ValueError("pred_schema does not contain k20")
            return prediction["k20"][:k]
        return prediction[:k]
    if "pred_schema_link" in row:
        return row["pred_schema_link"][:k]
    if "pred_schema_linking" in row:
        return row["pred_schema_linking"][:k]
    raise ValueError(f"No supported prediction field in row keys: {list(row)}")


def question(row: dict[str, Any]) -> str:
    return str(row.get("question", row.get("utterance", ""))).strip()


def table_bucket(table_count: int) -> str:
    return str(table_count) if table_count in (1, 2, 3) else "4+"


def evaluate(
    gold_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    k: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    """Project-compatible evaluation formerly provided by calculate_metrics.py."""

    if len(gold_rows) != len(pred_rows):
        raise ValueError(f"Gold/prediction length mismatch: {len(gold_rows)} != {len(pred_rows)}")

    recalls: list[float] = []
    complete: list[float] = []
    precisions: list[float] = []
    f1s: list[float] = []
    grouped: dict[str, list[int]] = {bucket: [0, 0] for bucket in ("1", "2", "3", "4+")}
    checks = {"question_mismatch": 0, "short_prediction": 0, "duplicate_prediction": 0}

    for gold, pred_row in zip(gold_rows, pred_rows):
        checks["question_mismatch"] += question(gold) != question(pred_row)
        gold_tables = {strict_name(table) for table in gold.get("gold_schema", gold.get("schema_link", []))}
        raw_predictions = prediction_list(pred_row, k)
        pred_tables = {strict_name(table) for table in raw_predictions}
        checks["short_prediction"] += len(raw_predictions) < k
        checks["duplicate_prediction"] += len(raw_predictions) != len(pred_tables)

        hits = len(gold_tables & pred_tables)
        recall = hits / len(gold_tables)
        precision = hits / len(pred_tables) if pred_tables else 0.0
        is_complete = float(gold_tables <= pred_tables)
        recalls.append(recall)
        complete.append(is_complete)
        precisions.append(precision)
        f1s.append(2 * recall * precision / (recall + precision) if recall + precision else 0.0)

        bucket = table_bucket(len(gold_tables))
        grouped[bucket][0] += int(is_complete)
        grouped[bucket][1] += 1

    overall = {
        "Table Rec.": mean(recalls),
        "Table CR": mean(complete),
        "Precision": mean(precisions),
        "F1": mean(f1s),
    }
    grouped_cr = {
        bucket: hits / count if count else 0.0
        for bucket, (hits, count) in grouped.items()
    }
    return overall, grouped_cr, {
        **checks,
        **{f"n_{bucket}": count for bucket, (_, count) in grouped.items()},
    }


def resolve_input(root: Path, value: Path) -> Path:
    if value.is_absolute():
        return value
    cwd_path = Path.cwd() / value
    if cwd_path.exists():
        return cwd_path.resolve()
    return (root / value).resolve()


def candidate_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            path
            for path in input_path.rglob("*.json")
            if not path.name.lower().endswith((".meta.json", ".metrics.json"))
        )
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def same_questions(gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]) -> bool:
    return len(gold_rows) == len(pred_rows) and all(
        question(gold) == question(prediction)
        for gold, prediction in zip(gold_rows, pred_rows)
    )


def detect_dataset(
    pred_rows: list[dict[str, Any]],
    gold_by_dataset: dict[str, list[dict[str, Any]]],
) -> str:
    matches = [
        dataset
        for dataset, gold_rows in gold_by_dataset.items()
        if same_questions(gold_rows, pred_rows)
    ]
    if len(matches) == 1:
        return matches[0]

    # The project's SynLink prediction files can contain minor question-text
    # differences while preserving the same row order used by evaluate().
    # Dataset sizes are distinct, so length is a safe project-local fallback.
    length_matches = [
        dataset
        for dataset, gold_rows in gold_by_dataset.items()
        if len(gold_rows) == len(pred_rows)
    ]
    if len(length_matches) == 1:
        return length_matches[0]
    raise ValueError(
        "could not uniquely match rows to Spider, Bird, or SynLink "
        f"(question_matches={matches}, length_matches={length_matches})"
    )


def load_runs(
    input_path: Path,
    gold_by_dataset: dict[str, list[dict[str, Any]]],
) -> dict[str, list[tuple[Path, list[dict[str, Any]]]]]:
    runs: dict[str, list[tuple[Path, list[dict[str, Any]]]]] = {
        dataset: [] for dataset in DATASETS
    }
    seen_hashes: dict[str, set[str]] = {dataset: set() for dataset in DATASETS}
    skipped: list[str] = []
    for path in candidate_files(input_path):
        try:
            pred_rows = load_json(path)
            dataset = detect_dataset(pred_rows, gold_by_dataset)
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            if input_path.is_file():
                raise ValueError(f"Cannot evaluate {path}: {exc}") from exc
            skipped.append(f"{path.name}: {exc}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen_hashes[dataset]:
            print(f"Skipped duplicate result file: {path.name}", file=sys.stderr)
            continue
        seen_hashes[dataset].add(digest)
        runs[dataset].append((path, pred_rows))

    if skipped:
        print(f"Skipped {len(skipped)} incompatible JSON file(s).", file=sys.stderr)
        for item in skipped:
            print(f"  - {item}", file=sys.stderr)
    if not any(runs.values()):
        raise ValueError(f"No compatible result JSON files found in {input_path}")
    return runs


def format_repeated(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{mean(values):.4f} ± {stdev(values):.4f}"


def markdown_table(rows: list[list[str]]) -> str:
    headers = ["Dataset", "Method", *METRICS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _resolve_gold_file(root: Path, relative_path: str) -> Path:
    candidates = [
        root / relative_path,
        root / "data" / Path(relative_path).name,
        root / "data" / relative_path.replace("dataset_v2/", ""),
        root / "schema_link" / relative_path,
        root / "dataset_v2" / relative_path.replace("dataset_v2/", ""),
    ]
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]


def _detect_project_root() -> Path:
    curr = Path(__file__).resolve().parent
    for p in [curr, curr.parent, curr.parents[1], curr.parents[2]]:
        if (p / "data").is_dir() or (p / "schema_link" / "dataset_v2").is_dir():
            return p
    return curr.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path, help="One result JSON file or a result directory")
    parser.add_argument("--root", type=Path, default=_detect_project_root())
    parser.add_argument("--k", type=int, default=15, help="Top-K table predictions (default: 15)")
    parser.add_argument("--method", help="Method label; defaults to the file's parent or directory name")
    parser.add_argument("--output", type=Path, help="Optional Markdown output file")
    args = parser.parse_args()
    if args.k <= 0:
        parser.error("--k must be positive")

    root = args.root.resolve()
    input_path = resolve_input(root, args.input_path)
    method = args.method or (input_path.parent.name if input_path.is_file() else input_path.name)
    gold_by_dataset = {
        dataset: load_json(_resolve_gold_file(root, relative_path))
        for dataset, relative_path in DATASETS.items()
    }
    runs = load_runs(input_path, gold_by_dataset)

    rows: list[list[str]] = []
    for dataset in ("Spider", "Bird", "SynLink"):
        if not runs[dataset]:
            continue
        per_run: list[dict[str, float]] = []
        for path, pred_rows in runs[dataset]:
            overall, _, checks = evaluate(gold_by_dataset[dataset], pred_rows, args.k)
            if checks["question_mismatch"]:
                print(
                    f"Warning: {path.name} has {checks['question_mismatch']} question-text "
                    "mismatch(es); metrics use the project-standard row alignment.",
                    file=sys.stderr,
                )
            per_run.append(overall)
        rows.append(
            [
                DATASET_LABELS[dataset],
                method,
                *[format_repeated([run[metric] for run in per_run]) for metric in METRICS],
            ]
        )

    output = markdown_table(rows)
    print(output)
    if args.output:
        output_path = args.output if args.output.is_absolute() else Path.cwd() / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
        print(f"\nWrote {output_path.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()

