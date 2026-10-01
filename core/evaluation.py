"""Evaluation utilities for schema-routing predictions.

Canonical metric names used in this project:

* Rec@K: macro-average per-query table recall.
* CR@K: complete recall, i.e. the fraction of queries for which every gold
  table is included in the top-K predictions.
"""

import json
import os
from typing import Dict, List, Optional, Sequence, Union

from tqdm import tqdm


CR_K_VALUES = [3, 5, 10, 20]
GoldInput = Sequence[Union[str, Sequence[str]]]
Predictions = Sequence[Sequence[str]]


def _normalize_ground_truth(ground_truth: GoldInput) -> List[List[str]]:
    """Normalize one-label-per-query and list-of-labels representations."""
    if not ground_truth:
        return []
    if isinstance(ground_truth[0], str):
        return [[str(item)] for item in ground_truth]
    return [list(item) for item in ground_truth]


def _validate_inputs(ground_truth: List[List[str]], predicted_res: Predictions) -> None:
    if len(ground_truth) != len(predicted_res):
        raise ValueError(
            "ground_truth and predicted_res must contain the same number of "
            f"queries, got {len(ground_truth)} and {len(predicted_res)}"
        )


def compute_complete_recall(
    ground_truth: GoldInput,
    predicted_res: Predictions,
) -> float:
    """Compute CR: fraction of non-empty queries with all gold tables hit."""
    normalized = _normalize_ground_truth(ground_truth)
    _validate_inputs(normalized, predicted_res)

    hits = 0
    valid_queries = 0
    for gold, predicted in zip(normalized, predicted_res):
        gold_set = set(gold)
        if not gold_set:
            continue
        valid_queries += 1
        hits += int(gold_set <= set(predicted))
    return hits / valid_queries if valid_queries else 0.0


def compute_average_recall(
    ground_truth: GoldInput,
    predicted_res: Predictions,
) -> float:
    """Compute Rec: macro-average of per-query table recall."""
    normalized = _normalize_ground_truth(ground_truth)
    _validate_inputs(normalized, predicted_res)

    total_recall = 0.0
    valid_queries = 0
    for gold, predicted in zip(normalized, predicted_res):
        gold_set = set(gold)
        if not gold_set:
            continue
        total_recall += len(gold_set & set(predicted)) / len(gold_set)
        valid_queries += 1
    return total_recall / valid_queries if valid_queries else 0.0


def _load_gt_and_predictions(json_file_path: str, verbose: bool = False):
    """Load ordered gold and predicted table lists from a prediction JSON."""
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    all_ground_truths: List[List[str]] = []
    all_predictions: List[List[str]] = []
    for item in tqdm(data, desc="Evaluating", disable=not verbose):
        routing = item.get("gold_schema", item.get("schema_link", item.get("schema_linking")))
        true_tables = list(routing) if isinstance(routing, (dict, list)) else []

        predictions = item.get(
            "pred_schema", item.get("pred_schema_link", item.get("pred_schema_linking", []))
        )
        if isinstance(predictions, dict):
            pred_tables = list(predictions)
        elif isinstance(predictions, list):
            pred_tables = predictions
        else:
            pred_tables = []

        all_ground_truths.append(true_tables)
        all_predictions.append(pred_tables)
    return data, all_ground_truths, all_predictions


def evaluate_routing_metrics(
    json_file_path: str,
    ks: Optional[List[int]] = None,
    save: bool = True,
    verbose: bool = True,
) -> Dict[str, float]:
    """Compute both CR@K and Rec@K using the canonical definitions."""
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"Prediction file not found: {json_file_path}")
    if ks is None:
        ks = CR_K_VALUES

    data, ground_truths, predictions = _load_gt_and_predictions(
        json_file_path, verbose=verbose
    )
    if verbose:
        print(f"Total Samples: {len(data)}")
        print("\n" + "=" * 34)
        print("SCHEMA ROUTING METRICS")
        print("=" * 34)
        print(f"{'K':<5} | {'CR@K':<10} | {'Rec@K':<10}")
        print("-" * 34)

    results: Dict[str, float] = {}
    for k in ks:
        predictions_at_k = [prediction[:k] for prediction in predictions]
        complete_recall = compute_complete_recall(ground_truths, predictions_at_k)
        average_recall = compute_average_recall(ground_truths, predictions_at_k)
        results[f"CR@{k}"] = complete_recall
        results[f"Rec@{k}"] = average_recall
        if verbose:
            print(f"{k:<5} | {complete_recall:<10.4f} | {average_recall:<10.4f}")

    if verbose:
        print("=" * 34)
    if save:
        output_file = os.path.join(
            os.path.dirname(json_file_path),
            "evaluation_" + os.path.basename(json_file_path),
        )
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2, ensure_ascii=False)
        if verbose:
            print(f"Results saved to {output_file}")
    return results


def print_metric_summary_table(
    param_label: str,
    entries: List[dict],
    param_key: str,
    ks: Optional[List[int]] = None,
    metrics: Sequence[str] = ("CR", "Rec"),
) -> None:
    """Print a sensitivity summary with unambiguous metric labels."""
    if ks is None:
        ks = CR_K_VALUES
    metric_keys = [f"{metric}@{k}" for metric in metrics for k in ks]
    header = f"{param_label:<12} | " + " | ".join(
        f"{key:<8}" for key in metric_keys
    )
    print(header)
    print("-" * len(header))
    for entry in entries:
        values = entry.get("evaluation_results", {})
        row = " | ".join(f"{values.get(key, 0.0):<8.4f}" for key in metric_keys)
        print(f"{entry.get(param_key, 'N/A')!s:<12} | {row}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_file")
    parser.add_argument("--no-save", action="store_true")
    arguments = parser.parse_args()
    evaluate_routing_metrics(
        arguments.prediction_file,
        save=not arguments.no_save,
        verbose=True,
    )
