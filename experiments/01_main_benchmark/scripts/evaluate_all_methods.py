import os
import glob
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

from calculate_overall_metrics_from_path import evaluate, load_json, _resolve_gold_file

spider_gold = load_json(_resolve_gold_file(REPO_ROOT, "spider/murre_spider_dev.json"))
bird_gold = load_json(_resolve_gold_file(REPO_ROOT, "bird/murre_bird_dev.json"))
synlink_gold = load_json(_resolve_gold_file(REPO_ROOT, "SynLink/Formal_moderate_1k_gd.json"))


def evaluate_dir(gold, dir_path, ds_name):
    rows = []
    if not os.path.exists(dir_path):
        return rows
    files = sorted(glob.glob(os.path.join(dir_path, "*.json")))
    for f in files:
        fname = os.path.basename(f)
        method_name = fname.replace(f"{ds_name}_", "").replace(".json", "")
        preds = load_json(Path(f))
        overall, grouped, _ = evaluate(gold, preds, 15)
        rows.append({
            "Dataset": ds_name,
            "Method": method_name,
            "Table Rec.": f"{overall['Table Rec.']:.4f}",
            "Table CR": f"{overall['Table CR']:.4f}",
            "Precision": f"{overall['Precision']:.4f}",
            "F1": f"{overall['F1']:.4f}",
        })
    return rows


def get_res_dir(sub: str) -> str:
    candidates = [
        SCRIPT_DIR.parent / "results" / sub,
        REPO_ROOT / "experiments/01_main_benchmark/results" / sub,
        REPO_ROOT / "schema_link/output" / sub,
        REPO_ROOT / "output" / sub,
    ]
    for c in candidates:
        if c.is_dir():
            return str(c)
    return str(candidates[0])


if __name__ == "__main__":
    all_rows = (
        evaluate_dir(spider_gold, get_res_dir("spider_results"), "SpiderUnion")
        + evaluate_dir(bird_gold, get_res_dir("bird_results"), "BirdUnion")
        + evaluate_dir(synlink_gold, get_res_dir("synlink_results"), "SynLink")
    )
    df = pd.DataFrame(all_rows)
    if not df.empty:
        print(df.to_markdown(index=False))
    else:
        print("No evaluation output directory found. Provide candidate directory or run benchmarks first.")
