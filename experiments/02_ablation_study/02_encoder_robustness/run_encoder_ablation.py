"""GERT encoder-swap ablation: rerun GERT with SGPT-1.3B instead of BGE-M3.

The pipeline is identical to the main GERT experiment (same KG text, same
retrieval query, same PPR parameters); the only change is that table and
query embeddings are produced by the local SGPT-1.3B sentence encoder that
also backs the SingleDPR (SGPT) baseline. Responds to the reviewer request
to disentangle the encoder difference from the PPR contribution.

Metrics at K=15: Table Recall, Table CR, Precision, F1, and the
average per-query routing time (query embedding + retrieval + PPR + fusion;
graph construction is excluded from the timing).

The Neo4j graph is cleared and rebuilt with SGPT-1.3B embeddings (table and
column nodes, matching the main pipeline) before retrieval.

Usage (from the project root):
    python experiments/02_ablation_study/02_encoder_robustness/run_encoder_ablation.py --dataset bird
"""

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

ABLATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ABLATION_DIR.parents[2]
DATA_ROOT = PROJECT_ROOT / "data"

# Load the project .env before importing modules that read config at import
# time (core.pipeline.common).
load_dotenv(PROJECT_ROOT / ".env")

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "core" / "pipeline"))
sys.path.insert(0, str(ABLATION_DIR))

from core.schemakg.schemakg_search_ppr import BaseSchemaLinkingRetrieverPPR  # noqa: E402
try:
    from core.pipeline.common import TABLE_VECTOR_INDEX_NAME, driver
except ImportError:
    from pipeline.common import TABLE_VECTOR_INDEX_NAME, driver  # noqa: E402
try:
    from core.pipeline.kg_construction import build_knowledge_graph
except ImportError:
    from pipeline.kg_construction import build_knowledge_graph  # noqa: E402

from sgpt_embedder import SGPTEmbedder  # noqa: E402

SGPT_MODEL = os.environ.get("SGPT_MODEL", r"D:\models\SGPT-1.3B-weightedmean-msmarco-specb-bitfit")

DATASETS = {
    "spider": {
        "schema": DATA_ROOT / "spider/spider_union_schema_FK.csv",
        "data": DATA_ROOT / "spider/murre_spider_dev.json",
    },
    "bird": {
        "schema": DATA_ROOT / "bird/bird_union_schema_FK.csv",
        "data": DATA_ROOT / "bird/murre_bird_dev.json",
    },
    "synlink": {
        "schema": DATA_ROOT / "SynLink/SynSQL_schema_csv_300.csv",
        "data": DATA_ROOT / "SynLink/Formal_moderate_1k_gd.json",
    },
}

# GERT main-experiment parameters (identical to the BGE-M3 runs).
PARAMS = dict(
    top_k=20,
    initial_top_k=20,
    alpha=0.85,
    seed_rank_decay=0.3,
    fusion_weight=0.7,
    max_ppr_iter=100,
    ppr_tol=1e-6,
)

METRICS_K = 15


PREDICTION_ROOT = ABLATION_DIR / "predictions_new"
RESULT_ROOT = ABLATION_DIR / "results_new"
DATASET_LABELS = {
    "spider": "SpiderUnion",
    "bird": "BirdUnion",
    "synlink": "SynLink",
}


def wait_for_index(timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    with driver.session() as session:
        while time.time() < deadline:
            index = session.run(
                "SHOW VECTOR INDEXES YIELD name, state "
                "WHERE name = $name RETURN state",
                name=TABLE_VECTOR_INDEX_NAME,
            ).single()
            if index and index["state"] == "ONLINE":
                return
            time.sleep(2)
    raise RuntimeError(f"Vector index {TABLE_VECTOR_INDEX_NAME} not ONLINE")


# ---------------------------------------------------------------------------
# Prediction with per-query timing (retrieval + PPR + fusion)
# ---------------------------------------------------------------------------

def search_text(item: dict) -> str:
    question = item.get("question", "")
    evidence = item.get("evidence", "")
    if evidence and str(evidence).strip():
        return f"{question} {str(evidence).strip()}"
    return question


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_checkpoint(path: Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def run_predictions(key: str) -> dict:
    """Rebuild the KG with SGPT embeddings, then run GERT over all queries."""
    paths = DATASETS[key]
    data = json.loads(paths["data"].read_text(encoding="utf-8"))
    print(f"[{key}] {len(data)} queries.")

    print(f"[{key}] Rebuilding Neo4j graph with SGPT-1.3B embeddings ...")
    build_knowledge_graph(
        str(paths["schema"]),
        clear_before_build=True,
        local_embedding_model_path=SGPT_MODEL,
    )
    wait_for_index()

    embedder = SGPTEmbedder(SGPT_MODEL)
    retriever = BaseSchemaLinkingRetrieverPPR(embedder=embedder)

    checkpoint_path = PREDICTION_ROOT / key / "checkpoint.json"
    done = load_checkpoint(checkpoint_path)
    errors = 0

    for index, item in enumerate(data, 1):
        qid = str(item.get("question_id", index - 1))
        if qid in done:
            continue
        try:
            t0 = time.perf_counter()
            _, post = retriever._retrieve_pre_post_tables(
                search_text(item), verbose=False, **PARAMS
            )
            elapsed = time.perf_counter() - t0
            done[qid] = {"pred": post, "query_time": elapsed}
        except Exception:
            errors += 1
            print(f"\n[Error] query {qid}: {traceback.format_exc(limit=3)}")
            done[qid] = {"pred": [], "query_time": 0.0}
        if len(done) % 50 == 0:
            save_checkpoint(checkpoint_path, done)
            print(f"[{key}] {len(done)}/{len(data)} done", flush=True)

    save_checkpoint(checkpoint_path, done)
    if errors:
        print(f"[{key}] {errors} queries failed (empty predictions recorded).")

    # Prediction file in the main-experiment format.
    records = []
    for index, item in enumerate(data):
        qid = str(item.get("question_id", index))
        records.append({
            "question_id": item.get("question_id", index),
            "db_id": item.get("db_id", ""),
            "question": item.get("question", ""),
            "SQL": item.get("SQL", ""),
            "gold_schema": list(item["gold_schema"].keys()) if isinstance(item.get("gold_schema"), dict) else item.get("gold_schema", item.get("schema_link", [])),
            "pred_schema": done[qid]["pred"],
            "query_time": done[qid]["query_time"],
        })
    out = PREDICTION_ROOT / key / "gert_sgpt.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{key}] Predictions saved to {out}")
    return done


# ---------------------------------------------------------------------------
# Metrics at K=15 (definitions verified against the BGE-M3 reference values)
# ---------------------------------------------------------------------------

def compute_metrics(key: str, done: dict) -> dict:
    data = json.loads(DATASETS[key]["data"].read_text(encoding="utf-8"))
    k = METRICS_K

    rec, cr, precision, f1 = [], [], [], []
    times = []
    for index, item in enumerate(data):
        qid = str(item.get("question_id", index))
        pred = list(dict.fromkeys(done[qid]["pred"]))[:k]
        gold = set(item.get("gold_schema", item.get("schema_link", [])))
        times.append(done[qid]["query_time"])

        if gold:
            hit = gold & set(pred)
            r = len(hit) / len(gold)
            p = len(hit) / len(pred) if pred else 0.0
            rec.append(r)
            cr.append(1.0 if len(hit) == len(gold) else 0.0)
            precision.append(p)
            f1.append(2 * p * r / (p + r) if p + r > 0 else 0.0)

    return {
        "n": len(data),
        "table_recall": statistics.mean(rec),
        "table_cr": statistics.mean(cr),
        "precision": statistics.mean(precision),
        "f1": statistics.mean(f1),
        "avg_query_time": statistics.mean(times),
        "median_query_time": statistics.median(times),
    }


def write_report(key: str, metrics: dict) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    label = DATASET_LABELS[key]
    lines = [
        f"# GERT encoder ablation ({label}, K={METRICS_K})",
        "",
        "| Dataset | Method | Table Recall | Table CR | Precision | F1 | avg(query_time) |",
        "|---|---|---:|---:|---:|---:|---:|",
        f"| {label} | GERT_PPR(SGPT-1.3B) | {metrics['table_recall']:.4f} | "
        f"{metrics['table_cr']:.4f} | {metrics['precision']:.4f} | {metrics['f1']:.4f} | "
        f"{metrics['avg_query_time']:.4f} |",
        "",
        f"n={metrics['n']} queries; "
        "query_time = embedding + retrieval + PPR + fusion per query (graph build excluded);",
    ]
    (RESULT_ROOT / f"{key}_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULT_ROOT / f"{key}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    args = parser.parse_args()

    done = run_predictions(args.dataset)
    metrics = compute_metrics(args.dataset, done)
    write_report(args.dataset, metrics)
    driver.close()


if __name__ == "__main__":
    main()

