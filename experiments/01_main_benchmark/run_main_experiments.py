"""Run GERT and optional repository baselines on full benchmark datasets."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Prefer this experiment's baselines package over any same-named root package.
if str(HERE) in sys.path:
    sys.path.remove(str(HERE))
sys.path.insert(0, str(HERE))

import argparse
import json
import time

import pandas as pd

from baselines.GERT_1hop import SchemaLinkingRetriever1Hop
from baselines.GERT_2hop import SchemaLinkingRetriever2Hop
try:
    from core.schemakg.schemakg_search_ppr import BaseSchemaLinkingRetrieverPPR
except ImportError:
    from core.schemakg_search_ppr import BaseSchemaLinkingRetrieverPPR
try:
    from core.pipeline.common import TABLE_VECTOR_INDEX_NAME, driver
except ImportError:
    from pipeline.common import TABLE_VECTOR_INDEX_NAME, driver
try:
    from core.pipeline.kg_construction import build_knowledge_graph
except ImportError:
    from pipeline.kg_construction import build_knowledge_graph
try:
    from core.pipeline.retrieval import OpenAIEmbedder
except ImportError:
    from pipeline.retrieval import OpenAIEmbedder


ROUTING_ROOT = REPO_ROOT
OUTPUT_ROOT = HERE / "main_experiment_test"

DATASETS = {
    "spider": {
        "schema": ROUTING_ROOT / "data/spider/spider_union_schema_FK.csv",
        "test": ROUTING_ROOT / "data/spider/murre_spider_dev.json",
    },
    "bird": {
        "schema": ROUTING_ROOT / "data/bird/bird_union_schema_FK.csv",
        "test": ROUTING_ROOT / "data/bird/murre_bird_dev.json",
    },
    "synlink": {
        "schema": ROUTING_ROOT / "data/SynLink/SynSQL_schema_csv_300.csv",
        "test": ROUTING_ROOT / "data/SynLink/Formal_moderate_1k_gd.json",
    },
}


class CachedRetryEmbedder:
    """OpenAI-compatible embedder with exact-query cache and transient retries."""

    def __init__(self, attempts=5):
        self.backend = OpenAIEmbedder()
        self.model = self.backend.model
        self.attempts = attempts
        self.cache = {}

    def embed_query(self, text):
        if text in self.cache:
            return self.cache[text]
        for attempt in range(self.attempts):
            try:
                vector = self.backend.embed_query(text)
                self.cache[text] = vector
                return vector
            except Exception:
                if attempt + 1 == self.attempts:
                    raise
                time.sleep(2 ** attempt)

    def get_token_usage(self):
        return self.backend.get_token_usage()

    def reset_token_usage(self):
        self.backend.reset_token_usage()


def expected_table_count(schema_path):
    frame = pd.read_csv(schema_path)
    return int(frame[["db_name", "table_name"]].drop_duplicates().shape[0])


def verify_graph(schema_path):
    expected = expected_table_count(schema_path)
    with driver.session() as session:
        row = session.run(
            "MATCH (t:Table) RETURN count(t) AS tables, "
            "count(t.table_embedding) AS embedded"
        ).single()
        deadline = time.time() + 120
        state = None
        while time.time() < deadline:
            index = session.run(
                "SHOW VECTOR INDEXES YIELD name, state "
                "WHERE name = $name RETURN state",
                name=TABLE_VECTOR_INDEX_NAME,
            ).single()
            state = index["state"] if index else None
            if state == "ONLINE":
                break
            time.sleep(2)
    if row["tables"] != expected or row["embedded"] != expected or state != "ONLINE":
        raise RuntimeError(
            f"Graph verification failed: expected={expected}, tables={row['tables']}, "
            f"embedded={row['embedded']}, index={state}"
        )
    print(f"[Verify] {expected} tables, all embedded, index ONLINE")


def valid_prediction_file(path, expected_rows):
    if not path.exists():
        return False
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        len(rows) == expected_rows
        and all(len(row.get("pred_schema", row.get("pred_schema_link", []))) >= 20 for row in rows)
    )


def run_dataset(key, config, force=False, gert_only=False):
    dataset_output = OUTPUT_ROOT / key
    dataset_output.mkdir(parents=True, exist_ok=True)
    test_rows = json.loads(config["test"].read_text(encoding="utf-8"))
    expected_rows = len(test_rows)
    paths = {
        "gert": dataset_output / "gert.json",
        "gert_1hop": dataset_output / "gert_1hop.json",
        "gert_2hop": dataset_output / "gert_2hop.json",
    }

    structural = ["gert"] if gert_only else ["gert", "gert_1hop", "gert_2hop"]
    if force or any(not valid_prediction_file(paths[name], expected_rows) for name in structural):
        print(f"\n[{key}] Building isolated Neo4j graph")
        build_knowledge_graph(str(config["schema"]), clear_before_build=True)
        verify_graph(config["schema"])
        embedder = CachedRetryEmbedder()

        if force or not valid_prediction_file(paths["gert"], expected_rows):
            BaseSchemaLinkingRetrieverPPR(embedder=embedder).process_dataset(
                str(config["test"]), str(paths["gert"]), top_k=20,
                verbose=False, initial_top_k=20, alpha=0.85,
                seed_rank_decay=0.3, fusion_weight=0.7,
            )
        if not gert_only and (force or not valid_prediction_file(paths["gert_1hop"], expected_rows)):
            SchemaLinkingRetriever1Hop(embedder=embedder).process_dataset(
                str(config["test"]), str(paths["gert_1hop"]), top_k=20,
                verbose=False, initial_top_k=20,
            )
        if not gert_only and (force or not valid_prediction_file(paths["gert_2hop"], expected_rows)):
            SchemaLinkingRetriever2Hop(embedder=embedder).process_dataset(
                str(config["test"]), str(paths["gert_2hop"]), top_k=20,
                verbose=False, initial_top_k=20,
            )

    for method in (["gert"] if gert_only else paths):
        path = paths[method]
        if not valid_prediction_file(path, expected_rows):
            raise RuntimeError(f"Incomplete prediction file: {method}: {path}")
    print(f"[{key}] Requested prediction files complete ({expected_rows} queries)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--gert-only", action="store_true", help="Run only GERT")
    args = parser.parse_args()
    selected = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}
    for key, config in selected.items():
        run_dataset(key, config, force=args.force, gert_only=args.gert_only)
    driver.close()


if __name__ == "__main__":
    main()
