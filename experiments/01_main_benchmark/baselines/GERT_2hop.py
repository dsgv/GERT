from pathlib import Path
"""
GERT 2-hop static neighbor expansion
"""
import json
import os
import sys
import re
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from tqdm import tqdm

# Configure path to import pipeline module
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[2]
pipeline_dir = project_root / "core" / "pipeline"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

try:
    from core.pipeline.common import driver, TABLE_VECTOR_INDEX_NAME
except ImportError:
    from pipeline.common import driver, TABLE_VECTOR_INDEX_NAME
try:
    from core.pipeline.retrieval import OpenAIEmbedder
except ImportError:
    from pipeline.retrieval import OpenAIEmbedder
try:
    from core.pipeline.result_parser import parse_retriever_result
except ImportError:
    from pipeline.result_parser import parse_retriever_result
try:
    from core.pipeline.kg_construction import build_knowledge_graph, reset_embed_token_usage, get_embed_token_usage
except ImportError:
    from pipeline.kg_construction import build_knowledge_graph, reset_embed_token_usage, get_embed_token_usage

from neo4j_graphrag.retrievers import VectorCypherRetriever


@dataclass
class TableInfo:
    """Table information"""
    table_name: str
    description: str
    columns: List[Dict]
    related_tables: List[str]
    score: float = 0.0


def escape_lucene_query(text: str) -> str:
    if not text:
        return text
    return re.sub(r'([\+\-\&\|\!\(\)\{\}\[\]\^\"\~\*\?\:\\\/])', r'\\\1', text)


# Export table-level foreign key edges
FK_TABLE_PAIR_QUERY = """
MATCH (t1:Table)-[:CONTAINS]->(c1:Column)-[:REFERENCES]->(c2:Column)<-[:CONTAINS]-(t2:Table)
WHERE t1.name <> t2.name
RETURN DISTINCT t1.name AS a, t2.name AS b
"""


class SchemaLinkingRetriever2Hop:
    """
    2-hop static neighbor expansion retriever.
    Fixed n_hop=2, retrieves seed tables' neighbors and their neighbors.
    """

    BASIC_RETRIEVAL_QUERY = """
    MATCH (node:Table)
    OPTIONAL MATCH (node)-[:CONTAINS]->(col:Column)
    RETURN
      node.name AS table_name,
      node.description AS table_description,
      collect(DISTINCT CASE
        WHEN col.name IS NOT NULL THEN {
          column_name: col.name,
          description: COALESCE(col.description, "No description available"),
          data_type: COALESCE(col.data_type, "Unknown"),
          column_sample_value: COALESCE(col.column_sample_value, "No sample value"),
          foreign_key_ref: col.foreign_key_ref
        }
      END) AS columns
    """

    def __init__(
        self,
        neo4j_driver=None,
        embedder=None,
        cache_fk_graph: bool = True,
    ):
        self.driver = neo4j_driver or driver
        self.embedder = embedder or OpenAIEmbedder()

        self.retriever = VectorCypherRetriever(
            driver=self.driver,
            index_name=TABLE_VECTOR_INDEX_NAME,
            embedder=self.embedder,
            retrieval_query=self.BASIC_RETRIEVAL_QUERY,
        )

        self._cache_fk_graph = cache_fk_graph
        self._fk_adj: Optional[Dict[str, Set[str]]] = None

    def invalidate_fk_cache(self) -> None:
        """Call after graph structure changes; re-fetches FK edges next time."""
        self._fk_adj = None

    def _load_fk_adjacency(self) -> Dict[str, Set[str]]:
        """Load FK adjacency list (undirected graph, bidirectionally stored)"""
        if self._fk_adj is not None:
            return self._fk_adj

        adj: Dict[str, Set[str]] = {}
        with self.driver.session() as session:
            result = session.run(FK_TABLE_PAIR_QUERY)
            for record in result:
                a = record.get("a")
                b = record.get("b")
                if a and b and a != b:
                    adj.setdefault(a, set()).add(b)
                    adj.setdefault(b, set()).add(a)

        if self._cache_fk_graph:
            self._fk_adj = adj
        return adj

    def _expand_2_hop(
        self,
        seed_tables: List[str],
    ) -> Tuple[Set[str], Dict[str, int]]:
        """
        Static 2-hop neighbor expansion.

        Args:
            seed_tables: Seed table list

        Returns:
            Tuple[Set[str], Dict[str, int]]: (expanded table set, {table_name: min hops from seed})
        """
        adj = self._load_fk_adjacency()
        expanded: Set[str] = set(seed_tables)
        hop_distance: Dict[str, int] = {t: 0 for t in seed_tables}

        current_frontier = set(seed_tables)

        for hop in range(1, 3):  # 1-hop and 2-hop
            next_frontier: Set[str] = set()
            for table in current_frontier:
                neighbors = adj.get(table, set())
                for neighbor in neighbors:
                    if neighbor not in expanded:
                        expanded.add(neighbor)
                        hop_distance[neighbor] = hop
                        next_frontier.add(neighbor)
            current_frontier = next_frontier

        return expanded, hop_distance

    def _get_retrieved_tables(
        self,
        query: str,
        top_k: int,
        verbose: bool,
    ) -> Tuple[List[TableInfo], Dict[str, float]]:
        """
        Execute vector retrieval, returning table info list and retrieval score dict.
        """
        safe_query = escape_lucene_query(query)
        retriever_result = self.retriever.search(query_text=safe_query, top_k=top_k)
        parsed_results = parse_retriever_result(retriever_result)

        tables = []
        retrieval_scores: Dict[str, float] = {}

        for i, result in enumerate(parsed_results):
            table_info = TableInfo(
                table_name=result.table_name,
                description=result.table_description or "No description",
                columns=result.columns,
                related_tables=result.get_all_related_tables() if result.related_columns else [],
            )
            tables.append(table_info)
            retrieval_scores[result.table_name] = 1.0 / (i + 1)

        if verbose:
            print(f"[Vector Retrieval] {len(tables)} tables: {', '.join([t.table_name for t in tables])}")

        return tables, retrieval_scores

    def _calc_hit_rate(self, pred_tables: List[str], gt_tables: List[str]) -> dict:
        pred_set = set(pred_tables)
        gt_set = set(gt_tables)

        hit_tables = pred_set & gt_set
        hit_count = len(hit_tables)
        gt_count = len(gt_set)

        recall = hit_count / gt_count if gt_count > 0 else 0.0

        return {
            "hit_count": hit_count,
            "gt_count": gt_count,
            "recall": recall,
            "hit_tables": sorted(hit_tables),
            "miss_tables": sorted(gt_set - pred_set),
        }

    def _retrieve_pre_post_tables(
        self,
        query: str,
        top_k: int,
        verbose: bool,
        initial_top_k: int,
        expansion_strategy: str = "merge",
    ) -> Tuple[List[str], List[str]]:
        """
        Perform retrieval + 2-hop static neighbor expansion.

        Args:
            query: Query text
            top_k: Number of tables to return in the final result
            verbose: Whether to print detailed information
            initial_top_k: Initial retrieval count
            expansion_strategy: Expansion strategy
                - "merge": Merge seeds and expanded neighbors, ordered by priority (seeds first)
                - "seed_only": Use seeds only (no expansion, same as basic retrieval)
                - "expanded_only": Use expanded tables only

        Returns:
            Tuple[List[str], List[str]]: (pre-expansion top_k table names, post-expansion top_k table names)
        """
        initial_tables, retrieval_scores = self._get_retrieved_tables(query, initial_top_k, verbose)
        if not initial_tables:
            return [], []

        seed_order = [t.table_name for t in initial_tables]
        pre_names = seed_order[:top_k]

        if expansion_strategy == "seed_only":
            return pre_names, pre_names

        # Perform 2-hop expansion
        expanded_tables, hop_distance = self._expand_2_hop(seed_order)

        if verbose:
            new_tables = expanded_tables - set(seed_order)
            print(f"[2-hop Expansion] Added {len(new_tables)} neighbor tables")

        # Sorting strategy: seed tables followed by their neighbor tables
        adj = self._load_fk_adjacency()

        sorted_tables: List[str] = []
        added: Set[str] = set()

        def add_table(table_name: str) -> None:
            """Add table to sorted list (dedup)"""
            if table_name not in added:
                sorted_tables.append(table_name)
                added.add(table_name)

        for seed in seed_order:
            add_table(seed)  # Add seed table first

            # Then add this seed's hop neighbors
            for hop in range(1, 3):  # 1-hop and 2-hop
                hop_neighbors: List[str] = []
                for table in expanded_tables:
                    if hop_distance.get(table) == hop:
                        if hop == 1:
                            if table in adj.get(seed, set()):
                                hop_neighbors.append(table)
                        else:
                            # 2-hop: check if connected to an already-added 1-hop layer table
                            for prev_table in added:
                                if hop_distance.get(prev_table) == 1:
                                    if table in adj.get(prev_table, set()):
                                        hop_neighbors.append(table)
                                        break

                # Sort by neighbor count
                hop_neighbors.sort(key=lambda t: -len(adj.get(t, set())))

                for neighbor in hop_neighbors:
                    add_table(neighbor)

        post_names = sorted_tables[:top_k]

        if verbose:
            top_show = min(10, len(sorted_tables))
            preview = []
            for t in sorted_tables[:top_show]:
                if t in retrieval_scores:
                    preview.append(f"{t}(seed)")
                else:
                    preview.append(f"{t}({hop_distance[t]}-hop)")
            print(f"[Sorted after expansion top] {', '.join(preview)}{' ...' if len(sorted_tables) > top_show else ''}")
            print(f"[Original retrieval top_{top_k}] {', '.join(pre_names)}")
            print(f"[Expansion output top_{top_k}] {', '.join(post_names)}")

            # Analyze newly added tables
            new_added = set(post_names) - set(pre_names)
            removed = set(pre_names) - set(post_names)
            if new_added:
                print(f"[Expansion added] {', '.join(new_added)}")
            if removed:
                print(f"[Expansion replaced out] {', '.join(removed)}")

        return pre_names, post_names

    def retrieve_tables(
        self,
        query: str,
        top_k: int = 15,
        ground_truth: List[str] = None,
        verbose: bool = True,
        initial_top_k: int = 15,
        expansion_strategy: str = "merge",
    ) -> List[str]:
        """
        Vector retrieval + 2-hop static neighbor expansion.

        Args:
            query: User query
            top_k: Number of tables to return in the final result
            ground_truth: Ground truth table list (for calculating hit rate)
            verbose: Whether to print detailed information
            initial_top_k: Initial retrieval count
            expansion_strategy: Expansion strategy
                - "merge": Merge seeds and expanded neighbors
                - "seed_only": Seeds only
        """
        if verbose:
            print(f"\n{'─'*60}")
            print(f"[Query] {query}")
            print(f"[Method] 2-hop static expansion")

        pre_names, post_names = self._retrieve_pre_post_tables(
            query,
            top_k=top_k,
            verbose=verbose,
            initial_top_k=initial_top_k,
            expansion_strategy=expansion_strategy,
        )

        if not post_names:
            if verbose:
                print(f"{'─'*60}")
            return []

        if ground_truth:
            hit_pre = self._calc_hit_rate(pre_names, ground_truth)
            hit_post = self._calc_hit_rate(post_names, ground_truth)
            st_pre = "[OK] Full recall" if hit_pre["recall"] == 1.0 else f"recall {hit_pre['recall']:.0%}"
            st_post = "[OK] Full recall" if hit_post["recall"] == 1.0 else f"recall {hit_post['recall']:.0%}"
            print(f"[Hit]   Before expansion: {st_pre} ({hit_pre['hit_count']}/{hit_pre['gt_count']})")
            if hit_pre["miss_tables"]:
                print(f"[Miss]   Before expansion not covered: {', '.join(hit_pre['miss_tables'])}")
            print(f"[Hit]   After expansion: {st_post} ({hit_post['hit_count']}/{hit_post['gt_count']})")
            if hit_post["miss_tables"]:
                print(f"[Miss]   After expansion not covered: {', '.join(hit_post['miss_tables'])}")

        if verbose:
            print(f"{'─'*60}")

        return post_names

    def process_dataset(
        self,
        ground_truth_file: str,
        output_file: str,
        top_k: int = 15,
        verbose: bool = True,
        initial_top_k: int = 15,
        expansion_strategy: str = "merge",
    ):
        """
        Process the entire dataset.

        Args:
            ground_truth_file: Input file path
            output_file: Output file path
            top_k: Number of tables to return in the final result
            verbose: Whether to print detailed information
            initial_top_k: Initial retrieval count
            expansion_strategy: Expansion strategy
        """
        print(f"Loading data from {ground_truth_file}...")

        with open(ground_truth_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Processing {len(data)} samples...")

        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Summary statistics
        n_with_gt = 0
        n_pre_full = 0
        n_post_full = 0
        sum_recall_pre = 0.0
        sum_recall_post = 0.0

        for i, item in enumerate(tqdm(data, desc="2-hop Schema Routing")):
            question = item.get("question", "")

            if not question:
                item["pred_schema"] = []
                continue

            evidence = item.get("evidence", "")
            if evidence and evidence.strip():
                search_query = f"{question} {evidence.strip()}"
            else:
                search_query = question

            gt_tables = []
            db_id = item.get("db_id", "")
            schema_link = item.get("gold_schema", item.get("schema_link", {}))
            if schema_link:
                for table_key in schema_link:
                    parts = table_key.split(".")
                    if len(parts) >= 2:
                        table_name = parts[1]
                    else:
                        table_name = table_key
                    gt_tables.append(f"{db_id}.{table_name}")

            try:
                pre_tables, post_tables = self._retrieve_pre_post_tables(
                    search_query,
                    top_k=top_k,
                    verbose=verbose,
                    initial_top_k=initial_top_k,
                    expansion_strategy=expansion_strategy,
                )
                item["pred_schema"] = post_tables

                if gt_tables:
                    hr_pre = self._calc_hit_rate(pre_tables, gt_tables)
                    hr_post = self._calc_hit_rate(post_tables, gt_tables)
                    n_with_gt += 1
                    sum_recall_pre += hr_pre["recall"]
                    sum_recall_post += hr_post["recall"]
                    if hr_pre["recall"] >= 1.0:
                        n_pre_full += 1
                    if hr_post["recall"] >= 1.0:
                        n_post_full += 1

            except Exception as e:
                print(f"\n[Error] Error processing question '{question}': {e}")
                item["pred_schema"] = []

            if (i + 1) % 10 == 0:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {output_file}")


def run_full_pipeline(
    schema_csv_path: str,
    ground_truth_file: str,
    output_file: str,
    summary_file: str = None,
    top_k: int = 20,
    initial_top_k: int = 20,
    clear_before_build: bool = False,
    verbose: bool = False,
):
    """
    Full pipeline: build KG + 2-hop retrieval and evaluation

    Args:
        schema_csv_path: Schema CSV file path
        ground_truth_file: Ground Truth JSON file path
        output_file: Retrieval result output path
        summary_file: Time/token summary JSON output path (default: same directory as output_file)
        top_k: Number of tables to return in the final result
        initial_top_k: Initial retrieval count
        clear_before_build: Whether to clear Neo4j before building
        verbose: Whether to print detailed information

    Returns:
        dict: Summary containing time statistics and token usage
    """
    total_t0 = time.time()

    # ========== Step 1: Build Knowledge Graph ==========
    print("\n" + "=" * 60)
    print("  [1/2] Build Schema Knowledge Graph")
    print("=" * 60)
    print(f"  Schema CSV: {schema_csv_path}")
    print(f"  Clear: {clear_before_build}")

    if not os.path.exists(schema_csv_path):
        raise FileNotFoundError(f"Schema CSV file not found: {schema_csv_path}")

    reset_embed_token_usage()
    build_t0 = time.time()

    try:
        build_knowledge_graph(
            schema_path=schema_csv_path,
            clear_before_build=clear_before_build,
        )
    except Exception as e:
        print(f"\n[Error] Knowledge Graph build failed: {e}")
        raise

    build_time = time.time() - build_t0
    build_token_usage = get_embed_token_usage()
    print(f"[Done] Knowledge Graph built successfully ({build_time:.2f}s)")

    # ========== Step 2: 2-hop retrieval and evaluation ==========
    print("\n" + "=" * 60)
    print("  [2/2] 2-hop retrieval and evaluation")
    print("=" * 60)
    print(f"  Ground Truth: {ground_truth_file}")
    print(f"  Output: {output_file}")
    print(f"  top_k / init: {top_k} / {initial_top_k}")

    if not os.path.exists(ground_truth_file):
        raise FileNotFoundError(f"Ground Truth file not found: {ground_truth_file}")

    retriever = SchemaLinkingRetriever2Hop()
    retriever.embedder.reset_token_usage()

    retrieval_t0 = time.time()
    retriever.process_dataset(
        ground_truth_file=ground_truth_file,
        output_file=output_file,
        top_k=top_k,
        verbose=verbose,
        initial_top_k=initial_top_k,
    )
    retrieval_time = time.time() - retrieval_t0
    retrieval_token_usage = retriever.embedder.get_token_usage()

    print(f"\n[Done] 2-hop retrieval and evaluation finished ({retrieval_time:.2f}s)")

    # ========== Summary statistics ==========
    total_time = time.time() - total_t0
    total_tokens = build_token_usage.get("total_tokens", 0) + retrieval_token_usage.get("total_tokens", 0)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "method": "2-hop static expansion",
        "schema_csv": schema_csv_path,
        "ground_truth_file": ground_truth_file,
        "output_file": output_file,
        "time_seconds": {
            "build_kg": round(build_time, 2),
            "retrieval_eval": round(retrieval_time, 2),
            "total": round(total_time, 2),
        },
        "embedding_tokens": {
            "build_kg": {
                "total_tokens": build_token_usage.get("total_tokens", 0),
                "total_requests": build_token_usage.get("total_requests", 0),
            },
            "retrieval_eval": {
                "total_tokens": retrieval_token_usage.get("total_tokens", 0),
                "total_requests": retrieval_token_usage.get("total_requests", 0),
            },
            "total_tokens": total_tokens,
        },
    }

    # Save summary file
    if summary_file is None:
        base, _ = os.path.splitext(output_file)
        summary_file = f"{base}_time_token_summary.json"

    summary_dir = os.path.dirname(summary_file)
    if summary_dir and not os.path.exists(summary_dir):
        os.makedirs(summary_dir)

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("  All pipelines completed!")
    print("=" * 60)
    print(f"  Build time       : {build_time:.2f}s")
    print(f"  Retrieval time   : {retrieval_time:.2f}s")
    print(f"  Total time       : {total_time:.2f}s")
    print(f"  Build tokens     : {build_token_usage.get('total_tokens', 0)}")
    print(f"  Retrieval tokens : {retrieval_token_usage.get('total_tokens', 0)}")
    print(f"  Total tokens     : {total_tokens}")
    print(f"  Summary file     : {summary_file}")

    return summary


def demo_batch_processing():
    """Demo batch processing - 2-hop expansion (retrieval only, no build)"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    ground_truth_file = os.path.join(
        project_root, "dataset_v2/splits/bird/test.json"
    )
    output_file = os.path.join(
        project_root,
        "dataset_v2/schema_link_Base/eval_result/Bird_union_Base_2hop_result_murre.json",
    )

    retriever = SchemaLinkingRetriever2Hop()

    retriever.process_dataset(
        ground_truth_file=ground_truth_file,
        output_file=output_file,
        top_k=20,
        verbose=False,
        initial_top_k=20,
    )


def demo_full_pipeline():
    """Demo full pipeline - build + 2-hop retrieval"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    schema_csv_path = os.path.join(
        project_root, "dataset_v2/SynLink/SynSQL_schema_csv_300.csv"
    )
    ground_truth_file = os.path.join(
        project_root, "dataset_v2/splits/synlink/test.json"
    )
    output_file = os.path.join(
        project_root,
        "dataset_v2/schema_link_Base/eval_result_synlink/SynLink_Base_2hop_result_murre.json",
    )

    run_full_pipeline(
        schema_csv_path=schema_csv_path,
        ground_truth_file=ground_truth_file,
        output_file=output_file,
        top_k=20,
        initial_top_k=20,
        clear_before_build=False,
        verbose=False,
    )


if __name__ == "__main__":
    # Use full pipeline (build + retrieval)
    demo_full_pipeline()
    # Or retrieval only:
    # demo_batch_processing()
