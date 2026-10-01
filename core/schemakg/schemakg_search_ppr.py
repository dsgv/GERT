"""
Schema Routing vector retrieval + PPR (Personalized PageRank) expansion module
"""
import json
import os
import sys
import re
import math
import time
import traceback
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from tqdm import tqdm

# Configure path to import pipeline module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
pipeline_dir = os.path.join(project_root, "pipeline")

if project_root not in sys.path:
    sys.path.append(project_root)
if pipeline_dir not in sys.path:
    sys.path.append(pipeline_dir)

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


# Export table-level foreign key edges (for undirected graph: each edge stored bidirectionally in adjacency list)
FK_TABLE_PAIR_QUERY = """
MATCH (t1:Table)-[:CONTAINS]->(c1:Column)-[:REFERENCES]->(c2:Column)<-[:CONTAINS]-(t2:Table)
WHERE t1.name <> t2.name
RETURN DISTINCT t1.name AS a, t2.name AS b
"""


def personalized_pagerank(
    nodes: List[str],
    adj: Dict[str, List[str]],
    personalization: Dict[str, float],
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Dict[str, float]:
    """
    Personalized PageRank (random walk on undirected graph: walks uniformly to neighbors).

    r^{t+1} = (1-alpha)*v + alpha * P^T * r^t
    where v is the personalization vector (normalized internally), P is row-stochastic, P[i,j]=1/deg(i) if i,j are adjacent.

    Args:
        nodes: All nodes (consistent with adj keys)
        adj: Adjacency list
        personalization: Non-negative weights, normalized to v inside the function
        alpha: Damping factor (defaults to common PageRank default 0.85)
        max_iter / tol: Convergence conditions

    Returns:
        Node -> PPR score
    """
    node_set = set(nodes)
    if not node_set:
        return {}

    v = {n: 0.0 for n in nodes}
    for name, w in personalization.items():
        if name in node_set and w > 0:
            v[name] += w
    s = sum(v.values())
    if s <= 0:
        n = len(nodes)
        for name in nodes:
            v[name] = 1.0 / n
    else:
        for name in nodes:
            v[name] /= s

    deg: Dict[str, int] = {}
    for u in nodes:
        neighbors = [x for x in adj.get(u, []) if x in node_set]
        deg[u] = len(neighbors)

    r = {n: v[n] for n in nodes}

    for _ in range(max_iter):
        r_new = {n: (1.0 - alpha) * v[n] for n in nodes}
        dangling_mass = 0.0
        for i in nodes:
            if deg[i] == 0:
                dangling_mass += r[i]
                continue
            for j in adj.get(i, []):
                if j in node_set:
                    r_new[j] += alpha * r[i] / deg[i]
        if dangling_mass > 0:
            for j in nodes:
                r_new[j] += alpha * dangling_mass * v[j]

        diff = sum(abs(r_new[n] - r[n]) for n in nodes)
        r = r_new
        if diff < tol:
            break

    return r


def build_seed_weights(
    ordered_seeds: List[str],
    decay: float = 0.0,
) -> Dict[str, float]:
    """
    Assign weights to seeds in retrieval order and normalize.
    decay==0: uniform; decay>0: weight ∝ exp(-decay * rank)
    """
    if not ordered_seeds:
        return {}
    seen: Set[str] = set()
    unique: List[str] = []
    for s in ordered_seeds:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    if not unique:
        return {}

    if decay <= 0:
        w = 1.0 / len(unique)
        return {name: w for name in unique}

    weights = [math.exp(-decay * i) for i in range(len(unique))]
    z = sum(weights)
    return {name: weights[i] / z for i, name in enumerate(unique)}


class BaseSchemaLinkingRetrieverPPR:
    """
    Vector retrieval + Personalized PageRank on table-level FK graph, for table ranking/expansion.
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
        self._fk_edges: Optional[List[Tuple[str, str]]] = None

    def invalidate_fk_cache(self) -> None:
        """Call after graph structure changes; re-fetches FK edges next time."""
        self._fk_edges = None

    def _load_fk_edges(self) -> List[Tuple[str, str]]:
        if self._fk_edges is not None:
            return self._fk_edges
        pairs: List[Tuple[str, str]] = []
        with self.driver.session() as session:
            result = session.run(FK_TABLE_PAIR_QUERY)
            for record in result:
                a = record.get("a")
                b = record.get("b")
                if a and b and a != b:
                    pairs.append((a, b))
        if self._cache_fk_graph:
            self._fk_edges = pairs
        return pairs

    def _build_graph_for_ppr(
        self,
        seed_tables: List[str],
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Build undirected adjacency list from all FK edges; node set = all tables on edges ∪ seed tables (isolated seeds can also participate in PPR).
        """
        edges = self._load_fk_edges()
        nodes: Set[str] = set(seed_tables)
        adj: Dict[str, Set[str]] = {}

        def add_edge(u: str, v: str) -> None:
            nodes.add(u)
            nodes.add(v)
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        for a, b in edges:
            add_edge(a, b)

        for s in seed_tables:
            nodes.add(s)
            adj.setdefault(s, set())

        node_list = sorted(nodes)
        adj_list: Dict[str, List[str]] = {n: sorted(adj.get(n, set())) for n in node_list}
        return node_list, adj_list

    def _get_retrieved_tables(
        self,
        query: str,
        top_k: int,
        verbose: bool,
    ) -> Tuple[List[TableInfo], Dict[str, float]]:
        """
        Execute hybrid retrieval, returning table info list and retrieval score dict.

        Returns:
            Tuple[List[TableInfo], Dict[str, float]]: (table list, {table_name: normalized retrieval score})
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

            # Normalized retrieval score: use reciprocal rank (higher rank = higher score)
            # Can also use exponential decay: exp(-decay * i)
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
        alpha: float,
        seed_rank_decay: float,
        max_ppr_iter: int,
        ppr_tol: float,
        fusion_weight: float = 0.5,
    ) -> Tuple[List[str], List[str]]:
        """
        Return (top_k table names before PPR, top_k table names after fusion).

        Fusion strategy: final_score = (1 - fusion_weight) * retrieval_score + fusion_weight * ppr_score

        Args:
            fusion_weight: Weight of PPR score, range [0, 1]
                - 0: Fully use original retrieval ranking (no PPR)
                - 0.5: Retrieval score and PPR score each contribute half
                - 1: Fully use PPR ranking (ignore original retrieval score)
        """
        initial_tables, retrieval_scores = self._get_retrieved_tables(query, initial_top_k, verbose)
        if not initial_tables:
            return [], []

        seed_order = [t.table_name for t in initial_tables]
        pre_names = seed_order[:top_k]

        pers = build_seed_weights(seed_order, decay=seed_rank_decay)
        nodes, adj = self._build_graph_for_ppr(seed_order)
        ppr_scores = personalized_pagerank(
            nodes,
            adj,
            pers,
            alpha=alpha,
            max_iter=max_ppr_iter,
            tol=ppr_tol,
        )

        # Normalize PPR scores (ensure sum = 1)
        ppr_total = sum(ppr_scores.values())
        if ppr_total > 0:
            ppr_scores = {k: v / ppr_total for k, v in ppr_scores.items()}

        # Normalize retrieval scores (ensure sum = 1)
        ret_total = sum(retrieval_scores.values())
        if ret_total > 0:
            retrieval_scores = {k: v / ret_total for k, v in retrieval_scores.items()}

        # Fusion scores
        fused_scores: Dict[str, float] = {}
        for name in nodes:
            ret_s = retrieval_scores.get(name, 0.0)
            ppr_s = ppr_scores.get(name, 0.0)
            fused_scores[name] = (1 - fusion_weight) * ret_s + fusion_weight * ppr_s

        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        post_names = [name for name, _ in ranked[:top_k]]

        if verbose:
            top_show = min(5, len(ranked))
            preview = ", ".join(f"{n}:{fused_scores[n]:.4f}" for n, _ in ranked[:top_show])
            print(f"[Fusion] fusion_weight={fusion_weight:.2f} (retrieval:{1-fusion_weight:.0%}, PPR:{fusion_weight:.0%})")
            print(f"[Fusion scores top] {preview}{' ...' if len(ranked) > top_show else ''}")
            print(f"[Original retrieval top_{top_k}] {', '.join(pre_names)}")
            print(f"[Fusion output top_{top_k}] {', '.join(post_names)}")

        return pre_names, post_names

    def retrieve_tables(
        self,
        query: str,
        top_k: int = 15,
        ground_truth: List[str] = None,
        verbose: bool = True,
        initial_top_k: int = 15,
        alpha: float = 0.85,
        seed_rank_decay: float = 0.3,
        max_ppr_iter: int = 100,
        ppr_tol: float = 1e-6,
        fusion_weight: float = 0.5,
    ) -> List[str]:
        """
        Vector retrieval to get seeds → run PPR on table-level FK undirected graph → fuse scores and return top_k table names.

        Args:
            query: User query
            top_k: Number of tables to return
            initial_top_k: Number of vector retrieval candidates (seed set; typically >= top_k)
            alpha: PPR damping factor
            seed_rank_decay: Seed weight decay by rank; 0 means uniform; ~0.1-0.2 gives higher-ranked tables more weight
            max_ppr_iter / ppr_tol: Power iteration parameters
            fusion_weight: PPR score weight [0, 1]
                - 0: Fully use original retrieval ranking
                - 0.5: Retrieval and PPR each contribute half
                - 1: Fully use PPR ranking
        """
        if verbose:
            print(f"\n{'─'*60}")
            print(f"[Query] {query}")

        pre_names, ranked_names = self._retrieve_pre_post_tables(
            query,
            top_k=top_k,
            verbose=verbose,
            initial_top_k=initial_top_k,
            alpha=alpha,
            seed_rank_decay=seed_rank_decay,
            max_ppr_iter=max_ppr_iter,
            ppr_tol=ppr_tol,
            fusion_weight=fusion_weight,
        )
        if not ranked_names:
            if verbose:
                print(f"{'─'*60}")
            return []

        if ground_truth:
            hit_pre = self._calc_hit_rate(pre_names, ground_truth)
            hit_post = self._calc_hit_rate(ranked_names, ground_truth)
            st_pre = "[OK] Full recall" if hit_pre["recall"] == 1.0 else f"recall {hit_pre['recall']:.0%}"
            st_post = "[OK] Full recall" if hit_post["recall"] == 1.0 else f"recall {hit_post['recall']:.0%}"
            print(f"[Hit]   Before PPR: {st_pre} ({hit_pre['hit_count']}/{hit_pre['gt_count']})")
            if hit_pre["miss_tables"]:
                print(f"[Miss]   Before PPR, GT tables not covered: {', '.join(hit_pre['miss_tables'])}")
            print(f"[Hit]   After PPR: {st_post} ({hit_post['hit_count']}/{hit_post['gt_count']})")
            if hit_post["miss_tables"]:
                print(f"[Miss]   After PPR, GT tables not covered: {', '.join(hit_post['miss_tables'])}")

        if verbose:
            print(f"{'─'*60}")

        return ranked_names

    def process_dataset(
        self,
        ground_truth_file: str,
        output_file: str,
        top_k: int = 15,
        verbose: bool = True,
        initial_top_k: int = 15,
        alpha: float = 0.85,
        seed_rank_decay: float = 0.3,
        max_ppr_iter: int = 100,
        ppr_tol: float = 1e-6,
        fusion_weight: float = 0.5,
    ):
        print(f"Loading data from {ground_truth_file}...")

        with open(ground_truth_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        total_samples = len(data)
        print(f"Processing {total_samples} samples...")
        start_time = time.time()

        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Summary statistics (only samples with GT table annotations)
        n_with_gt = 0
        n_pre_full = 0
        n_post_full = 0
        sum_recall_pre = 0.0
        sum_recall_post = 0.0

        for i, item in enumerate(tqdm(data, desc="Base+PPR Schema Routing")):
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
                    alpha=alpha,
                    seed_rank_decay=seed_rank_decay,
                    max_ppr_iter=max_ppr_iter,
                    ppr_tol=ppr_tol,
                    fusion_weight=fusion_weight,
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
                err_type = type(e).__name__
                err_msg = repr(e)
                print(f"\n[Error] Error processing question '{question}': {err_type} {err_msg}")
                print(traceback.format_exc())
                item["pred_schema"] = []

            if (i + 1) % 10 == 0:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        elapsed = time.time() - start_time
        avg_per_query = elapsed / total_samples

        print(f"\nResults saved to: {output_file}")
        print(f"Total retrieval time: {elapsed:.2f} s")
        print(f"Average time per query: {avg_per_query:.2f} s ({total_samples} samples)")

        return {
            "n_with_gt": n_with_gt,
            "n_pre_full": n_pre_full,
            "n_post_full": n_post_full,
            "avg_recall_pre": sum_recall_pre / n_with_gt if n_with_gt > 0 else 0.0,
            "avg_recall_post": sum_recall_post / n_with_gt if n_with_gt > 0 else 0.0,
        }

def demo_batch_processing():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    ground_truth_file = os.path.join(
        project_root, "data/bird/murre_bird_dev.json"
    )
    output_file = os.path.join(
        project_root,
        "output/Bird_result_murre_restart.json",
    )
    retriever = BaseSchemaLinkingRetrieverPPR()

    retriever.process_dataset(
        ground_truth_file=ground_truth_file,
        output_file=output_file,
        top_k=20,
        verbose=False,
        initial_top_k=20,
        alpha=0.85,
        seed_rank_decay=0.3,
        fusion_weight=0.7,
    )
if __name__ == "__main__":
    demo_batch_processing()

