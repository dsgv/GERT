"""
Unified GERT Runner with Integrated Recovery, Rescue, and Structural Distance Stratification
Strictly uses dataset_v2 without heuristic bridge table metrics.
In a single pass, simultaneously records:
  - Dense semantic retrieval (pre_names)
  - GERT structural propagation (post_names)
  - Ablation variants (w/o Vector, w/o SeedDecay, 1-Hop, BM25)
Directly computes:
  - Standard metrics: Table Rec., Perfect Recall (Table CR), Precision, F1
  - Recovery@K, RescueRate@K, DamageRate@K
  - Structural distance stratification (1-hop, 2-hop, 3-hop+, Disconnected)
"""

import argparse
import copy
from collections import defaultdict, deque
import json
import math
import os
import sys
import time
from typing import Dict, List, Set, Tuple, Any

import pandas as pd
from tqdm import tqdm

from pathlib import Path
current_dir = Path(__file__).resolve().parent
repo_root = current_dir
for p in [current_dir, current_dir.parent, current_dir.parents[1], current_dir.parents[2]]:
    if (p / "core").is_dir() and (p / "data").is_dir():
        repo_root = p
        break

project_root = str(repo_root)

if str(repo_root) in sys.path:
    sys.path.remove(str(repo_root))
sys.path.insert(0, str(repo_root))

baselines_dir = str(repo_root / "experiments" / "01_main_benchmark" / "baselines")
if baselines_dir not in sys.path and os.path.exists(baselines_dir):
    sys.path.insert(1, baselines_dir)

scripts_dir = str(repo_root / "experiments" / "01_main_benchmark" / "scripts")
if scripts_dir not in sys.path and os.path.exists(scripts_dir):
    sys.path.insert(1, scripts_dir)

from core.schemakg.schemakg_search_ppr import (
    BaseSchemaLinkingRetrieverPPR,
    build_seed_weights,
    personalized_pagerank,
)
from bm25_retrieval import BM25SchemaLinker
from core.pipeline.kg_construction import build_knowledge_graph, clear_neo4j_database
from calculate_overall_metrics_from_path import strict_name, evaluate


def _resolve_path(rel_path: str) -> str:
    candidates = [
        os.path.join(project_root, "data", rel_path),
        os.path.join(project_root, "dataset_v2", rel_path),
        os.path.join(project_root, "schema_link", "dataset_v2", rel_path),
        os.path.join(project_root, rel_path),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def load_json(path: Any) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def build_db_adj(schema_csv_path: str):
    """Build table-level foreign key adjacency graph per database."""
    df = pd.read_csv(schema_csv_path)
    adj = defaultdict(lambda: defaultdict(set))
    for _, row in df.iterrows():
        fk = str(row.get("foreign_key_ref", ""))
        if "." in fk and pd.notna(row.get("column_name")):
            db = str(row["db_name"])
            t1 = strict_name(f"{db}.{row['table_name']}")
            ref_t = fk.split(".")[0]
            t2 = strict_name(f"{db}.{ref_t}")
            if t1 != t2:
                adj[db][t1].add(t2)
                adj[db][t2].add(t1)
    return adj


def get_distance(adj_graph, source, targets):
    """Compute BFS shortest path distance from source to nearest target."""
    if not targets:
        return -1
    if source in targets:
        return 0
    queue = deque([(source, 0)])
    visited = {source}
    while queue:
        curr, d = queue.popleft()
        if curr in targets:
            return d
        for nbr in adj_graph.get(curr, set()):
            if nbr not in visited:
                visited.add(nbr)
                queue.append((nbr, d + 1))
    return -1


def run_dataset_pipeline(
    dataset_name: str,
    schema_csv_path: str,
    data_json_path: str,
    output_dir: str,
    top_k: int = 20,
    eval_k: int = 15,
    build_kg: bool = True,
    sample_limit: int = None,
) -> Dict[str, Any]:
    print(f"\n{'='*75}")
    print(f"PROCESSING DATASET: {dataset_name.upper()} (dataset_v2)")
    print(f"{'='*75}")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Build Knowledge Graph if requested
    if build_kg:
        print(f"\n[1/4] Clearing Neo4j and building Knowledge Graph from {schema_csv_path}...")
        t0 = time.time()
        build_knowledge_graph(schema_csv_path, clear_before_build=True)
        print(f"Knowledge Graph built in {time.time() - t0:.2f} s")
    else:
        print("[1/4] Skipping KG build (using existing Neo4j graph).")

    # 2. Build local FK adjacency for distance stratification
    print("\n[2/4] Building Foreign-Key adjacency graphs...")
    db_adj = build_db_adj(schema_csv_path)

    # 3. Setup retrievers
    print("\n[3/4] Initializing retrievers (Dense, PPR, BM25)...")
    bm25_linker = BM25SchemaLinker(schema_csv_path)
    retriever = BaseSchemaLinkingRetrieverPPR()
    retriever.invalidate_fk_cache()
    fk_edges = retriever._load_fk_edges()

    # Pre-build undirected adjacency list for PPR / 1-hop
    fk_adj_dict: Dict[str, Set[str]] = {}
    for a, b in fk_edges:
        fk_adj_dict.setdefault(a, set()).add(b)
        fk_adj_dict.setdefault(b, set()).add(a)

    # Load dataset JSON
    samples = load_json(data_json_path)
    if sample_limit:
        samples = samples[:sample_limit]
        print(f"[Notice] Limiting execution to first {sample_limit} samples.")

    total_samples = len(samples)
    print(f"\n[4/4] Running single-pass inference on {total_samples} queries...")

    methods = [
        "GERT_Full",
        "w_o_PPR",
        "w_o_Vector",
        "w_o_SeedDecay",
        "Baseline_1Hop",
        "Baseline_BM25",
    ]
    results_data = {m: copy.deepcopy(samples) for m in methods}

    # Tracking for recovery and stratification metrics
    total_omitted = 0
    recovered_omitted = 0
    dense_fail_queries = 0
    rescued_queries = 0
    dense_succ_queries = 0
    damaged_queries = 0

    bucket_omitted = defaultdict(int)
    bucket_recovered = defaultdict(int)

    start_time = time.time()
    for idx in tqdm(range(total_samples), desc=f"Evaluating {dataset_name}"):
        item = samples[idx]
        question = item.get("question", "")
        evidence = item.get("evidence") or item.get("external_knowledge") or ""
        if evidence and str(evidence).strip():
            search_query = f"{question} {str(evidence).strip()}"
        else:
            search_query = question

        if not search_query.strip():
            for m in methods:
                results_data[m][idx]["pred_schema"] = []
            continue

        gt_tables = {strict_name(t) for t in item.get("gold_schema", item.get("schema_link", []))}
        db_name = str(item.get("db_id", ""))
        graph = db_adj[db_name]

        # 1. Single-pass Dense Retrieval
        try:
            initial_tables, retrieval_scores = retriever._get_retrieved_tables(
                search_query, top_k=top_k, verbose=False
            )
            seed_order = [t.table_name for t in initial_tables]
            dense_top = seed_order[:top_k]
            results_data["w_o_PPR"][idx]["pred_schema"] = dense_top

            # Normalized retrieval scores
            ret_total = sum(retrieval_scores.values())
            norm_ret = (
                {k: v / ret_total for k, v in retrieval_scores.items()}
                if ret_total > 0
                else {}
            )

            # Build graph for PPR
            nodes, adj = retriever._build_graph_for_ppr(seed_order)

            def get_fused_ranking(pers_weights: Dict[str, float], beta: float) -> List[str]:
                ppr_s = personalized_pagerank(
                    nodes, adj, pers_weights, alpha=0.85, max_iter=100, tol=1e-6
                )
                p_sum = sum(ppr_s.values())
                if p_sum > 0:
                    ppr_s = {k: v / p_sum for k, v in ppr_s.items()}
                fused = {}
                for n in nodes:
                    fused[n] = (1.0 - beta) * norm_ret.get(n, 0.0) + beta * ppr_s.get(n, 0.0)
                ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
                return [name for name, _ in ranked[:top_k]]

            # GERT Full (decay=0.3, beta=0.7)
            pers_decay = build_seed_weights(seed_order, decay=0.3)
            gert_top = get_fused_ranking(pers_decay, beta=0.7)
            results_data["GERT_Full"][idx]["pred_schema"] = gert_top

            # w/o Vector (beta=1.0, pure PPR)
            results_data["w_o_Vector"][idx]["pred_schema"] = get_fused_ranking(
                pers_decay, beta=1.0
            )

            # w/o SeedDecay (decay=0.0, uniform seeds)
            pers_uniform = build_seed_weights(seed_order, decay=0.0)
            results_data["w_o_SeedDecay"][idx]["pred_schema"] = get_fused_ranking(
                pers_uniform, beta=0.7
            )

            # 1-Hop Expansion
            sorted_1hop: List[str] = []
            added_set: Set[str] = set()
            for s in seed_order:
                if s not in added_set:
                    sorted_1hop.append(s)
                    added_set.add(s)
                nbrs = list(fk_adj_dict.get(s, set()))
                nbrs.sort(key=lambda t: -len(fk_adj_dict.get(t, set())))
                for nb in nbrs:
                    if nb not in added_set:
                        sorted_1hop.append(nb)
                        added_set.add(nb)
            results_data["Baseline_1Hop"][idx]["pred_schema"] = sorted_1hop[:top_k]

        except Exception as e:
            print(f"\nError in GERT retrieval at query {idx}: {e}")
            for m in ["GERT_Full", "w_o_PPR", "w_o_Vector", "w_o_SeedDecay", "Baseline_1Hop"]:
                results_data[m][idx]["pred_schema"] = []
            dense_top, gert_top = [], []

        # 2. BM25 Baseline
        try:
            bm25_tables = bm25_linker.retrieve(search_query, top_k=top_k)
            results_data["Baseline_BM25"][idx]["pred_schema"] = bm25_tables
        except Exception as e:
            results_data["Baseline_BM25"][idx]["pred_schema"] = []

        # 3. Compute in-depth Recovery & Stratification metrics (at eval_k)
        d_k = {strict_name(t) for t in dense_top[:eval_k]}
        s_k = {strict_name(t) for t in gert_top[:eval_k]}

        pr_d = 1 if gt_tables <= d_k else 0
        pr_s = 1 if gt_tables <= s_k else 0

        if pr_d == 0:
            dense_fail_queries += 1
            if pr_s == 1:
                rescued_queries += 1
        else:
            dense_succ_queries += 1
            if pr_s == 0:
                damaged_queries += 1

        recalled_by_dense = gt_tables & d_k
        omitted = gt_tables - d_k
        total_omitted += len(omitted)
        recovered_omitted += len(omitted & s_k)

        for t in omitted:
            dist = get_distance(graph, t, recalled_by_dense)
            if dist == 1:
                b_name = "1-hop"
            elif dist == 2:
                b_name = "2-hop"
            elif dist >= 3:
                b_name = "3+ hops"
            else:
                b_name = "N/A"

            bucket_omitted[b_name] += 1
            if t in s_k:
                bucket_recovered[b_name] += 1

    elapsed = time.time() - start_time
    print(f"\nInference completed in {elapsed:.2f} s ({elapsed/total_samples:.3f} s/query).")

    # Save output JSONs
    output_files = {}
    for m in methods:
        out_path = os.path.join(output_dir, f"{dataset_name}_{m}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results_data[m], f, indent=2, ensure_ascii=False)
        output_files[m] = out_path

    # Standard metrics across all methods
    method_metrics = []
    gold_rows = samples
    for m in methods:
        overall, _, _ = evaluate(gold_rows, results_data[m], k=eval_k)
        method_metrics.append({
            "Dataset": dataset_name,
            "Method": m,
            "Table Rec.": overall["Table Rec."],
            "Table CR": overall["Table CR"],
            "Precision": overall["Precision"],
            "F1": overall["F1"],
        })

    # Recovery metrics
    rec_rate = recovered_omitted / total_omitted if total_omitted else 0.0
    rescue_rate = rescued_queries / dense_fail_queries if dense_fail_queries else 0.0
    damage_rate = damaged_queries / dense_succ_queries if dense_succ_queries else 0.0

    pr_dense = dense_succ_queries / total_samples
    pr_gert = (dense_succ_queries - damaged_queries + rescued_queries) / total_samples

    recovery_summary = {
        "Dataset": dataset_name,
        "DR Table CR (w/o PPR)": f"{pr_dense:.2%}",
        "GERT Table CR (Full)": f"{pr_gert:.2%}",
        "ΔCR": f"+{pr_gert - pr_dense:.2%}",
        "Total gold tables missed by dense retrieval": total_omitted,
        "Tables recovered by PPR": recovered_omitted,
        f"Table Recovery@{eval_k}": f"{rec_rate:.2%}",
        "Dense retrieval failures": dense_fail_queries,
        "Queries rescued by GERT": rescued_queries,
        f"Query Gain@{eval_k}": f"{rescue_rate:.2%}",
        f"Query Loss@{eval_k}": f"{damage_rate:.2%}",
    }

    # Stratification table (Recovery rate on omitted tables - Table 8)
    strat_omitted = []
    for b in ["1-hop", "2-hop", "3+ hops", "N/A"]:
        tot = bucket_omitted[b]
        rec = bucket_recovered[b]
        r = rec / tot if tot else 0.0
        strat_omitted.append({
            "Foreign-key distance": b,
            "Total gold tables missed by dense retrieval": tot,
            "Tables recovered by PPR": rec,
            "Recovery rate": f"{r:.2%}",
        })

    return {
        "dataset_name": dataset_name,
        "method_metrics": method_metrics,
        "recovery_summary": recovery_summary,
        "strat_omitted": strat_omitted,
        "output_files": output_files,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["spider", "bird", "synlink", "all"], default="all")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--eval_k", type=int, default=15)
    parser.add_argument("--skip_build", action="store_true")
    parser.add_argument("--sample_limit", type=int, default=None)
    args = parser.parse_args()

    spider_schema = _resolve_path("spider/spider_union_schema_FK.csv")
    spider_json = _resolve_path("spider/murre_spider_dev.json")

    bird_schema = _resolve_path("bird/bird_union_schema_FK.csv")
    bird_json = _resolve_path("bird/murre_bird_dev.json")

    synlink_schema = _resolve_path("SynLink/SynSQL_schema_csv_300.csv")
    synlink_json = _resolve_path("SynLink/Formal_moderate_1k_gd.json")

    all_results = {}

    # Run Spider
    if args.dataset in ["spider", "all"]:
        spider_out = os.path.join(project_root, "output/spider_results")
        all_results["SpiderUnion"] = run_dataset_pipeline(
            dataset_name="SpiderUnion",
            schema_csv_path=spider_schema,
            data_json_path=spider_json,
            output_dir=spider_out,
            top_k=args.top_k,
            eval_k=args.eval_k,
            build_kg=not args.skip_build,
            sample_limit=args.sample_limit,
        )

    # Run Bird
    if args.dataset in ["bird", "all"]:
        bird_out = os.path.join(project_root, "output/bird_results")
        all_results["BirdUnion"] = run_dataset_pipeline(
            dataset_name="BirdUnion",
            schema_csv_path=bird_schema,
            data_json_path=bird_json,
            output_dir=bird_out,
            top_k=args.top_k,
            eval_k=args.eval_k,
            build_kg=not args.skip_build,
            sample_limit=args.sample_limit,
        )

    # Run SynLink
    if args.dataset in ["synlink", "all"]:
        synlink_out = os.path.join(project_root, "output/synlink_results")
        all_results["SynLink"] = run_dataset_pipeline(
            dataset_name="SynLink",
            schema_csv_path=synlink_schema,
            data_json_path=synlink_json,
            output_dir=synlink_out,
            top_k=args.top_k,
            eval_k=args.eval_k,
            build_kg=not args.skip_build,
            sample_limit=args.sample_limit,
        )

    # Aggregate & format full markdown report
    all_method_metrics = []
    all_recovery_summaries = []

    for name, r in all_results.items():
        all_method_metrics.extend(r["method_metrics"])
        all_recovery_summaries.append(r["recovery_summary"])

    df_main = pd.DataFrame(all_method_metrics)
    df_rec = pd.DataFrame(all_recovery_summaries)

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    print("\n### 1. Main & Ablation Performance")
    print(df_main.to_markdown(index=False))
    print("\n### 2. Table Recovery & Query Rescue (Tables 6 & 7)")
    print(df_rec.to_markdown(index=False))

    for name, r in all_results.items():
        print(f"\n### 3. {name} Recovery by FK Distance (Table 8)")
        print(pd.DataFrame(r["strat_omitted"]).to_markdown(index=False))

    summary_file = os.path.join(project_root, "output/gert_recovery_analysis_summary.md")
    report_sections = [
        f"# GERT Comprehensive Evaluation & In-Depth Analysis\nEvaluated with candidate budget Top-K = {args.eval_k}\n",
        "## 1. Main & Ablation Performance\n" + df_main.to_markdown(index=False),
        "## 2. In-Depth Analysis: Table Recovery & Query Rescue\n" + df_rec.to_markdown(index=False),
    ]
    for name, r in all_results.items():
        report_sections.append(
            f"## 3. {name} Recovery by FK Distance (Table 8)\n"
            + pd.DataFrame(r["strat_omitted"]).to_markdown(index=False)
        )
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(report_sections) + "\n")
    print(f"\nReport successfully written to {summary_file}")


if __name__ == "__main__":
    main()



