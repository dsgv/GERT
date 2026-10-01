"""
Evaluation of Gold Table Recovery, Query Rescue/Damage, and Structural Distance Stratification
Based on dataset_v2
"""

import json
import os
import sys
from collections import defaultdict, deque
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
if project_root not in sys.path:
    sys.path.append(project_root)

from calculate_overall_metrics_from_path import strict_name


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def build_db_adj(schema_csv_path):
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
    """Compute BFS shortest path distance from source table to nearest target table."""
    if not targets:
        return -1  # Disconnected
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
    return -1  # Disconnected


def analyze_dataset_recovery(schema_csv, gold_path, dense_path, gert_path, k=15):
    gold_data = load_json(gold_path)
    dense_data = load_json(dense_path)
    gert_data = load_json(gert_path)

    adj = build_db_adj(schema_csv)

    total_queries = len(gold_data)
    total_omitted = 0
    recovered_omitted = 0

    dense_fail_queries = 0
    rescued_queries = 0

    dense_succ_queries = 0
    damaged_queries = 0

    # Distance stratification: on omitted gold tables
    bucket_omitted = defaultdict(int)
    bucket_recovered = defaultdict(int)

    # Distance stratification: on ALL gold tables
    bucket_all_total = defaultdict(int)
    bucket_all_dense_hit = defaultdict(int)
    bucket_all_gert_hit = defaultdict(int)

    for i, g in enumerate(gold_data):
        db = str(g.get("db_id", ""))
        db_graph = adj[db]
        gt = {strict_name(t) for t in g.get("gold_schema", g.get("schema_link", []))}
        d_preds = {strict_name(t) for t in (dense_data[i].get("pred_schema") or dense_data[i].get("pred_schema_link", []))[:k]}
        s_preds = {strict_name(t) for t in (gert_data[i].get("pred_schema") or gert_data[i].get("pred_schema_link", []))[:k]}

        pr_dense = 1 if gt <= d_preds else 0
        pr_gert = 1 if gt <= s_preds else 0

        # Query-level rescue / damage
        if pr_dense == 0:
            dense_fail_queries += 1
            if pr_gert == 1:
                rescued_queries += 1
        else:
            dense_succ_queries += 1
            if pr_gert == 0:
                damaged_queries += 1

        # Table-level omitted
        recalled_by_dense = gt & d_preds
        m = gt - d_preds
        total_omitted += len(m)
        recovered_omitted += len(m & s_preds)

        for t in m:
            dist = get_distance(db_graph, t, recalled_by_dense)
            if dist == 1:
                b_name = "1-hop"
            elif dist == 2:
                b_name = "2-hop"
            elif dist >= 3:
                b_name = "3-hop+"
            else:
                b_name = "disconnected"

            bucket_omitted[b_name] += 1
            if t in s_preds:
                bucket_recovered[b_name] += 1

        # All gold tables stratification (distance to ANY seed table in d_preds)
        for t in gt:
            # Distance from t to the nearest seed retrieved by Dense
            dist_seed = get_distance(db_graph, t, d_preds)
            if dist_seed == 0:
                b_s = "0-hop (direct hit)"
            elif dist_seed == 1:
                b_s = "1-hop"
            elif dist_seed == 2:
                b_s = "2-hop"
            elif dist_seed >= 3:
                b_s = "3-hop+"
            else:
                b_s = "disconnected"

            bucket_all_total[b_s] += 1
            if t in d_preds:
                bucket_all_dense_hit[b_s] += 1
            if t in s_preds:
                bucket_all_gert_hit[b_s] += 1

    recovery_rate = recovered_omitted / total_omitted if total_omitted else 0.0
    rescue_rate = rescued_queries / dense_fail_queries if dense_fail_queries else 0.0
    damage_rate = damaged_queries / dense_succ_queries if dense_succ_queries else 0.0

    pr_dense_overall = dense_succ_queries / total_queries
    pr_gert_overall = (total_queries - dense_fail_queries + rescued_queries - damaged_queries) / total_queries

    # Stratification on omitted gold tables
    omitted_strat = []
    for b in ["1-hop", "2-hop", "3-hop+", "disconnected"]:
        tot = bucket_omitted[b]
        rec = bucket_recovered[b]
        rec_rate = rec / tot if tot else 0.0
        omitted_strat.append({
            "Structural distance": b,
            "Omitted gold tables": tot,
            "Tables recovered by PPR": rec,
            "Recovery rate": f"{rec_rate:.2%}",
        })

    # Stratification across all gold tables (Dense Recall vs GERT Recall vs Gain)
    all_strat = []
    for b in ["1-hop", "2-hop", "3-hop+", "disconnected"]:
        tot = bucket_all_total[b]
        d_hit = bucket_all_dense_hit[b]
        s_hit = bucket_all_gert_hit[b]
        d_rec = d_hit / tot if tot else 0.0
        s_rec = s_hit / tot if tot else 0.0
        gain = s_rec - d_rec
        all_strat.append({
            "Structural distance": b,
            "Total gold tables": tot,
            "Dense Recall": f"{d_rec:.2%}",
            "GERT Recall": f"{s_rec:.2%}",
            "Gain": f"+{gain:.2%}" if gain >= 0 else f"{gain:.2%}",
        })

    return {
        "dataset_size": total_queries,
        "pr_dense": pr_dense_overall,
        "pr_gert": pr_gert_overall,
        "pr_gain": pr_gert_overall - pr_dense_overall,
        "total_omitted": total_omitted,
        "recovered_omitted": recovered_omitted,
        "recovery_rate": recovery_rate,
        "dense_fail_queries": dense_fail_queries,
        "rescued_queries": rescued_queries,
        "rescue_rate": rescue_rate,
        "damage_rate": damage_rate,
        "omitted_strat": omitted_strat,
        "all_strat": all_strat,
    }


def main():
    all_ds = [
        {
            "name": "SpiderUnion",
            "schema": os.path.join(project_root, "dataset_v2/spider/spider_union_schema_FK.csv"),
            "gold": os.path.join(project_root, "dataset_v2/spider/murre_spider_dev.json"),
            "dense": os.path.join(project_root, "output/spider_results/SpiderUnion_w_o_PPR.json"),
            "gert": os.path.join(project_root, "output/spider_results/SpiderUnion_GERT_Full.json"),
        },
        {
            "name": "BirdUnion",
            "schema": os.path.join(project_root, "dataset_v2/bird/bird_union_schema_FK.csv"),
            "gold": os.path.join(project_root, "dataset_v2/bird/murre_bird_dev.json"),
            "dense": os.path.join(project_root, "output/bird_results/BirdUnion_w_o_PPR.json"),
            "gert": os.path.join(project_root, "output/bird_results/BirdUnion_GERT_Full.json"),
        },
        {
            "name": "SynLink",
            "schema": os.path.join(project_root, "dataset_v2/SynLink/SynSQL_schema_csv_300.csv"),
            "gold": os.path.join(project_root, "dataset_v2/SynLink/Formal_moderate_1k_gd.json"),
            "dense": os.path.join(project_root, "output/synlink_results/SynLink_w_o_PPR.json"),
            "gert": os.path.join(project_root, "output/synlink_results/SynLink_GERT_Full.json"),
        },
    ]
    datasets = [d for d in all_ds if os.path.exists(d["gert"])]

    print("=" * 80)
    print("GOLD TABLE RECOVERY & STRUCTURAL DISTANCE ANALYSIS (dataset_v2)")
    print("=" * 80)

    summary_rows = []
    all_strats = {}

    for d in datasets:
        res = analyze_dataset_recovery(d["schema"], d["gold"], d["dense"], d["gert"], k=15)
        summary_rows.append({
            "Dataset": d["name"],
            "w/o PPR (Dense PR)": f"{res['pr_dense']:.2%}",
            "GERT (PR)": f"{res['pr_gert']:.2%}",
            "PR gain": f"+{res['pr_gain']:.2%}",
            "Recovery@15": f"{res['recovery_rate']:.2%}",
            "RescueRate@15": f"{res['rescue_rate']:.2%}",
            "DamageRate@15": f"{res['damage_rate']:.2%}",
        })
        all_strats[d["name"]] = res

    df_summary = pd.DataFrame(summary_rows)
    print("\n### 1. Overall Perfect Recall, Recovery & Rescue Analysis")
    print(df_summary.to_markdown(index=False))

    report_lines = [
        "# GERT Comprehensive In-Depth Recovery & Stratification Analysis (dataset_v2)\n",
        "Evaluated with candidate budget Top-K = 15\n",
        "## 1. Overall Perfect Recall, Recovery & Rescue Analysis\n",
        df_summary.to_markdown(index=False),
    ]

    for name, res in all_strats.items():
        print(f"\n### 2. {name} Structural distance stratification (Structural Distance Stratification)")
        print(f"#### (a) Recovery of gold tables missed by dense retrieval (Omitted Tables Recovery by Distance to Recalled Tables)")
        print(pd.DataFrame(res["omitted_strat"]).to_markdown(index=False))
        print(f"\n#### (b) Gold-table recall by structural distance (Recall & Gain by Structural Distance to Nearest Seed)")
        print(pd.DataFrame(res["all_strat"]).to_markdown(index=False))

        report_lines.append(f"\n\n## 2. {name} Structural Distance Stratification\n")
        report_lines.append("### (a) Recovery of gold tables missed by dense retrieval (Omitted Tables Recovery by Distance to Recalled Tables)\n")
        report_lines.append(pd.DataFrame(res["omitted_strat"]).to_markdown(index=False))
        report_lines.append("\n\n### (b) Gold-table recall gain by distance (Recall Gain by Distance to Nearest Seed)\n")
        report_lines.append(pd.DataFrame(res["all_strat"]).to_markdown(index=False))

    summary_file = os.path.join(project_root, "output/gert_recovery_analysis_summary.md")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nSummary report saved to {summary_file}")


if __name__ == "__main__":
    main()
