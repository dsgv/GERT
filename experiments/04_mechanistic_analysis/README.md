# Mechanistic analysis

This suite covers gold-table recovery, query-level rescue and damage, foreign-key distance strata (Tables 6–8), and induced subgraph connectivity (Table 9). SpiderUnion, BirdUnion, and SynLink use the gold JSON and schema CSV listed in the [main benchmark](../01_main_benchmark/README.md). Compare dense and GERT predictions for the same queries, encoder, and top-15 budget.

## Recovery and foreign-key distance

Use archived dense (`w_o_PPR`) and GERT (`GERT_Full`) predictions to recompute the report:

```bash
python experiments/04_mechanistic_analysis/01_recovery_and_distance/evaluate_recovery_and_stratification.py
```

The script writes `01_recovery_and_distance/reports/RECOVERY_AND_STRATIFICATION_REPORT.md`; copy that report before rerunning if you need the previous version. To regenerate predictions, use the following dataset-specific commands after configuring Neo4j and the embedding endpoint. They may rebuild the graph and write prediction files:

```bash
python experiments/04_mechanistic_analysis/01_recovery_and_distance/run_gert_with_recovery_analysis.py --dataset spider --top_k 20 --eval_k 15
python experiments/04_mechanistic_analysis/01_recovery_and_distance/run_gert_with_recovery_analysis.py --dataset bird --top_k 20 --eval_k 15
python experiments/04_mechanistic_analysis/01_recovery_and_distance/run_gert_with_recovery_analysis.py --dataset synlink --top_k 20 --eval_k 15
```

Gold-table recovery divides the number of gold tables missed by dense top-15 but retrieved by GERT top-15 by all gold tables missed by dense top-15. Query rescue divides dense failures converted to complete recall by all dense failures; query damage divides dense successes converted to failures by all dense successes. Foreign-key distance is the shortest undirected path from a missed gold table to a gold table retrieved by dense search, grouped as 1, 2, 3+ hops, or no reachable anchor. Report each dataset separately.

## Connectivity

The connectivity inputs are `02_connectivity_experiment/data_inputs/{SpiderUnion,BirdUnion,SynLink}.json`, with foreign-key edges from each schema CSV:

```bash
python experiments/04_mechanistic_analysis/02_connectivity_experiment/run_connectivity_experiment.py --k 15 --seed-budget 20
```

This script writes predictions and tables under `02_connectivity_experiment/`. It reports the full dataset, and separately the subset with at least two gold tables. The subset is defined in advance because single-table queries have trivial connectivity; both populations remain available. Table recall, complete recall, precision, and F1 are macro-averaged over queries. Conn@15 requires complete gold-table coverage and a single connected component for the gold tables in the induced undirected foreign-key graph of the 15 selected tables. Keep the full and multi-table denominators separate. Inspect the generated tables rather than assuming one method is best in every metric.
