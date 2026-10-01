# Main benchmark

This suite evaluates table-level schema linking on SpiderUnion, BirdUnion, and SynLink at `K=15`. The paper's Table 2 reports table recall, complete recall (Table CR), precision, and F1. Figure 2 groups complete recall by the number of gold tables.

## Inputs and GERT runs

Run from the repository root. Each command uses the indicated gold JSON and schema CSV and writes predictions under `main_experiment_test/<dataset>/`. Configure Neo4j and the embedding API using the [root README](../../README.md). Use a dedicated Neo4j database: graph construction deletes existing nodes, relationships, and removable indexes. Complete prediction files are reused; append `--force` to rebuild the graph and overwrite predictions. The documented commands use API embeddings and do not require local model paths.

| Dataset | Gold JSON | Schema CSV | Command |
| --- | --- | --- | --- |
| SpiderUnion | `data/spider/murre_spider_dev.json` | `data/spider/spider_union_schema_FK.csv` | `python experiments/01_main_benchmark/run_main_experiments.py --dataset spider --gert-only` |
| BirdUnion | `data/bird/murre_bird_dev.json` | `data/bird/bird_union_schema_FK.csv` | `python experiments/01_main_benchmark/run_main_experiments.py --dataset bird --gert-only` |
| SynLink | `data/SynLink/Formal_moderate_1k_gd.json` | `data/SynLink/SynSQL_schema_csv_300.csv` | `python experiments/01_main_benchmark/run_main_experiments.py --dataset synlink --gert-only` |

## Evaluation

Evaluate each prediction file separately:

```bash
python experiments/01_main_benchmark/scripts/calculate_overall_metrics_from_path.py experiments/01_main_benchmark/main_experiment_test/spider/gert.json --k 15 --method GERT
python experiments/01_main_benchmark/scripts/calculate_overall_metrics_from_path.py experiments/01_main_benchmark/main_experiment_test/bird/gert.json --k 15 --method GERT
python experiments/01_main_benchmark/scripts/calculate_overall_metrics_from_path.py experiments/01_main_benchmark/main_experiment_test/synlink/gert.json --k 15 --method GERT
```

For each query, compare the first 15 predicted tables with its gold table set. Table recall is the fraction of gold tables retrieved; Table CR is 1 only when all gold tables are retrieved; precision is the fraction of retrieved tables that are gold; F1 is the harmonic mean of query-level precision and recall. Report the mean over queries. For complexity strata, compute Table CR separately for 1, 2, 3, and 4+ gold tables. The archived results are retained separately from new predictions.
