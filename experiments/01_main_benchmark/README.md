# Main benchmark

This suite evaluates table-level schema linking on SpiderUnion, BirdUnion, and SynLink at `K=15`.
- **Main Experiment (Table 2)**: Evaluates Table Recall, Complete Recall (Table CR), Precision, and F1.
- **Multi-Table Experiment (Figure 2)**: Groups Table CR by the number of gold tables (1, 2, 3, and 4+).

## Data & Results Structure

- `paper_baseline_result/`: Contains paper baseline predictions across SpiderUnion, BirdUnion, and SynLink (`BM25`, `CORE-T`, `CRED-SQL`, `CRUSH4SQL`, `GERT`, `GESR_1hop`, `GESR_2hop`, `JAR`, `LinkAlign`, `MURRE`, `sgpt`).
- `main_experiment_test/`: Output directory when running new experiments via `run_main_experiments.py`.

## Running GERT

Run from the repository root:

| Dataset | Gold JSON | Schema CSV | Command |
| --- | --- | --- | --- |
| SpiderUnion | `data/spider/murre_spider_dev.json` | `data/spider/spider_union_schema_FK.csv` | `python experiments/01_main_benchmark/run_main_experiments.py --dataset spider --gert-only` |
| BirdUnion | `data/bird/murre_bird_dev.json` | `data/bird/bird_union_schema_FK.csv` | `python experiments/01_main_benchmark/run_main_experiments.py --dataset bird --gert-only` |
| SynLink | `data/SynLink/Formal_moderate_1k_gd.json` | `data/SynLink/SynSQL_schema_csv_300.csv` | `python experiments/01_main_benchmark/run_main_experiments.py --dataset synlink --gert-only` |

Complete prediction files are reused; append `--force` to rebuild the graph and overwrite predictions.

## Evaluation Commands

Both evaluation scripts support passing either an entire directory (automatically matches and evaluates SpiderUnion, BirdUnion, and SynLink) or a single prediction JSON.

### 1. Main Experiment Metrics (Table 2: Recall, CR, Precision, F1)

Evaluate all 3 datasets for a method in a single command:
```bash
# Evaluate GERT archived results
python experiments/01_main_benchmark/scripts/calculate_overall_metrics_from_path.py experiments/01_main_benchmark/paper_baseline_result/GERT --k 15

# Or evaluate any baseline (e.g., CRED-SQL, CRUSH4SQL)
python experiments/01_main_benchmark/scripts/calculate_overall_metrics_from_path.py experiments/01_main_benchmark/paper_baseline_result/CRED-SQL --k 15
```

Or evaluate newly generated prediction files individually:
```bash
python experiments/01_main_benchmark/scripts/calculate_overall_metrics_from_path.py experiments/01_main_benchmark/main_experiment_test/spider/gert.json --k 15 --method GERT
```

### 2. Multi-Table Grouped Experiment (Figure 2: Table CR by 1, 2, 3, 4+ Gold Tables)

Evaluate complete recall stratified by gold table count:
```bash
# Evaluate GERT across 1, 2, 3, and 4+ gold tables
python experiments/01_main_benchmark/scripts/calculate_grouped_cr_from_path.py experiments/01_main_benchmark/paper_baseline_result/GERT --k 15

# Or evaluate newly generated predictions
python experiments/01_main_benchmark/scripts/calculate_grouped_cr_from_path.py experiments/01_main_benchmark/main_experiment_test/spider/gert.json --k 15 --method GERT
```

> **Tip**: Append `--output <filename>.md` to any command to export the markdown table directly to a file.
