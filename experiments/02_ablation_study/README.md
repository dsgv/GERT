# Ablation study

This suite evaluates GERT component variants and encoder robustness on SpiderUnion, BirdUnion, and SynLink. Gold JSON and schema CSV paths are listed in the [main benchmark](../01_main_benchmark/README.md). All table metrics use the first 15 predicted tables and are macro-averaged over queries.

## Component ablation

Archived predictions are under `01_component_ablation/archive_results/`. Recompute their table recall, complete recall, precision, and F1 without inference:

```bash
python experiments/02_ablation_study/01_component_ablation/evaluate_component_ablation.py
```

Generate new predictions for each dataset using the same graph and embedding configuration as the main benchmark:

```bash
python experiments/02_ablation_study/01_component_ablation/run_component_ablation.py --dataset spider
python experiments/02_ablation_study/01_component_ablation/run_component_ablation.py --dataset bird
python experiments/02_ablation_study/01_component_ablation/run_component_ablation.py --dataset synlink
```

New outputs go to `01_component_ablation/predictions_new/<dataset>/`; the script refuses to overwrite existing files unless `--force` is supplied. Variants are full GERT, without PPR, without semantic scoring, and without seed decay. Evaluate a new prediction JSON with `experiments/01_main_benchmark/scripts/calculate_overall_metrics_from_path.py <file> --k 15`.

## Encoder robustness

Run SGPT encoder comparisons with a configured local model, Neo4j, and embedding endpoint:

```bash
python experiments/02_ablation_study/02_encoder_robustness/run_encoder_ablation.py --dataset spider
python experiments/02_ablation_study/02_encoder_robustness/run_encoder_ablation.py --dataset bird
python experiments/02_ablation_study/02_encoder_robustness/run_encoder_ablation.py --dataset synlink
```

The script writes new predictions under `02_encoder_robustness/predictions_new/` and metrics under `02_encoder_robustness/results_new/`. The existing `predictions/` and `results/` contain archived outputs. Compare encoders using the same query sets, retrieval budget, and evaluation definitions.
