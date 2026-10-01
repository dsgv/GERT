# Downstream Text-to-SQL

This suite evaluates SQL generation on the fixed 100-query BirdUnion subset represented by the frozen files in `data/prepared_prompts/`. Each record includes the question, gold SQL, gold tables, selected tables, schema text, and final prompt. The `gert.json` file is also the shared 100-query input for the MAC-SQL and DIN-SQL adapters. Databases are under `data/dev_database/` and the full union schema is `data/bird_union_schema_FK.txt`. The subset contains 85 three-table and 15 four-table queries; 227 source queries have three or four gold tables. `data/prepared_prompts/sample_manifest.json` records the published IDs and checksum. All conditions use the same question IDs and gold SQL.

The frozen prompts and three-run zero-shot, MAC-SQL, and DIN-SQL metrics are included under `data/prepared_prompts/` and `runs_and_results/`. Recompute the paper's downstream table from those archived metrics with:

```bash
python experiments/03_downstream_text2sql/scripts/calculate_downstream_metrics.py
```

That script reads each of the three run reports, checks that each condition has 100 queries, and prints mean and sample standard deviation. The DIN-SQL native-linker and GERT rows use separate archived three-run series, as specified in the script. It does not make API calls. The Bird SQLite databases are distributed separately; see `data/README.md` before rerunning SQL execution.

The compared pipelines are zero-shot generation with GERT candidate schemas, MAC-SQL with its native selector or GERT, and DIN-SQL with its native linker or GERT. Configure model API credentials before running. Use a distinct output directory for each repetition.

## GERT zero-shot generation

Copy `configs/experiment.yaml` and set `paths.output_dir` and `paths.result_dir` to a new run directory. Keep `paths.prepared_dir` pointing at the frozen prompts unless you regenerate them. Replace `<config.yaml>` below with that copy:

```bash
python experiments/03_downstream_text2sql/scripts/text2sql_experiment.py --config <config.yaml> validate
python experiments/03_downstream_text2sql/scripts/text2sql_experiment.py --config <config.yaml> run --conditions gert
python experiments/03_downstream_text2sql/scripts/text2sql_experiment.py --config <config.yaml> evaluate --conditions gert
```

For a direct full-schema comparison, run and evaluate with `--conditions full gert` in the same repetition. The optional `prepare` command can regenerate frozen prompts into a separate `paths.prepared_dir` from the `*_input` paths, which point to the checked-in prepared files; it requires the Bird SQLite databases. The paper does not use the optional gold-table oracle condition, so no `gold.json` is included.

## MAC-SQL and DIN-SQL

```bash
python experiments/03_downstream_text2sql/scripts/macsql_birdunion_experiment.py --config experiments/03_downstream_text2sql/configs/macsql_birdunion.yaml validate
python experiments/03_downstream_text2sql/scripts/macsql_birdunion_experiment.py --config experiments/03_downstream_text2sql/configs/macsql_birdunion.yaml run --conditions mac_selector gert_selector --output-dir experiments/03_downstream_text2sql/runs_and_results/new_mac_round/outputs
python experiments/03_downstream_text2sql/scripts/macsql_birdunion_experiment.py --config experiments/03_downstream_text2sql/configs/macsql_birdunion.yaml evaluate --conditions mac_selector gert_selector --output-dir experiments/03_downstream_text2sql/runs_and_results/new_mac_round/outputs --result-dir experiments/03_downstream_text2sql/runs_and_results/new_mac_round/results
python experiments/03_downstream_text2sql/scripts/dinsql_birdunion_experiment.py --config experiments/03_downstream_text2sql/configs/dinsql_birdunion.yaml validate
python experiments/03_downstream_text2sql/scripts/dinsql_birdunion_experiment.py --config experiments/03_downstream_text2sql/configs/dinsql_birdunion.yaml run --conditions din_linker gert_selector --output-dir experiments/03_downstream_text2sql/runs_and_results/new_din_round/outputs
python experiments/03_downstream_text2sql/scripts/dinsql_birdunion_experiment.py --config experiments/03_downstream_text2sql/configs/dinsql_birdunion.yaml evaluate --conditions din_linker gert_selector --output-dir experiments/03_downstream_text2sql/runs_and_results/new_din_round/outputs --result-dir experiments/03_downstream_text2sql/runs_and_results/new_din_round/results
```

## Metrics

Execution accuracy (EX) compares the execution results of generated SQL and gold SQL in the recorded Bird SQLite database. Parse errors, execution errors, and timeouts count as incorrect. For DIN-SQL, distinguish the Bird EX measure used in the paper from any stricter comparison also reported by the script. Context reduction in the archived reports is computed from estimated schema tokens relative to the full-schema reference; the native full-schema selectors are reported as zero reduction because their selector stage reads the full union schema. Tokens/query includes prompt and completion tokens across the full MAC-SQL or DIN-SQL pipeline, including native selection/linking when applicable. For repeated runs, report the mean and sample standard deviation of independent repetitions. `scripts/calculate_downstream_metrics.py` recomputes the table from the archived per-run metric reports; rerun the individual evaluation scripts to recompute those reports from raw SQL outputs.
