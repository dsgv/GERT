# GERT: Graph-Enhanced Retrieval of Tables over Complex Multi-Database Candidate Spaces

Code and experiment artifacts for GERT, a training-free table retrieval method for multi-database Text-to-SQL. GERT combines semantic retrieval with Personalized PageRank over schema foreign-key links.

## Environment

- Python 3.10+
- A running Neo4j server with vector index support
- An OpenAI-compatible embedding API (the supplied `.env.template` uses `BAAI/bge-m3` with 1024 dimensions)

Run all commands from the repository root. Create a virtual environment, activate it, and install the Python dependencies.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
if (-not (Test-Path -LiteralPath .env)) { Copy-Item -LiteralPath .env.template -Destination .env }
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
if [ ! -e .env ]; then cp .env.template .env; fi
```

Copy the template only when creating a new `.env`; preserve an existing configured file. Edit `.env` and set `NEO4J_BOLT_URL`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `Embed_API_KEY`, and `Embed_API_BASE` for your environment. Replace the placeholder credentials with working credentials and start Neo4j before running the benchmark.

Keep `OPENAI_EMBEDDING_MODEL` and `EMBEDDING_DIMENSIONS` consistent with the embedding endpoint. Model identifiers must match the provider's identifier, including capitalization. The template sets `EMBEDDING_MAX_BATCH_SIZE=10` to cap the number of texts in each embedding request; use a value supported by your provider. Retain the template's index and embedding-property names for the documented benchmark commands.

Full runs of all three datasets have been verified with Python 3.12.1 and the following installed package versions: `neo4j==6.2.0`, `neo4j-graphrag==1.19.0`, `openai==2.48.0`, `numpy==2.3.5`, `pandas==3.0.3`, and `python-dotenv==1.2.2`. These describe the tested environment; `requirements.txt` specifies minimum versions rather than a dependency lock. Installation into a fresh environment was not part of that verification.

Use a dedicated Neo4j database for reproduction. Whenever graph construction runs, it deletes all nodes and relationships in the configured database and attempts to drop its indexes before building the selected dataset's graph. The benchmark runner does not automatically back up or restore existing data. Embedding generation and query retrieval call the configured API and may incur provider charges.

## Run and evaluate GERT

The benchmark inputs are included under `data/`; no separate graph-building command is required. Run the datasets in Bird → Spider → SynLink order:

```bash
python experiments/01_main_benchmark/run_main_experiments.py --dataset bird --gert-only
python experiments/01_main_benchmark/run_main_experiments.py --dataset spider --gert-only
python experiments/01_main_benchmark/run_main_experiments.py --dataset synlink --gert-only
```

`--gert-only` runs GERT alone. Omitting it also runs the repository's GERT 1-hop and 2-hop baselines. Each GERT command writes `experiments/01_main_benchmark/main_experiment_test/<dataset>/gert.json`.

If a complete prediction file already exists, the runner skips its computation. To force a new computation and graph rebuild, append `--force` to the corresponding command, for example:

```bash
python experiments/01_main_benchmark/run_main_experiments.py --dataset bird --gert-only --force
```

This overwrites the selected dataset's existing predictions. An incomplete prediction file triggers a new run from the beginning; it is not resumed from its last saved query.

During graph construction, look for `[Verify] <table count> tables, all embedded, index ONLINE`. A successful run finishes with `[<dataset>] Requested prediction files complete (<query count> queries)` and exits without an error. Expected counts are:

| Dataset argument | Tables in the graph | Queries | Predictions per query |
| --- | ---: | ---: | ---: |
| `bird` | 597 | 1534 | 20 |
| `spider` | 876 | 658 | 20 |
| `synlink` | 2965 | 1000 | 20 |

Check the console for embedding failures or `[Error]` messages.

Evaluate the first 15 retrieved tables for each query:

```bash
python experiments/01_main_benchmark/scripts/calculate_overall_metrics_from_path.py experiments/01_main_benchmark/main_experiment_test/bird/gert.json --k 15 --method GERT
python experiments/01_main_benchmark/scripts/calculate_overall_metrics_from_path.py experiments/01_main_benchmark/main_experiment_test/spider/gert.json --k 15 --method GERT
python experiments/01_main_benchmark/scripts/calculate_overall_metrics_from_path.py experiments/01_main_benchmark/main_experiment_test/synlink/gert.json --k 15 --method GERT
```

Evaluation prints Table Recall, Table CR, precision, and F1 to the console. It writes a Markdown report only when `--output <path>` is supplied.

## Other experiments

- [Main benchmark](experiments/01_main_benchmark/README.md)
- [Ablation study](experiments/02_ablation_study/README.md)
- [Downstream Text-to-SQL](experiments/03_downstream_text2sql/README.md)
- [Mechanistic analysis](experiments/04_mechanistic_analysis/README.md)

To summarize the archived downstream results without rerunning SQL generation:

```bash
python experiments/03_downstream_text2sql/scripts/calculate_downstream_metrics.py
```

Rerunning downstream SQL execution additionally requires the Bird SQLite databases and model API credentials; see the [downstream instructions](experiments/03_downstream_text2sql/README.md).

## License

Original GERT code is licensed under the [Apache License, Version 2.0](LICENSE).

Third-party code retains its original licenses and copyright notices. Benchmark datasets and externally downloaded model weights remain subject to their respective providers' terms; the project license does not override those terms.
