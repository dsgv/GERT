# GERT: Graph-Enhanced Retrieval of Tables over Complex Multi-Database Candidate Spaces

Official code and experiment artifacts for **GERT (Graph-Enhanced Retrieval of Tables)**, a training-free table retrieval method for multi-database Text-to-SQL. GERT combines dense semantic retrieval with Personalized PageRank (PPR) over within-database foreign-key graphs to recover weakly signaled bridge tables without task-specific training or online LLM inference.

## Method Overview

Given an open candidate space without target database identifiers, GERT retrieves the top-$K$ ($K=15$) tables in four steps:

1. **Dense Semantic Retrieval**: Encodes the question and serialized table texts using BGE-M3 (1024-dim dense vectors) to retrieve top-$k_s$ ($k_s=20$) semantic seeds across the global pool.
2. **Rank-Decayed Personalization Vector**: Assigns exponential rank-decayed weights ($v_i \propto \exp(-\mu \cdot l_i)$, default $\mu=0.3$) based on seed ranks $l_i$ to prioritize high-confidence anchors while avoiding seed dilution.
3. **PPR Structural Propagation**: Diffuses relevance over the undirected within-database foreign-key graph using Personalized PageRank ($\alpha=0.85$, convergence threshold $10^{-6}$, max 100 iterations).
4. **Semantic-Structural Score Fusion**: Fuses normalized reciprocal-rank semantic scores with the stationary PPR distribution ($s_{\text{final}} = (1-\beta)\bar{s}_{\text{ret}} + \beta\bar{s}_{\text{ppr}}$, default $\beta=0.7$) to rank candidates.

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

Copy the template only when creating a new `.env`; preserve an existing configured file. Edit `.env` and set `NEO4J_BOLT_URL`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `Embed_API_KEY`, and `Embed_API_BASE` for your environment. Replace placeholder credentials with working credentials and start Neo4j before running the benchmark.

Keep `OPENAI_EMBEDDING_MODEL` and `EMBEDDING_DIMENSIONS` consistent with the embedding endpoint. Model identifiers must match the provider's identifier, including capitalization. The template sets `EMBEDDING_MAX_BATCH_SIZE=10` to cap the number of texts in each embedding request; use a value supported by your provider. Retain the template's index and embedding-property names for the documented benchmark commands.

Full runs of all three datasets have been verified with Python 3.12.1 and the following installed package versions: `neo4j==6.2.0`, `neo4j-graphrag==1.19.0`, `openai==2.48.0`, `numpy==2.3.5`, `pandas==3.0.3`, and `python-dotenv==1.2.2`. These describe the tested environment; `requirements.txt` specifies minimum versions rather than a dependency lock.

> **Caution**: Use a dedicated Neo4j database for reproduction. Whenever graph construction runs, it deletes all nodes and relationships in the configured database and attempts to drop its indexes before building the selected dataset's graph. The benchmark runner does not automatically back up or restore existing data. Embedding generation and query retrieval call the configured API and may incur provider charges.

## Run and Evaluate GERT

### 1. Benchmark Execution

The benchmark inputs are included under `data/`; no separate graph-building command is required. Run the datasets in Bird → Spider → SynLink order:

```bash
python experiments/01_main_benchmark/run_main_experiments.py --dataset bird --gert-only
python experiments/01_main_benchmark/run_main_experiments.py --dataset spider --gert-only
python experiments/01_main_benchmark/run_main_experiments.py --dataset synlink --gert-only
```

`--gert-only` runs GERT alone (recommended for fast reproduction). Omitting it also runs the repository's GERT 1-hop and 2-hop baselines for reproducing Table 2 comparisons. Each GERT command writes `experiments/01_main_benchmark/main_experiment_test/<dataset>/gert.json`.

If a complete prediction file already exists, the runner skips its computation. To force a new computation and graph rebuild, append `--force` to the corresponding command, for example:

```bash
python experiments/01_main_benchmark/run_main_experiments.py --dataset bird --gert-only --force
```

This overwrites the selected dataset's existing predictions. An incomplete prediction file triggers a new run from the beginning; it is not resumed from its last saved query.

During graph construction, look for `[Verify] <table count> tables, all embedded, index ONLINE`. A successful run finishes with `[<dataset>] Requested prediction files complete (<query count> queries)` and exits without an error. Expected counts are:

| Dataset argument | Tables in the graph | Queries | Predictions per query |
| --- | ---: | ---: | ---: |
| `bird` (BirdUnion) | 597 | 1534 | 20 |
| `spider` (SpiderUnion) | 876 | 658 | 20 |
| `synlink` (SynLink) | 2965 | 1000 | 20 |

Check the console for embedding failures or `[Error]` messages.

### 2. Evaluation ($K=15$)

Evaluate the first 15 retrieved tables for each query:

```bash
python experiments/01_main_benchmark/scripts/calculate_overall_metrics_from_path.py experiments/01_main_benchmark/main_experiment_test/bird/gert.json --k 15 --method GERT
python experiments/01_main_benchmark/scripts/calculate_overall_metrics_from_path.py experiments/01_main_benchmark/main_experiment_test/spider/gert.json --k 15 --method GERT
python experiments/01_main_benchmark/scripts/calculate_overall_metrics_from_path.py experiments/01_main_benchmark/main_experiment_test/synlink/gert.json --k 15 --method GERT
```

Evaluation prints Table Recall, Table CR (Complete Recall), Precision, and F1 to the console. It writes a Markdown report only when `--output <path>` is supplied.

### 3. Main Results (Table 2 in Paper, $K=15$)

| Dataset | Method | Table Rec. (%) | Table CR (%) | Precision (%) | F1-Score (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **SpiderUnion** | GERT | 95.24 | 94.98 | 9.44 | 16.92 |
| **BirdUnion** | **GERT** | **92.85** | **90.55** | **11.89** | **20.84** |
| **SynLink** | **GERT** | **89.43** | **78.80** | **18.16** | **29.77** |

*Table CR measures the proportion of queries where all gold tables are completely retrieved within the top-15 budget. On the multi-table-intensive benchmarks BirdUnion and SynLink, GERT outperforms all baselines (including LLM-based iterative and selection approaches) by 3.02% and 4.23% in Table CR, respectively.*

## Other Experiments

Each experiment module corresponds directly to the paper's analyses:

- [**Main Benchmark**](experiments/01_main_benchmark/README.md) (Paper §4.2, Table 2 & Figure 2): Main comparative evaluation and breakdown by gold table count.
- [**Ablation Study**](experiments/02_ablation_study/README.md) (Paper §4.3 & §4.5, Tables 3–4): Component ablations (`w/o Graph Propagation`, `w/o Semantic Score`, `w/o Seed Decay`) and encoder robustness across BGE-M3 and SGPT-1.3B.
- [**Downstream Text-to-SQL**](experiments/03_downstream_text2sql/README.md) (Paper §4.4, Table 5): Evaluates end-to-end SQL execution accuracy (EX) and token reduction across Zero-shot, MAC-SQL, and DIN-SQL pipelines on multi-table queries from BirdUnion.
  - To summarize archived downstream metrics without rerun:
    ```bash
    python experiments/03_downstream_text2sql/scripts/calculate_downstream_metrics.py
    ```
  - Rerunning downstream SQL execution additionally requires the Bird SQLite databases and model API credentials; see the [downstream instructions](experiments/03_downstream_text2sql/README.md).
- [**Mechanistic Analysis**](experiments/04_mechanistic_analysis/README.md) (Paper §4.6, Tables 6–9): Gold table recovery, foreign-key distance decay, and candidate schema connectivity.

## License

Original GERT code is licensed under the [Apache License, Version 2.0](LICENSE).

Third-party code retains its original licenses and copyright notices. Benchmark datasets and externally downloaded model weights remain subject to their respective providers' terms; the project license does not override those terms.
