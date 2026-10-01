# MAC-SQL with GERT on BirdUnion-100

This directory can be installed and run independently. The `mac_selector` condition lets the native MAC-SQL selector read the full BirdUnion union schema and choose 15 tables. The `gert_selector` condition supplies the frozen top-15 table list in `birdunion_data/GERT_t2sql.json` to the same downstream generation and correction pipeline. LLM requests specify the model and messages but leave sampling parameters at provider defaults.

## Data and setup

The 100 queries use nine SQLite databases for execution accuracy; those databases contain full data. The remaining databases retain their table, column, primary-key, and foreign-key structure for global selection and schema serialization. The selector does not receive the gold `db_id`. See `birdunion_data/DATA_MANIFEST.json`.

Use Python 3.10 or 3.11 from this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `MACSQL_API_KEY` in `.env`. The default model is `deepseek-v3.2` through a DashScope OpenAI-compatible endpoint. `README_UPSTREAM.md` and `requirements-upstream.txt` describe the upstream framework.

## Run and evaluate

```powershell
python experiment\run_birdunion.py validate
python experiment\run_birdunion.py run --conditions mac_selector gert_selector --output-dir new_runs\run1\outputs
python experiment\run_birdunion.py evaluate --conditions mac_selector gert_selector --output-dir new_runs\run1\outputs --result-dir new_runs\run1\results
```

Use separate `run2` and `run3` directories for further repetitions. JSONL supports resuming. The run summary includes prompt, completion, and total tokens across the downstream pipeline. Aggregate independent repetitions with:

```powershell
python experiment\run_birdunion.py aggregate --result-dirs new_runs\run1\results new_runs\run2\results new_runs\run3\results --conditions mac_selector gert_selector --output-dir new_runs\aggregate
```

The archived three-run paper metrics are in `../../runs_and_results/macsql_runs/macsql_selector_runs/`. The framework's `experiment_results/three_rounds/` retains results that are not duplicated there, including the first round's raw SQL outputs. The reported standard deviation is the sample standard deviation (`n-1`). Frozen GERT routing tokens are not counted as part of the MAC-SQL downstream pipeline; account for that distinction when comparing total system cost.
