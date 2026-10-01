# BIRD Development Databases

This directory stores the SQLite databases from the official **BIRD Benchmark (dev set)** used for downstream Text-to-SQL evaluation. Due to GitHub's file size limits (>100 MB), raw `.sqlite` database files are excluded from this repository.

### Download Links

Please download `dev_databases.zip` from the official BIRD sources and extract the database folders directly into this directory (`data/dev_database/`):

* **Official Website**: [https://bird-bench.github.io/](https://bird-bench.github.io/)
* **Google Drive (Official)**: [BIRD Dev Databases](https://drive.google.com/file/d/13AM9AWv_V_m1Vvyda_GStnpkwTF9q65r/view?usp=sharing)
* **Hugging Face**: [https://huggingface.co/datasets/birdsql/bird](https://huggingface.co/datasets/birdsql/bird)

### Verification

Once extracted, verify that the databases are in place by running:

```bash
python experiments/03_downstream_text2sql/scripts/text2sql_experiment.py validate
```
