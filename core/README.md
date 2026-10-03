# GERT Core Modules

This directory contains the foundational algorithms, graph construction engines, and Personalized PageRank (PPR) routing logic for **GERT**.

---

## Modules

- **`schemakg/`**:
  - `schemakg_search_ppr.py`: Core routing engine combining dense embedding retrieval with Personalized PageRank graph propagation on foreign-key topology.
  - `schemakg_build.py`: Graph entity indexing and search structure builder.
- **`pipeline/`**:
  - `kg_construction.py`: Constructs the schema graph in Neo4j from relational database schema CSVs.
  - `retrieval.py`: Embedding retrieval interface for OpenAI-compatible and local embedding endpoints.
  - `local_embedding.py`: Local embedding model wrappers.
  - `common.py`: Shared Neo4j driver connection and index configuration constants.
- **`clear_neo4j.py`**:
  - Utility to purge existing graph nodes, foreign-key relationships, and vector indexes in Neo4j.
