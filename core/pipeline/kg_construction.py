import datetime
import yaml
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from neo4j import GraphDatabase
from typing import List, Optional

# Import config and client from common module
from common import (
    driver, Embed_client,
    OPENAI_EMBEDDING_MODEL, EMBEDDING_DIMENSIONS,
    EMBEDDING_MAX_BATCH_SIZE,
    TABLE_VECTOR_INDEX_NAME,
    TABLE_EMBEDDING_PROPERTY_NAME
)

# Import LLM client for concept generation
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)




from local_embedding import load_sentence_transformer, embed_batch_sentence_transformer


# Global token counter, accumulated by _embed_batch_openai
_embed_token_counter = {"total_tokens": 0, "total_requests": 0}


def _embed_batch_openai(batch_texts: List[str], embedding_model: str) -> List[List[float]]:
    global _embed_token_counter
    response = Embed_client.embeddings.create(
        input=batch_texts,
        model=embedding_model
    )
    _embed_token_counter["total_tokens"] += response.usage.total_tokens
    _embed_token_counter["total_requests"] += 1
    return [data.embedding for data in response.data]


def get_embed_token_usage() -> dict:
    """Get cumulative embedding token usage during KG construction"""
    return dict(_embed_token_counter)


def reset_embed_token_usage():
    """Reset KG construction token counters"""
    global _embed_token_counter
    _embed_token_counter = {"total_tokens": 0, "total_requests": 0}


def make_serializable(obj):
    """
    Utility that handles unserializable types (e.g. numpy int64 etc.).
    Converts them to Python native types safe for Neo4j queries.
    """
    if isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, np.generic):  # Handle NumPy scalar types
        return obj.item()
    elif isinstance(obj, (pd.Timestamp, datetime.datetime)):  # Handle datetime and pd.Timestamp
        return obj.isoformat()
    elif isinstance(obj, datetime.time):  # Handle time objects
        return obj.isoformat()  # Convert to ISO 8601 string
    else:
        return obj


class DatabaseSchemaToGraph:
    """
    Schema Knowledge Graph builder
    """

    def __init__(self, concepts_filepath: str = None):
        self.driver = driver
        (
            self.column_concept_mapping,
            self.concept_definitions,
            self.concept_relationships
        ) = self.load_concept_mapping(concepts_filepath) if concepts_filepath else ({}, {}, [])

    def clear_graph(self, drop_indexes: bool = True):
        """
        Delete nodes and relationships from the graph.

        Args:
            drop_indexes: Whether to also drop vector indexes, defaults to True
        """
        with self.driver.session() as session:
            # Drop indexes
            if drop_indexes:
                for idx_name in [TABLE_VECTOR_INDEX_NAME]:
                    try:
                        session.run(f"DROP INDEX {idx_name} IF EXISTS")
                        print(f"  [Clear] Dropped index: {idx_name}")
                    except Exception as e:
                        print(f"  [Clear] Warning: Could not drop index {idx_name}: {e}")

            # Delete all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")
            print("  [Clear] All nodes and relationships deleted.")

    @staticmethod
    def load_concept_mapping(concept_mapping_file):
        """Load concept mapping, concept definitions, and concept relationships from YAML file."""
        with open(concept_mapping_file, 'r', encoding='utf-8') as file:
            mapping = yaml.safe_load(file)
        concept_mapping = mapping.get('column_concept_mapping', {})
        concept_definitions = mapping.get('concepts', {})
        concept_relationships = mapping.get('concept_relationships', [])
        return concept_mapping, concept_definitions, concept_relationships

    def ingest_schema(self, df, clear_graph: bool = False):
        """
        Orchestrate schema ingestion into Neo4j.

        Args:
            df: schema DataFrame
            clear_graph: Whether to clear the graph before ingestion (default False, decided by caller)

        Process:
          1) Clear graph (optional)
          2) Create/merge table nodes
          3) Create/merge column nodes and link them to tables
        """
        if clear_graph:
            self.clear_graph()
        self.create_table_nodes(df)
        self.create_column_nodes(df)

    def create_table_nodes(self, df):
        """
        Create/merge Table nodes.
        """
        with self.driver.session() as session:
            groups = list(df.groupby(['db_name', 'table_name']))
            for (db_name, table_name), group in tqdm(groups, desc="Creating Table Nodes", unit="table"):
                try:
                    # Extract table-level properties
                    unique_table_name = f"{db_name}.{table_name}"
                    column_names = group['column_name'].fillna('').astype(str).tolist()
                    description = f"Table {unique_table_name} contains columns: {', '.join(column_names)}"
                    cluster = "General"
                    title = table_name
                    remarks = ""
                    source_filename = ""

                    # Identify primary key columns
                    primary_keys = (
                        group.loc[group['is_primary_key'].str.lower() == 'yes', 'column_name']
                        .tolist()
                    )

                    # Create Table node
                    session.run(
                        """
                        MERGE (t:Table {name: $name})
                        SET t.title = $title,
                            t.description = $description,
                            t.cluster = $cluster,
                            t.remarks = $remarks,
                            t.source_filename = $source_filename,
                            t.primary_keys = $primary_keys
                        """,
                        name=unique_table_name,
                        title=title,
                        description=description,
                        cluster=cluster,
                        remarks=remarks,
                        source_filename=source_filename,
                        primary_keys=primary_keys
                    )
                except Exception as e:
                    print(f"Error creating table node {db_name}.{table_name}: {e}")

    def create_column_nodes(self, df):
        """
        Create/merge Column nodes for each table, linked via CONTAINS relationship:
            (Table)-[:CONTAINS]->(Column)
        """
        with self.driver.session() as session:
            groups = list(df.groupby(['db_name', 'table_name']))
            for (db_name, table_name), group in tqdm(groups, desc="Creating Column Nodes", unit="table"):
                try:
                    unique_table_name = f"{db_name}.{table_name}"
                    cols_to_keep = [
                        'column_name', 'data_type', 'is_primary_key', 'foreign_key_ref'
                    ]
                    for _, row in group.iterrows():
                        column_name = row['column_name']

                        # Skip if column_name is NaN or empty
                        if pd.isna(column_name) or str(column_name).strip() == "":
                            continue

                        column_name = str(column_name)

                        # Collect related column properties
                        column_properties = row[cols_to_keep].dropna().to_dict()
                        column_properties['description'] = column_name
                        column_properties['column_sample_value'] = ""
                        column_properties = make_serializable(column_properties)

                        # Create Column node
                        session.run(
                            """
                            MATCH (t:Table {name: $table_name})
                            MERGE (c:Column {name: $column_name, table_name: $table_name})
                            SET c += $properties
                            MERGE (t)-[:CONTAINS]->(c)
                            """,
                            table_name=unique_table_name,
                            column_name=column_name,
                            properties=column_properties
                        )

                        # Create foreign key relationship
                        if 'foreign_key_ref' in column_properties and column_properties['foreign_key_ref']:
                            try:
                                ref_table, ref_col = column_properties['foreign_key_ref'].split('.')
                                ref_table_full = f"{db_name}.{ref_table}"
                                session.run(
                                    """
                                    MATCH (c:Column {name: $column_name, table_name: $table_name})
                                    MERGE (ref_t:Table {name: $ref_table})
                                    MERGE (ref_c:Column {name: $ref_col, table_name: $ref_table})
                                    MERGE (ref_t)-[:CONTAINS]->(ref_c)
                                    MERGE (c)-[:REFERENCES]->(ref_c)
                                    """,
                                    column_name=column_name,
                                    table_name=unique_table_name,
                                    ref_table=ref_table_full,
                                    ref_col=ref_col
                                )
                            except ValueError:
                                pass
                except Exception as e:
                    print(f"Error creating column nodes for table {db_name}.{table_name}: {e}")

    def generate_table_node_embeddings(self,
                                       embedding_model=None,
                                       properties=None,
                                       embedding_property=None,
                                       batch_size: int = 32,
                                       local_st_model=None):
        """
        Batch-generate embeddings for each Table node and write to the node.

        Args:
            embedding_model: OpenAI API embedding model name (only effective when using API)
            local_st_model: Optional, pre-loaded ``sentence_transformers.SentenceTransformer``;
                If provided, use local inference instead of calling the Embed API.
        """
        if embedding_model is None:
            embedding_model = OPENAI_EMBEDDING_MODEL
        if properties is None:
            properties = ["title", "description", "remarks", "concept"]
        if embedding_property is None:
            embedding_property = TABLE_EMBEDDING_PROPERTY_NAME

        effective_batch_size = batch_size
        if local_st_model is None and batch_size > EMBEDDING_MAX_BATCH_SIZE:
            effective_batch_size = EMBEDDING_MAX_BATCH_SIZE
            print(
                f"[Embedding] batch_size={batch_size} exceeds provider limit; "
                f"clamped to {effective_batch_size} (EMBEDDING_MAX_BATCH_SIZE)."
            )

        with self.driver.session() as session:
            query = f"""
            MATCH (t:Table)
            RETURN elementId(t) AS node_id, {", ".join([f"t.{p} AS {p}" for p in properties])}
            """
            results = list(session.run(query))

            if not results:
                print("No Table nodes found for embedding.")
                return

            # Prepare all texts to embed and their corresponding node IDs
            node_ids = []
            texts_to_embed = []

            for record in results:
                node_id = record["node_id"]
                # Concatenate relevant text fields (filter out None or empty)
                texts = [str(record[prop]) if record[prop] is not None else "" for prop in properties]
                text_to_embed = " ".join(texts).strip()

                if text_to_embed.strip():
                    node_ids.append(node_id)
                    texts_to_embed.append(text_to_embed)

            if not texts_to_embed:
                print("No valid text content found for embedding.")
                return

            print(f"Generating embeddings for {len(texts_to_embed)} tables in batches of {effective_batch_size}...")

            # Batch generate embeddings
            total_batches = (len(texts_to_embed) + effective_batch_size - 1) // effective_batch_size
            for i in tqdm(range(0, len(texts_to_embed), effective_batch_size),
                          desc="Batch Embedding",
                          total=total_batches,
                          unit="batch"):
                batch_texts = texts_to_embed[i:i + effective_batch_size]
                batch_ids = node_ids[i:i + effective_batch_size]

                try:
                    if local_st_model is not None:
                        vecs = embed_batch_sentence_transformer(local_st_model, batch_texts)
                    else:
                        vecs = _embed_batch_openai(batch_texts, embedding_model)

                    embeddings_data = [
                        {"node_id": batch_ids[j], "embedding": vecs[j]}
                        for j in range(len(batch_ids))
                    ]

                    update_query = f"""
                    UNWIND $embeddings_data AS item
                    MATCH (t:Table)
                    WHERE elementId(t) = item.node_id
                    SET t.{embedding_property} = item.embedding
                    """
                    session.run(update_query, embeddings_data=embeddings_data)

                except Exception as e:
                    print(f"Error in batch {i // effective_batch_size + 1}: {e}")
                    for j, (node_id, text) in enumerate(zip(batch_ids, batch_texts)):
                        try:
                            if local_st_model is not None:
                                embedding_vector = embed_batch_sentence_transformer(local_st_model, [text])[0]
                            else:
                                embedding_vector = _embed_batch_openai([text], embedding_model)[0]
                            update_query = f"""
                            MATCH (t:Table)
                            WHERE elementId(t) = $node_id
                            SET t.{embedding_property} = $embedding
                            """
                            session.run(update_query, node_id=node_id, embedding=embedding_vector)
                        except Exception as inner_e:
                            print(f"Error embedding node {node_id}: {inner_e}")

            print(f"Embeddings stored in node property `{embedding_property}`.")

    def generate_column_node_embeddings(self,
                                        embedding_model=None,
                                        embedding_property: str = "embedding",
                                        batch_size: int = 32,
                                        local_st_model=None):
        """
        Batch-generate embeddings for each Column node and write to the node.

        Args:
            embedding_model: OpenAI API embedding model name (only effective when using API)
            local_st_model: Optional, local ``SentenceTransformer``; If provided, skip API calls.
        """
        if embedding_model is None:
            embedding_model = OPENAI_EMBEDDING_MODEL

        effective_batch_size = batch_size
        if local_st_model is None and batch_size > EMBEDDING_MAX_BATCH_SIZE:
            effective_batch_size = EMBEDDING_MAX_BATCH_SIZE
            print(
                f"[Embedding] batch_size={batch_size} exceeds provider limit; "
                f"clamped to {effective_batch_size} (EMBEDDING_MAX_BATCH_SIZE)."
            )

        with self.driver.session() as session:
            query = """
            MATCH (c:Column)
            RETURN elementId(c) AS node_id,
                   c.name AS column_name,
                   c.table_name AS table_name,
                   c.description AS description,
                   c.data_type AS data_type,
                   c.column_sample_value AS column_sample_value
            """
            results = list(session.run(query))

            if not results:
                print("No Column nodes found for embedding.")
                return

            # Prepare all texts to embed and their corresponding node IDs
            node_ids = []
            texts_to_embed = []

            for record in results:
                node_id = record["node_id"]
                column_name = record["column_name"] or ""
                table_name = record["table_name"] or ""
                description = record["description"] or ""
                data_type = record["data_type"] or ""
                sample_value = record["column_sample_value"] or ""

                # Build text for embedding (includes column name, table name, description, data type, and sample value)
                text_parts = []
                if column_name:
                    text_parts.append(f"Column: {column_name}")
                if table_name:
                    text_parts.append(f"Table: {table_name}")
                if description and description != column_name:
                    text_parts.append(f"Description: {description}")
                if data_type:
                    text_parts.append(f"Type: {data_type}")
                if sample_value:
                    text_parts.append(f"Sample: {sample_value}")

                text_to_embed = ". ".join(text_parts).strip()

                if text_to_embed:
                    node_ids.append(node_id)
                    texts_to_embed.append(text_to_embed)

            if not texts_to_embed:
                print("No valid text content found for column embedding.")
                return

            print(f"Generating embeddings for {len(texts_to_embed)} columns in batches of {effective_batch_size}...")

            # Batch generate embeddings
            total_batches = (len(texts_to_embed) + effective_batch_size - 1) // effective_batch_size
            for i in tqdm(range(0, len(texts_to_embed), effective_batch_size),
                          desc="Batch Column Embedding",
                          total=total_batches,
                          unit="batch"):
                batch_texts = texts_to_embed[i:i + effective_batch_size]
                batch_ids = node_ids[i:i + effective_batch_size]

                try:
                    if local_st_model is not None:
                        vecs = embed_batch_sentence_transformer(local_st_model, batch_texts)
                    else:
                        vecs = _embed_batch_openai(batch_texts, embedding_model)

                    embeddings_data = [
                        {"node_id": batch_ids[j], "embedding": vecs[j]}
                        for j in range(len(batch_ids))
                    ]

                    update_query = f"""
                    UNWIND $embeddings_data AS item
                    MATCH (c:Column)
                    WHERE elementId(c) = item.node_id
                    SET c.{embedding_property} = item.embedding
                    """
                    session.run(update_query, embeddings_data=embeddings_data)

                except Exception as e:
                    print(f"Error in batch {i // effective_batch_size + 1}: {e}")
                    for j, (node_id, text) in enumerate(zip(batch_ids, batch_texts)):
                        try:
                            if local_st_model is not None:
                                embedding_vector = embed_batch_sentence_transformer(local_st_model, [text])[0]
                            else:
                                embedding_vector = _embed_batch_openai([text], embedding_model)[0]
                            update_query = f"""
                            MATCH (c:Column)
                            WHERE elementId(c) = $node_id
                            SET c.{embedding_property} = $embedding
                            """
                            session.run(update_query, node_id=node_id, embedding=embedding_vector)
                        except Exception as inner_e:
                            print(f"Error embedding column node {node_id}: {inner_e}")

            print(f"Column embeddings stored in node property `{embedding_property}`.")

    def create_vector_index(self,
                            index_name=None,
                            embedding_property=None,
                            dimensions=None):
        """
        Create Neo4j vector index for fast embedding-based similarity search.
        """
        if index_name is None:
            index_name = TABLE_VECTOR_INDEX_NAME
        if embedding_property is None:
            embedding_property = TABLE_EMBEDDING_PROPERTY_NAME
        if dimensions is None:
            dimensions = EMBEDDING_DIMENSIONS

        query = f"""
        CREATE VECTOR INDEX {index_name}
        FOR (t:Table)
        ON (t.{embedding_property})
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {dimensions},
                `vector.similarity_function`: "cosine"
            }}
        }}
        """
        with self.driver.session() as session:
            session.run(f"DROP INDEX {index_name} IF EXISTS")
            session.run(query)
            print(f"Vector index `{index_name}` created for Table on `{embedding_property}` with {dimensions} dimensions.")

    def create_column_vector_index(self,
                                   index_name=None,
                                   embedding_property: str = "embedding",
                                   dimensions=None):
        """
        Create Neo4j vector index for Column nodes.
        """
        if index_name is None:
            index_name = "column_vector_index"
        if dimensions is None:
            dimensions = EMBEDDING_DIMENSIONS

        query = f"""
        CREATE VECTOR INDEX {index_name}
        FOR (c:Column)
        ON (c.{embedding_property})
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {dimensions},
                `vector.similarity_function`: "cosine"
            }}
        }}
        """
        with self.driver.session() as session:
            session.run(f"DROP INDEX {index_name} IF EXISTS")
            session.run(query)
            print(f"Column vector index `{index_name}` created for Column on `{embedding_property}` with {dimensions} dimensions.")

    def run_cypher_query(self, query):
        with self.driver.session() as session:
            result = session.run(query)
            return [record for record in result]


def build_knowledge_graph(schema_path: str,
                          clear_before_build: bool = False,
                          local_embedding_model_path: Optional[str] = None):
    """
    Build knowledge graph from CSV schema file.

    Args:
        schema_path: Path to CSV schema file
        clear_before_build: Whether to clear the entire database before building (default False)
        local_embedding_model_path: If provided, load ``sentence_transformers`` model from this directory for local embedding,
            without calling the Embed API; vector index dimensions match the model output.

    Process:
        1. Clear existing database (optional) - if clear_before_build=True
        2. Create Table and Column nodes
        3. Generate embedding vectors
        4. Create vector index and fulltext index
    """
    # 1. Clear database (optional)
    if clear_before_build:
        print("\n" + "=" * 50)
        print("[KG Build] Step 1: Clearing existing database...")
        print("=" * 50)
        clear_neo4j_database(drop_indexes=True, verify=True)

    # 2. Read schema
    print("\n" + "=" * 50)
    print("[KG Build] Step 2: Loading schema...")
    print("=" * 50)
    df_schema = pd.read_csv(schema_path)
    df_db_details = df_schema
    print(f"[Schema] Loaded {len(df_db_details)} rows from {schema_path}")

    local_st_model = None
    embedding_dims = EMBEDDING_DIMENSIONS
    if local_embedding_model_path:
        local_st_model = load_sentence_transformer(local_embedding_model_path)
        embedding_dims = int(local_st_model.get_sentence_embedding_dimension())
        print(
            f"[KG Build] Local embedding: {os.path.abspath(local_embedding_model_path)} "
            f"(dimensions={embedding_dims})"
        )
    else:
        print(
            f"[KG Build] Embedding API: model={OPENAI_EMBEDDING_MODEL}, "
            f"index dimensions={embedding_dims}"
        )

    neo4j_kg = DatabaseSchemaToGraph()

    # 3. Create nodes (skip clear_graph since already cleared)
    print("\n" + "=" * 50)
    print("[KG Build] Step 3: Creating table and column nodes...")
    print("=" * 50)
    neo4j_kg.create_table_nodes(df_db_details)
    neo4j_kg.create_column_nodes(df_db_details)

    # 3. Generate embeddings
    print("\n" + "=" * 50)
    print("[KG Build] Step 3: Generating embeddings...")
    print("=" * 50)
    neo4j_kg.generate_table_node_embeddings(local_st_model=local_st_model)
    neo4j_kg.generate_column_node_embeddings(local_st_model=local_st_model)

    # 4. Create indexes
    print("\n" + "=" * 50)
    print("[KG Build] Step 4: Creating indexes...")
    print("=" * 50)
    neo4j_kg.create_vector_index(dimensions=embedding_dims)
    neo4j_kg.create_column_vector_index(dimensions=embedding_dims)

    print("\n" + "=" * 50)
    print("[KG Build] [OK] Knowledge graph construction completed!")
    print("=" * 50)

    return get_embed_token_usage()


def clear_neo4j_database(drop_indexes: bool = True, verify: bool = True):
    """
    Standalone function: Clear Neo4j database (nodes, relationships, indexes)

    Args:
        drop_indexes: Whether to drop indexes
        verify: Whether to verify the result
    """
    print("[Clear DB] Starting Neo4j database cleanup...")

    with driver.session() as session:
        # 1. Show state before clearing
        if verify:
            node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            print(f"  [Before] Nodes: {node_count}, Relationships: {rel_count}")

        # 2. Delete all nodes and relationships
        print("  [Clear] Deleting all nodes and relationships...")
        session.run("MATCH (n) DETACH DELETE n")

        # 3. Dynamically retrieve and drop all indexes (more thorough approach)
        if drop_indexes:
            try:
                indexes = session.run("SHOW INDEXES")
                index_names = [record["name"] for record in indexes]
                dropped_indexes = []
                for idx_name in index_names:
                    try:
                        session.run(f"DROP INDEX `{idx_name}` IF EXISTS")
                        dropped_indexes.append(idx_name)
                        print(f"  [Clear] Dropped index: {idx_name}")
                    except Exception as e:
                        print(f"  [Clear] Warning: Could not drop index {idx_name}: {e}")
                print(f"  [Clear] Total indexes dropped: {len(dropped_indexes)}")
            except Exception as e:
                print(f"  [Clear] Warning: Could not retrieve indexes: {e}")

        # 4. Verify clearing result
        if verify:
            node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            print(f"  [After] Nodes: {node_count}, Relationships: {rel_count}")

            if node_count == 0 and rel_count == 0:
                print("[Clear DB] [OK] Database cleared successfully!")
            else:
                print("[Clear DB] [WARNING] Database may not be fully cleared!")

    return True
