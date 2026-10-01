"""
Retrieval module - execute hybrid search (Vector + Fulltext)
"""
import json
import os
import sys
from typing import Optional

import numpy as np
from neo4j_graphrag.retrievers import HybridCypherRetriever

# Ensure sibling modules can be found
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from common import (
    driver, Embed_client,
    OPENAI_EMBEDDING_MODEL,
    TABLE_VECTOR_INDEX_NAME,
    TABLE_FULLTEXT_INDEX_NAME,
    TOP_K
)
from local_embedding import load_sentence_transformer
from result_parser import get_detailed_results


# Custom class for OpenAI embeddings
class OpenAIEmbedder:
    def __init__(self, model=None, api_key=None):
        # Use default values from env vars
        if model is None:
            model = OPENAI_EMBEDDING_MODEL
        self.model = model
        # Token usage statistics
        self.total_tokens = 0
        self.total_requests = 0

    def embed_query(self, text):
        response = Embed_client.embeddings.create(
            input=text,
            model=self.model
        )
        # Accumulate token usage
        self.total_tokens += response.usage.total_tokens
        self.total_requests += 1
        return response.data[0].embedding

    def get_token_usage(self):
        """Get token usage statistics"""
        return {
            "total_tokens": self.total_tokens,
            "total_requests": self.total_requests
        }

    def reset_token_usage(self):
        """Reset token usage statistics"""
        self.total_tokens = 0
        self.total_requests = 0


class LocalSentenceTransformerEmbedder:
    """
    Same interface as OpenAIEmbedder, for local vector encoding of queries during retrieval.
    Must use the same local model path as during KG construction.
    """

    def __init__(self, model_path: str):
        self._st = load_sentence_transformer(model_path)
        self.model = f"local:{os.path.abspath(model_path)}"
        self.total_requests = 0

    def embed_query(self, text: str):
        vec = self._st.encode(
            text,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        if isinstance(vec, np.ndarray) and vec.ndim > 1:
            vec = vec[0]
        self.total_requests += 1
        return vec.tolist()

    def get_token_usage(self):
        return {
            "total_tokens": 0,
            "total_requests": self.total_requests,
        }

    def reset_token_usage(self):
        self.total_requests = 0


def make_embedder(local_embedding_model_path: Optional[str] = None):
    """
    If ``local_embedding_model_path`` is provided, return a local Embedder; otherwise return API Embedder.
    """
    if local_embedding_model_path:
        return LocalSentenceTransformerEmbedder(local_embedding_model_path)
    return OpenAIEmbedder()


# Set up Cypher query to retrieve necessary context from knowledge graph
retrieval_query = """
MATCH (node:Table)
OPTIONAL MATCH (node)-[:CONTAINS]->(col:Column)
OPTIONAL MATCH (col)-[:MAPS_TO_CONCEPT]->(concept:Concept)
OPTIONAL MATCH (concept)<-[:MAPS_TO_CONCEPT]-(relatedCol:Column)

// Get columns related via foreign keys (direct association)
OPTIONAL MATCH (col)-[:REFERENCES]->(refCol:Column)
OPTIONAL MATCH (col)<-[:REFERENCES]-(referencingCol:Column)

RETURN
  node.name AS table_name,
  node.description AS table_description,

  // Only include columns if col.name is not null
  collect(DISTINCT CASE
    WHEN col.name IS NOT NULL THEN {
      column_name: col.name,
      description: COALESCE(col.description, "No description available"),
      data_type: COALESCE(col.data_type, "Unknown"),
      column_sample_value: COALESCE(col.column_sample_value, "No sample value"),
      foreign_key_ref: col.foreign_key_ref
    }
  END) AS columns,

  collect(DISTINCT concept.name) AS concepts,

  // Only include related columns if relatedCol.name is not null
  collect(DISTINCT CASE
    WHEN relatedCol.name IS NOT NULL THEN {
      column_name: relatedCol.name,
      description: COALESCE(relatedCol.description, "No description available"),
      data_type: COALESCE(relatedCol.data_type, "Unknown"),
      column_sample_value: COALESCE(relatedCol.column_sample_value, "No sample value"),
      source: "concept_linked"
    }
  END) +

  // Include columns referenced by this table's columns (FK targets)
  collect(DISTINCT CASE
    WHEN refCol.name IS NOT NULL THEN {
      column_name: refCol.name,
      table_name: refCol.table_name,
      data_type: COALESCE(refCol.data_type, "Unknown"),
      source: "foreign_key_target"
    }
  END) +

  // Include columns that reference this table's columns (FK sources)
  collect(DISTINCT CASE
    WHEN referencingCol.name IS NOT NULL THEN {
      column_name: referencingCol.name,
      table_name: referencingCol.table_name,
      data_type: COALESCE(referencingCol.data_type, "Unknown"),
      source: "foreign_key_source"
    }
  END) AS related_columns
"""


def perform_retrieval(query_text, top_k=None, local_embedding_model_path: Optional[str] = None):
    """
    Execute hybrid retrieval

    Args:
        query_text: Query text
        top_k: Number of top-k results to return, defaults to env var value
        local_embedding_model_path: If set, consistent with KG build, use local sentence-transformers to encode query
    """
    if top_k is None:
        top_k = TOP_K

    # Set up hybrid retriever
    retriever = HybridCypherRetriever(
        driver=driver,
        vector_index_name=TABLE_VECTOR_INDEX_NAME,
        fulltext_index_name=TABLE_FULLTEXT_INDEX_NAME,
        embedder=make_embedder(local_embedding_model_path),
        retrieval_query=retrieval_query
    )

    retriever_result = retriever.search(query_text=query_text, top_k=top_k)

    # Use structured parsing instead of regex parsing
    tables_data = get_detailed_results(retriever_result)

    retrieved_contents = json.dumps({"table": tables_data}, indent=2, ensure_ascii=False)
    return retrieved_contents
