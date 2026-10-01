"""
Common configuration module - load all configs from environment variables

All config values are loaded from .env file or system environment variables
"""
import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# Load environment variables
load_dotenv()


# ===========================================
# Config loading helper functions
# ===========================================

def get_env_str(key: str, default: str = "") -> str:
    """Get string-type environment variable"""
    value = os.getenv(key, default)
    if not value:
        print(f"Warning: Environment variable '{key}' is not set, using default: '{default}'")
    return value


def get_env_int(key: str, default: int = 0) -> int:
    """Get integer-type environment variable"""
    value = os.getenv(key)
    if value is None:
        print(f"Warning: Environment variable '{key}' is not set, using default: {default}")
        return default
    try:
        return int(value)
    except ValueError:
        print(f"Warning: Cannot convert '{key}={value}' to int, using default: {default}")
        return default


def get_env_float(key: str, default: float = 0.0) -> float:
    """Get float-type environment variable"""
    value = os.getenv(key)
    if value is None:
        print(f"Warning: Environment variable '{key}' is not set, using default: {default}")
        return default
    try:
        return float(value)
    except ValueError:
        print(f"Warning: Cannot convert '{key}={value}' to float, using default: {default}")
        return default


# ===========================================
# Embedding config
# ===========================================

Embed_API_KEY = get_env_str("Embed_API_KEY")
Embed_API_BASE = get_env_str("Embed_API_BASE", "https://api.openai.com/v1")
OPENAI_EMBEDDING_MODEL = get_env_str("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = get_env_int("EMBEDDING_DIMENSIONS", 1536)

# Some OpenAI-compatible providers have stricter limits on embedding batch size (e.g. DashScope: <= 10).
# This provides a uniform upper bound, used by modules when batching embeddings.
_default_embed_max_batch = 10 if "dashscope" in (Embed_API_BASE or "").lower() else 128
EMBEDDING_MAX_BATCH_SIZE = get_env_int("EMBEDDING_MAX_BATCH_SIZE", _default_embed_max_batch)

# Embed Client (for vector embeddings)
Embed_client = OpenAI(
    api_key=Embed_API_KEY,
    base_url=Embed_API_BASE,
)

# ===========================================
# Neo4j config
# ===========================================

NEO4J_USERNAME = get_env_str("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = get_env_str("NEO4J_PASSWORD")
NEO4J_BOLT_URL = get_env_str("NEO4J_BOLT_URL", "bolt://localhost:7687")

# Neo4j Driver
driver = GraphDatabase.driver(
    NEO4J_BOLT_URL,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

# Vector Index config
TABLE_VECTOR_INDEX_NAME = get_env_str("TABLE_VECTOR_INDEX_NAME", "table_vector_index")
TABLE_FULLTEXT_INDEX_NAME = get_env_str("TABLE_FULLTEXT_INDEX_NAME", "table_fulltext_index")
TABLE_EMBEDDING_PROPERTY_NAME = get_env_str("TABLE_EMBEDDING_PROPERTY_NAME", "table_embedding")

# ===========================================
# Schema Routing config
# ===========================================

TOP_K = get_env_int("TOP_K", 15)
