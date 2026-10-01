"""
Schema Knowledge Graph construction module
"""
import argparse
import os
import sys
from typing import Optional

# Configure path to import pipeline module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
pipeline_dir = os.path.join(project_root, "pipeline")

if project_root not in sys.path:
    sys.path.append(project_root)
if pipeline_dir not in sys.path:
    sys.path.append(pipeline_dir)

# Import config and modules
try:
    from core.pipeline.common import driver
except ImportError:
    from pipeline.common import driver
try:
    from core.pipeline.kg_construction import build_knowledge_graph
except ImportError:
    from pipeline.kg_construction import build_knowledge_graph


def build_schema_kg(
    schema_csv_path: str,
    clear_before_build: bool = False,
    local_embedding_model_path: Optional[str] = None,
) -> dict:
    """
    Build knowledge graph from CSV Schema file

    Args:
        schema_csv_path: Schema CSV file path
        clear_before_build: Whether to clear entire database before building (default False)
        local_embedding_model_path: If set, load local sentence-transformers model for embedding, skip API

    Returns:
        dict: {"success": bool, "token_usage": {"total_tokens": int, "total_requests": int}}
    """
    try:
        from core.pipeline.kg_construction import reset_embed_token_usage
    except ImportError:
        from pipeline.kg_construction import reset_embed_token_usage

    reset_embed_token_usage()
    print(f"Start building Knowledge Graph from {schema_csv_path}...")

    try:
        token_usage = build_knowledge_graph(
            schema_path=schema_csv_path,
            clear_before_build=clear_before_build,
            local_embedding_model_path=local_embedding_model_path,
        )
        print("Knowledge Graph built successfully.")
        return {"success": True, "token_usage": token_usage}
    except Exception as e:
        print(f"Error building Knowledge Graph: {e}")
        return {"success": False, "token_usage": {"total_tokens": 0, "total_requests": 0}}


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description="Build Schema KG from CSV")
    parser.add_argument(
        "schema_csv",
        nargs="?",
        default=os.path.join(project_root, "data/bird/bird_union_schema_FK.csv"),
        help="Schema CSV path",
    )
    parser.add_argument("--clear", action="store_true", help="Clear Neo4j before build")
    args = parser.parse_args()

    local_path = None


    if os.path.exists(args.schema_csv):
        result = build_schema_kg(
            args.schema_csv,
            clear_before_build=args.clear,
            local_embedding_model_path=local_path,
        )
        if not result["success"]:
            sys.exit(1)
    else:
        print(f"Schema file not found: {args.schema_csv}")
        sys.exit(1)

