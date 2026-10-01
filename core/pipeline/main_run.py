import sys
import os

# Ensure we can import from local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kg_construction import build_knowledge_graph
from retrieval import perform_retrieval
from sql_generation import generate_sql

def main():
    print("Starting Pipeline...")
    
    # 1. Knowledge Graph Construction
    print("\n--- Step 1: Building Knowledge Graph ---")
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    schema_path = os.environ.get(
        "SCHEMA_PATH",
        os.path.join(repo_root, "data", "spider", "spider_union_schema_FK.csv"),
    )
    build_knowledge_graph(schema_path)
    print("Knowledge Graph built successfully.")
    
    # 2. Retrieval
    print("\n--- Step 2: Retrieval ---")
    query_text = "What is the highest eligible free rate for K-12 students in the schools in Alameda County?"
    print(f"User Query: {query_text}")
    
    retrieved_contents = perform_retrieval(query_text)
    print("Retrieved Context:")
    print(retrieved_contents)
    
    # 3. SQL Generation
    print("\n--- Step 3: SQL Generation ---")
    final_sql = generate_sql(query_text, retrieved_contents)
    print("\nFinal Result:")
    print(final_sql)

if __name__ == "__main__":
    main()
