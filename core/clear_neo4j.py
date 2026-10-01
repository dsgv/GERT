"""
Neo4j database management utility
Used to clear all data in the Neo4j database
"""

import os
import sys

# Configure path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from core.pipeline.common import driver
except ImportError:
    from pipeline.common import driver


def clear_neo4j_database(confirm=True):
    """
    Clear all nodes, relationships, and indexes in the Neo4j database

    Args:
        confirm: Whether user confirmation is needed (default True)

    Returns:
        bool: Whether cleared successfully
    """
    if confirm:
        print("\n" + "="*50)
        print("⚠️  WARNING: This will DELETE ALL data in Neo4j!")
        print("="*50)
        response = input("Are you sure? (yes/no): ").strip().lower()
        if response != 'yes':
            print("Operation cancelled.")
            return False

    try:
        with driver.session() as session:
            # 1. Count data before deletion
            node_result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = node_result.single()["count"]

            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = rel_result.single()["count"]

            print(f"\nDeleting {node_count} nodes and {rel_count} relationships...")

            # 2. Delete all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")

            # 3. Drop all indexes
            indexes = session.run("SHOW INDEXES")
            index_names = [record["name"] for record in indexes]

            dropped_indexes = []
            for idx_name in index_names:
                try:
                    session.run(f"DROP INDEX `{idx_name}` IF EXISTS")
                    dropped_indexes.append(idx_name)
                except Exception:
                    pass  # Some system indexes may not be droppable

            print(f"\n✅ Neo4j database cleared successfully!")
            print(f"   - Deleted nodes: {node_count}")
            print(f"   - Deleted relationships: {rel_count}")
            print(f"   - Dropped indexes: {len(dropped_indexes)}")

            return True

    except Exception as e:
        print(f"❌ Error clearing Neo4j database: {e}")
        return False


def get_neo4j_stats():
    """
    Get Neo4j database statistics

    Returns:
        dict: Contains node count, relationship count, index count, etc.
    """
    try:
        with driver.session() as session:
            # Node count
            node_result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = node_result.single()["count"]

            # Relationship count
            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = rel_result.single()["count"]

            # Node count by label
            labels_result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as label, count(n) as count
                ORDER BY count DESC
            """)
            label_counts = {record["label"]: record["count"] for record in labels_result}

            # Indexes
            indexes_result = session.run("SHOW INDEXES")
            indexes = [record["name"] for record in indexes_result]

            print(f"\n{'='*50}")
            print("Neo4j Database Statistics")
            print(f"{'='*50}")
            print(f"Total nodes: {node_count}")
            print(f"Total relationships: {rel_count}")
            print(f"Total indexes: {len(indexes)}")

            if label_counts:
                print(f"\nNodes by label:")
                for label, count in label_counts.items():
                    print(f"  - {label}: {count}")

            if indexes:
                print(f"\nIndexes:")
                for idx in indexes:
                    print(f"  - {idx}")

            print(f"{'='*50}")

            return {
                "total_nodes": node_count,
                "total_relationships": rel_count,
                "nodes_by_label": label_counts,
                "indexes": indexes
            }

    except Exception as e:
        print(f"❌ Error getting Neo4j stats: {e}")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Neo4j Database Management Tool')
    parser.add_argument('--yes', action='store_true', help='Skip confirmation prompt',default="yes")
    parser.add_argument('--stats', action='store_true', help='Show database statistics only')

    args = parser.parse_args()

    if args.stats:
        get_neo4j_stats()
    else:
        clear_neo4j_database(confirm=not args.yes)
