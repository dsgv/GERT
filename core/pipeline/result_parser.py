"""
Structured parsing module - for parsing Neo4j retrieval results

Provides stable structured parsing methods to replace fragile regex parsing.
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field


@dataclass
class ColumnInfo:
    """Column information dataclass"""
    column_name: str
    description: str = "No description available"
    data_type: str = "Unknown"
    column_sample_value: str = "No sample value"
    foreign_key_ref: Optional[str] = None


@dataclass
class RelatedColumn:
    """Related column information (via foreign key association)"""
    column_name: str
    table_name: str
    data_type: str = "Unknown"
    source: str = "unknown"  # "foreign_key_target" or "foreign_key_source"


@dataclass
class TableRetrievalResult:
    """Complete result for a single retrieval"""
    table_name: str
    table_description: str = ""
    columns: List[ColumnInfo] = field(default_factory=list)
    related_columns: List[RelatedColumn] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    raw_score: float = 0.0

    def get_all_related_tables(self) -> List[str]:
        """Get all related table names (via foreign key association)"""
        tables = set()
        for rc in self.related_columns:
            if rc.table_name:
                tables.add(rc.table_name)
        return list(tables)


def parse_retrieval_records(records) -> List[TableRetrievalResult]:
    """
    Parse structured results from a list of neo4j.Record

    Args:
        records: List of neo4j.Record (from RawSearchResult.records)

    Returns:
        List[TableRetrievalResult]: List of structured retrieval results
    """
    results = []

    for record in records:
        try:
            # Use neo4j.Record.data() method to get dict
            data = record.data() if hasattr(record, 'data') else dict(record)

            # Extract main table info
            table_name = data.get('table_name', '')
            if not table_name:
                continue

            table_result = TableRetrievalResult(
                table_name=table_name,
                table_description=data.get('table_description', '') or '',
                raw_score=float(data.get('score', 0.0) or 0.0),
            )

            # Parse column info
            columns_data = data.get('columns', [])
            if columns_data:
                for col in columns_data:
                    if col is None:
                        continue
                    col_info = ColumnInfo(
                        column_name=col.get('column_name', ''),
                        description=col.get('description', 'No description available') or 'No description available',
                        data_type=col.get('data_type', 'Unknown') or 'Unknown',
                        column_sample_value=col.get('column_sample_value', 'No sample value') or 'No sample value',
                        foreign_key_ref=col.get('foreign_key_ref'),
                    )
                    table_result.columns.append(col_info)

            # Parse related column info (foreign key association)
            related_columns_data = data.get('related_columns', [])
            if related_columns_data:
                for rc in related_columns_data:
                    if rc is None:
                        continue
                    related_col = RelatedColumn(
                        column_name=rc.get('column_name', ''),
                        table_name=rc.get('table_name', ''),
                        data_type=rc.get('data_type', 'Unknown') or 'Unknown',
                        source=rc.get('source', 'unknown'),
                    )
                    table_result.related_columns.append(related_col)

            # Parse concept info (if exists)
            concepts_data = data.get('concepts', [])
            if concepts_data:
                table_result.concepts = [c for c in concepts_data if c]

            results.append(table_result)

        except Exception as e:
            print(f"Warning: Failed to parse record: {e}")
            continue

    return results


def extract_tables_from_results(
    results: List[TableRetrievalResult],
    top_k: int = 30,
    include_fk_related: bool = True
) -> List[str]:
    """
    Extract table name list from retrieval results

    Args:
        results: List of structured retrieval results
        top_k: Maximum number of tables to return
        include_fk_related: Whether to include tables linked via foreign keys

    Returns:
        List[str]: List of table names (preserving order)
    """
    ordered_tables = []
    seen_tables: Set[str] = set()

    for result in results:
        # Add main table
        if result.table_name and result.table_name not in seen_tables:
            ordered_tables.append(result.table_name)
            seen_tables.add(result.table_name)

        # Add tables linked via foreign keys
        if include_fk_related:
            for related_table in result.get_all_related_tables():
                if related_table and related_table not in seen_tables:
                    ordered_tables.append(related_table)
                    seen_tables.add(related_table)

    return ordered_tables[:top_k]


def parse_retriever_result(retriever_result) -> List[TableRetrievalResult]:
    """
    Parse structured data from HybridCypherRetriever.search() results

    This function is the main entry point, handling RawSearchResult objects.

    Args:
        retriever_result: RawSearchResult object (from retriever.search())

    Returns:
        List[TableRetrievalResult]: List of structured retrieval results
    """
    # Prefer using records attribute (raw neo4j.Record list)
    if hasattr(retriever_result, 'records') and retriever_result.records:
        return parse_retrieval_records(retriever_result.records)

    # Fall back to items attribute
    if hasattr(retriever_result, 'items') and retriever_result.items:
        return parse_retriever_items(retriever_result.items)

    return []


def parse_retriever_items(items) -> List[TableRetrievalResult]:
    """
    Parse structured data from RetrieverResultItem list (fallback method)

    Used when raw records are not accessible.
    Attempts to parse content as structured data.
    """
    import json
    import re

    results = []

    for item in items:
        try:
            content = item.content

            # If content is already a dict
            if isinstance(content, dict):
                results.append(_parse_dict_content(content))
                continue

            # If content is a string, try multiple parsing methods
            if isinstance(content, str):
                parsed = _parse_string_content(content)
                if parsed:
                    results.append(parsed)

        except Exception as e:
            print(f"Warning: Failed to parse item: {e}")
            continue

    return results


def _parse_dict_content(data: dict) -> TableRetrievalResult:
    """Parse content in dictionary format"""
    table_result = TableRetrievalResult(
        table_name=data.get('table_name', ''),
        table_description=data.get('table_description', '') or '',
    )

    # Parse columns
    for col in data.get('columns', []):
        if col:
            table_result.columns.append(ColumnInfo(
                column_name=col.get('column_name', ''),
                description=col.get('description', '') or '',
                data_type=col.get('data_type', 'Unknown') or 'Unknown',
                column_sample_value=col.get('column_sample_value', '') or '',
                foreign_key_ref=col.get('foreign_key_ref'),
            ))

    # Parse related columns
    for rc in data.get('related_columns', []):
        if rc:
            table_result.related_columns.append(RelatedColumn(
                column_name=rc.get('column_name', ''),
                table_name=rc.get('table_name', ''),
                data_type=rc.get('data_type', 'Unknown') or 'Unknown',
                source=rc.get('source', 'unknown'),
            ))

    return table_result


def _parse_string_content(content: str) -> Optional[TableRetrievalResult]:
    """
    Parse content in string format

    This is the last resort fallback method, attempting to extract structured info from strings.
    Supports JSON format and Neo4j Record string format.
    """
    import json
    import re

    # Try to parse as JSON
    try:
        data = json.loads(content)
        return _parse_dict_content(data)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to parse Neo4j Record string format
    # Format: <Record table_name='...' ...>
    if content.startswith('<Record ') or 'table_name=' in content:
        return _parse_record_string(content)

    return None


def _parse_record_string(content: str) -> Optional[TableRetrievalResult]:
    """
    Parse Neo4j Record string format

    Format example:
    <Record table_name='accounts' table_description='...' columns=[{...}] related_columns=[{...}]>
    """
    import re

    # Clean string
    text = content.strip()
    if text.startswith('<Record '):
        text = text[8:]
    if text.endswith('>'):
        text = text[:-1]

    table_result = TableRetrievalResult(table_name='')

    # Extract table_name
    m = re.search(r"table_name='([^']*)'", text)
    if m:
        table_result.table_name = m.group(1)

    # Extract table_description
    m = re.search(r"table_description='(.*?)'\s+columns=", text)
    if m:
        table_result.table_description = m.group(1)

    # Extract columns - use more robust bracket matching
    columns_str = _extract_nested_structure(text, 'columns=')
    if columns_str:
        table_result.columns = _parse_columns_string(columns_str)

    # Extract related_columns
    related_str = _extract_nested_structure(text, 'related_columns=')
    if related_str:
        table_result.related_columns = _parse_related_columns_string(related_str)

    # Extract concepts
    concepts_str = _extract_nested_structure(text, 'concepts=')
    if concepts_str:
        try:
            import ast
            concepts = ast.literal_eval(concepts_str)
            if isinstance(concepts, list):
                table_result.concepts = [c for c in concepts if isinstance(c, str)]
        except:
            pass

    return table_result if table_result.table_name else None


def _extract_nested_structure(text: str, prefix: str) -> Optional[str]:
    """
    Extract nested structure string (correctly handles bracket matching)

    e.g.: [{...}, {...}] from columns=[{...}, {...}]
    """
    import re

    # Find prefix position
    start_idx = text.find(prefix)
    if start_idx == -1:
        return None

    start_idx += len(prefix)
    if start_idx >= len(text):
        return None

    # Skip spaces
    while start_idx < len(text) and text[start_idx] == ' ':
        start_idx += 1

    if start_idx >= len(text):
        return None

    # Determine closing character from opening character
    open_char = text[start_idx]
    close_char = {'[': ']', '{': '}', '(': ')'}.get(open_char)

    if not close_char:
        return None

    # Bracket matching
    depth = 0
    in_string = False
    escape_next = False

    for i in range(start_idx, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == '\\':
            escape_next = True
            continue

        if char == "'" and not in_string:
            in_string = True
        elif char == "'" and in_string:
            in_string = False

        if not in_string:
            if char == open_char:
                depth += 1
            elif char == close_char:
                depth -= 1
                if depth == 0:
                    return text[start_idx:i + 1]

    return None


def _parse_columns_string(columns_str: str) -> List[ColumnInfo]:
    """Parse columns string into ColumnInfo list"""
    import ast

    columns = []
    try:
        data = ast.literal_eval(columns_str)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    columns.append(ColumnInfo(
                        column_name=item.get('column_name', ''),
                        description=item.get('description', '') or 'No description available',
                        data_type=item.get('data_type', 'Unknown') or 'Unknown',
                        column_sample_value=item.get('column_sample_value', '') or 'No sample value',
                        foreign_key_ref=item.get('foreign_key_ref'),
                    ))
    except Exception as e:
        print(f"Warning: Failed to parse columns: {e}")

    return columns


def _parse_related_columns_string(related_str: str) -> List[RelatedColumn]:
    """Parse related_columns string into RelatedColumn list"""
    import ast

    related = []
    try:
        data = ast.literal_eval(related_str)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    related.append(RelatedColumn(
                        column_name=item.get('column_name', ''),
                        table_name=item.get('table_name', ''),
                        data_type=item.get('data_type', 'Unknown') or 'Unknown',
                        source=item.get('source', 'unknown'),
                    ))
    except Exception as e:
        print(f"Warning: Failed to parse related_columns: {e}")

    return related


# ============ Convenience functions ============

def get_tables_from_retriever(retriever_result, top_k: int = 30) -> List[str]:
    """
    Get table name list directly from retrieval results (convenience function)

    Args:
        retriever_result: Return value of HybridCypherRetriever.search()
        top_k: Maximum number to return

    Returns:
        List[str]: List of table names
    """
    results = parse_retriever_result(retriever_result)
    return extract_tables_from_results(results, top_k=top_k)


def get_detailed_results(retriever_result) -> List[Dict[str, Any]]:
    """
    Get detailed retrieval results (containing all information)

    Returns:
        List[Dict]: List of dicts containing complete table information
    """
    results = parse_retriever_result(retriever_result)

    detailed = []
    for r in results:
        detailed.append({
            'table_name': r.table_name,
            'table_description': r.table_description,
            'columns': [
                {
                    'column_name': c.column_name,
                    'description': c.description,
                    'data_type': c.data_type,
                    'column_sample_value': c.column_sample_value,
                    'foreign_key_ref': c.foreign_key_ref,
                }
                for c in r.columns
            ],
            'related_columns': [
                {
                    'column_name': rc.column_name,
                    'table_name': rc.table_name,
                    'data_type': rc.data_type,
                    'source': rc.source,
                }
                for rc in r.related_columns
            ],
            'related_tables': r.get_all_related_tables(),
        })

    return detailed
