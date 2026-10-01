#!/usr/bin/env python3
"""Prepare frozen BirdUnion inputs for the official DIN-SQL and MAC-SQL code."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "prepared_prompts" / "gert.json"
DB_ROOT = ROOT / "data" / "dev_database"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def sqlite_path(db_id: str) -> Path:
    expected = DB_ROOT / db_id / f"{db_id}.sqlite"
    if expected.exists():
        return expected
    matches = sorted((DB_ROOT / db_id).glob("*.sqlite"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one SQLite file for {db_id}: {matches}")
    return matches[0]


def normalized_type(declared: str) -> str:
    value = (declared or "").upper()
    if "INT" in value:
        return "number"
    if any(token in value for token in ("REAL", "FLOA", "DOUB", "NUM", "DEC")):
        return "number"
    if any(token in value for token in ("DATE", "TIME")):
        return "time"
    if "BOOL" in value:
        return "boolean"
    return "text"


def build_schema(db_id: str) -> dict:
    connection = sqlite3.connect(f"file:{sqlite_path(db_id).as_posix()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name COLLATE NOCASE"
        ).fetchall()
        tables = [row[0] for row in rows]
        original: list[list[object]] = [[-1, "*"]]
        normalized: list[list[object]] = [[-1, "*"]]
        types = ["text"]
        primary_keys: list[int] = []
        column_index: dict[tuple[str, str], int] = {}
        for table_id, table in enumerate(tables):
            for row in connection.execute(
                f"PRAGMA table_info({quote_identifier(table)})"
            ).fetchall():
                name = row[1]
                column_index[(table.lower(), name.lower())] = len(original)
                original.append([table_id, name])
                normalized.append([table_id, name])
                types.append(normalized_type(row[2]))
                if row[5]:
                    primary_keys.append(len(original) - 1)
        foreign_keys: list[list[int]] = []
        for table in tables:
            for row in connection.execute(
                f"PRAGMA foreign_key_list({quote_identifier(table)})"
            ).fetchall():
                source = column_index.get((table.lower(), str(row[3]).lower()))
                target = column_index.get((str(row[2]).lower(), str(row[4]).lower()))
                if source is not None and target is not None:
                    foreign_keys.append([source, target])
        return {
            "db_id": db_id,
            "table_names_original": tables,
            "table_names": tables,
            "column_names_original": original,
            "column_names": normalized,
            "column_types": types,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys,
        }
    finally:
        connection.close()


def main() -> None:
    source = read_json(SOURCE)
    adapted = [
        {
            "question_id": pipeline_id,
            "source_question_id": item["question_id"],
            "db_id": item["db_id"],
            "question": item["question"],
            "SQL": item["gold_sql"],
            "difficulty": item.get("difficulty", "unknown"),
        }
        for pipeline_id, item in enumerate(source)
    ]
    schemas = [build_schema(db_id) for db_id in sorted({x["db_id"] for x in adapted})]
    id_map = [
        {
            "pipeline_question_id": x["question_id"],
            "source_question_id": x["source_question_id"],
        }
        for x in adapted
    ]
    for method in ("DIN-SQL", "MAC-SQL"):
        write_json(ROOT / "frameworks" / method / "data" / "dev.json", adapted)
        write_json(ROOT / "frameworks" / method / "data" / "tables.json", schemas)
        write_json(ROOT / "frameworks" / method / "data" / "question_id_map.json", id_map)
    print(f"Prepared {len(adapted)} questions and {len(schemas)} database schemas")


if __name__ == "__main__":
    main()
