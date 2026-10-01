"""Prepare, run, and evaluate the BirdUnion downstream Text-to-SQL experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sqlite3
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml
from dotenv import load_dotenv


SCRIPT_DIR = Path(__file__).resolve().parent
TEXT2SQL_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = TEXT2SQL_ROOT / "configs" / "experiment.yaml"


def load_config(path: Path) -> dict[str, Any]:
    load_dotenv(TEXT2SQL_ROOT / ".env")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    generation = config["generation"]
    generation["model"] = os.getenv("TEXT2SQL_MODEL", generation["model"])
    generation["api_base"] = os.getenv("TEXT2SQL_API_BASE", generation["api_base"])
    return config


def resolve_path(value: str) -> Path:
    return (TEXT2SQL_ROOT / value).resolve()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


class TokenCounter:
    def __init__(self, tokenizer_name: str):
        self.method = "chars_div_4_estimate"
        self.tokenizer = None
        if tokenizer_name:
            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                self.method = f"huggingface:{tokenizer_name}"
            except Exception as exc:  # fallback is intentionally explicit
                print(f"Warning: tokenizer unavailable ({exc}); using chars/4 estimate.", file=sys.stderr)

    def count(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return math.ceil(len(text) / 4)


class SQLiteCatalog:
    def __init__(self, database_root: Path, aliases: dict[str, str]):
        self.database_root = database_root
        self.aliases = {key.lower(): value for key, value in aliases.items()}
        self._tables: dict[str, list[str]] = {}

    def database_path(self, db_id: str) -> Path:
        path = self.database_root / db_id / f"{db_id}.sqlite"
        if not path.is_file():
            raise FileNotFoundError(f"SQLite database not found: {path}")
        return path

    def tables(self, db_id: str) -> list[str]:
        if db_id not in self._tables:
            connection = sqlite3.connect(f"file:{self.database_path(db_id)}?mode=ro", uri=True)
            try:
                rows = connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()
            finally:
                connection.close()
            self._tables[db_id] = [row[0] for row in rows if not row[0].lower().startswith("sqlite_")]
        return self._tables[db_id]

    def resolve_table(self, qualified_name: str) -> tuple[str, str, str]:
        if "." not in qualified_name:
            raise ValueError(f"Malformed candidate table: {qualified_name}")
        db_id, requested = qualified_name.split(".", 1)
        available = self.tables(db_id)
        if requested in available:
            return db_id, requested, "exact"
        case_matches = [name for name in available if name.lower() == requested.lower()]
        if len(case_matches) == 1:
            return db_id, case_matches[0], "case_insensitive"
        alias = self.aliases.get(qualified_name.lower())
        if alias:
            alias_matches = [name for name in available if name.lower() == alias.lower()]
            if len(alias_matches) == 1:
                return db_id, alias_matches[0], "explicit_alias"
        raise KeyError(
            f"Cannot map {qualified_name} to a physical table; available={available}"
        )

    @staticmethod
    def quote_identifier(value: str) -> str:
        return '"' + value.replace('"', '""') + '"'

    def ddl(self, db_id: str, table_name: str, allowed_tables: set[str]) -> str:
        """Build canonical DDL without exposing non-candidate FK targets."""
        connection = sqlite3.connect(f"file:{self.database_path(db_id)}?mode=ro", uri=True)
        try:
            quoted_table = self.quote_identifier(table_name)
            columns = connection.execute(f"PRAGMA table_info({quoted_table})").fetchall()
            foreign_keys = connection.execute(
                f"PRAGMA foreign_key_list({quoted_table})"
            ).fetchall()
        finally:
            connection.close()
        if not columns:
            raise KeyError(f"Columns not found for {db_id}.{table_name}")

        primary_columns = sorted(
            [row for row in columns if int(row[5] or 0) > 0],
            key=lambda row: int(row[5]),
        )
        definitions: list[str] = []
        for _, name, column_type, not_null, _, primary_position in columns:
            parts = [self.quote_identifier(name), column_type or "TEXT"]
            if not_null:
                parts.append("NOT NULL")
            if len(primary_columns) == 1 and primary_position:
                parts.append("PRIMARY KEY")
            definitions.append(" ".join(parts))
        if len(primary_columns) > 1:
            names = ", ".join(self.quote_identifier(row[1]) for row in primary_columns)
            definitions.append(f"PRIMARY KEY ({names})")

        allowed_lower = {name.lower() for name in allowed_tables}
        for _, _, target_table, source_column, target_column, *_ in foreign_keys:
            if target_table.lower() not in allowed_lower:
                continue
            reference = f"REFERENCES {self.quote_identifier(target_table)}"
            if target_column:
                reference += f" ({self.quote_identifier(target_column)})"
            definitions.append(
                f"FOREIGN KEY ({self.quote_identifier(source_column)}) {reference}"
            )
        body = ",\n    ".join(definitions)
        return f"CREATE TABLE {quoted_table} (\n    {body}\n);"

    def serialize(self, candidates: list[str]) -> tuple[str, list[dict[str, str]]]:
        mappings: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for candidate in candidates:
            db_id, physical, match_type = self.resolve_table(candidate)
            identity = (db_id, physical.lower())
            if identity in seen:
                continue
            seen.add(identity)
            mappings.append(
                {
                    "requested": candidate,
                    "database": db_id,
                    "physical_table": physical,
                    "match_type": match_type,
                }
            )
        allowed_by_database: dict[str, set[str]] = {}
        for mapping in mappings:
            allowed_by_database.setdefault(mapping["database"], set()).add(
                mapping["physical_table"]
            )
        sections: list[str] = []
        for mapping in mappings:
            db_id = mapping["database"]
            physical = mapping["physical_table"]
            sections.append(
                f"Candidate table: {db_id}.{physical}\n"
                f"Executable SQLite table name: {physical}\n"
                f"{self.ddl(db_id, physical, allowed_by_database[db_id])}"
            )
        return "\n\n".join(sections), mappings


def validate_aligned(reference: list[dict], other: list[dict], label: str) -> None:
    if len(reference) != len(other):
        raise ValueError(f"{label} has {len(other)} records; expected {len(reference)}")
    for index, (left, right) in enumerate(zip(reference, other)):
        if left["question_id"] != right["question_id"]:
            raise ValueError(
                f"Question mismatch at index {index}: reference={left['question_id']} "
                f"{label}={right['question_id']}"
            )


def load_method_inputs(config: dict[str, Any]) -> dict[str, list[dict]]:
    paths = config["paths"]
    records = {
        "gert": read_json(resolve_path(paths["gert_input"])),
        "murre": read_json(resolve_path(paths["murre_input"])),
        "dense": read_json(resolve_path(paths["dense_input"])),
    }
    reference = records["gert"]
    validate_aligned(reference, records["murre"], "murre")
    validate_aligned(reference, records["dense"], "dense")
    return records


def condition_candidates(
    condition: str,
    index: int,
    records: dict[str, list[dict]],
    catalog: SQLiteCatalog,
) -> list[str]:
    reference = records["gert"][index]
    if condition == "gold":
        return list(reference.get("gold_schema", reference.get("schema_link", [])))
    return list(records[condition][index]["predicted_schemas_at15"])


def load_full_source_schema(config: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    """Load the pooled BirdUnion schema verbatim and index its database.table names."""
    path = resolve_path(config["paths"]["full_source_schema"])
    schema_text = path.read_text(encoding="utf-8").strip()
    current_database: str | None = None
    mappings: list[dict[str, str]] = []
    seen_tables: set[tuple[str, str]] = set()
    database_pattern = re.compile(r"^\s*--\s*Database:\s*(\S+)\s*$", re.I)
    table_pattern = re.compile(
        r'^\s*CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?["`\[]?([^\s("`\]]+)',
        re.I,
    )
    for line_number, line in enumerate(schema_text.splitlines(), start=1):
        database_match = database_pattern.match(line)
        if database_match:
            current_database = database_match.group(1)
            continue
        table_match = table_pattern.match(line)
        if table_match:
            if current_database is None:
                raise ValueError(
                    f"CREATE TABLE before a Database header at {path}:{line_number}"
                )
            table = table_match.group(1)
            table_key = (current_database.lower(), table.lower())
            if table_key in seen_tables:
                continue
            seen_tables.add(table_key)
            mappings.append(
                {
                    "input": f"{current_database}.{table}",
                    "database": current_database,
                    "physical_table": table,
                    "match_type": "full_source_file",
                }
            )
    if not mappings:
        raise ValueError(f"No database.table entries found in full schema: {path}")
    return schema_text, mappings


def validate_project(config: dict[str, Any]) -> None:
    """Validate the standalone inputs without calling an external API."""

    paths = config["paths"]
    required_path_keys = [
        "prompt",
        "database_root",
        "full_source_schema",
        "gert_input",
        "murre_input",
        "dense_input",
    ]
    for key in required_path_keys:
        path = resolve_path(paths[key])
        if not path.exists():
            raise FileNotFoundError(f"Missing paths.{key}: {path}")

    records = load_method_inputs(config)
    reference = records["gert"]
    catalog = SQLiteCatalog(
        resolve_path(paths["database_root"]), config.get("table_aliases", {})
    )
    database_ids: set[str] = set()
    resolved_candidates = 0
    for index, record in enumerate(reference):
        database_ids.add(record["db_id"])
        catalog.database_path(record["db_id"])
        for table in record.get("gold_schema", record.get("schema_link", [])):
            catalog.resolve_table(table)
        for condition in ("dense", "murre", "gert"):
            for table in condition_candidates(condition, index, records, catalog):
                catalog.resolve_table(table)
                resolved_candidates += 1

    _, full_mappings = load_full_source_schema(config)
    print("Standalone project validation passed")
    print(f"Queries: {len(reference)}")
    print(f"SQLite databases used: {len(database_ids)}")
    print(f"Resolved predicted candidates: {resolved_candidates}")
    print(f"Full-source schema tables: {len(full_mappings)}")


def prepare(config: dict[str, Any], conditions: list[str]) -> None:
    records = load_method_inputs(config)
    reference = records["gert"]
    catalog = SQLiteCatalog(
        resolve_path(config["paths"]["database_root"]), config.get("table_aliases", {})
    )
    counter = TokenCounter(config["tokenization"].get("tokenizer_name", ""))
    prompt_template = resolve_path(config["paths"]["prompt"]).read_text(encoding="utf-8")
    prepared_dir = resolve_path(config["paths"]["prepared_dir"])
    full_schema_text, full_mappings = load_full_source_schema(config)

    for condition in conditions:
        prepared: list[dict[str, Any]] = []
        for index, record in enumerate(reference):
            if condition == "full":
                schema_text = full_schema_text
                mappings = full_mappings
                candidates = [mapping["input"] for mapping in mappings]
            else:
                candidates = condition_candidates(condition, index, records, catalog)
                schema_text, mappings = catalog.serialize(candidates)
            prompt = prompt_template.replace("{SCHEMA}", schema_text).replace(
                "{QUESTION}", record["question"]
            )
            gold_tables = set()
            for gold_table in record.get("gold_schema", record.get("schema_link", [])):
                gold_db, gold_physical, _ = catalog.resolve_table(gold_table)
                gold_tables.add(f"{gold_db}.{gold_physical}".lower())
            canonical_candidates = {
                f"{mapping['database']}.{mapping['physical_table']}".lower()
                for mapping in mappings
            }
            prepared.append(
                {
                    "question_id": record["question_id"],
                    "db_id": record["db_id"],
                    "question": record["question"],
                    "gold_sql": record["SQL"],
                    "gold_schema": record.get("gold_schema", record.get("schema_link")),
                    "condition": condition,
                    "candidate_tables": candidates,
                    "canonical_mappings": mappings,
                    "candidate_count": len(mappings),
                    "gold_table_covered": gold_tables <= canonical_candidates,
                    "schema_text": schema_text,
                    "prompt": prompt,
                    "schema_chars": len(schema_text),
                    "schema_tokens": counter.count(schema_text),
                    "prompt_tokens_estimate": counter.count(prompt),
                    "token_count_method": counter.method,
                }
            )
        output = prepared_dir / f"{condition}.json"
        write_json(output, prepared)
        print(f"Prepared {condition}: {len(prepared)} queries -> {output}")


def clean_sql_response(content: str) -> str:
    text = (content or "").strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.I | re.S)
    if fenced:
        text = fenced.group(1).strip()
    text = re.sub(r"^\s*SQL\s*:\s*", "", text, flags=re.I)
    return text.strip()


def load_completed(path: Path) -> dict[int, dict]:
    completed: dict[int, dict] = {}
    if not path.exists():
        return completed
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            completed[int(item["question_id"])] = item
        except Exception as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return completed


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def create_client(config: dict[str, Any]):
    from openai import OpenAI

    api_key = os.getenv("TEXT2SQL_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing TEXT2SQL_API_KEY (or OPENAI_API_KEY). Copy .env.template to .env."
        )
    generation = config["generation"]
    return OpenAI(
        api_key=api_key,
        base_url=generation["api_base"],
        timeout=float(generation["timeout_seconds"]),
        max_retries=0,
    )


def generate_one(client: Any, config: dict[str, Any], prompt: str) -> dict[str, Any]:
    generation = config["generation"]
    attempts = int(generation["api_retries"]) + 1
    last_error: Exception | None = None
    for attempt in range(attempts):
        started = time.perf_counter()
        try:
            call_kwargs: dict[str, Any] = {
                "model": generation["model"],
                "messages": [{"role": "user", "content": prompt}],
            }
            if "temperature" in generation and generation["temperature"] is not None:
                call_kwargs["temperature"] = float(generation["temperature"])
            if "top_p" in generation and generation["top_p"] is not None:
                call_kwargs["top_p"] = float(generation["top_p"])
            if "max_output_tokens" in generation and generation["max_output_tokens"] is not None:
                call_kwargs["max_tokens"] = int(generation["max_output_tokens"])
            if "seed" in generation and generation["seed"] is not None:
                call_kwargs["seed"] = int(generation["seed"])
            response = client.chat.completions.create(**call_kwargs)
            elapsed_ms = (time.perf_counter() - started) * 1000
            content = response.choices[0].message.content or ""
            usage = response.usage
            return {
                "api_status": "success",
                "raw_response": content,
                "predicted_sql": clean_sql_response(content),
                "generation_ms": elapsed_ms,
                "api_attempts": attempt + 1,
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                "provider_request_id": getattr(response, "id", None),
                "finish_reason": response.choices[0].finish_reason,
            }
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(float(generation["retry_backoff_seconds"]) * (2**attempt))
    return {
        "api_status": "error",
        "api_error": f"{type(last_error).__name__}: {last_error}",
        "raw_response": "",
        "predicted_sql": "",
        "generation_ms": 0.0,
        "api_attempts": attempts,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "provider_request_id": None,
        "finish_reason": None,
    }


def run_generation(
    config: dict[str, Any],
    conditions: list[str],
    limit: int | None,
    question_id: int | None,
    resume: bool,
    db_id: str | None = None,
) -> None:
    client = create_client(config)
    prepared_dir = resolve_path(config["paths"]["prepared_dir"])
    output_dir = resolve_path(config["paths"]["output_dir"])
    pricing = config.get("pricing")

    for condition in conditions:
        prepared_path = prepared_dir / f"{condition}.json"
        if not prepared_path.exists():
            raise FileNotFoundError(f"Run prepare first: {prepared_path}")
        queries = read_json(prepared_path)
        if question_id is not None:
            queries = [item for item in queries if int(item["question_id"]) == question_id]
        if db_id is not None:
            queries = [item for item in queries if item["db_id"] == db_id]
        if limit is not None:
            queries = queries[:limit]
        output_path = output_dir / f"{condition}.jsonl"
        completed = load_completed(output_path) if resume else {}
        if not resume and output_path.exists():
            output_path.unlink()

        for position, item in enumerate(queries, 1):
            qid = int(item["question_id"])
            if qid in completed:
                print(f"[{condition} {position}/{len(queries)}] qid={qid} already complete")
                continue
            print(f"[{condition} {position}/{len(queries)}] qid={qid} generating")
            generated = generate_one(client, config, item["prompt"])
            result = {
                "question_id": qid,
                "db_id": item["db_id"],
                "condition": condition,
                "model": config["generation"]["model"],
                "temperature": config["generation"].get("temperature", "service_default"),
                "top_p": config["generation"].get("top_p", "service_default"),
                "max_output_tokens": config["generation"].get("max_output_tokens", "service_default"),
                "schema_chars": item["schema_chars"],
                "schema_tokens": item["schema_tokens"],
                "schema_token_method": item["token_count_method"],
                **generated,
            }
            if pricing and "input_per_million_tokens" in pricing and "output_per_million_tokens" in pricing:
                input_cost = generated["prompt_tokens"] / 1_000_000 * float(
                    pricing["input_per_million_tokens"]
                )
                output_cost = generated["completion_tokens"] / 1_000_000 * float(
                    pricing["output_per_million_tokens"]
                )
                result.update(
                    {
                        "cost_currency": pricing.get("currency", "CNY"),
                        "input_cost": input_cost,
                        "output_cost": output_cost,
                        "total_cost": input_cost + output_cost,
                    }
                )
            append_jsonl(output_path, result)


def inspect_prompt(config: dict[str, Any], condition: str, question_id: int) -> None:
    path = resolve_path(config["paths"]["prepared_dir"]) / f"{condition}.json"
    if not path.exists():
        raise FileNotFoundError(f"Run prepare first: {path}")
    matches = [item for item in read_json(path) if int(item["question_id"]) == question_id]
    if not matches:
        raise KeyError(f"question_id={question_id} not found in {condition}")
    print(matches[0]["prompt"])


def parse_read_only_sql(sql: str) -> tuple[bool, str | None, str | None]:
    if not sql.strip():
        return False, "EMPTY_SQL", "empty SQL"
    try:
        import sqlglot
        from sqlglot import exp

        expressions = sqlglot.parse(sql, read="sqlite")
        if len(expressions) != 1:
            return False, "MULTIPLE_STATEMENTS", "expected one SQL statement"
        expression = expressions[0]
        prohibited = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter)
        if isinstance(expression, prohibited) or any(expression.find(kind) for kind in prohibited):
            return False, "NON_READ_ONLY_SQL", "write/DDL statement is not allowed"
        if not expression.find(exp.Select) and not isinstance(expression, exp.Select):
            return False, "NON_QUERY_SQL", "statement does not contain SELECT"
        try:
            normalized = expression.sql(dialect="sqlite", normalize=True, pretty=False)
        except TypeError:
            normalized = expression.sql(dialect="sqlite", pretty=False).lower()
        return True, normalized, None
    except Exception as exc:
        return False, "SQL_PARSE_ERROR", f"{type(exc).__name__}: {exc}"


def normalize_cell(value: Any, digits: int) -> Any:
    if value is None:
        return ("null", None)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)):
        return ("number", round(float(value), digits))
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    return ("text", str(value))


def normalize_rows(rows: Iterable[tuple], digits: int) -> list[tuple]:
    return [tuple(normalize_cell(value, digits) for value in row) for row in rows]


def execute_sql(
    database_path: Path,
    sql: str,
    timeout_seconds: float,
    max_rows: int,
) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    deadline = time.perf_counter() + timeout_seconds
    connection.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 10000)
    started = time.perf_counter()
    try:
        cursor = connection.execute(sql)
        rows = cursor.fetchmany(max_rows + 1)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if len(rows) > max_rows:
            return {
                "success": False,
                "category": "RESULT_LIMIT_EXCEEDED",
                "error": f"more than {max_rows} rows",
                "execution_ms": elapsed_ms,
                "rows": [],
            }
        return {
            "success": True,
            "category": "EXECUTABLE",
            "error": None,
            "execution_ms": elapsed_ms,
            "rows": rows,
        }
    except sqlite3.OperationalError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        message = str(exc)
        lower = message.lower()
        if "interrupted" in lower:
            category = "SQL_TIMEOUT"
        elif "no such table" in lower:
            category = "HALLUCINATED_TABLE"
        elif "no such column" in lower or "ambiguous column" in lower:
            category = "HALLUCINATED_COLUMN"
        else:
            category = "SQL_EXECUTION_ERROR"
        return {
            "success": False,
            "category": category,
            "error": message,
            "execution_ms": elapsed_ms,
            "rows": [],
        }
    except Exception as exc:
        return {
            "success": False,
            "category": "SQL_EXECUTION_ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "execution_ms": (time.perf_counter() - started) * 1000,
            "rows": [],
        }
    finally:
        connection.close()


def result_sets_equal(
    gold_rows: list[tuple], predicted_rows: list[tuple], gold_sql: str, digits: int
) -> bool:
    gold = normalize_rows(gold_rows, digits)
    predicted = normalize_rows(predicted_rows, digits)
    if re.search(r"\border\s+by\b", gold_sql, flags=re.I):
        return gold == predicted
    return Counter(gold) == Counter(predicted)


def evaluate_query(
    prepared: dict[str, Any],
    generated: dict[str, Any] | None,
    catalog: SQLiteCatalog,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    base = {
        "question_id": prepared["question_id"],
        "db_id": prepared["db_id"],
        "condition": prepared["condition"],
        "gold_table_covered": bool(prepared["gold_table_covered"]),
        "candidate_count": prepared["candidate_count"],
        "schema_chars": prepared["schema_chars"],
        "schema_tokens": prepared["schema_tokens"],
        "schema_token_method": prepared["token_count_method"],
    }
    if generated is None:
        return {
            **base,
            "category": "MISSING_GENERATION",
            "parse_valid": False,
            "executable": False,
            "execution_correct": False,
            "normalized_exact_match": False,
            "execution_ms": 0.0,
            "generation_ms": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "predicted_sql": "",
            "error": "missing generation output",
        }
    if generated.get("api_status") != "success":
        return {
            **base,
            "category": "API_ERROR",
            "parse_valid": False,
            "executable": False,
            "execution_correct": False,
            "normalized_exact_match": False,
            "execution_ms": 0.0,
            "error": generated.get("api_error"),
            **{key: generated.get(key, 0) for key in ["generation_ms", "prompt_tokens", "completion_tokens", "total_tokens", "total_cost"]},
            "predicted_sql": generated.get("predicted_sql", ""),
        }

    predicted_sql = generated.get("predicted_sql", "")
    parse_valid, normalized_predicted, parse_error = parse_read_only_sql(predicted_sql)
    _, normalized_gold, _ = parse_read_only_sql(prepared["gold_sql"])
    common = {
        **base,
        "generation_ms": generated.get("generation_ms", 0.0),
        "prompt_tokens": generated.get("prompt_tokens", 0),
        "completion_tokens": generated.get("completion_tokens", 0),
        "total_tokens": generated.get("total_tokens", 0),
        "total_cost": generated.get("total_cost", 0.0),
        "predicted_sql": predicted_sql,
        "parse_valid": parse_valid,
        "normalized_exact_match": bool(
            parse_valid and normalized_predicted == normalized_gold
        ),
    }
    if not parse_valid:
        return {
            **common,
            "category": normalized_predicted or "SQL_PARSE_ERROR",
            "executable": False,
            "execution_correct": False,
            "execution_ms": 0.0,
            "error": parse_error,
        }

    database_path = catalog.database_path(prepared["db_id"])
    timeout = float(evaluation["sql_timeout_seconds"])
    max_rows = int(evaluation["max_result_rows"])
    gold_execution = execute_sql(database_path, prepared["gold_sql"], timeout, max_rows)
    if not gold_execution["success"]:
        return {
            **common,
            "category": "GOLD_SQL_EXECUTION_ERROR",
            "executable": False,
            "execution_correct": False,
            "execution_ms": 0.0,
            "error": gold_execution["error"],
        }
    prediction_execution = execute_sql(database_path, predicted_sql, timeout, max_rows)
    if not prediction_execution["success"]:
        return {
            **common,
            "category": prediction_execution["category"],
            "executable": False,
            "execution_correct": False,
            "execution_ms": prediction_execution["execution_ms"],
            "error": prediction_execution["error"],
        }
    correct = result_sets_equal(
        gold_execution["rows"],
        prediction_execution["rows"],
        prepared["gold_sql"],
        int(evaluation["float_round_digits"]),
    )
    return {
        **common,
        "category": "CORRECT" if correct else "VALID_BUT_WRONG_RESULT",
        "executable": True,
        "execution_correct": correct,
        "execution_ms": prediction_execution["execution_ms"],
        "gold_result_rows": len(gold_execution["rows"]),
        "predicted_result_rows": len(prediction_execution["rows"]),
        "error": None,
    }


def aggregate(items: list[dict[str, Any]], full_schema_tokens: dict[int, int]) -> dict[str, Any]:
    count = len(items)
    mean = lambda key: sum(float(item.get(key, 0) or 0) for item in items) / count if count else 0.0
    generation = [float(item.get("generation_ms", 0) or 0) for item in items]
    execution = [float(item.get("execution_ms", 0) or 0) for item in items]
    downstream = [g + e for g, e in zip(generation, execution)]
    reductions = []
    for item in items:
        full = full_schema_tokens.get(int(item["question_id"]), 0)
        if full:
            reductions.append(1 - float(item["schema_tokens"]) / full)
    return {
        "n": count,
        "execution_accuracy": mean("execution_correct"),
        "normalized_exact_match": mean("normalized_exact_match"),
        "parse_valid_rate": mean("parse_valid"),
        "executable_rate": mean("executable"),
        "invalid_sql_rate": 1 - mean("executable"),
        "timeout_rate": sum(item["category"] == "SQL_TIMEOUT" for item in items) / count if count else 0.0,
        "gold_table_coverage_rate": mean("gold_table_covered"),
        "avg_candidate_tables": mean("candidate_count"),
        "avg_schema_chars": mean("schema_chars"),
        "avg_schema_tokens": mean("schema_tokens"),
        "avg_prompt_tokens": mean("prompt_tokens"),
        "avg_completion_tokens": mean("completion_tokens"),
        "avg_total_tokens": mean("total_tokens"),
        "schema_token_reduction_vs_full": statistics.mean(reductions) if reductions else 0.0,
        "avg_cost_per_query": mean("total_cost"),
        "total_cost": sum(float(item.get("total_cost", 0) or 0) for item in items),
        "generation_latency_ms": {
            "mean": statistics.mean(generation) if generation else 0.0,
            "median": statistics.median(generation) if generation else 0.0,
            "p95": percentile(generation, 0.95),
        },
        "execution_latency_ms": {
            "mean": statistics.mean(execution) if execution else 0.0,
            "median": statistics.median(execution) if execution else 0.0,
            "p95": percentile(execution, 0.95),
        },
        "downstream_latency_ms": {
            "mean": statistics.mean(downstream) if downstream else 0.0,
            "median": statistics.median(downstream) if downstream else 0.0,
            "p95": percentile(downstream, 0.95),
        },
        "failure_categories": dict(Counter(item["category"] for item in items)),
    }


def paired_bootstrap(
    left: dict[int, bool], right: dict[int, bool], samples: int, seed: int
) -> dict[str, Any]:
    ids = sorted(set(left) & set(right))
    observed = statistics.mean(float(left[i]) - float(right[i]) for i in ids) if ids else 0.0
    rng = random.Random(seed)
    differences = []
    for _ in range(samples):
        chosen = [rng.choice(ids) for _ in ids]
        differences.append(statistics.mean(float(left[i]) - float(right[i]) for i in chosen))
    return {
        "n": len(ids),
        "difference": observed,
        "ci95": [percentile(differences, 0.025), percentile(differences, 0.975)],
    }


def mcnemar_exact(left: dict[int, bool], right: dict[int, bool]) -> dict[str, Any]:
    ids = sorted(set(left) & set(right))
    left_only = sum(left[i] and not right[i] for i in ids)
    right_only = sum(right[i] and not left[i] for i in ids)
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1)) / (2**discordant)
        p_value = min(1.0, 2 * tail)
    return {
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "discordant": discordant,
        "p_value_two_sided": p_value,
    }


def markdown_main_table(metrics: dict[str, Any]) -> str:
    labels = {"gold": "Gold Tables", "full": "Full Schema", "dense": "SingleDPR (SGPT)", "murre": "MURRE", "gert": "GERT"}
    lines = [
        "# Downstream Text-to-SQL results",
        "",
        # Reviewer-required metrics only: execution accuracy, exact match,
        # invalid-SQL rate, context (token) reduction, token/API cost, latency.
        "| Schema Input | EX | EM | Invalid-SQL | Context Red. | Total Tokens | Cost/query | Latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in ["gold", "full", "dense", "murre", "gert"]:
        if condition not in metrics:
            continue
        value = metrics[condition]
        lines.append(
            f"| {labels[condition]} | {value['execution_accuracy']:.4f} | "
            f"{value['normalized_exact_match']:.4f} | {value['invalid_sql_rate']:.4f} | "
            f"{value['schema_token_reduction_vs_full']:.4f} | "
            f"{value['avg_total_tokens']:.1f} | "
            f"{value['avg_cost_per_query']:.6f} | {value['downstream_latency_ms']['mean']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def markdown_error_table(per_condition: dict[str, list[dict]]) -> str:
    categories = sorted({item["category"] for values in per_condition.values() for item in values})
    lines = ["# Error analysis", "", "| Method | " + " | ".join(categories) + " |", "|---|" + "---:|" * len(categories)]
    for condition in ["gold", "full", "dense", "murre", "gert"]:
        if condition not in per_condition:
            continue
        counts = Counter(item["category"] for item in per_condition[condition])
        lines.append(f"| {condition.upper()} | " + " | ".join(str(counts.get(category, 0)) for category in categories) + " |")
    return "\n".join(lines) + "\n"


def evaluate(
    config: dict[str, Any],
    conditions: list[str],
    limit: int | None = None,
    db_id: str | None = None,
) -> None:
    prepared_dir = resolve_path(config["paths"]["prepared_dir"])
    output_dir = resolve_path(config["paths"]["output_dir"])
    result_dir = resolve_path(config["paths"]["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    catalog = SQLiteCatalog(
        resolve_path(config["paths"]["database_root"]), config.get("table_aliases", {})
    )
    per_condition: dict[str, list[dict]] = {}
    all_per_query: list[dict] = []
    full_prepared_path = prepared_dir / "full.json"
    full_schema_tokens = {
        int(item["question_id"]): int(item["schema_tokens"])
        for item in read_json(full_prepared_path)
    } if full_prepared_path.exists() else {}

    for condition in conditions:
        prepared_path = prepared_dir / f"{condition}.json"
        if not prepared_path.exists():
            print(f"Skipping {condition}: missing {prepared_path}", file=sys.stderr)
            continue
        output_path = output_dir / f"{condition}.jsonl"
        generated = load_completed(output_path)
        values = []
        prepared_items = read_json(prepared_path)
        if db_id is not None:
            prepared_items = [item for item in prepared_items if item["db_id"] == db_id]
        if limit is not None:
            prepared_items = prepared_items[:limit]
        for item in prepared_items:
            result = evaluate_query(
                item, generated.get(int(item["question_id"])), catalog, config["evaluation"]
            )
            values.append(result)
            all_per_query.append(result)
        per_condition[condition] = values

    metrics = {
        condition: aggregate(values, full_schema_tokens)
        for condition, values in per_condition.items()
    }
    significance = {}
    if "gert" in per_condition:
        gert_scores = {int(item["question_id"]): bool(item["execution_correct"]) for item in per_condition["gert"]}
        for baseline in ["dense", "murre"]:
            if baseline not in per_condition:
                continue
            baseline_scores = {int(item["question_id"]): bool(item["execution_correct"]) for item in per_condition[baseline]}
            key = f"gert_vs_{baseline}"
            significance[key] = {
                "paired_bootstrap": paired_bootstrap(
                    gert_scores,
                    baseline_scores,
                    int(config["evaluation"]["bootstrap_samples"]),
                    int(config["evaluation"]["bootstrap_seed"]),
                ),
                "mcnemar_exact": mcnemar_exact(gert_scores, baseline_scores),
            }

    report = {
        "experiment": config["experiment"],
        "generation": config["generation"],
        "metrics": metrics,
        "significance": significance,
        "notes": [
            "Predicted SQL is executed on the sample's recorded db_id; database selection is not evaluated.",
            "Latency excludes the already-completed schema-routing stage.",
        ],
    }
    if "pricing" in config:
        report["pricing"] = config["pricing"]
    write_json(result_dir / "per_query.json", all_per_query)
    write_json(result_dir / "downstream_metrics.json", report)
    (result_dir / "downstream_main_table.md").write_text(markdown_main_table(metrics), encoding="utf-8")
    (result_dir / "error_analysis.md").write_text(markdown_error_table(per_condition), encoding="utf-8")
    print(f"Evaluation written to {result_dir}")


def selected_conditions(config: dict[str, Any], requested: list[str] | None) -> list[str]:
    conditions = requested or list(config["experiment"]["conditions"])
    valid = {"gold", "full", "dense", "murre", "gert"}
    invalid = set(conditions) - valid
    if invalid:
        raise ValueError(f"Unknown conditions: {sorted(invalid)}")
    return conditions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "validate", help="Check local data, schemas, and SQLite files without API calls"
    )

    prepare_parser = subparsers.add_parser("prepare", help="Build canonical schema prompts")
    prepare_parser.add_argument("--conditions", nargs="+")

    inspect_parser = subparsers.add_parser("inspect", help="Print one prepared prompt")
    inspect_parser.add_argument("--condition", required=True)
    inspect_parser.add_argument("--question-id", type=int, required=True)

    run_parser = subparsers.add_parser("run", help="Call the SQL generation model")
    run_parser.add_argument("--conditions", nargs="+")
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--question-id", type=int)
    run_parser.add_argument("--db-id", help="Restrict to one database (e.g. student_club)")
    run_parser.add_argument("--no-resume", action="store_true")
    run_parser.add_argument(
        "--output-dir", type=Path,
        help="Override outputs directory (relative paths are based on the project root)",
    )

    evaluate_parser = subparsers.add_parser("evaluate", help="Execute SQL and aggregate metrics")
    evaluate_parser.add_argument("--conditions", nargs="+")
    evaluate_parser.add_argument("--limit", type=int, help="Evaluate only the first N prepared queries")
    evaluate_parser.add_argument("--db-id", help="Restrict to one database (e.g. student_club)")
    evaluate_parser.add_argument(
        "--output-dir", type=Path,
        help="Read generations from another outputs directory",
    )
    evaluate_parser.add_argument(
        "--result-dir", type=Path,
        help="Write reports to another results directory",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    if getattr(args, "output_dir", None) is not None:
        config["paths"]["output_dir"] = str(args.output_dir)
    if getattr(args, "result_dir", None) is not None:
        config["paths"]["result_dir"] = str(args.result_dir)
    if args.command == "validate":
        validate_project(config)
    elif args.command == "prepare":
        prepare(config, selected_conditions(config, args.conditions))
    elif args.command == "inspect":
        inspect_prompt(config, args.condition, args.question_id)
    elif args.command == "run":
        run_generation(
            config,
            selected_conditions(config, args.conditions),
            args.limit,
            args.question_id,
            not args.no_resume,
            args.db_id,
        )
    elif args.command == "evaluate":
        evaluate(config, selected_conditions(config, args.conditions), args.limit, args.db_id)


if __name__ == "__main__":
    main()
