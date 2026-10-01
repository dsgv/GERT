#!/usr/bin/env python3
"""Compare DIN-SQL table selection with frozen GERT routing on BirdUnion-100."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import statistics
import threading
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import yaml
from prepared_inputs import load_prepared_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = ROOT / "configs" / "dinsql_birdunion.yaml"

from text2sql_experiment import (  # noqa: E402
    SQLiteCatalog,
    execute_sql,
    parse_read_only_sql,
    result_sets_equal,
)


CONDITIONS = ("din_linker", "gert_selector")
CONDITION_LABELS = {
    "din_linker": "DIN-SQL Selector (full union schema input)",
    "gert_selector": "DIN-SQL + GERT (ours)",
}
UPSTREAM_COMMIT = "0474801616130413b024fd008b016a190684649e"
PROTOCOL_VERSION = "dinsql_selector_vs_frozen_gert_k15_v3"
PROMPT_NAMES = (
    "SYSTEM_SCHEMA_LINKING_TEMPLATE",
    "HUMAN_SCHEMA_LINKING_TEMPLATE",
    "SYSTEM_CLASSIFICATION_TEMPLATE",
    "HUMAN_CLASSIFICATION_TEMPLATE",
    "SYSTEM_EASY_CLASS_TEMPLATE",
    "HUMAN_EASY_CLASS_TEMPLATE",
    "SYSTEM_NON_NESTED_CLASS_TEMPLATE",
    "HUMAN_NON_NESTED_CLASS_TEMPLATE",
    "SYSTEM_NESTED_CLASS_TEMPLATE",
    "HUMAN_NESTED_CLASS_TEMPLATE",
    "SYSTEM_SELF_CORRECTION_PROMPT",
    "HUMAN_SELF_CORRECTION_PROMPT",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, value: dict[str, Any], lock: threading.Lock) -> None:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def load_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            rows[int(row["question_id"])] = row
        except Exception as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def report_path(path: Path) -> str:
    """Keep report paths portable when outputs live inside this project."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid YAML root: {path}")
    return value


def chars_div_4(text: str) -> int:
    return math.ceil(len(text) / 4)


def safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def safe_stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


class DefaultParameterLLM:
    """OpenAI-compatible client that sends model and messages only."""

    def __init__(self, config: dict[str, Any]) -> None:
        load_dotenv(ROOT / ".env")
        api_key = (
            os.getenv("DINSQL_API_KEY")
            or os.getenv("TEXT2SQL_API_KEY")
            or os.getenv("API_KEY")
        )
        api_base = (
            os.getenv("DINSQL_API_BASE")
            or os.getenv("TEXT2SQL_API_BASE")
            or os.getenv("API_BASE")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = (
            os.getenv("DINSQL_MODEL")
            or os.getenv("TEXT2SQL_MODEL")
            or os.getenv("MODEL")
            or "deepseek-v3.2"
        )
        if not api_key:
            raise RuntimeError(
                "DINSQL_API_KEY, TEXT2SQL_API_KEY, or API_KEY is required in Text2sql_ex/.env"
            )
        runtime = config["runtime"]
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            max_retries=0,
            timeout=float(runtime.get("request_timeout_seconds", 180)),
        )
        self.retries = int(runtime["api_retries"])
        self.backoff = float(runtime["retry_backoff_seconds"])

    def complete(self, messages: list[dict[str, str]], stage: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            started = time.perf_counter()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
                usage = response.usage
                return {
                    "stage": stage,
                    "content": response.choices[0].message.content or "",
                    "prompt_tokens": int(usage.prompt_tokens) if usage else 0,
                    "completion_tokens": int(usage.completion_tokens) if usage else 0,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "request_id": getattr(response, "id", None),
                    "attempts": attempt,
                }
            except Exception as exc:
                last_error = exc
                if getattr(exc, "status_code", None) in {401, 403}:
                    raise RuntimeError("LLM authentication/authorization failed; check .env") from exc
                if attempt < self.retries:
                    time.sleep(self.backoff * (2 ** (attempt - 1)))
        raise RuntimeError(f"LLM request failed after {self.retries} attempts") from last_error


def token_totals(calls: list[dict[str, Any]]) -> dict[str, int]:
    prompt = sum(int(call.get("prompt_tokens", 0)) for call in calls)
    completion = sum(int(call.get("completion_tokens", 0)) for call in calls)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def canonical_table_map(
    catalog: SQLiteCatalog,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    exact: dict[str, str] = {}
    by_table: dict[str, list[str]] = {}
    for db_dir in sorted(catalog.database_root.iterdir()):
        if not db_dir.is_dir():
            continue
        db_id = db_dir.name
        sqlite_path = db_dir / f"{db_id}.sqlite"
        if not sqlite_path.is_file():
            continue
        for table in catalog.tables(db_id):
            qualified = f"{db_id}.{table}"
            exact[qualified.casefold()] = qualified
            by_table.setdefault(table.casefold(), []).append(qualified)
    return exact, by_table


def load_upstream_prompts(repo: Path) -> dict[str, str]:
    source_path = repo / "DIN-SQL_BIRD.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    prompts: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in PROMPT_NAMES:
            continue
        value = ast.literal_eval(node.value)
        if isinstance(value, str):
            prompts[target.id] = value
    missing = sorted(set(PROMPT_NAMES) - set(prompts))
    if missing:
        raise ValueError(f"Missing DIN-SQL prompt constants: {missing}")
    return prompts


def find_linked_tables(
    response: str,
    exact_tables: dict[str, str],
    tables_by_name: dict[str, list[str]],
) -> list[str]:
    lowered = response.casefold()
    located: list[tuple[int, str]] = []
    for key, canonical in exact_tables.items():
        pattern = rf"(?<![a-z0-9_]){re.escape(key)}(?![a-z0-9_])"
        match = re.search(pattern, lowered)
        if match:
            located.append((match.start(), canonical))
    located.sort(key=lambda item: item[0])
    selected: list[str] = []
    seen: set[str] = set()
    for _, table in located:
        if table.casefold() not in seen:
            seen.add(table.casefold())
            selected.append(table)

    # Tolerate an unqualified table.column only when that table name is globally unique.
    for match in re.finditer(r"(?<![a-z0-9_])([a-z_][a-z0-9_ ]*)\s*\.", lowered):
        table_name = match.group(1).strip().strip("`\"'")
        candidates = tables_by_name.get(table_name, [])
        if len(candidates) == 1 and candidates[0].casefold() not in seen:
            seen.add(candidates[0].casefold())
            selected.append(candidates[0])
    return selected


def extract_schema_links(text: str) -> list[str]:
    matches = re.findall(r"Schema_links\s*:\s*\[(.*?)\]", text, flags=re.I | re.S)
    if not matches:
        return []
    return [item.strip().strip("`\"'") for item in matches[-1].split(",") if item.strip()]


def build_linker_messages(
    prompts: dict[str, str], schema: str, question: str, correction: str = ""
) -> list[dict[str, str]]:
    human = prompts["HUMAN_SCHEMA_LINKING_TEMPLATE"].format(
        schema=schema,
        columns_descriptions="",
        question=question,
        hint="",
    )
    human += """

Additional BirdUnion global protocol:
- The schema may pool independent databases. The nearest `-- Database:` header or
  `Candidate table:` line defines a table's database.
- Write every schema identifier as `database.table.column`.
- Finish with exactly one line `Schema_links: [item1, item2, ...]`.
- Do not assume a target database identifier outside the schema and question.
"""
    if correction:
        human += "\n" + correction
    return [
        {"role": "system", "content": prompts["SYSTEM_SCHEMA_LINKING_TEMPLATE"]},
        {"role": "user", "content": human},
    ]


def extract_label_and_subquestions(text: str) -> tuple[str, list[str], bool]:
    upper = text.upper()
    parsed = True
    if "NON-NESTED" in upper or "NON NESTED" in upper:
        label = "NON-NESTED"
    elif re.search(r"\bEASY\b", upper):
        label = "EASY"
    elif re.search(r"\bNESTED\b", upper):
        label = "NESTED"
    else:
        label = "NESTED"
        parsed = False
    matches = re.findall(r"sub_questions\s*:\s*\[(.*?)\]", text, flags=re.I | re.S)
    subquestions = []
    if matches:
        subquestions = [item.strip().strip("`\"'") for item in matches[-1].split(",") if item.strip()]
    return label, subquestions, parsed


def extract_sql(text: str, revised: bool = False) -> str:
    tag = "Revised_SQL" if revised else "SQL"
    tagged = re.findall(
        rf"(?:^|\n)\s*(?:\*\*)?{tag}(?:\*\*)?\s*:\s*((?:SELECT|WITH)\b[\s\S]*)",
        text,
        flags=re.I,
    )
    if tagged:
        candidate = tagged[-1].strip()
        candidate = re.split(
            r"\n\s*(?:\*\*)?(?:Explanation|Reasoning|Analysis|Step-by-step)(?:\*\*)?\s*:",
            candidate,
            maxsplit=1,
            flags=re.I,
        )[0]
        candidate = candidate.split("```")[0]
        return candidate.strip().strip("`")
    fenced = re.findall(r"```sql\s*(.*?)```", text, flags=re.I | re.S)
    if fenced:
        return fenced[-1].strip()
    # Some compatible models return a bare query, especially during correction.
    # Only accept SELECT/WITH at the beginning of the response (or a line).  A
    # loose word search can otherwise mistake prose such as "the SQL SELECT..."
    # for an executable query and inflate Invalid-SQL.
    bare = re.search(r"(?:\A|\n)\s*((?:SELECT|WITH)\b[\s\S]*)", text, flags=re.I)
    if bare:
        candidate = bare.group(1).strip()
        candidate = re.split(
            r"\n\s*(?:\*\*)?(?:Explanation|Reasoning|Analysis)(?:\*\*)?\s*:",
            candidate,
            maxsplit=1,
            flags=re.I,
        )[0]
        return candidate.split("```")[0].strip().strip("`")
    return ""


def process_query(
    row: dict[str, Any],
    condition: str,
    config: dict[str, Any],
    llm: DefaultParameterLLM,
    catalog: SQLiteCatalog,
    union_schema: str,
    prompts: dict[str, str],
    exact_tables: dict[str, str],
    tables_by_name: dict[str, list[str]],
) -> dict[str, Any]:
    started = time.perf_counter()
    k = int(config["routing"]["k"])
    linker_calls: list[dict[str, Any]] = []
    downstream_calls: list[dict[str, Any]] = []
    linker_fillers: list[str] = []

    if condition == "gert_selector":
        selected_tables = list(row["predicted_schemas_at15"])
        if len(selected_tables) != k or len({x.casefold() for x in selected_tables}) != k:
            raise ValueError(f"GERT question {row['question_id']} does not contain K={k}")
        linker_schema, pre_mappings = catalog.serialize(selected_tables)
        if len(pre_mappings) != k:
            raise ValueError(f"GERT question {row['question_id']} resolved fewer than K={k} tables")
    else:
        # The DIN-SQL schema linker sees the complete union schema.  In the
        # global-schema adaptation, the physical tables named by Schema_links
        # are its selector output and become the candidate schema downstream.
        selected_tables: list[str] = []
        linker_schema = union_schema

    linker_response = ""
    schema_links: list[str] = []
    linked_tables: list[str] = []
    attempts = int(config["routing"]["linker_parse_attempts"])
    correction = ""
    for _ in range(attempts):
        call = llm.complete(
            build_linker_messages(prompts, linker_schema, row["question"], correction),
            "schema_linking",
        )
        linker_calls.append(call)
        linker_response = call["content"]
        schema_links = extract_schema_links(linker_response)
        linked_tables = find_linked_tables(linker_response, exact_tables, tables_by_name)
        if schema_links and (condition == "gert_selector" or linked_tables):
            break
        correction = (
            "The previous response was not parseable. Use fully-qualified identifiers and end "
            "with `Schema_links: [...]`."
        )
    if not schema_links:
        raise ValueError(f"DIN-SQL schema linking was not parseable for question {row['question_id']}")

    if condition == "din_linker":
        if not linked_tables:
            raise ValueError(f"DIN-SQL linker linked no valid table for question {row['question_id']}")
        selected_tables = linked_tables
        schema_text, mappings = catalog.serialize(selected_tables)
        if len(mappings) != len(selected_tables):
            raise ValueError("DIN-SQL selector output contains unresolved tables")
    else:
        schema_text, mappings = catalog.serialize(selected_tables)
        if len(mappings) != k:
            raise ValueError(f"Resolved {len(mappings)} tables; expected K={k}")

    common_values = {
        "schema": schema_text,
        "columns_descriptions": "",
        "question": row["question"],
        "hint": "",
        "schema_links": str(schema_links),
    }
    classification_human = prompts["HUMAN_CLASSIFICATION_TEMPLATE"].format(**common_values)
    classification_human += (
        "\nUse only the candidate schema above. Do not infer a hidden target database."
    )
    classification_call = llm.complete(
        [
            {"role": "system", "content": prompts["SYSTEM_CLASSIFICATION_TEMPLATE"]},
            {"role": "user", "content": classification_human},
        ],
        "classification",
    )
    downstream_calls.append(classification_call)
    label, subquestions, classification_parsed = extract_label_and_subquestions(
        classification_call["content"]
    )

    if label == "EASY":
        system_name, human_name = "SYSTEM_EASY_CLASS_TEMPLATE", "HUMAN_EASY_CLASS_TEMPLATE"
    elif label == "NON-NESTED":
        system_name, human_name = (
            "SYSTEM_NON_NESTED_CLASS_TEMPLATE",
            "HUMAN_NON_NESTED_CLASS_TEMPLATE",
        )
    else:
        system_name, human_name = "SYSTEM_NESTED_CLASS_TEMPLATE", "HUMAN_NESTED_CLASS_TEMPLATE"
    generation_values = {**common_values, "sub_questions": str(subquestions)}
    generation_human = prompts[human_name].format(**generation_values)
    generation_human += (
        "\nThe final SQL must use the `Executable SQLite table name` shown in the schema, "
        "not a database-qualified table name."
    )
    generation_call = llm.complete(
        [
            {"role": "system", "content": prompts[system_name]},
            {"role": "user", "content": generation_human},
        ],
        f"sql_generation:{label}",
    )
    downstream_calls.append(generation_call)
    generated_sql = extract_sql(generation_call["content"])

    correction_values = {
        "schema": schema_text,
        "columns_descriptions": "",
        "question": row["question"],
        "hint": "",
        "sql_query": generated_sql,
    }
    correction_human = prompts["HUMAN_SELF_CORRECTION_PROMPT"].format(**correction_values)
    correction_human += (
        "\nUse only the candidate schema. Return one read-only SQLite query and use executable "
        "physical table names."
    )
    correction_call = llm.complete(
        [
            {"role": "system", "content": prompts["SYSTEM_SELF_CORRECTION_PROMPT"]},
            {"role": "user", "content": correction_human},
        ],
        "self_correction",
    )
    downstream_calls.append(correction_call)
    corrected_sql = extract_sql(correction_call["content"], revised=True)
    predicted_sql = corrected_sql or generated_sql
    if not predicted_sql:
        predicted_sql = "error: No SQL found in DIN-SQL response"

    execution = execute_sql(
        catalog.database_path(row["db_id"]),
        predicted_sql,
        float(config["evaluation"]["sql_timeout_seconds"]),
        int(config["evaluation"]["max_result_rows"]),
    )
    linker_usage = token_totals(linker_calls)
    downstream_usage = token_totals(downstream_calls)
    pipeline_usage = token_totals([*linker_calls, *downstream_calls])
    stage_usage = {
        "schema_linking": token_totals(linker_calls),
        "classification": token_totals([classification_call]),
        "sql_generation": token_totals([generation_call]),
        "self_correction": token_totals([correction_call]),
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "question_id": int(row["question_id"]),
        "db_id": row["db_id"],
        "question": row["question"],
        "gold_sql": row["SQL"],
        "gold_schema": row.get("gold_schema", row.get("schema_link")),
        "condition": condition,
        "schema_scope": (
            "din_selector_from_full_union"
            if condition == "din_linker"
            else "frozen_gert_k15"
        ),
        "schema_table_count": len(selected_tables),
        "model": llm.model,
        "model_parameters": "provider_defaults",
        "gold_db_id_visibility": "execution_and_evaluation_only",
        "evidence": "",
        "selected_tables": selected_tables,
        "linked_tables": linked_tables,
        "linker_fillers": linker_fillers,
        "schema_links": schema_links,
        "linker_response": linker_response,
        "linker_call_count": len(linker_calls),
        "classification": label,
        "classification_parsed": classification_parsed,
        "sub_questions": subquestions,
        "classification_response": classification_call["content"],
        "generation_response": generation_call["content"],
        "generated_sql": generated_sql,
        "correction_response": correction_call["content"],
        "corrected_sql": corrected_sql,
        "predicted_sql": predicted_sql,
        "final_execution": {
            "success": bool(execution["success"]),
            "category": execution["category"],
            "error": execution.get("error"),
            "row_count": len(execution.get("rows", [])),
        },
        "schema_tokens_estimate": chars_div_4(schema_text),
        "linker_usage": linker_usage,
        "downstream_usage": downstream_usage,
        "pipeline_usage": pipeline_usage,
        "stage_usage": stage_usage,
        "request_ids": [
            call.get("request_id") for call in [*linker_calls, *downstream_calls]
        ],
        "wall_time_ms": (time.perf_counter() - started) * 1000,
    }


def selected_conditions(requested: list[str] | None) -> list[str]:
    conditions = requested or list(CONDITIONS)
    invalid = set(conditions) - set(CONDITIONS)
    if invalid:
        raise ValueError(f"Unknown conditions: {sorted(invalid)}")
    return conditions


def filter_rows(
    rows: list[dict[str, Any]], question_ids: list[int] | None, limit: int | None
) -> list[dict[str, Any]]:
    if question_ids:
        wanted = set(question_ids)
        rows = [row for row in rows if int(row["question_id"]) in wanted]
        found = {int(row["question_id"]) for row in rows}
        if found != wanted:
            raise ValueError(f"Unknown question IDs: {sorted(wanted - found)}")
    if limit is not None:
        rows = rows[:limit]
    return rows


def build_run_summary(
    records_by_condition: dict[str, list[dict[str, Any]]],
    expected_count: int,
    model: str,
    output_dir: Path,
) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    for condition, records in records_by_condition.items():
        stage_names = (
            "schema_linking",
            "classification",
            "sql_generation",
            "self_correction",
        )
        stage_totals: dict[str, Any] = {}
        for stage in stage_names:
            usages = [record.get("stage_usage", {}).get(stage, {}) for record in records]
            prompt = sum(int(usage.get("prompt_tokens", 0)) for usage in usages)
            completion = sum(int(usage.get("completion_tokens", 0)) for usage in usages)
            stage_totals[stage] = {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
                "avg_tokens_per_question": (prompt + completion) / len(records)
                if records
                else 0.0,
            }
        prompt_tokens = sum(
            int(record.get("pipeline_usage", {}).get("prompt_tokens", 0))
            for record in records
        )
        completion_tokens = sum(
            int(record.get("pipeline_usage", {}).get("completion_tokens", 0))
            for record in records
        )
        total_tokens = prompt_tokens + completion_tokens
        conditions[condition] = {
            "label": CONDITION_LABELS[condition],
            "completed_questions": len(records),
            "expected_questions": expected_count,
            "complete": len(records) == expected_count,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "avg_prompt_tokens_per_question": prompt_tokens / len(records) if records else 0.0,
            "avg_completion_tokens_per_question": completion_tokens / len(records)
            if records
            else 0.0,
            "avg_total_tokens_per_question": total_tokens / len(records) if records else 0.0,
            "avg_wall_time_seconds": safe_mean(
                [float(record.get("wall_time_ms", 0)) / 1000 for record in records]
            ),
            "api_calls": sum(int(record.get("linker_call_count", 0)) + 3 for record in records),
            "stage_usage": stage_totals,
        }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "run_name": output_dir.parent.name if output_dir.name == "outputs" else output_dir.name,
        "output_dir": report_path(output_dir),
        "model": model,
        "model_parameters": "provider_defaults",
        "token_definition": (
            "Provider-reported prompt + completion tokens across schema linking, "
            "classification, SQL generation, and self-correction"
        ),
        "conditions": conditions,
    }


def write_run_summary(summary: dict[str, Any], output_dir: Path) -> None:
    write_json(output_dir / "run_summary.json", summary)
    lines = [
        "# DIN-SQL run token summary",
        "",
        f"- Protocol: `{summary['protocol_version']}`",
        f"- Run: `{summary['run_name']}`",
        f"- Model: `{summary['model']}` (provider defaults)",
        "",
        "| Condition | Completed | Prompt Tokens | Completion Tokens | Total Tokens | Avg Tokens / Question | Avg Wall Time |",
        "| :-- | --: | --: | --: | --: | --: | --: |",
    ]
    for condition in CONDITIONS:
        if condition not in summary["conditions"]:
            continue
        metric = summary["conditions"][condition]
        lines.append(
            f"| {metric['label']} | {metric['completed_questions']}/{metric['expected_questions']} | "
            f"{metric['prompt_tokens']} | {metric['completion_tokens']} | "
            f"{metric['total_tokens']} | {metric['avg_total_tokens_per_question']:.2f} | "
            f"{metric['avg_wall_time_seconds']:.2f}s |"
        )
    lines.extend(
        [
            "",
            "`Total Tokens` includes every successful LLM call in all four DIN-SQL stages.",
        ]
    )
    (output_dir / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(config: dict[str, Any]) -> None:
    paths = config["paths"]
    dataset = load_prepared_dataset(resolve(paths["dataset"]))
    if len(dataset) != 100:
        raise ValueError(f"Expected BirdUnion-100, found {len(dataset)} rows")
    k = int(config["routing"]["k"])
    for row in dataset:
        route = list(row["predicted_schemas_at15"])
        if len(route) != k or len({x.casefold() for x in route}) != k:
            raise ValueError(f"Question {row['question_id']} does not have unique K={k}")
    repo = resolve(paths["dinsql_repo"])
    prompts = load_upstream_prompts(repo)
    catalog = SQLiteCatalog(resolve(paths["database_root"]), config.get("table_aliases", {}))
    exact, _ = canonical_table_map(catalog)
    for row in dataset:
        _, mappings = catalog.serialize(list(row["predicted_schemas_at15"]))
        if len(mappings) != k:
            raise ValueError(f"Question {row['question_id']} route does not resolve to K={k}")
    union_schema = resolve(paths["union_schema"]).read_text(encoding="utf-8")
    if "-- Database:" not in union_schema:
        raise ValueError("Union schema lacks database identity headers")
    print("DIN-SQL BirdUnion adapter validation passed")
    print(f"Questions: {len(dataset)}; K={k}; physical tables={len(exact)}")
    print(f"Upstream commit: {UPSTREAM_COMMIT}; prompt constants={len(prompts)}")
    print(f"Protocol version: {PROTOCOL_VERSION}")
    print("Protocol: DIN selector from full union schema vs frozen GERT K=15; gold db_id hidden; evidence empty")
    print("Pipeline: schema linking -> classification -> generation -> self-correction")
    print("Model sampling parameters: provider defaults")


def run(
    config: dict[str, Any],
    conditions: list[str],
    output_dir_override: Path | None,
    workers: int | None,
    question_ids: list[int] | None,
    limit: int | None,
) -> None:
    paths = config["paths"]
    rows = filter_rows(load_prepared_dataset(resolve(paths["dataset"])), question_ids, limit)
    output_dir = resolve(output_dir_override or paths["output_dir"])
    catalog = SQLiteCatalog(resolve(paths["database_root"]), config.get("table_aliases", {}))
    union_schema = resolve(paths["union_schema"]).read_text(encoding="utf-8")
    prompts = load_upstream_prompts(resolve(paths["dinsql_repo"]))
    exact, by_name = canonical_table_map(catalog)
    llm = DefaultParameterLLM(config)
    worker_count = int(workers or config["runtime"]["workers"])
    for known_condition in CONDITIONS:
        existing_path = output_dir / f"{known_condition}.jsonl"
        existing = load_jsonl(existing_path)
        incompatible = sorted(
            qid
            for qid, record in existing.items()
            if record.get("protocol_version") != PROTOCOL_VERSION
        )
        if incompatible:
            raise ValueError(
                f"{output_dir} contains incompatible records in {existing_path.name}. "
                "Use a new per-run output directory."
            )
    for condition in conditions:
        output_path = output_dir / f"{condition}.jsonl"
        completed = load_jsonl(output_path)
        pending = [row for row in rows if int(row["question_id"]) not in completed]
        print(
            f"[{condition}] total={len(rows)}, resumed={len(rows)-len(pending)}, "
            f"pending={len(pending)}, workers={worker_count}"
        )
        lock = threading.Lock()
        failed: list[tuple[int, str]] = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    process_query,
                    row,
                    condition,
                    config,
                    llm,
                    catalog,
                    union_schema,
                    prompts,
                    exact,
                    by_name,
                ): row
                for row in pending
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=condition):
                row = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    qid = int(row["question_id"])
                    failed.append((qid, f"{type(exc).__name__}: {exc}"))
                    tqdm.write(
                        f"[{condition}] question {qid} failed; left pending for next resume run"
                    )
                    continue
                completed[int(result["question_id"])] = result
                append_jsonl(output_path, result, lock)
        if failed:
            print(
                f"[{condition}] transient failures={len(failed)}; "
                f"pending_ids={[qid for qid, _ in failed]}"
            )
        expected = {int(row["question_id"]) for row in rows}
        if expected <= set(completed):
            ordered = [completed[int(row["question_id"])] for row in rows]
            write_json(output_dir / f"{condition}.json", ordered)
            print(f"Wrote {output_dir / f'{condition}.json'}")
        records_by_condition: dict[str, list[dict[str, Any]]] = {}
        for known_condition in CONDITIONS:
            known = load_jsonl(output_dir / f"{known_condition}.jsonl")
            compatible_records = [
                known[qid]
                for qid in sorted(expected & set(known))
                if known[qid].get("protocol_version") == PROTOCOL_VERSION
            ]
            if compatible_records:
                records_by_condition[known_condition] = compatible_records
        summary = build_run_summary(records_by_condition, len(rows), llm.model, output_dir)
        write_run_summary(summary, output_dir)
        print(f"Wrote token summary to {output_dir / 'run_summary.json'}")


def official_bird_equal(gold_rows: list[tuple], predicted_rows: list[tuple]) -> bool:
    """Match the official DIN-SQL/BIRD repository evaluator: set(row) equality."""
    return set(gold_rows) == set(predicted_rows)


def evaluate_condition(
    source_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    catalog: SQLiteCatalog,
    config: dict[str, Any],
    full_schema_chars: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_id = {int(row["question_id"]): row for row in predictions}
    expected = {int(row["question_id"]) for row in source_rows}
    if set(by_id) != expected:
        raise ValueError("Prediction question IDs do not exactly match requested source rows")
    evaluation = config["evaluation"]
    per_query: list[dict[str, Any]] = []
    for source in source_rows:
        predicted = by_id[int(source["question_id"])]
        sql = predicted["predicted_sql"]
        parse_valid, normalized_predicted, parse_error = parse_read_only_sql(sql)
        _, normalized_gold, _ = parse_read_only_sql(source["SQL"])
        executable = False
        bird_correct = False
        strict_correct = False
        category = normalized_predicted if not parse_valid else "SQL_EXECUTION_ERROR"
        error = parse_error
        gold_rows: list[tuple] = []
        predicted_rows: list[tuple] = []
        if parse_valid:
            db_path = catalog.database_path(source["db_id"])
            gold_result = execute_sql(
                db_path,
                source["SQL"],
                float(evaluation["sql_timeout_seconds"]),
                int(evaluation["max_result_rows"]),
            )
            pred_result = execute_sql(
                db_path,
                sql,
                float(evaluation["sql_timeout_seconds"]),
                int(evaluation["max_result_rows"]),
            )
            executable = bool(pred_result["success"])
            error = pred_result.get("error")
            category = pred_result.get("category", "SQL_EXECUTION_ERROR")
            if gold_result["success"] and pred_result["success"]:
                gold_rows = gold_result["rows"]
                predicted_rows = pred_result["rows"]
                bird_correct = official_bird_equal(gold_rows, predicted_rows)
                strict_correct = result_sets_equal(
                    gold_rows,
                    predicted_rows,
                    source["SQL"],
                    int(evaluation["float_round_digits"]),
                )
                category = "CORRECT" if bird_correct else "VALID_BUT_WRONG_RESULT"
        gold_tables = {table.casefold() for table in source.get("gold_schema", source.get("schema_link", []))}
        selected = {table.casefold() for table in predicted["selected_tables"]}
        per_query.append(
            {
                "question_id": int(source["question_id"]),
                "db_id": source["db_id"],
                "condition": predicted["condition"],
                "predicted_sql": sql,
                "parse_valid": parse_valid,
                "executable": executable,
                "bird_execution_correct": bird_correct,
                "strict_execution_correct": strict_correct,
                "normalized_exact_match": bool(
                    parse_valid and normalized_predicted == normalized_gold
                ),
                "category": category,
                "error": error,
                "gold_result_rows": len(gold_rows),
                "predicted_result_rows": len(predicted_rows),
                # End-to-end context reduction counts the selector call too.
                # DIN reads the full union schema in that call, so its headline
                # Context Red. is zero even though downstream stages are smaller.
                "context_reduction": (
                    0.0
                    if predicted["condition"] == "din_linker"
                    else 1 - len(source["schema_text"]) / full_schema_chars
                ),
                "schema_tokens_estimate": predicted["schema_tokens_estimate"],
                "selected_table_count": len(predicted["selected_tables"]),
                "linker_tokens": predicted["linker_usage"]["total_tokens"],
                "downstream_tokens": predicted["downstream_usage"]["total_tokens"],
                "pipeline_tokens": predicted["pipeline_usage"]["total_tokens"],
                "gold_table_complete": gold_tables <= selected,
                "linker_fill_count": len(predicted.get("linker_fillers", [])),
                "linker_call_count": int(predicted.get("linker_call_count", 0)),
                "classification": predicted.get("classification"),
                "classification_parsed": bool(predicted.get("classification_parsed")),
            }
        )
    metrics = {
        "n": len(per_query),
        "bird_execution_accuracy": safe_mean(
            [float(x["bird_execution_correct"]) for x in per_query]
        ),
        "context_reduction": safe_mean([x["context_reduction"] for x in per_query]),
        "avg_pipeline_tokens": safe_mean([x["pipeline_tokens"] for x in per_query]),
    }
    return metrics, per_query


def evaluate_all(
    config: dict[str, Any],
    conditions: list[str],
    output_dir_override: Path | None,
    result_dir_override: Path | None,
    question_ids: list[int] | None,
    limit: int | None,
) -> None:
    paths = config["paths"]
    source_rows = filter_rows(load_prepared_dataset(resolve(paths["dataset"])), question_ids, limit)
    output_dir = resolve(output_dir_override or paths["output_dir"])
    result_dir = resolve(result_dir_override or paths["result_dir"])
    catalog = SQLiteCatalog(resolve(paths["database_root"]), config.get("table_aliases", {}))
    full_schema_chars = len(resolve(paths["union_schema"]).read_text(encoding="utf-8"))
    full_schema_tokens = math.ceil(full_schema_chars / 4)
    all_metrics: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for condition in conditions:
        json_path = output_dir / f"{condition}.json"
        predictions = (
            read_json(json_path)
            if json_path.exists()
            else list(load_jsonl(output_dir / f"{condition}.jsonl").values())
        )
        incompatible = sorted(
            int(predicted["question_id"])
            for predicted in predictions
            if predicted.get("protocol_version") != PROTOCOL_VERSION
        )
        if incompatible:
            raise ValueError(
                "Predictions use an incompatible DIN-SQL protocol for question IDs: "
                f"{incompatible[:10]}. Evaluate a v2 per-run output directory instead."
            )
        if condition == "din_linker":
            invalid_selector = sorted(
                int(predicted["question_id"])
                for predicted in predictions
                if predicted.get("schema_scope") != "din_selector_from_full_union"
                or not predicted.get("selected_tables")
                or int(predicted.get("schema_tokens_estimate", full_schema_tokens))
                >= full_schema_tokens
            )
            if invalid_selector:
                raise ValueError(
                    "DIN-SQL baseline did not pass its selector output downstream for "
                    f"question IDs: {invalid_selector[:10]}"
                )
        if condition == "gert_selector":
            source_by_id = {int(row["question_id"]): row for row in source_rows}
            route_mismatches = [
                int(predicted["question_id"])
                for predicted in predictions
                if int(predicted["question_id"]) in source_by_id
                and predicted.get("selected_tables")
                != source_by_id[int(predicted["question_id"])]["predicted_schemas_at15"]
            ]
            if route_mismatches:
                raise ValueError(
                    "GERT selected_tables differs from frozen predicted_schemas_at15 for "
                    f"question IDs: {sorted(route_mismatches)}"
                )
        metrics, per_query = evaluate_condition(
            source_rows, predictions, catalog, config, full_schema_chars
        )
        all_metrics[condition] = metrics
        all_rows.extend(per_query)

    labels = CONDITION_LABELS
    lines = [
        "# DIN-SQL schema-linking replacement on BirdUnion-100",
        "",
        "| Schema Input | BIRD EX | Context Red. | Tokens/Query |",
        "| :-- | --: | --: | --: |",
    ]
    for condition in conditions:
        metric = all_metrics[condition]
        lines.append(
            f"| {labels[condition]} | {metric['bird_execution_accuracy']:.2f} | "
            f"{metric['context_reduction']:.2%} | {metric['avg_pipeline_tokens']:.0f} |"
        )

    report = {
        "experiment": config["experiment"],
        "protocol": {
            "protocol_version": PROTOCOL_VERSION,
            "upstream_commit": UPSTREAM_COMMIT,
            "k": int(config["routing"]["k"]),
            "din_schema_input": (
                "full union schema for schema linking; tables named by Schema_links "
                "are serialized and passed to classification, generation, and self-correction"
            ),
            "gert_schema_input": "frozen predicted_schemas_at15 in all four LLM stages",
            "evidence": "empty",
            "gold_db_id": "hidden from prompts; execution/evaluation only",
            "pipeline": "schema linking -> classification -> generation -> self-correction",
            "total_tokens": "provider-reported prompt + completion tokens across all four stages",
            "context_reduction": (
                "end-to-end: zero for DIN because its selector reads the full union schema; "
                "GERT is measured against the full union schema"
            ),
            "bird_ex": "set(predicted_rows) == set(gold_rows), matching upstream evaluator",
            "gert_routing_cost": "excluded because GERT predictions are frozen inputs",
            "model_parameters": "provider defaults; no sampling parameters sent",
        },
        "metrics": all_metrics,
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    write_json(result_dir / "metrics.json", report)
    write_json(result_dir / "per_query.json", all_rows)
    (result_dir / "main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    round_summary = {
        "protocol_version": PROTOCOL_VERSION,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "output_dir": report_path(output_dir),
        "result_dir": report_path(result_dir),
        "metrics": all_metrics,
    }
    write_json(result_dir / "round_summary.json", round_summary)
    (result_dir / "round_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))
    print(f"Wrote reports to {result_dir}")


def aggregate_runs(
    result_dirs: list[Path], output_dir: Path, conditions: list[str]
) -> None:
    if len(result_dirs) < 2:
        raise ValueError("Aggregate requires at least two completed result directories")
    resolved_dirs = [resolve(path) for path in result_dirs]
    reports: list[dict[str, Any]] = []
    for result_dir in resolved_dirs:
        path = result_dir / "metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing per-round metrics: {path}")
        report = read_json(path)
        if report.get("protocol", {}).get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError(f"Incompatible protocol in {path}")
        if not set(conditions) <= set(report.get("metrics", {})):
            raise ValueError(f"Requested conditions are missing in {path}")
        reports.append(report)

    metric_names = (
        "bird_execution_accuracy",
        "context_reduction",
        "avg_pipeline_tokens",
    )
    aggregate: dict[str, Any] = {}
    for condition in conditions:
        condition_metrics: dict[str, Any] = {}
        for metric_name in metric_names:
            values = [float(report["metrics"][condition][metric_name]) for report in reports]
            condition_metrics[metric_name] = {
                "values": values,
                "mean": safe_mean(values),
                "sample_std": safe_stdev(values),
            }
        aggregate[condition] = condition_metrics

    output_dir = resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "round_count": len(reports),
        "result_dirs": [report_path(path) for path in resolved_dirs],
        "std_definition": "sample standard deviation (n-1)",
        "metrics": aggregate,
    }
    write_json(output_dir / "aggregate_metrics.json", payload)

    lines = [
        f"# DIN-SQL {len(reports)}-round mean ± std",
        "",
        "| Schema Input | BIRD EX | Context Red. | Tokens/Query |",
        "| :-- | --: | --: | --: |",
    ]
    for condition in conditions:
        metric = aggregate[condition]
        bird = metric["bird_execution_accuracy"]
        context = metric["context_reduction"]
        tokens = metric["avg_pipeline_tokens"]
        lines.append(
            f"| {CONDITION_LABELS[condition]} | {bird['mean']:.3f} ± {bird['sample_std']:.3f} | "
            f"{context['mean']:.2%} ± {context['sample_std']:.2%} | "
            f"{tokens['mean']:.0f} ± {tokens['sample_std']:.0f} |"
        )
    (output_dir / "main_table_mean_std.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))
    print(f"Wrote aggregate reports to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="Offline protocol and path validation")
    run_parser = subparsers.add_parser("run", help="Run one or both routing conditions")
    run_parser.add_argument("--conditions", nargs="+", choices=CONDITIONS)
    run_parser.add_argument("--output-dir", type=Path)
    run_parser.add_argument("--workers", type=int)
    run_parser.add_argument("--question-ids", nargs="+", type=int)
    run_parser.add_argument("--limit", type=int)
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate completed outputs")
    eval_parser.add_argument("--conditions", nargs="+", choices=CONDITIONS)
    eval_parser.add_argument("--output-dir", type=Path)
    eval_parser.add_argument("--result-dir", type=Path)
    eval_parser.add_argument("--question-ids", nargs="+", type=int)
    eval_parser.add_argument("--limit", type=int)
    aggregate_parser = subparsers.add_parser(
        "aggregate", help="Aggregate completed rounds as mean +/- sample std"
    )
    aggregate_parser.add_argument("--result-dirs", nargs="+", type=Path, required=True)
    aggregate_parser.add_argument("--conditions", nargs="+", choices=CONDITIONS)
    aggregate_parser.add_argument(
        "--output-dir", type=Path, default=Path("dinsql_runs/aggregate")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config.resolve())
    if args.command == "validate":
        validate(config)
    elif args.command == "run":
        run(
            config,
            selected_conditions(args.conditions),
            args.output_dir,
            args.workers,
            args.question_ids,
            args.limit,
        )
    elif args.command == "evaluate":
        evaluate_all(
            config,
            selected_conditions(args.conditions),
            args.output_dir,
            args.result_dir,
            args.question_ids,
            args.limit,
        )
    elif args.command == "aggregate":
        aggregate_runs(
            args.result_dirs,
            args.output_dir,
            selected_conditions(args.conditions),
        )


if __name__ == "__main__":
    main()
