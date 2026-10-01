#!/usr/bin/env python3
"""Compare MAC-SQL's adapted global Selector with frozen GERT routing on BirdUnion."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
import re
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.yaml"
sys.path.insert(0, str(ROOT))

from core.const import (  # noqa: E402
    decompose_template_bird,
    refiner_template,
    selector_template,
)
from text2sql_experiment import (  # noqa: E402
    SQLiteCatalog,
    execute_sql,
    parse_read_only_sql,
    result_sets_equal,
)


CONDITIONS = ("mac_selector", "gert_selector")
PROTOCOL_VERSION = "macsql_selector_vs_frozen_gert_k15_v2"


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


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML root: {path}")
    return config


def chars_div_4(text: str) -> int:
    return math.ceil(len(text) / 4)


class DefaultParameterLLM:
    """OpenAI-compatible client that omits every model sampling parameter."""

    def __init__(self, config: dict[str, Any]) -> None:
        load_dotenv(ROOT / ".env")
        api_key = os.getenv("MACSQL_API_KEY") or os.getenv("TEXT2SQL_API_KEY")
        api_base = (
            os.getenv("MACSQL_API_BASE")
            or os.getenv("TEXT2SQL_API_BASE")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = os.getenv("MACSQL_MODEL") or os.getenv("TEXT2SQL_MODEL") or "deepseek-v3.2"
        if not api_key:
            raise RuntimeError(
                "MACSQL_API_KEY or TEXT2SQL_API_KEY is required in MAC-SQL/.env"
            )
        self.client = OpenAI(api_key=api_key, base_url=api_base, max_retries=0)
        runtime = config["runtime"]
        self.retries = int(runtime["api_retries"])
        self.backoff = float(runtime["retry_backoff_seconds"])

    def complete(self, prompt: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            started = time.perf_counter()
            try:
                # Per experiment protocol, do not set temperature, top_p,
                # max_tokens, seed, or any other model generation parameter.
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                usage = response.usage
                return {
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


def parse_json_object(text: str) -> dict[str, Any]:
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.I | re.S)
    candidates = [fenced.group(1)] if fenced else []
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    raise ValueError("Selector response does not contain a valid JSON object")


def routing_tokens(text: str) -> set[str]:
    expanded = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return set(re.findall(r"[a-z0-9]+", expanded.casefold().replace("_", " ")))


def canonical_table_map(
    catalog: SQLiteCatalog,
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, set[str]]]:
    exact: dict[str, str] = {}
    by_table: dict[str, list[str]] = {}
    search_tokens: dict[str, set[str]] = {}
    for db_dir in sorted(catalog.database_root.iterdir()):
        if not db_dir.is_dir():
            continue
        db_id = db_dir.name
        sqlite_path = db_dir / f"{db_id}.sqlite"
        if not sqlite_path.is_file():
            continue
        connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
        try:
            for table in catalog.tables(db_id):
                qualified = f"{db_id}.{table}"
                exact[qualified.casefold()] = qualified
                by_table.setdefault(table.casefold(), []).append(qualified)
                quoted = catalog.quote_identifier(table)
                columns = [
                    str(row[1])
                    for row in connection.execute(f"PRAGMA table_info({quoted})").fetchall()
                ]
                search_tokens[qualified] = routing_tokens(
                    " ".join([db_id, table, *columns])
                )
        finally:
            connection.close()
    return exact, by_table, search_tokens


def normalize_selector_tables(
    response: str,
    exact: dict[str, str],
    by_table: dict[str, list[str]],
) -> list[str]:
    value = parse_json_object(response)
    selected: list[str] = []
    seen: set[str] = set()
    for raw_key, column_choice in value.items():
        if isinstance(column_choice, str) and column_choice.casefold() == "drop_all":
            continue
        key = str(raw_key).strip().strip("`\"'")
        canonical = exact.get(key.casefold())
        if canonical is None and "." not in key:
            matches = by_table.get(key.casefold(), [])
            if len(matches) == 1:
                canonical = matches[0]
        if canonical is None or canonical.casefold() in seen:
            continue
        seen.add(canonical.casefold())
        selected.append(canonical)
    return selected


def normalize_fixed_budget(
    selected: list[str],
    question: str,
    search_tokens: dict[str, set[str]],
    k: int,
) -> tuple[list[str], list[str]]:
    selected = selected[:k]
    seen = {table.casefold() for table in selected}
    query_tokens = routing_tokens(question)
    ranked_fallback = sorted(
        (table for table in search_tokens if table.casefold() not in seen),
        key=lambda table: (
            -len(query_tokens & search_tokens[table]),
            table.casefold(),
        ),
    )
    fillers = ranked_fallback[: max(0, k - len(selected))]
    normalized = [*selected, *fillers]
    if len(normalized) != k:
        raise ValueError(f"Could not normalize Selector output to K={k}")
    return normalized, fillers


def build_selector_prompt(union_schema: str, question: str, k: int) -> str:
    fixed_budget_template = selector_template.replace(
        "Ensure that at least 3 tables are included in the final output JSON.",
        f"Ensure that exactly {k} tables are included in the final output JSON.",
    )
    prompt = fixed_budget_template.format(
        db_id="BirdUnion_global_pool",
        query=question,
        evidence="",
        desc_str=union_schema,
        fk_str="Foreign-key clauses are included in the CREATE TABLE statements above.",
    )
    protocol = f"""

Additional BirdUnion fixed-budget protocol (required for this experiment):
1. The schema above pools many databases. A table identity is `database.table`,
   where `database` comes from the nearest `-- Database:` header.
2. Return exactly {k} relevant unique tables, ranked from most to least relevant.
3. JSON keys must be fully-qualified `database.table` identities. Do not emit
   irrelevant tables with `drop_all`; emit only the selected {k} tables.
4. JSON values must still follow the original MAC-SQL Selector protocol:
   `keep_all` or a relevant-column list. Output JSON only.
"""
    marker = "\u3010Answer\u3011"
    position = prompt.rfind(marker)
    return prompt[:position] + protocol + "\n" + prompt[position:] if position >= 0 else prompt + protocol


def sql_from_mac_response(text: str) -> str:
    matches = re.findall(r"```sql\s*(.*?)```", text, flags=re.I | re.S)
    return matches[-1].strip() if matches else "error: No SQL found in the input string"


def prefixed_sql(sql: str) -> str:
    return "\n".join(f"-- {line}" for line in sql.splitlines())


def needs_refinement(execution: dict[str, Any]) -> bool:
    if not execution["success"]:
        return True
    rows = execution.get("rows", [])
    if not rows:
        return True
    return any(value is None for row in rows for value in row)


def token_totals(calls: list[dict[str, Any]]) -> dict[str, int]:
    prompt = sum(int(call.get("prompt_tokens", 0)) for call in calls)
    completion = sum(int(call.get("completion_tokens", 0)) for call in calls)
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def process_query(
    row: dict[str, Any],
    condition: str,
    config: dict[str, Any],
    llm: DefaultParameterLLM,
    catalog: SQLiteCatalog,
    union_schema: str,
    exact_tables: dict[str, str],
    tables_by_name: dict[str, list[str]],
    table_search_tokens: dict[str, set[str]],
) -> dict[str, Any]:
    started = time.perf_counter()
    k = int(config["routing"]["k"])
    selector_calls: list[dict[str, Any]] = []
    selector_response = ""
    selector_fillers: list[str] = []
    selector_llm_table_count = 0
    selector_below_minimum = False
    if condition == "gert_selector":
        selected_tables = list(row["predicted_schemas_at15"])
        if len(selected_tables) != k or len({x.casefold() for x in selected_tables}) != k:
            raise ValueError(f"GERT question {row['question_id']} does not contain exactly K={k} tables")
    else:
        selector_prompt = build_selector_prompt(union_schema, row["question"], k)
        last_error: Exception | None = None
        best_tables: list[str] = []
        best_response = ""
        minimum = int(config["routing"]["selector_min_llm_tables"])
        for _ in range(int(config["routing"]["selector_format_attempts"])):
            call = llm.complete(selector_prompt)
            selector_calls.append(call)
            try:
                llm_tables = normalize_selector_tables(
                    call["content"], exact_tables, tables_by_name
                )
                if len(llm_tables) > len(best_tables):
                    best_tables = llm_tables
                    best_response = call["content"]
                if len(llm_tables) < minimum:
                    last_error = ValueError(
                        f"Selector returned only {len(llm_tables)} valid tables; minimum is {minimum}"
                    )
                    selector_prompt += (
                        "\nYour previous response selected too few valid tables. "
                        f"Return exactly {k} valid unique fully-qualified tables as one JSON object."
                    )
                    continue
                selector_response = call["content"]
                selector_llm_table_count = len(llm_tables)
                selected_tables, selector_fillers = normalize_fixed_budget(
                    llm_tables, row["question"], table_search_tokens, k
                )
                break
            except Exception as exc:
                last_error = exc
                selector_prompt += (
                    "\nYour previous response violated the fixed-budget protocol. "
                    f"Return exactly {k} valid unique fully-qualified tables as one JSON object."
                )
        else:
            if not best_tables:
                raise ValueError(f"Invalid MAC-SQL Selector output: {last_error}")
            selector_response = best_response
            selector_llm_table_count = len(best_tables)
            selector_below_minimum = selector_llm_table_count < minimum
            selected_tables, selector_fillers = normalize_fixed_budget(
                best_tables, row["question"], table_search_tokens, k
            )

    schema_text, mappings = catalog.serialize(selected_tables)
    if len(mappings) != k:
        raise ValueError(f"Resolved {len(mappings)} schema tables; expected {k}")
    fk_text = "Foreign-key constraints between selected tables are included in the DDL above."
    decomposer_prompt = decompose_template_bird.format(
        desc_str=schema_text,
        fk_str=fk_text,
        query=row["question"],
        evidence="",
    )
    decomposer_call = llm.complete(decomposer_prompt)
    initial_sql = sql_from_mac_response(decomposer_call["content"])

    evaluation = config["evaluation"]
    db_path = catalog.database_path(row["db_id"])
    initial_execution = execute_sql(
        db_path,
        initial_sql,
        float(evaluation["sql_timeout_seconds"]),
        int(evaluation["max_result_rows"]),
    )
    refiner_calls: list[dict[str, Any]] = []
    final_sql = initial_sql
    if "error" not in initial_sql.casefold() and needs_refinement(initial_execution):
        refiner_prompt = refiner_template.format(
            query=row["question"],
            evidence="",
            desc_str=schema_text,
            fk_str=fk_text,
            sql=prefixed_sql(initial_sql),
            sqlite_error=initial_execution.get("error") or "no data selected",
            exception_class=initial_execution.get("category", ""),
        )
        refiner_call = llm.complete(refiner_prompt)
        refiner_calls.append(refiner_call)
        final_sql = sql_from_mac_response(refiner_call["content"])

    downstream_calls = [decomposer_call, *refiner_calls]
    all_calls = [*selector_calls, *downstream_calls]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "question_id": int(row["question_id"]),
        "db_id": row["db_id"],
        "question": row["question"],
        "gold_sql": row["SQL"],
        "gold_schema": row.get("gold_schema", row.get("schema_link")),
        "condition": condition,
        "model": llm.model,
        "model_parameters": "provider_defaults",
        "selected_tables": selected_tables,
        "selector_llm_table_count": selector_llm_table_count,
        "selector_fillers": selector_fillers,
        "selector_call_count": len(selector_calls),
        "selector_below_minimum": selector_below_minimum,
        "candidate_count": len(mappings),
        "schema_text": schema_text,
        "schema_chars": len(schema_text),
        "schema_tokens_estimate": chars_div_4(schema_text),
        "selector_response": selector_response,
        "decomposer_response": decomposer_call["content"],
        "refiner_responses": [call["content"] for call in refiner_calls],
        "initial_sql": initial_sql,
        "predicted_sql": final_sql,
        "refiner_called": bool(refiner_calls),
        "selector_usage": token_totals(selector_calls),
        "downstream_usage": token_totals(downstream_calls),
        "pipeline_usage": token_totals(all_calls),
        "selector_latency_ms": sum(call["latency_ms"] for call in selector_calls),
        "downstream_latency_ms": sum(call["latency_ms"] for call in downstream_calls),
        "pipeline_latency_ms": sum(call["latency_ms"] for call in all_calls),
        "request_ids": [call.get("request_id") for call in all_calls],
        "wall_time_ms": (time.perf_counter() - started) * 1000,
    }


def selected_conditions(requested: list[str] | None) -> list[str]:
    conditions = requested or list(CONDITIONS)
    invalid = set(conditions) - set(CONDITIONS)
    if invalid:
        raise ValueError(f"Unknown conditions: {sorted(invalid)}")
    return conditions


def validate(config: dict[str, Any]) -> None:
    paths = config["paths"]
    required = ["dataset", "database_root", "union_schema", "macsql_repo"]
    resolved = {key: resolve(ROOT, paths[key]) for key in required}
    for key, path in resolved.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing paths.{key}: {path}")
    for name in ("core/const.py", "README.md", ".git"):
        if not (resolved["macsql_repo"] / name).exists():
            raise FileNotFoundError(f"Incomplete MAC-SQL clone: {resolved['macsql_repo'] / name}")
    rows = read_json(resolved["dataset"])
    if len(rows) != 100:
        raise ValueError(f"Expected 100 questions, got {len(rows)}")
    catalog = SQLiteCatalog(resolved["database_root"], config.get("table_aliases", {}))
    exact, _, _ = canonical_table_map(catalog)
    k = int(config["routing"]["k"])
    for row in rows:
        predictions = row["predicted_schemas_at15"]
        if len(predictions) != k or len({x.casefold() for x in predictions}) != k:
            raise ValueError(f"Invalid GERT K for question {row['question_id']}")
        for table in predictions:
            catalog.resolve_table(table)
        catalog.database_path(row["db_id"])
    commit = subprocess.check_output(
        ["git", "-C", str(resolved["macsql_repo"]), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    print("MAC-SQL BirdUnion validation passed")
    print(f"Questions: {len(rows)}; K={k}; physical tables={len(exact)}")
    print(f"Upstream commit: {commit}")
    print("Selector protocol: global union schema; gold db_id hidden")
    print("Model sampling parameters: provider defaults")


def run(
    config: dict[str, Any],
    conditions: list[str],
    limit: int | None,
    question_id: int | None,
    workers: int | None,
    output_dir_override: Path | None,
) -> None:
    paths = config["paths"]
    rows = read_json(resolve(ROOT, paths["dataset"]))
    if question_id is not None:
        rows = [row for row in rows if int(row["question_id"]) == question_id]
    if limit is not None:
        rows = rows[:limit]
    output_dir = resolve(ROOT, output_dir_override or paths["output_dir"])
    catalog = SQLiteCatalog(
        resolve(ROOT, paths["database_root"]), config.get("table_aliases", {})
    )
    union_schema = resolve(ROOT, paths["union_schema"]).read_text(encoding="utf-8")
    exact_tables, tables_by_name, table_search_tokens = canonical_table_map(catalog)
    llm = DefaultParameterLLM(config)
    worker_count = workers or int(config["runtime"]["workers"])

    for condition in conditions:
        output_path = output_dir / f"{condition}.jsonl"
        completed = load_jsonl(output_path)
        pending = [row for row in rows if int(row["question_id"]) not in completed]
        print(
            f"[{condition}] total={len(rows)}, resumed={len(rows)-len(pending)}, "
            f"pending={len(pending)}, workers={worker_count}"
        )
        lock = threading.Lock()
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
                    exact_tables,
                    tables_by_name,
                    table_search_tokens,
                ): row
                for row in pending
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=condition):
                row = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    for other in futures:
                        other.cancel()
                    raise RuntimeError(f"Question {row['question_id']} failed") from exc
                completed[int(result["question_id"])] = result
                append_jsonl(output_path, result, lock)
        if len(completed) >= len(rows):
            ordered = [completed[int(row["question_id"])] for row in rows]
            write_json(output_dir / f"{condition}.json", ordered)
            print(f"Wrote {output_dir / f'{condition}.json'}")


def safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def safe_stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def evaluate_condition(
    source_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    catalog: SQLiteCatalog,
    config: dict[str, Any],
    full_schema_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_id = {int(row["question_id"]): row for row in predictions}
    if set(by_id) != {int(row["question_id"]) for row in source_rows}:
        raise ValueError("Prediction question IDs do not exactly match the 100-query source")
    per_query: list[dict[str, Any]] = []
    evaluation = config["evaluation"]
    for source in source_rows:
        predicted = by_id[int(source["question_id"])]
        sql = predicted["predicted_sql"]
        parse_valid, normalized_predicted, parse_error = parse_read_only_sql(sql)
        _, normalized_gold, _ = parse_read_only_sql(source["SQL"])
        executable = False
        execution_correct = False
        category = normalized_predicted if not parse_valid else "SQL_EXECUTION_ERROR"
        error = parse_error
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
                execution_correct = result_sets_equal(
                    gold_result["rows"],
                    pred_result["rows"],
                    source["SQL"],
                    int(evaluation["float_round_digits"]),
                )
                category = "CORRECT" if execution_correct else "VALID_BUT_WRONG_RESULT"
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
                "execution_correct": execution_correct,
                "normalized_exact_match": bool(parse_valid and normalized_predicted == normalized_gold),
                "category": category,
                "error": error,
                # End-to-end context reduction includes the selector call.
                # MAC-SQL's selector reads the full union schema, so the
                # headline Context Red. for this condition is zero.
                "context_reduction": (
                    0.0
                    if predicted["condition"] == "mac_selector"
                    else 1 - predicted["schema_tokens_estimate"] / full_schema_tokens
                ),
                "downstream_context_reduction": 1
                - predicted["schema_tokens_estimate"] / full_schema_tokens,
                "schema_tokens_estimate": predicted["schema_tokens_estimate"],
                "selected_table_count": len(predicted["selected_tables"]),
                "selector_tokens": predicted["selector_usage"]["total_tokens"],
                "downstream_total_tokens": predicted["downstream_usage"]["total_tokens"],
                "pipeline_total_tokens": predicted["pipeline_usage"]["total_tokens"],
                "gold_table_complete": gold_tables <= selected,
                "refiner_called": predicted["refiner_called"],
                "selector_fill_count": len(predicted.get("selector_fillers", [])),
                "selector_call_count": int(predicted.get("selector_call_count", 0)),
                "selector_below_minimum": bool(predicted.get("selector_below_minimum", False)),
            }
        )
    metrics = {
        "n": len(per_query),
        "execution_accuracy": safe_mean([float(x["execution_correct"]) for x in per_query]),
        "invalid_sql_rate": 1 - safe_mean([float(x["executable"]) for x in per_query]),
        "normalized_exact_match": safe_mean([float(x["normalized_exact_match"]) for x in per_query]),
        "context_reduction": safe_mean([x["context_reduction"] for x in per_query]),
        "downstream_context_reduction": safe_mean(
            [x["downstream_context_reduction"] for x in per_query]
        ),
        "avg_schema_tokens_estimate": safe_mean([x["schema_tokens_estimate"] for x in per_query]),
        "avg_selected_table_count": safe_mean(
            [x["selected_table_count"] for x in per_query]
        ),
        "avg_total_tokens": safe_mean([x["downstream_total_tokens"] for x in per_query]),
        "avg_selector_tokens": safe_mean([x["selector_tokens"] for x in per_query]),
        "avg_pipeline_tokens": safe_mean([x["pipeline_total_tokens"] for x in per_query]),
        "gold_table_complete_rate": safe_mean([float(x["gold_table_complete"]) for x in per_query]),
        "refiner_call_rate": safe_mean([float(x["refiner_called"]) for x in per_query]),
        "avg_selector_fill_count": safe_mean([x["selector_fill_count"] for x in per_query]),
        "avg_selector_call_count": safe_mean([x["selector_call_count"] for x in per_query]),
        "selector_below_minimum_rate": safe_mean(
            [float(x["selector_below_minimum"]) for x in per_query]
        ),
    }
    return metrics, per_query


def evaluate_all(
    config: dict[str, Any],
    conditions: list[str],
    output_dir_override: Path | None,
    result_dir_override: Path | None,
    limit: int | None,
    question_id: int | None,
) -> None:
    paths = config["paths"]
    source_rows = read_json(resolve(ROOT, paths["dataset"]))
    if question_id is not None:
        source_rows = [row for row in source_rows if int(row["question_id"]) == question_id]
    if limit is not None:
        source_rows = source_rows[:limit]
    output_dir = resolve(ROOT, output_dir_override or paths["output_dir"])
    result_dir = resolve(ROOT, result_dir_override or paths["result_dir"])
    catalog = SQLiteCatalog(
        resolve(ROOT, paths["database_root"]), config.get("table_aliases", {})
    )
    union_schema = resolve(ROOT, paths["union_schema"]).read_text(encoding="utf-8")
    full_schema_tokens = chars_div_4(union_schema)
    all_metrics: dict[str, Any] = {}
    all_per_query: list[dict[str, Any]] = []
    for condition in conditions:
        json_path = output_dir / f"{condition}.json"
        predictions = read_json(json_path) if json_path.exists() else list(load_jsonl(output_dir / f"{condition}.jsonl").values())
        metrics, per_query = evaluate_condition(
            source_rows, predictions, catalog, config, full_schema_tokens
        )
        all_metrics[condition] = metrics
        all_per_query.extend(per_query)

    labels = {
        "mac_selector": "MAC-SQL Selector (full union schema)",
        "gert_selector": "MAC-SQL + GERT (ours)",
    }
    lines = [
        "# MAC-SQL Selector replacement on BirdUnion-100",
        "",
        "| Schema Input | EX | Invalid-SQL | Context Red. | Total Tokens |",
        "| :-- | --: | --: | --: | --: |",
    ]
    for condition in conditions:
        metric = all_metrics[condition]
        lines.append(
            f"| {labels[condition]} | {metric['execution_accuracy']:.2f} | "
            f"{metric['invalid_sql_rate']:.2f} | {metric['context_reduction']:.2%} | "
            f"{metric['avg_pipeline_tokens']:.0f} |"
        )
    paired: dict[str, Any] = {}
    if set(CONDITIONS) <= set(conditions):
        mac = {
            int(row["question_id"]): bool(row["execution_correct"])
            for row in all_per_query
            if row["condition"] == "mac_selector"
        }
        gert = {
            int(row["question_id"]): bool(row["execution_correct"])
            for row in all_per_query
            if row["condition"] == "gert_selector"
        }
        ids = sorted(set(mac) & set(gert))
        mac_only = sum(mac[i] and not gert[i] for i in ids)
        gert_only = sum(gert[i] and not mac[i] for i in ids)
        discordant = mac_only + gert_only
        if discordant:
            tail = sum(
                math.comb(discordant, j)
                for j in range(min(mac_only, gert_only) + 1)
            ) / (2**discordant)
            p_value = min(1.0, 2 * tail)
        else:
            p_value = 1.0
        paired = {
            "n": len(ids),
            "both_correct": sum(mac[i] and gert[i] for i in ids),
            "mac_only_correct": mac_only,
            "gert_only_correct": gert_only,
            "neither_correct": sum(not mac[i] and not gert[i] for i in ids),
            "mcnemar_exact_p_two_sided": p_value,
        }

    diagnostic_lines = [
        "# MAC-SQL routing and pipeline diagnostics",
        "",
        "| Condition | Avg Selected Tables | Downstream Context Red. | Gold Table CR | Refiner Rate | Avg Selector Fill | Avg Selector Calls | Avg Selector Tokens | Avg Downstream Tokens | Avg Pipeline Tokens |",
        "| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |",
    ]
    for condition in conditions:
        metric = all_metrics[condition]
        diagnostic_lines.append(
            f"| {labels[condition]} | {metric['avg_selected_table_count']:.2f} | "
            f"{metric['downstream_context_reduction']:.2%} | "
            f"{metric['gold_table_complete_rate']:.2f} | "
            f"{metric['refiner_call_rate']:.2f} | {metric['avg_selector_fill_count']:.2f} | "
            f"{metric['avg_selector_call_count']:.2f} | {metric['avg_selector_tokens']:.0f} | "
            f"{metric['avg_total_tokens']:.0f} | {metric['avg_pipeline_tokens']:.0f} |"
        )
    diagnostic_lines.extend(["", "## Final SQL categories", ""])
    categories = sorted({row["category"] for row in all_per_query})
    diagnostic_lines.append("| Condition | " + " | ".join(categories) + " |")
    diagnostic_lines.append("| :-- | " + " | ".join("--:" for _ in categories) + " |")
    for condition in conditions:
        counts = Counter(
            row["category"] for row in all_per_query if row["condition"] == condition
        )
        diagnostic_lines.append(
            f"| {labels[condition]} | "
            + " | ".join(str(counts.get(category, 0)) for category in categories)
            + " |"
        )
    if paired:
        diagnostic_lines.extend(
            [
                "",
                "## Paired EX comparison",
                "",
                f"- Both correct: {paired['both_correct']}",
                f"- MAC-SQL only correct: {paired['mac_only_correct']}",
                f"- GERT replacement only correct: {paired['gert_only_correct']}",
                f"- Neither correct: {paired['neither_correct']}",
                f"- Exact McNemar p-value (two-sided): {paired['mcnemar_exact_p_two_sided']:.6f}",
            ]
        )
    result_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        result_dir / "metrics.json",
        {
            "experiment": config["experiment"],
            "protocol": {
                "protocol_version": PROTOCOL_VERSION,
                "k": config["routing"]["k"],
                "full_schema_tokens_estimate": full_schema_tokens,
                "primary_total_tokens": (
                    "Full pipeline: Selector + Decomposer + optional Refiner for MAC; "
                    "Decomposer + optional Refiner for frozen GERT"
                ),
                "context_reduction": (
                    "End-to-end: zero for MAC because its Selector reads the full union schema; "
                    "GERT is measured against the full union schema"
                ),
                "downstream_context_reduction": (
                    "Diagnostic only: candidate schema reduction after selection"
                ),
                "gert_routing_cost": "Excluded because GERT predictions are frozen inputs",
                "model_parameters": "Provider defaults; no sampling parameters sent",
                "selector_db_id": "Gold db_id hidden; used only for SQLite execution/evaluation",
            },
            "metrics": all_metrics,
            "paired_comparison": paired,
        },
    )
    write_json(result_dir / "per_query.json", all_per_query)
    (result_dir / "main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (result_dir / "diagnostics.md").write_text(
        "\n".join(diagnostic_lines) + "\n", encoding="utf-8"
    )
    round_summary = {
        "protocol_version": PROTOCOL_VERSION,
        "metrics": all_metrics,
        "token_definition": "Provider-reported prompt + completion tokens for the full pipeline",
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
        raise ValueError("Aggregate requires at least two result directories")
    reports: list[dict[str, Any]] = []
    resolved_dirs = [resolve(ROOT, path) for path in result_dirs]
    for result_dir in resolved_dirs:
        path = result_dir / "metrics.json"
        report = read_json(path)
        if report.get("protocol", {}).get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError(f"Incompatible protocol in {path}")
        if not set(conditions) <= set(report.get("metrics", {})):
            raise ValueError(f"Requested conditions are missing in {path}")
        reports.append(report)

    metric_names = (
        "execution_accuracy",
        "invalid_sql_rate",
        "context_reduction",
        "downstream_context_reduction",
        "avg_pipeline_tokens",
        "avg_selector_tokens",
        "avg_total_tokens",
        "gold_table_complete_rate",
        "avg_selected_table_count",
    )
    aggregate: dict[str, Any] = {}
    for condition in conditions:
        aggregate[condition] = {}
        for metric_name in metric_names:
            values = [float(report["metrics"][condition][metric_name]) for report in reports]
            aggregate[condition][metric_name] = {
                "values": values,
                "mean": safe_mean(values),
                "sample_std": safe_stdev(values),
            }

    output_dir = resolve(ROOT, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "round_count": len(reports),
        "std_definition": "sample standard deviation (n-1)",
        "result_dirs": [str(path) for path in resolved_dirs],
        "metrics": aggregate,
    }
    write_json(output_dir / "aggregate_metrics.json", payload)
    labels = {
        "mac_selector": "MAC-SQL Selector (full union schema input)",
        "gert_selector": "MAC-SQL + GERT (ours)",
    }
    lines = [
        f"# MAC-SQL {len(reports)}-round mean ± std",
        "",
        "| Schema Input | EX | Invalid-SQL | Context Red. | Total Tokens |",
        "| :-- | --: | --: | --: | --: |",
    ]
    for condition in conditions:
        metric = aggregate[condition]
        ex = metric["execution_accuracy"]
        invalid = metric["invalid_sql_rate"]
        context = metric["context_reduction"]
        tokens = metric["avg_pipeline_tokens"]
        lines.append(
            f"| {labels[condition]} | {ex['mean']:.3f} ± {ex['sample_std']:.3f} | "
            f"{invalid['mean']:.3f} ± {invalid['sample_std']:.3f} | "
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
    subparsers.add_parser("validate", help="Validate local inputs without calling the API")
    run_parser = subparsers.add_parser("run", help="Run one or both routing conditions")
    run_parser.add_argument("--conditions", nargs="+", choices=CONDITIONS)
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--question-id", type=int)
    run_parser.add_argument("--workers", type=int)
    run_parser.add_argument("--output-dir", type=Path)
    evaluate_parser = subparsers.add_parser("evaluate", help="Execute SQL and aggregate metrics")
    evaluate_parser.add_argument("--conditions", nargs="+", choices=CONDITIONS)
    evaluate_parser.add_argument("--output-dir", type=Path)
    evaluate_parser.add_argument("--result-dir", type=Path)
    evaluate_parser.add_argument("--limit", type=int)
    evaluate_parser.add_argument("--question-id", type=int)
    aggregate_parser = subparsers.add_parser(
        "aggregate", help="Aggregate completed rounds as mean +/- sample std"
    )
    aggregate_parser.add_argument("--conditions", nargs="+", choices=CONDITIONS)
    aggregate_parser.add_argument("--result-dirs", nargs="+", type=Path, required=True)
    aggregate_parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config.resolve())
    conditions = selected_conditions(getattr(args, "conditions", None))
    if args.command == "validate":
        validate(config)
    elif args.command == "run":
        run(
            config,
            conditions,
            args.limit,
            args.question_id,
            args.workers,
            args.output_dir,
        )
    elif args.command == "evaluate":
        evaluate_all(
            config,
            conditions,
            args.output_dir,
            args.result_dir,
            args.limit,
            args.question_id,
        )
    elif args.command == "aggregate":
        aggregate_runs(args.result_dirs, args.output_dir, conditions)


if __name__ == "__main__":
    main()
