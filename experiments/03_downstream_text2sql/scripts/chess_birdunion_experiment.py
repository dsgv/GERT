#!/usr/bin/env python3
"""Compare adapted CHESS table selection with frozen GERT routing on BirdUnion-100."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import statistics
import threading
import time
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import yaml
from prepared_inputs import load_prepared_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = ROOT / "configs" / "chess_birdunion.yaml"

from text2sql_experiment import (  # noqa: E402
    SQLiteCatalog,
    execute_sql,
    parse_read_only_sql,
    result_sets_equal,
)


CONDITIONS = ("chess_selector", "gert_selector")
UPSTREAM_COMMIT = "3d6e835f858d26885d21d4bc0215aeecf855efbe"
CHESS_HARDCODED_TEST = (
    "Only the best answer from the set of candidates that most accurately answers "
    "the question, given the database schema and hint should pass this test."
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


def safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


class DefaultParameterLLM:
    """OpenAI-compatible client that sends no sampling/generation parameters."""

    def __init__(self, config: dict[str, Any]) -> None:
        load_dotenv(ROOT / ".env")
        api_key = (
            os.getenv("CHESS_API_KEY")
            or os.getenv("TEXT2SQL_API_KEY")
            or os.getenv("API_KEY")
        )
        api_base = (
            os.getenv("CHESS_API_BASE")
            or os.getenv("TEXT2SQL_API_BASE")
            or os.getenv("API_BASE")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = (
            os.getenv("CHESS_MODEL")
            or os.getenv("TEXT2SQL_MODEL")
            or os.getenv("MODEL")
            or "deepseek-v3.2"
        )
        if not api_key:
            raise RuntimeError(
                "CHESS_API_KEY, TEXT2SQL_API_KEY, or API_KEY is required in Text2sql_ex/.env"
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

    def complete(self, prompt: str, stage: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            started = time.perf_counter()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
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
                search_tokens[qualified] = routing_tokens(" ".join([db_id, table, *columns]))
        finally:
            connection.close()
    return exact, by_table, search_tokens


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
    raise ValueError("Response does not contain a valid JSON object")


def normalize_selector_tables(
    response: str,
    exact: dict[str, str],
    by_table: dict[str, list[str]],
) -> list[str]:
    value = parse_json_object(response)
    raw_tables = value.get("table_names", [])
    if not isinstance(raw_tables, list):
        raise ValueError("CHESS table_names must be a JSON list")
    selected: list[str] = []
    seen: set[str] = set()
    for raw in raw_tables:
        key = str(raw).strip().strip("`\"'")
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
    fallback = sorted(
        (table for table in search_tokens if table.casefold() not in seen),
        key=lambda table: (-len(query_tokens & search_tokens[table]), table.casefold()),
    )
    fillers = fallback[: max(0, k - len(selected))]
    normalized = [*selected, *fillers]
    if len(normalized) != k:
        raise ValueError(f"Could not normalize Selector output to K={k}")
    return normalized, fillers


def build_selector_prompt(template: str, union_schema: str, question: str, k: int) -> str:
    prompt = template.format(DATABASE_SCHEMA=union_schema, QUESTION=question, HINT="")
    return prompt + f"""

Additional BirdUnion global fixed-budget protocol (mandatory):
1. The schema pools many independent databases. A table identity is `database.table`,
   where `database` comes from the nearest `-- Database:` header.
2. Select exactly {k} unique tables, ranked from most to least relevant.
3. Every item in `table_names` must be a fully-qualified `database.table` identity.
4. Return one JSON object only with exactly these keys:
   `chain_of_thought_reasoning` and `table_names`.
5. Do not assume or infer a gold database identifier outside the schema and question.
"""


def extract_sql(text: str) -> str:
    tagged = re.findall(r"<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>", text, flags=re.I | re.S)
    if tagged:
        return tagged[-1].strip()
    fenced = re.findall(r"```sql\s*(.*?)```", text, flags=re.I | re.S)
    if fenced:
        return fenced[-1].strip()
    match = re.search(r"\b(SELECT|WITH)\b[\s\S]*", text, flags=re.I)
    return match.group(0).strip() if match else "error: No SQL found in the model response"


def parse_unit_tests(text: str, limit: int) -> list[str]:
    match = re.search(r"<Answer>\s*(.*?)\s*</Answer>", text, flags=re.I | re.S)
    if match:
        payload = match.group(1).strip()
    else:
        fenced = re.search(r"```(?:python|json)?\s*(\[.*?\])\s*```", text, flags=re.I | re.S)
        if fenced:
            payload = fenced.group(1)
        else:
            start, end = text.find("["), text.rfind("]")
            if start < 0 or end <= start:
                raise ValueError("Unit-test response contains no list")
            payload = text[start : end + 1]
    try:
        value = ast.literal_eval(payload)
    except Exception:
        value = json.loads(payload)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("Unit-test answer is not a list of strings")
    tests = [item.strip() for item in value if item.strip()]
    if not tests:
        raise ValueError("Unit-test answer is empty")
    return tests[:limit]


def parse_unit_scores(text: str, candidate_count: int) -> list[int]:
    match = re.search(r"<Answer>\s*(.*?)\s*</Answer>", text, flags=re.I | re.S)
    payload = match.group(1) if match else text
    scores: list[int] = []
    for line in payload.splitlines():
        if re.search(r"Candidate\s+Response\s*#?\d+", line, flags=re.I):
            scores.append(1 if "[passed]" in line.casefold() else 0)
    if len(scores) != candidate_count:
        raise ValueError(f"Expected {candidate_count} verdicts, parsed {len(scores)}")
    return scores


def execution_summary(execution: dict[str, Any], preview_rows: int) -> dict[str, Any]:
    rows = execution.get("rows", []) if execution.get("success") else []
    return {
        "success": bool(execution.get("success")),
        "category": execution.get("category"),
        "error": execution.get("error"),
        "row_count": len(rows),
        "rows_preview": rows[:preview_rows],
    }


def formatted_execution(execution: dict[str, Any], preview_rows: int) -> str:
    summary = execution_summary(execution, preview_rows)
    if not summary["success"]:
        return f"Error: {summary['error']}"
    column_count = len(summary["rows_preview"][0]) if summary["rows_preview"] else 0
    return (
        f"Rows: {summary['row_count']}, Columns: {column_count}, "
        f"Results: {summary['rows_preview']}"
    )


def execution_cluster_key(execution: dict[str, Any]) -> str:
    if not execution.get("success"):
        return f"ERROR::{execution.get('category')}::{execution.get('error')}"
    return repr(execution.get("rows", []))


def render_candidate_clusters(
    candidates: list[dict[str, Any]], preview_rows: int
) -> tuple[str, dict[str, list[int]]]:
    clusters: dict[str, list[int]] = {}
    for index, candidate in enumerate(candidates):
        clusters.setdefault(execution_cluster_key(candidate["execution"]), []).append(index)
    lines: list[str] = []
    for cluster_index, indices in enumerate(clusters.values(), 1):
        lines.append(f"Cluster #{cluster_index}:")
        for candidate_index in indices:
            lines.append(f"Query: {candidates[candidate_index]['sql']}")
            lines.append("########")
        lines.append(
            "Execution result: "
            + formatted_execution(candidates[indices[-1]]["execution"], preview_rows)
        )
        lines.append("=====================")
    return "\n".join(lines), clusters


def render_candidate_responses(candidates: list[dict[str, Any]], preview_rows: int) -> str:
    lines: list[str] = []
    for index, candidate in enumerate(candidates, 1):
        lines.append(f"Candidate Response #{index}: Query: {candidate['sql']}")
        lines.append(
            "Execution Result: "
            + formatted_execution(candidate["execution"], preview_rows)
        )
    return "\n".join(lines)


def call_and_parse(
    llm: DefaultParameterLLM,
    prompt: str,
    stage: str,
    parser: Callable[[str], Any],
    attempts: int,
    calls: list[dict[str, Any]],
) -> tuple[Any, list[str]]:
    errors: list[str] = []
    current_prompt = prompt
    for attempt in range(1, attempts + 1):
        call = llm.complete(current_prompt, stage)
        calls.append(call)
        try:
            return parser(call["content"]), errors
        except Exception as exc:
            errors.append(str(exc))
            current_prompt = (
                prompt
                + "\n\nYour previous response could not be parsed. Follow the required output "
                + f"format exactly. Parser error: {exc}"
            )
    raise ValueError(f"Could not parse {stage} after {attempts} attempts: {errors[-1]}")


def choose_candidate(
    candidates: list[dict[str, Any]], scores: list[int], clusters: dict[str, list[int]]
) -> int:
    maximum = max(scores)
    best = [index for index, score in enumerate(scores) if score == maximum]
    if len(best) == 1:
        return best[0]
    largest_cluster = max(clusters.values(), key=len)
    for index in best:
        if index in largest_cluster:
            return index
    return best[0]


def process_query(
    row: dict[str, Any],
    condition: str,
    config: dict[str, Any],
    llm: DefaultParameterLLM,
    catalog: SQLiteCatalog,
    union_schema: str,
    templates: dict[str, str],
    exact_tables: dict[str, str],
    tables_by_name: dict[str, list[str]],
    table_search_tokens: dict[str, set[str]],
) -> dict[str, Any]:
    started = time.perf_counter()
    routing = config["routing"]
    pipeline = config["pipeline"]
    evaluation = config["evaluation"]
    k = int(routing["k"])
    selector_calls: list[dict[str, Any]] = []
    downstream_calls: list[dict[str, Any]] = []
    selector_response = ""
    selector_fillers: list[str] = []
    selector_llm_table_count = 0
    selector_below_minimum = False

    if condition == "gert_selector":
        selected_tables = list(row["predicted_schemas_at15"])
        if len(selected_tables) != k or len({x.casefold() for x in selected_tables}) != k:
            raise ValueError(f"GERT question {row['question_id']} does not contain K={k}")
    else:
        selector_prompt = build_selector_prompt(
            templates["template_select_tables.txt"], union_schema, row["question"], k
        )
        best_tables: list[str] = []
        best_response = ""
        last_error: Exception | None = None
        minimum = int(routing["selector_min_llm_tables"])
        for _ in range(int(routing["selector_format_attempts"])):
            call = llm.complete(selector_prompt, "selector")
            selector_calls.append(call)
            try:
                llm_tables = normalize_selector_tables(
                    call["content"], exact_tables, tables_by_name
                )
                if len(llm_tables) > len(best_tables):
                    best_tables = llm_tables
                    best_response = call["content"]
                if len(llm_tables) < minimum:
                    raise ValueError(
                        f"Selector returned {len(llm_tables)} valid tables; minimum={minimum}"
                    )
                selector_response = call["content"]
                selector_llm_table_count = len(llm_tables)
                selected_tables, selector_fillers = normalize_fixed_budget(
                    llm_tables, row["question"], table_search_tokens, k
                )
                break
            except Exception as exc:
                last_error = exc
                selector_prompt += (
                    "\nThe previous output violated the protocol. Return exactly "
                    f"{k} valid unique fully-qualified tables in table_names. JSON only."
                )
        else:
            if not best_tables:
                raise ValueError(f"Invalid CHESS Selector output: {last_error}")
            selector_response = best_response
            selector_llm_table_count = len(best_tables)
            selector_below_minimum = selector_llm_table_count < minimum
            selected_tables, selector_fillers = normalize_fixed_budget(
                best_tables, row["question"], table_search_tokens, k
            )

    schema_text, mappings = catalog.serialize(selected_tables)
    if len(mappings) != k:
        raise ValueError(f"Resolved {len(mappings)} candidate tables; expected K={k}")
    db_path = catalog.database_path(row["db_id"])
    sql_timeout = float(evaluation["sql_timeout_seconds"])
    max_rows = int(evaluation["max_result_rows"])
    preview_rows = int(evaluation["result_preview_rows"])

    candidates: list[dict[str, Any]] = []
    for template_name in pipeline["generator_templates"]:
        template = templates[template_name]
        prompt = template.format(
            DATABASE_SCHEMA=schema_text,
            QUESTION=row["question"],
            HINT="",
        )
        for sample_index in range(int(pipeline["samples_per_generator"])):
            call = llm.complete(prompt, f"candidate:{template_name}:{sample_index + 1}")
            downstream_calls.append(call)
            sql = extract_sql(call["content"])
            execution = execute_sql(db_path, sql, sql_timeout, max_rows)
            candidates.append(
                {
                    "template": template_name,
                    "sample_index": sample_index + 1,
                    "initial_response": call["content"],
                    "initial_sql": sql,
                    "initial_execution": execution_summary(execution, preview_rows),
                    "revision_response": "",
                    "revised": False,
                    "sql": sql,
                    "execution": execution,
                }
            )

    if bool(pipeline["revise_invalid_or_empty_once"]):
        revise_template = templates[pipeline["revise_template"]]
        for index, candidate in enumerate(candidates):
            execution = candidate["execution"]
            if execution.get("success") and execution.get("rows"):
                continue
            revise_prompt = revise_template.format(
                DATABASE_SCHEMA=schema_text,
                QUESTION=row["question"],
                HINT="",
                QUERY=candidate["sql"],
                RESULT=formatted_execution(execution, preview_rows),
            )
            call = llm.complete(revise_prompt, f"revise:{index + 1}")
            downstream_calls.append(call)
            revised_sql = extract_sql(call["content"])
            revised_execution = execute_sql(db_path, revised_sql, sql_timeout, max_rows)
            candidate.update(
                {
                    "revision_response": call["content"],
                    "revised": True,
                    "sql": revised_sql,
                    "execution": revised_execution,
                }
            )

    clusters_text, clusters = render_candidate_clusters(candidates, preview_rows)
    unit_tests: list[str] = []
    unit_generation_responses: list[str] = []
    unit_generation_errors: list[str] = []
    unit_evaluations: list[dict[str, Any]] = []
    scores = [0] * len(candidates)

    if len(candidates) > 1 and len(clusters) > 1:
        unit_count = int(pipeline["generated_unit_test_count"])
        unit_prompt = templates["template_generate_unit_tests.txt"].format(
            UNIT_TEST_CAP=unit_count,
            QUESTION=row["question"],
            HINT="",
            DATABASE_SCHEMA=schema_text,
            CANDIDATE_QUERIES=clusters_text,
        )
        before = len(downstream_calls)
        try:
            parsed_tests, unit_generation_errors = call_and_parse(
                llm,
                unit_prompt,
                "unit_test_generation",
                lambda text: parse_unit_tests(text, unit_count),
                int(pipeline["unit_test_parse_attempts"]),
                downstream_calls,
            )
        except Exception as exc:
            parsed_tests = []
            unit_generation_errors = [str(exc)]
        unit_generation_responses = [
            call["content"] for call in downstream_calls[before:]
        ]
        unit_tests.extend(parsed_tests)
        if bool(pipeline["include_chess_hardcoded_test"]):
            unit_tests.append(CHESS_HARDCODED_TEST)

        formatted_candidates = render_candidate_responses(candidates, preview_rows)
        successful_evaluations = 0
        for test_index, unit_test in enumerate(unit_tests, 1):
            evaluation_prompt = templates["template_evaluate.txt"].format(
                DATABASE_SCHEMA=schema_text,
                QUESTION=row["question"],
                HINT="",
                CANDIDATE_RESPONSES=formatted_candidates,
                UNIT_TEST=unit_test,
            )
            before = len(downstream_calls)
            try:
                parsed_scores, parse_errors = call_and_parse(
                    llm,
                    evaluation_prompt,
                    f"unit_test_evaluation:{test_index}",
                    lambda text: parse_unit_scores(text, len(candidates)),
                    int(pipeline["evaluation_parse_attempts"]),
                    downstream_calls,
                )
                successful_evaluations += 1
            except Exception as exc:
                parsed_scores = [0] * len(candidates)
                parse_errors = [str(exc)]
            responses = [call["content"] for call in downstream_calls[before:]]
            scores = [left + right for left, right in zip(scores, parsed_scores)]
            unit_evaluations.append(
                {
                    "unit_test": unit_test,
                    "scores": parsed_scores,
                    "responses": responses,
                    "parse_errors": parse_errors,
                }
            )
        selected_index = (
            choose_candidate(candidates, scores, clusters)
            if successful_evaluations
            else 0
        )
    else:
        selected_index = 0

    final_candidate = candidates[selected_index]
    selector_usage = token_totals(selector_calls)
    downstream_usage = token_totals(downstream_calls)
    all_calls = [*selector_calls, *downstream_calls]
    return {
        "question_id": int(row["question_id"]),
        "db_id": row["db_id"],
        "question": row["question"],
        "evidence": "",
        "gold_sql": row["SQL"],
        "gold_schema": row.get("gold_schema", row.get("schema_link")),
        "condition": condition,
        "model": llm.model,
        "model_parameters": "provider_defaults",
        "gold_db_id_visibility": "execution_and_evaluation_only",
        "information_retriever_enabled": False,
        "selected_tables": selected_tables,
        "selector_llm_table_count": selector_llm_table_count,
        "selector_fillers": selector_fillers,
        "selector_call_count": len(selector_calls),
        "selector_below_minimum": selector_below_minimum,
        "selector_response": selector_response,
        "candidate_count": len(candidates),
        "schema_text": schema_text,
        "schema_chars": len(schema_text),
        "schema_tokens_estimate": chars_div_4(schema_text),
        "candidates": [
            {
                **{key: value for key, value in candidate.items() if key != "execution"},
                "final_execution": execution_summary(candidate["execution"], preview_rows),
            }
            for candidate in candidates
        ],
        "execution_clusters": list(clusters.values()),
        "unit_tests": unit_tests,
        "unit_generation_responses": unit_generation_responses,
        "unit_generation_parse_errors": unit_generation_errors,
        "unit_evaluations": unit_evaluations,
        "candidate_scores": scores,
        "selected_candidate_index": selected_index,
        "predicted_sql": final_candidate["sql"],
        "final_execution": execution_summary(final_candidate["execution"], preview_rows),
        "revision_call_count": sum(call["stage"].startswith("revise:") for call in downstream_calls),
        "unit_test_call_count": sum(call["stage"].startswith("unit_test") for call in downstream_calls),
        "selector_usage": selector_usage,
        "downstream_usage": downstream_usage,
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


def load_templates(chess_repo: Path, config: dict[str, Any]) -> dict[str, str]:
    names = {
        "template_select_tables.txt",
        "template_generate_unit_tests.txt",
        "template_evaluate.txt",
        config["pipeline"]["revise_template"],
        *config["pipeline"]["generator_templates"],
    }
    templates: dict[str, str] = {}
    for name in names:
        path = chess_repo / "templates" / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing CHESS template: {path}")
        templates[name] = path.read_text(encoding="utf-8")
    return templates


def validate(config: dict[str, Any]) -> None:
    paths = config["paths"]
    resolved = {
        key: resolve(ROOT, paths[key])
        for key in ("dataset", "database_root", "union_schema", "chess_repo")
    }
    for key, path in resolved.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing paths.{key}: {path}")
    marker = resolved["chess_repo"] / "UPSTREAM_SNAPSHOT.md"
    if not marker.is_file() or UPSTREAM_COMMIT not in marker.read_text(encoding="utf-8"):
        raise ValueError("CHESS upstream commit marker is missing or unexpected")
    templates = load_templates(resolved["chess_repo"], config)
    rows = load_prepared_dataset(resolved["dataset"])
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
    sample_prompt = templates["template_generate_candidate_one.txt"].format(
        DATABASE_SCHEMA="CREATE TABLE sample(id INTEGER);", QUESTION="Return ids.", HINT=""
    )
    if "CREATE TABLE sample" not in sample_prompt:
        raise ValueError("CHESS candidate template formatting failed")
    print("CHESS BirdUnion adapter validation passed")
    print(f"Questions: {len(rows)}; K={k}; physical tables={len(exact)}")
    print(f"Upstream commit: {UPSTREAM_COMMIT}")
    print("Protocol: global union selector; gold db_id hidden from prompts; IR disabled")
    print("Downstream: 2 templates x 2 samples; one revision; 5 generated unit tests")
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
    rows = load_prepared_dataset(resolve(ROOT, paths["dataset"]))
    if question_id is not None:
        rows = [row for row in rows if int(row["question_id"]) == question_id]
    if limit is not None:
        rows = rows[:limit]
    output_dir = resolve(ROOT, output_dir_override or paths["output_dir"])
    catalog = SQLiteCatalog(
        resolve(ROOT, paths["database_root"]), config.get("table_aliases", {})
    )
    union_schema = resolve(ROOT, paths["union_schema"]).read_text(encoding="utf-8")
    templates = load_templates(resolve(ROOT, paths["chess_repo"]), config)
    exact, by_name, search_tokens = canonical_table_map(catalog)
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
                    templates,
                    exact,
                    by_name,
                    search_tokens,
                ): row
                for row in pending
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=condition):
                row = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    # A transient provider/network failure for one question must
                    # not discard the rest of a long-running batch.  Successful
                    # questions are checkpointed below; a later invocation will
                    # pick up only the missing question IDs from the JSONL file.
                    question_id = int(row["question_id"])
                    failed.append((question_id, f"{type(exc).__name__}: {exc}"))
                    tqdm.write(
                        f"[{condition}] question {question_id} failed; "
                        "left pending for the next resume run"
                    )
                    continue
                completed[int(result["question_id"])] = result
                append_jsonl(output_path, result, lock)
        if failed:
            print(
                f"[{condition}] transient failures={len(failed)}; "
                f"pending_ids={[question_id for question_id, _ in failed]}"
            )
        if len(completed) >= len(rows):
            ordered = [completed[int(row["question_id"])] for row in rows]
            write_json(output_dir / f"{condition}.json", ordered)
            print(f"Wrote {output_dir / f'{condition}.json'}")


def evaluate_condition(
    source_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    catalog: SQLiteCatalog,
    config: dict[str, Any],
    full_schema_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_id = {int(row["question_id"]): row for row in predictions}
    expected = {int(row["question_id"]) for row in source_rows}
    if set(by_id) != expected:
        raise ValueError("Prediction question IDs do not exactly match the requested source rows")
    evaluation = config["evaluation"]
    per_query: list[dict[str, Any]] = []
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
                "normalized_exact_match": bool(
                    parse_valid and normalized_predicted == normalized_gold
                ),
                "category": category,
                "error": error,
                "context_reduction": 1
                - predicted["schema_tokens_estimate"] / full_schema_tokens,
                "schema_tokens_estimate": predicted["schema_tokens_estimate"],
                "selector_tokens": predicted["selector_usage"]["total_tokens"],
                "downstream_total_tokens": predicted["downstream_usage"]["total_tokens"],
                "pipeline_total_tokens": predicted["pipeline_usage"]["total_tokens"],
                "gold_table_complete": gold_tables <= selected,
                "revision_call_count": int(predicted["revision_call_count"]),
                "unit_test_call_count": int(predicted["unit_test_call_count"]),
                "selector_fill_count": len(predicted.get("selector_fillers", [])),
                "selector_call_count": int(predicted.get("selector_call_count", 0)),
            }
        )
    metrics = {
        "n": len(per_query),
        "execution_accuracy": safe_mean([float(x["execution_correct"]) for x in per_query]),
        "invalid_sql_rate": 1 - safe_mean([float(x["executable"]) for x in per_query]),
        "normalized_exact_match": safe_mean(
            [float(x["normalized_exact_match"]) for x in per_query]
        ),
        "context_reduction": safe_mean([x["context_reduction"] for x in per_query]),
        "avg_schema_tokens_estimate": safe_mean(
            [x["schema_tokens_estimate"] for x in per_query]
        ),
        "avg_total_tokens": safe_mean([x["downstream_total_tokens"] for x in per_query]),
        "avg_selector_tokens": safe_mean([x["selector_tokens"] for x in per_query]),
        "avg_pipeline_tokens": safe_mean([x["pipeline_total_tokens"] for x in per_query]),
        "gold_table_complete_rate": safe_mean(
            [float(x["gold_table_complete"]) for x in per_query]
        ),
        "avg_revision_calls": safe_mean([x["revision_call_count"] for x in per_query]),
        "avg_unit_test_calls": safe_mean([x["unit_test_call_count"] for x in per_query]),
        "avg_selector_fill_count": safe_mean([x["selector_fill_count"] for x in per_query]),
        "avg_selector_call_count": safe_mean([x["selector_call_count"] for x in per_query]),
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
    source_rows = load_prepared_dataset(resolve(ROOT, paths["dataset"]))
    if question_id is not None:
        source_rows = [row for row in source_rows if int(row["question_id"]) == question_id]
    if limit is not None:
        source_rows = source_rows[:limit]
    output_dir = resolve(ROOT, output_dir_override or paths["output_dir"])
    result_dir = resolve(ROOT, result_dir_override or paths["result_dir"])
    catalog = SQLiteCatalog(
        resolve(ROOT, paths["database_root"]), config.get("table_aliases", {})
    )
    full_schema_tokens = chars_div_4(
        resolve(ROOT, paths["union_schema"]).read_text(encoding="utf-8")
    )
    all_metrics: dict[str, Any] = {}
    all_per_query: list[dict[str, Any]] = []
    for condition in conditions:
        json_path = output_dir / f"{condition}.json"
        predictions = (
            read_json(json_path)
            if json_path.exists()
            else list(load_jsonl(output_dir / f"{condition}.jsonl").values())
        )
        metrics, per_query = evaluate_condition(
            source_rows, predictions, catalog, config, full_schema_tokens
        )
        all_metrics[condition] = metrics
        all_per_query.extend(per_query)

    labels = {
        "chess_selector": "CHESS Selector (full union schema)",
        "gert_selector": "CHESS + GERT (ours)",
    }
    lines = [
        "# CHESS Selector replacement on BirdUnion-100",
        "",
        "| Schema Input | EX | Invalid-SQL | Context Red. | Total Tokens |",
        "| :-- | --: | --: | --: | --: |",
    ]
    for condition in conditions:
        metric = all_metrics[condition]
        lines.append(
            f"| {labels[condition]} | {metric['execution_accuracy']:.2f} | "
            f"{metric['invalid_sql_rate']:.2f} | {metric['context_reduction']:.2%} | "
            f"{metric['avg_total_tokens']:.0f} |"
        )

    paired: dict[str, Any] = {}
    if set(CONDITIONS) <= set(conditions):
        left = {
            int(row["question_id"]): bool(row["execution_correct"])
            for row in all_per_query
            if row["condition"] == "chess_selector"
        }
        right = {
            int(row["question_id"]): bool(row["execution_correct"])
            for row in all_per_query
            if row["condition"] == "gert_selector"
        }
        ids = sorted(set(left) & set(right))
        left_only = sum(left[i] and not right[i] for i in ids)
        right_only = sum(right[i] and not left[i] for i in ids)
        discordant = left_only + right_only
        if discordant:
            tail = sum(
                math.comb(discordant, j)
                for j in range(min(left_only, right_only) + 1)
            ) / (2**discordant)
            p_value = min(1.0, 2 * tail)
        else:
            p_value = 1.0
        paired = {
            "n": len(ids),
            "both_correct": sum(left[i] and right[i] for i in ids),
            "chess_only_correct": left_only,
            "gert_only_correct": right_only,
            "neither_correct": sum(not left[i] and not right[i] for i in ids),
            "mcnemar_exact_p_two_sided": p_value,
        }

    diagnostic_lines = [
        "# CHESS routing and pipeline diagnostics",
        "",
        "| Condition | Gold Table CR | Avg Revise Calls | Avg Unit Calls | Avg Selector Fill | Avg Selector Tokens | Avg Pipeline Tokens |",
        "| :-- | --: | --: | --: | --: | --: | --: |",
    ]
    for condition in conditions:
        metric = all_metrics[condition]
        diagnostic_lines.append(
            f"| {labels[condition]} | {metric['gold_table_complete_rate']:.2f} | "
            f"{metric['avg_revision_calls']:.2f} | {metric['avg_unit_test_calls']:.2f} | "
            f"{metric['avg_selector_fill_count']:.2f} | "
            f"{metric['avg_selector_tokens']:.0f} | {metric['avg_pipeline_tokens']:.0f} |"
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
                f"- CHESS only correct: {paired['chess_only_correct']}",
                f"- GERT only correct: {paired['gert_only_correct']}",
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
                "upstream_commit": UPSTREAM_COMMIT,
                "k": config["routing"]["k"],
                "information_retriever": "disabled to prevent gold-db leakage",
                "evidence": "empty",
                "gold_db_id": "hidden from prompts; execution/evaluation only",
                "candidate_budget": "2 official prompts x 2 provider-default samples",
                "revision": "one pass for invalid or empty candidates",
                "generated_unit_tests": config["pipeline"]["generated_unit_test_count"],
                "primary_total_tokens": "Candidate generation + revise + unit-test generation/evaluation",
                "gert_routing_cost": "Excluded because GERT predictions are frozen inputs",
                "model_parameters": "Provider defaults; no sampling parameters sent",
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
    print("\n".join(lines))
    print(f"Wrote reports to {result_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="Validate inputs without calling an API")
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


if __name__ == "__main__":
    main()
