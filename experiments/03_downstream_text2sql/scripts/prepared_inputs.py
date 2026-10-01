"""Read frozen prepared prompts as the downstream benchmark input."""

from __future__ import annotations

import json
from pathlib import Path


def load_prepared_dataset(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 100:
        raise ValueError(f"Expected 100 prepared queries: {path}")
    ids: set[int] = set()
    normalized = []
    for row in rows:
        if row.get("condition") != "gert":
            raise ValueError(f"Expected GERT prepared prompts: {path}")
        qid = int(row["question_id"])
        if qid in ids:
            raise ValueError(f"Duplicate question_id={qid}: {path}")
        ids.add(qid)
        candidates = list(row["candidate_tables"])
        if len(candidates) != 15 or len({t.casefold() for t in candidates}) != 15:
            raise ValueError(f"Expected 15 distinct candidate tables for question_id={qid}")
        normalized.append({
            **row,
            "SQL": row["gold_sql"],
            "predicted_schemas_at15": candidates,
        })
    return normalized
