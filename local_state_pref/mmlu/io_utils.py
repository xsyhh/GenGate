from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any


def iter_jsonl(path: str) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def is_completed_rollout(row: dict[str, Any]) -> bool:
    ratio = row.get("ratio")
    if ratio is None:
        return True
    try:
        return abs(float(ratio) - 1.0) < 1e-8
    except (TypeError, ValueError):
        return False


def iter_completed_jsonl(path: str) -> Iterator[dict[str, Any]]:
    for row in iter_jsonl(path):
        if is_completed_rollout(row):
            yield row


def load_jsonl(path: str) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def dump_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_mmlu_rows(
    jsonl_path: str,
    *,
    task_id_key: str = "unique_id",
    problem_key: str = "problem",
    answer_key: str = "answer",
) -> dict[str, dict[str, Any]]:
    raw_map: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(iter_jsonl(jsonl_path)):
        task_id = str(row.get(task_id_key, "")).strip() or str(idx)
        raw_map[task_id] = {
            "problem": str(row.get(problem_key, "")).strip(),
            "answer": str(row.get(answer_key, "")).strip(),
            "solution": str(row.get("solution", "")).strip(),
            "subject": str(row.get("subject", "")).strip(),
        }
    return raw_map
