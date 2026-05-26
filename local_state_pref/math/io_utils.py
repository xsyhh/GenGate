from __future__ import annotations

import json
from typing import Any


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_math_rows(jsonl_path: str, task_id_key: str = "unique_id", problem_key: str = "problem") -> dict[str, dict[str, Any]]:
    raw_map = {}
    for row in load_jsonl(jsonl_path):
        task_id = str(row.get(task_id_key, "")).strip()
        if task_id:
            raw_map[task_id] = {
                "problem": str(row[problem_key]),
            }
    return raw_map
