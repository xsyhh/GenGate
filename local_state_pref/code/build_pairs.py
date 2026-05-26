from __future__ import annotations

from collections import defaultdict
from typing import Any

from common import ACTIONS, build_s0_context, build_s1_context


def build_pair_records(raw_map: dict[str, dict[str, Any]], evaluated_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated_rows:
        grouped[str(row["task_id"])].append(row)

    records = []
    for task_id, rows in grouped.items():
        if task_id not in raw_map:
            continue

        valid_rows = [row for row in rows if str(row.get("extracted_code", "")).strip()]
        if not valid_rows:
            continue

        valid_rows.sort(key=lambda row: int(row.get("sample_index", 0)))
        raw = raw_map[task_id]
        rollout_count = len(valid_rows)
        p_hat = sum(1.0 if row.get("self_passed") else 0.0 for row in valid_rows) / rollout_count

        records.append(
            {
                "state_type": "s0",
                "task_id": task_id,
                "sample_index": None,
                "context": build_s0_context(raw["problem"], raw["starter_code"]),
                "code": "",
                "action_a": ACTIONS["attempt"],
                "action_b": ACTIONS["defer"],
                "target_prob": float(p_hat),
                "state_weight": 1.0,
                "rollout_count": rollout_count,
            }
        )

        per_rollout_weight = 1.0 / rollout_count
        for row in valid_rows:
            records.append(
                {
                    "state_type": "s1",
                    "task_id": task_id,
                    "sample_index": row.get("sample_index"),
                    "context": build_s1_context(
                        raw["problem"],
                        raw["starter_code"],
                        str(row["extracted_code"]),
                    ),
                    "code": str(row["extracted_code"]),
                    "action_a": ACTIONS["self"],
                    "action_b": ACTIONS["defer"],
                    "target_prob": 1.0 if row.get("self_passed") else 0.0,
                    "state_weight": per_rollout_weight,
                    "rollout_count": rollout_count,
                }
            )

    return records
