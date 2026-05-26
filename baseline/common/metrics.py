from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from .progress import progress_iter


METADATA_FIELDNAMES = [
    "task_id",
    "route",
    "model_decision",
    "first_p_defer",
    "first_p_self",
    "first_margin",
    "post_p_defer",
    "post_p_self",
    "post_margin",
    "p_defer",
    "p_self",
    "margin",
    "self_passed",
    "expert_passed",
    "answer_len",
    "actual_local_tokens",
    "method",
    "domain",
    "model_slug",
    "dataset_slug",
    "score",
]


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def row_from_score(
    *,
    task_id: str,
    score: float,
    self_passed: int,
    threshold: float,
    method: str,
    domain: str,
    model_slug: str,
    dataset_slug: str,
    answer_len: int = 0,
    actual_local_tokens: int = 0,
    route_self: str = "post_self",
    route_defer: str = "post_defer",
    expert_passed: int | float | str = "",
    stage: str = "post",
) -> dict[str, Any]:
    p_self = max(0.0, min(1.0, float(score)))
    p_defer = 1.0 - p_self
    margin = math.log(max(p_defer, 1e-12)) - math.log(max(p_self, 1e-12))
    model_decision = "self" if p_self >= threshold else "defer"
    route = route_self if model_decision == "self" else route_defer
    row = {
        "task_id": task_id,
        "route": route,
        "model_decision": model_decision,
        "first_p_defer": "",
        "first_p_self": "",
        "first_margin": "",
        "post_p_defer": "",
        "post_p_self": "",
        "post_margin": "",
        "p_defer": round(p_defer, 6),
        "p_self": round(p_self, 6),
        "margin": round(margin, 4),
        "self_passed": int(self_passed),
        "expert_passed": expert_passed,
        "answer_len": int(answer_len),
        "actual_local_tokens": int(actual_local_tokens),
        "method": method,
        "domain": domain,
        "model_slug": model_slug,
        "dataset_slug": dataset_slug,
        "score": round(p_self, 6),
    }
    if stage == "pre":
        row["first_p_defer"] = round(p_defer, 6)
        row["first_p_self"] = round(p_self, 6)
        row["first_margin"] = round(margin, 4)
    else:
        row["post_p_defer"] = round(p_defer, 6)
        row["post_p_self"] = round(p_self, 6)
        row["post_margin"] = round(margin, 4)
    return row


def write_metadata(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(METADATA_FIELDNAMES)
    extras = sorted({key for row in rows for key in row.keys()} - set(fieldnames))
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + extras)
        writer.writeheader()
        writer.writerows(rows)


def robust_read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return [{str(k).strip(): v for k, v in row.items()} for row in csv.DictReader(f)]


def load_expert_map(expert_csv: str | Path | None, *, assume_correct: bool = False) -> dict[str, float] | None:
    if expert_csv is None:
        return {} if assume_correct else None
    rows = robust_read_csv(expert_csv)
    out = {}
    for row in rows:
        task_id = str(row.get("task_id", "")).strip()
        if not task_id:
            continue
        value = row.get("expert_passed", row.get("self_passed", "0"))
        try:
            out[task_id] = float(value)
        except (TypeError, ValueError):
            out[task_id] = 1.0 if str(value).strip().lower() == "true" else 0.0
    return out


def attach_expert(rows: list[dict[str, Any]], expert_map: dict[str, float] | None, *, assume_correct: bool = False) -> list[dict[str, Any]]:
    if expert_map is None and not assume_correct:
        return rows
    out = []
    for row in rows:
        updated = dict(row)
        if assume_correct and not expert_map:
            updated["expert_passed"] = 1.0
        else:
            updated["expert_passed"] = float(expert_map.get(str(row["task_id"]), 0.0))
        out.append(updated)
    return out


def sweep_single_threshold(records: list[dict[str, Any]], thresholds: list[float] | None = None) -> list[dict[str, Any]]:
    if thresholds is None:
        thresholds = [idx / 100.0 for idx in range(101)]
    rows = []
    n = len(records)
    for threshold in progress_iter(thresholds, desc="single threshold sweep", total=len(thresholds)):
        n_defer = 0
        correct = 0.0
        for record in records:
            score = float(record["score"])
            self_passed = float(record.get("self_passed", 0))
            expert_passed = float(record.get("expert_passed", 1.0))
            if score >= threshold:
                correct += self_passed
            else:
                n_defer += 1
                correct += expert_passed
        rows.append(
            {
                "threshold": threshold,
                "expert_rate": n_defer / max(n, 1),
                "accuracy": correct / max(n, 1),
            }
        )
    return rows


def write_rows_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_threshold_for_budget(records: list[dict[str, Any]], target_expert_rate: float = 0.5) -> float:
    curve = sweep_single_threshold(records)
    target = float(target_expert_rate)
    best = min(curve, key=lambda row: (abs(float(row["expert_rate"]) - target), -float(row["accuracy"])))
    return float(best["threshold"])


def cascade_grid(
    records: list[dict[str, Any]],
    pre_thresholds: list[float] | None = None,
    post_thresholds: list[float] | None = None,
) -> list[dict[str, Any]]:
    if pre_thresholds is None:
        pre_thresholds = [idx / 100.0 for idx in range(101)]
    if post_thresholds is None:
        post_thresholds = [idx / 100.0 for idx in range(101)]
    n = len(records)
    rows = []
    for pre_t in progress_iter(pre_thresholds, desc="cascade grid", total=len(pre_thresholds)):
        for post_t in post_thresholds:
            correct = 0.0
            early_defer = 0
            post_defer = 0
            post_self = 0
            attempted = 0
            for record in records:
                pre = float(record["pre_score"])
                post = float(record["post_score"])
                self_passed = float(record.get("self_passed", 0))
                expert_passed = float(record.get("expert_passed", 1.0))
                if pre < pre_t:
                    early_defer += 1
                    correct += expert_passed
                    continue
                attempted += 1
                if post < post_t:
                    post_defer += 1
                    correct += expert_passed
                else:
                    post_self += 1
                    correct += self_passed
            rows.append(
                {
                    "pre_threshold": pre_t,
                    "post_threshold": post_t,
                    "accuracy": correct / max(n, 1),
                    "expert_rate": (early_defer + post_defer) / max(n, 1),
                    "attempt_rate": attempted / max(n, 1),
                    "early_defer": early_defer,
                    "post_defer": post_defer,
                    "post_self": post_self,
                }
            )
    return rows


def pareto_envelope(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (float(row["expert_rate"]), -float(row["accuracy"])))
    out = []
    best_acc = -1.0
    for row in ordered:
        acc = float(row["accuracy"])
        if acc > best_acc + 1e-12:
            out.append(row)
            best_acc = acc
    return out
