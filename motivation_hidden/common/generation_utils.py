"""Shared streaming helpers for hidden-state step1 scripts."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Iterable


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch
    except ImportError:
        return

    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_ratios(text: str) -> list[float]:
    ratios = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not ratios:
        raise ValueError("ratio list is empty")
    return ratios


def slice_raw_text_by_tokens(raw_text: str, ratios: Iterable[float], tokenizer: Any) -> dict[float, dict[str, Any]]:
    if not isinstance(raw_text, str):
        raw_text = ""
    token_ids = tokenizer.encode(raw_text, add_special_tokens=False)
    total_tokens = len(token_ids)
    output: dict[float, dict[str, Any]] = {}
    for ratio in ratios:
        ratio = float(ratio)
        if ratio <= 0.0:
            prefix_ids = []
        elif ratio >= 1.0:
            prefix_ids = token_ids
        else:
            cut_idx = max(1, int(math.ceil(total_tokens * ratio))) if total_tokens else 0
            prefix_ids = token_ids[:cut_idx]
        output[ratio] = {
            "prefix_raw": tokenizer.decode(prefix_ids, skip_special_tokens=False),
            "prefix_token_len": len(prefix_ids),
            "full_token_len": total_tokens,
        }
    return output


def load_completed_task_ids(out_jsonl: Path, expected_k: int) -> set[str]:
    if not out_jsonl.exists():
        return set()
    counts: dict[str, int] = {}
    with out_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                ratio = float(row.get("ratio", -1.0))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if abs(ratio - 1.0) > 1e-9:
                continue
            task_id = str(row.get("task_id", ""))
            if task_id:
                counts[task_id] = counts.get(task_id, 0) + 1
    return {task_id for task_id, count in counts.items() if count >= expected_k}


def iter_chunks(rows: Iterable[dict[str, Any]], chunk_size: int, start: int, limit: int | None, completed: set[str]):
    chunk = []
    considered = 0
    for idx, row in enumerate(rows):
        if idx < start:
            continue
        if limit is not None and considered >= limit:
            break
        considered += 1
        if str(row.get("task_id", "")) in completed:
            continue
        chunk.append(row)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def select_rows(
    rows: Iterable[dict[str, Any]],
    limit: int | None,
    seed: int,
    sample_mode: str = "head",
    stratify_key: str = "subject",
) -> list[dict[str, Any]]:
    selected = list(rows)
    if limit is None or limit >= len(selected):
        return selected
    if limit <= 0:
        return []
    if sample_mode == "head":
        return selected[:limit]

    indexed = list(enumerate(selected))
    rng = random.Random(seed)
    if sample_mode == "random":
        sampled_indices = sorted(rng.sample(range(len(selected)), limit))
        return [selected[idx] for idx in sampled_indices]
    if sample_mode != "stratified":
        raise ValueError(f"unknown sample_mode: {sample_mode}")

    groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for idx, row in indexed:
        key = str(row.get(stratify_key, "") or "__missing__")
        groups.setdefault(key, []).append((idx, row))

    total = len(selected)
    allocations: dict[str, int] = {}
    remainders = []
    for key, group in groups.items():
        exact = limit * len(group) / total
        base = min(len(group), int(exact))
        allocations[key] = base
        remainders.append((exact - base, key))

    remaining = limit - sum(allocations.values())
    for _, key in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        if allocations[key] < len(groups[key]):
            allocations[key] += 1
            remaining -= 1

    sampled: list[tuple[int, dict[str, Any]]] = []
    for key, group in groups.items():
        n = allocations[key]
        if n <= 0:
            continue
        sampled.extend(rng.sample(group, n))
    sampled.sort(key=lambda item: item[0])
    return [row for _, row in sampled]


def build_chat_prompts(rows: list[dict[str, Any]], tokenizer: Any) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt_text"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]


def write_sliced_records(
    f_out,
    row: dict[str, Any],
    sample_idx: int,
    full_raw: str,
    ratios: list[float],
    tokenizer: Any,
    y_final: int,
    pred_answer: str = "",
    extra: dict[str, Any] | None = None,
) -> int:
    extra = extra or {}
    count = 0
    for ratio, sliced in slice_raw_text_by_tokens(full_raw, ratios, tokenizer).items():
        record = {
            "domain": row["domain"],
            "task_id": row["task_id"],
            "sample_idx": int(sample_idx),
            "ratio": float(ratio),
            "prompt_text": row["prompt_text"],
            "problem_text": row.get("problem", ""),
            "starter_code": row.get("starter_code", ""),
            "subject": row.get("subject", ""),
            "level": row.get("level", ""),
            "ground_truth_answer": row.get("answer", ""),
            "pred_answer": pred_answer,
            "prefix_raw": sliced["prefix_raw"],
            "full_raw": full_raw,
            "prefix_token_len": sliced["prefix_token_len"],
            "full_token_len": sliced["full_token_len"],
            "y_final": int(y_final),
        }
        record.update(extra)
        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1
    return count


def write_pass1_summary(out_jsonl: Path, summary_path: Path, domain: str) -> dict[str, Any]:
    n_rollouts = 0
    n_pass = 0
    n_first = 0
    n_first_pass = 0
    task_any: dict[str, int] = {}
    with out_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                ratio = float(row.get("ratio", -1.0))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if abs(ratio - 1.0) > 1e-9:
                continue
            y_final = int(row.get("y_final", 0))
            task_id = str(row.get("task_id", ""))
            n_rollouts += 1
            n_pass += y_final
            if int(row.get("sample_idx", -1)) == 0:
                n_first += 1
                n_first_pass += y_final
            if task_id:
                task_any[task_id] = max(task_any.get(task_id, 0), y_final)
    summary = {
        "domain": domain,
        "source_jsonl": str(out_jsonl),
        "n_ratio1_rollouts": n_rollouts,
        "n_ratio1_pass": n_pass,
        "raw_pass1": n_pass / n_rollouts if n_rollouts else 0.0,
        "n_first_sample_rollouts": n_first,
        "n_first_sample_pass": n_first_pass,
        "raw_pass1_first_sample": n_first_pass / n_first if n_first else 0.0,
        "n_tasks_with_ratio1": len(task_any),
        "n_tasks_any_pass": sum(task_any.values()),
        "task_any_pass_rate": sum(task_any.values()) / len(task_any) if task_any else 0.0,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
