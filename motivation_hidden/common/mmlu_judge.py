"""Standalone MMLU final option extraction."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable


def normalize_choice(value: object) -> str:
    text = "" if value is None else str(value).strip().upper()
    match = re.fullmatch(r"\(?\s*([A-Z])\s*\)?", text)
    return match.group(1) if match else text


def coerce_options(options: Any) -> list[Any]:
    if isinstance(options, list):
        return options
    if isinstance(options, str):
        try:
            parsed = json.loads(options)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def valid_labels_for_row(row: dict[str, Any]) -> list[str]:
    options = coerce_options(row.get("options"))
    if options:
        return [chr(ord("A") + idx) for idx in range(len(options))]
    answer = normalize_choice(row.get("answer"))
    if len(answer) == 1 and "A" <= answer <= "Z":
        max_idx = max(ord(answer) - ord("A"), 3)
        return [chr(ord("A") + idx) for idx in range(max_idx + 1)]
    return [chr(ord("A") + idx) for idx in range(10)]


def extract_mmlu_choice(text: str, valid_labels: Iterable[str]) -> str:
    if not isinstance(text, str):
        return ""
    labels = {str(label).strip().upper() for label in valid_labels if str(label).strip()}
    if not labels:
        return ""
    label_pattern = "".join(sorted(labels))
    tail = text[-1600:]

    final_matches = re.findall(
        rf"final\s+answer\s*[:：]?\s*\(?([{label_pattern}])\)?",
        tail,
        flags=re.IGNORECASE,
    )
    if final_matches:
        return str(final_matches[-1]).upper()
    return ""


def is_correct_choice(text: str, row: dict[str, Any]) -> tuple[str, int]:
    pred = extract_mmlu_choice(text, valid_labels_for_row(row))
    gold = normalize_choice(row.get("answer", row.get("ground_truth_answer")))
    return pred, int(bool(pred) and pred == gold)
