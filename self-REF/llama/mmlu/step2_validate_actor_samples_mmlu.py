from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm


HASH_ANSWER_RE = re.compile(r"####\s*([A-D])\s*[\).:-]\s*([^\n\r]+)", flags=re.IGNORECASE)
OPTION_LINE_RE = re.compile(r"^\s*([A-D])\s*[\).:-]\s*(.+?)\s*$", flags=re.IGNORECASE)


def _load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _normalize_text(raw: str) -> str:
    text = str(raw or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .。;；,，:：!?！？\"'`[](){}")
    return text


def _reference_option_from_answer(answer: str, options: Dict[str, str]) -> str:
    a = str(answer or "").strip().upper()
    if a in {"A", "B", "C", "D"}:
        return a

    ref_norm = _normalize_text(answer)
    if not ref_norm:
        return ""

    for letter, text in options.items():
        if _normalize_text(text) == ref_norm:
            return str(letter).upper()
    return ""


def _parse_options(problem: str) -> Dict[str, str]:
    options: Dict[str, str] = {}
    for ln in str(problem or "").splitlines():
        m = OPTION_LINE_RE.match(ln)
        if not m:
            continue
        options[m.group(1).upper()] = m.group(2).strip()
    return options


def extract_mmlu_prediction(solution: str) -> Tuple[str, str]:
    text = str(solution or "").strip()
    if not text:
        return "", ""

    matches = HASH_ANSWER_RE.findall(text)
    if matches:
        letter, content = matches[-1]
        return letter.upper(), str(content).strip()

    # Fallback: find last line like "A. xxx"
    for ln in reversed(text.splitlines()):
        m = OPTION_LINE_RE.match(ln)
        if m:
            return m.group(1).upper(), m.group(2).strip()

    return "", ""


def _mmlu_equal(pred_letter: str, pred_content: str, reference: str, options: Dict[str, str]) -> bool:
    del pred_content  # decision-only eval: judge by option letter only
    pred = str(pred_letter or "").upper().strip()
    ref = _reference_option_from_answer(reference, options)
    if not pred or not ref:
        return False
    return pred == ref


def _build_maps(path: str, task_id_key: str, answer_key: str, problem_key: str) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    answer_map: Dict[str, str] = {}
    option_map: Dict[str, Dict[str, str]] = {}
    for i, row in enumerate(_load_jsonl(path)):
        task_id = str(row.get(task_id_key, row.get("id", i)))
        answer_map[task_id] = str(row.get(answer_key, "")).strip()
        option_map[task_id] = _parse_options(str(row.get(problem_key, "")))
    return answer_map, option_map


def _eval_self_passed(pred_text: str, answer: str, options: Dict[str, str]) -> Tuple[bool, str]:
    if answer is None or str(answer).strip() == "":
        return False, ""

    letter, content = extract_mmlu_prediction(str(pred_text))
    passed = _mmlu_equal(letter, content, str(answer), options)
    pred = letter if letter else ""
    return bool(passed), pred


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--actor_samples_jsonl", type=str, default="llama/mmlu/output/step1.jsonl")
    p.add_argument("--data_jsonl", type=str, default="data/mmlu/mmlu_all_train.jsonl")
    p.add_argument("--output_jsonl", type=str, default="llama/mmlu/output/step2.jsonl")
    p.add_argument("--task_id_key", type=str, default="unique_id")
    p.add_argument("--answer_key", type=str, default="answer")
    p.add_argument("--problem_key", type=str, default="problem")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    answer_map, option_map = _build_maps(
        args.data_jsonl,
        task_id_key=args.task_id_key,
        answer_key=args.answer_key,
        problem_key=args.problem_key,
    )
    samples = _load_jsonl(args.actor_samples_jsonl)

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for item in tqdm(samples, desc="Validate"):
            task_id = str(item["id"])
            reasoning = str(item.get("extracted_reasoning", ""))
            answer = answer_map.get(task_id, "")
            options = option_map.get(task_id, {})

            self_passed, prediction = _eval_self_passed(
                pred_text=reasoning,
                answer=answer,
                options=options,
            )

            out = {
                "id": task_id,
                "sample_index": int(item.get("sample_index", 0)),
                "raw_output": item.get("raw_output", ""),
                "extracted_reasoning": reasoning,
                "self_passed": bool(self_passed),
                "prediction": prediction,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[step2] wrote validation to {args.output_jsonl}")


if __name__ == "__main__":
    main()
