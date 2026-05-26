"""Standalone math final-answer extraction and judging."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path


DECISION_RE = re.compile(r"<\s*(?:self|defer|attempt)\s*>", flags=re.IGNORECASE)
NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def normalize_answer(raw: str) -> str:
    text = str(raw or "").strip()
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r"\s+", "", text)
    text = text.rstrip(".。;；,，")
    return text


def extract_final_answer(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""

    parts = text.rsplit("Final answer:", 1)
    if len(parts) > 1:
        answer = parts[1].strip()
    else:
        lower_parts = text.lower().rsplit("final answer:", 1)
        answer = lower_parts[1].strip() if len(lower_parts) > 1 else text

    decision_match = DECISION_RE.search(answer)
    if decision_match:
        answer = answer[: decision_match.start()].strip()

    answer = answer.strip().strip("*").strip().strip("$").strip()
    if answer.startswith("\\(") and answer.endswith("\\)"):
        answer = answer[2:-2].strip()
    elif answer.startswith("\\[") and answer.endswith("\\]"):
        answer = answer[2:-2].strip()
    answer = answer.strip(" \n.。;；")

    boxed_match = re.search(r"\\boxed\s*\{", answer)
    if boxed_match:
        i = boxed_match.end()
        depth = 1
        chars: list[str] = []
        while i < len(answer):
            ch = answer[i]
            if ch == "{":
                depth += 1
                chars.append(ch)
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
                chars.append(ch)
            else:
                chars.append(ch)
            i += 1
        answer = "".join(chars).strip()

    if not answer:
        numbers = NUMBER_RE.findall(text.replace(" ", ""))
        if numbers:
            return normalize_answer(numbers[-1].replace(",", ""))
        return ""

    return normalize_answer(answer)


def load_qwen_math_tools(eval_dir: str | None):
    if eval_dir is None:
        print("[WARN] Qwen math evaluator path not provided; using normalized string match.")
        return None, None

    qwen_eval_dir = Path(eval_dir)
    parser_path = qwen_eval_dir / "parser.py"
    grader_path = qwen_eval_dir / "grader.py"
    if not parser_path.exists() or not grader_path.exists():
        print(f"[WARN] Qwen math evaluator not found at {qwen_eval_dir}; using normalized string match.")
        return None, None

    parser_spec = importlib.util.spec_from_file_location("anonymous_qwen_math_parser", parser_path)
    grader_spec = importlib.util.spec_from_file_location("anonymous_qwen_math_grader", grader_path)
    if parser_spec is None or parser_spec.loader is None or grader_spec is None or grader_spec.loader is None:
        print(f"[WARN] Failed to load Qwen math evaluator from {qwen_eval_dir}; using normalized string match.")
        return None, None

    parser_mod = importlib.util.module_from_spec(parser_spec)
    grader_mod = importlib.util.module_from_spec(grader_spec)
    eval_dir_text = str(qwen_eval_dir)
    inserted = False
    if eval_dir_text not in sys.path:
        sys.path.insert(0, eval_dir_text)
        inserted = True
    try:
        try:
            parser_spec.loader.exec_module(parser_mod)
            grader_spec.loader.exec_module(grader_mod)
        except Exception as exc:
            print(f"[WARN] Failed to import Qwen math evaluator ({exc}); using normalized string match.")
            return None, None
    finally:
        if inserted:
            try:
                sys.path.remove(eval_dir_text)
            except ValueError:
                pass
    return getattr(parser_mod, "strip_string", None), getattr(grader_mod, "math_equal", None)


def answers_match(prediction_text: str, gold_answer: str, strip_string=None, math_equal=None) -> bool:
    prediction = extract_final_answer(prediction_text)
    gold = normalize_answer(gold_answer)
    if not prediction or not gold:
        return False

    if math_equal is None:
        return prediction == gold

    try:
        prediction_stripped = strip_string(prediction) if strip_string else prediction
        gold_stripped = strip_string(gold) if strip_string else gold
        return bool(
            math_equal(
                prediction_stripped,
                gold_stripped,
                include_percentage=True,
                is_close=True,
                timeout=True,
            )
        )
    except TypeError:
        return bool(math_equal(prediction, gold))
    except Exception:
        return False
