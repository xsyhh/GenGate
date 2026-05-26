"""Shared helpers for multiple-choice dataset normalization."""

from __future__ import annotations

from numbers import Integral


OPTION_LABELS = tuple(chr(ord("A") + i) for i in range(26))


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_options(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [normalize_text(x) for x in value]
    if isinstance(value, tuple):
        return [normalize_text(x) for x in value]
    if isinstance(value, str):
        return [normalize_text(value)]
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return [normalize_text(x) for x in converted]
        return [normalize_text(converted)]
    try:
        return [normalize_text(x) for x in list(value)]  # type: ignore[arg-type]
    except Exception:
        return []


def label_for_index(index: int) -> str:
    if 0 <= index < len(OPTION_LABELS):
        return OPTION_LABELS[index]
    return str(index)


def answer_to_index(answer_raw: object) -> int:
    if isinstance(answer_raw, bool):
        return int(answer_raw)
    if isinstance(answer_raw, Integral):
        return int(answer_raw)
    if hasattr(answer_raw, "item"):
        try:
            return answer_to_index(answer_raw.item())
        except Exception:
            pass

    text = normalize_text(answer_raw)
    if not text:
        return -1

    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        return int(text)

    up = text.upper()
    if up in OPTION_LABELS:
        return OPTION_LABELS.index(up)

    return -1


def build_problem_with_options(question: str, options: list[str]) -> str:
    if not options:
        return question

    option_lines = [f"{label_for_index(i)}. {option}" for i, option in enumerate(options)]
    return f"{question}\n" + "\n".join(option_lines)


def extract_labeled_answer(options: list[str], answer_raw: object) -> tuple[str, str]:
    idx = answer_to_index(answer_raw)
    if 0 <= idx < len(options):
        return label_for_index(idx), options[idx]

    text = normalize_text(answer_raw)
    if text:
        return text, text
    return "", ""
