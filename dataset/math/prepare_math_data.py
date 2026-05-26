#!/usr/bin/env python3
"""Prepare unified math datasets from hendrycks_math and MATH-500.

Output schema for every record:
- problem
- solution
- subject
- answer
- level
- unique_id
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable

import pandas as pd


def _fallback_extract_answer(solution: str) -> str:
    text = (solution or "").strip()
    if not text:
        return ""

    if "boxed" in text:
        ans = text.split("boxed")[-1]
        if ans.startswith("{"):
            stack = 1
            chars = []
            for ch in ans[1:]:
                if ch == "{":
                    stack += 1
                    chars.append(ch)
                elif ch == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    chars.append(ch)
                else:
                    chars.append(ch)
            return "".join(chars).strip()
        return ans.split("$")[0].strip()

    m = re.findall(r"-?\d*\.?\d+", text.replace(",", ""))
    if m:
        return m[-1]
    return ""


def build_answer_extractor(repo_root: Path) -> Callable[[str], str]:
    eval_dir = repo_root / "Qwen2.5-Math" / "evaluation"
    latex_dir = eval_dir / "latex2sympy"

    if str(eval_dir) not in sys.path:
        sys.path.append(str(eval_dir))
    if str(latex_dir) not in sys.path:
        sys.path.append(str(latex_dir))

    try:
        from parser import extract_answer as qwen_extract_answer  # type: ignore

        def _extract(solution: str) -> str:
            ans = qwen_extract_answer(solution or "", "math")
            if ans:
                return ans
            return _fallback_extract_answer(solution)

        return _extract
    except Exception:
        return _fallback_extract_answer


def normalize_subject(subject_raw: str | None, default_subject: str) -> str:
    subject = (subject_raw or "").strip()
    if subject:
        return subject
    return default_subject.replace("_", " ").title()


def make_hendrycks_rows(
    hendrycks_dir: Path,
    split: str,
    extract_answer: Callable[[str], str],
) -> list[dict]:
    rows: list[dict] = []
    idx = 0

    for subject_dir in sorted(p for p in hendrycks_dir.iterdir() if p.is_dir()):
        parquet_files = sorted(subject_dir.glob(f"{split}-*.parquet"))
        for pq in parquet_files:
            df = pd.read_parquet(pq)
            for rec in df.to_dict(orient="records"):
                problem = str(rec.get("problem", "") or "").strip()
                solution = str(rec.get("solution", "") or "").strip()
                subject = normalize_subject(rec.get("type"), subject_dir.name)
                level = rec.get("level", "")

                rows.append(
                    {
                        "problem": problem,
                        "solution": solution,
                        "subject": subject,
                        "answer": extract_answer(solution),
                        "level": level,
                        "unique_id": f"hendrycks_math/{split}/{idx:06d}",
                    }
                )
                idx += 1

    return rows


def make_math500_rows(math500_dir: Path) -> list[dict]:
    test_file = math500_dir / "test.jsonl"
    rows: list[dict] = []

    with test_file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            problem = str(rec.get("problem", "") or "").strip()
            solution = str(rec.get("solution", "") or "").strip()
            subject = str(rec.get("subject", "") or "").strip()
            level = rec.get("level", "")
            answer = str(rec.get("answer", "") or "").strip()
            if not answer:
                raise ValueError(f"MATH-500 record missing answer at line {i + 1}")

            rows.append(
                {
                    "problem": problem,
                    "solution": solution,
                    "subject": subject,
                    "answer": answer,
                    "level": level,
                    "unique_id": f"math500/test/{i:06d}",
                }
            )

    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def ensure_global_unique_ids(*datasets: list[dict]) -> None:
    all_ids = []
    for ds in datasets:
        all_ids.extend(rec["unique_id"] for rec in ds)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Found duplicated unique_id across output datasets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--math500-dir", type=Path, default=Path("data/MATH-500"))
    parser.add_argument("--hendrycks-dir", type=Path, default=Path("data/hendrycks_math"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/math"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    extract_answer = build_answer_extractor(repo_root)

    hm_train = make_hendrycks_rows(args.hendrycks_dir, "train", extract_answer)
    hm_test = make_hendrycks_rows(args.hendrycks_dir, "test", extract_answer)
    m500_test = make_math500_rows(args.math500_dir)

    ensure_global_unique_ids(hm_train, hm_test, m500_test)

    write_jsonl(args.output_dir / "hendrycks_math_train.jsonl", hm_train)
    write_jsonl(args.output_dir / "hendrycks_math_test.jsonl", hm_test)
    write_jsonl(args.output_dir / "math500_test.jsonl", m500_test)

    print(f"hendrycks_math train: {len(hm_train)}")
    print(f"hendrycks_math test: {len(hm_test)}")
    print(f"MATH-500 test: {len(m500_test)}")
    print(f"output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
