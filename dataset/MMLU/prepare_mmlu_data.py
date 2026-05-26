#!/usr/bin/env python3
"""Prepare MMLU(all) parquet files into unified jsonl files.

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
import sys
from pathlib import Path
from typing import Iterable


DATASET_ROOT = Path(__file__).resolve().parents[1]
if str(DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASET_ROOT))

from common.multiple_choice import (  # noqa: E402
    build_problem_with_options,
    extract_labeled_answer,
    normalize_options,
    normalize_text,
)


def rows_from_records(
    records: Iterable[dict],
    split: str,
    start_idx: int = 0,
    id_prefix: str = "mmlu/all",
) -> list[dict]:
    rows: list[dict] = []
    idx = int(start_idx)

    for rec in records:
        question = normalize_text(rec.get("question"))
        subject = normalize_text(rec.get("subject")) or "MMLU"
        choices = normalize_options(rec.get("choices"))

        answer_label, answer_text = extract_labeled_answer(choices, rec.get("answer"))
        problem = build_problem_with_options(question, choices)
        if answer_label and answer_text:
            solution = f"Correct option: {answer_label}\nAnswer: {answer_text}"
        elif answer_text:
            solution = f"Answer: {answer_text}"
        else:
            solution = ""

        rows.append(
            {
                "problem": problem,
                "solution": solution,
                "subject": subject,
                # Keep MC gold answer as option letter for downstream letter-based validation.
                "answer": answer_label if answer_label else answer_text,
                "level": "",
                "unique_id": f"{id_prefix}/{split}/{idx:06d}",
            }
        )
        idx += 1

    return rows


def _read_parquet_records(path: Path) -> list[dict]:
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError(
            "Reading parquet requires pandas (+ pyarrow/fastparquet). "
            "Please install dependencies first."
        ) from exc

    df = pd.read_parquet(path)
    return df.to_dict(orient="records")


def make_split_rows(mmlu_dir: Path, split: str, source_split: str) -> list[dict]:
    rows: list[dict] = []
    idx = 0

    split_dir = mmlu_dir / "all"
    parquet_files = sorted(split_dir.glob(f"{source_split}-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found for split={source_split} in {split_dir}")

    for pq in parquet_files:
        recs = _read_parquet_records(pq)
        split_rows = rows_from_records(recs, split=split, start_idx=idx)
        rows.extend(split_rows)
        idx += len(split_rows)

    return rows


def ensure_global_unique_ids(*datasets: list[dict]) -> None:
    all_ids = []
    for ds in datasets:
        all_ids.extend(rec["unique_id"] for rec in ds)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Found duplicated unique_id across output datasets")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmlu-dir", type=Path, default=Path("data/mmlu"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mmlu"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_rows = make_split_rows(args.mmlu_dir, split="train", source_split="auxiliary_train")
    dev_rows = make_split_rows(args.mmlu_dir, split="dev", source_split="dev")
    validation_rows = make_split_rows(args.mmlu_dir, split="validation", source_split="validation")
    test_rows = make_split_rows(args.mmlu_dir, split="test", source_split="test")

    ensure_global_unique_ids(train_rows, dev_rows, validation_rows, test_rows)

    write_jsonl(args.output_dir / "mmlu_all_train.jsonl", train_rows)
    write_jsonl(args.output_dir / "mmlu_all_dev.jsonl", dev_rows)
    write_jsonl(args.output_dir / "mmlu_all_validation.jsonl", validation_rows)
    write_jsonl(args.output_dir / "mmlu_all_test.jsonl", test_rows)

    print(f"mmlu all train: {len(train_rows)}")
    print(f"mmlu all dev: {len(dev_rows)}")
    print(f"mmlu all validation: {len(validation_rows)}")
    print(f"mmlu all test: {len(test_rows)}")
    print(f"output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
