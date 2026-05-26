#!/usr/bin/env python3
"""Extract hendrycks_math train/test parquet files into unified jsonl files."""

from __future__ import annotations

import argparse
from pathlib import Path

from prepare_math_data import build_answer_extractor, make_hendrycks_rows, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hendrycks-dir", type=Path, default=Path("data/hendrycks_math"))
    parser.add_argument(
        "--train-output", type=Path, default=Path("data/processed/math/hendrycks_math_train.jsonl")
    )
    parser.add_argument(
        "--test-output", type=Path, default=Path("data/processed/math/hendrycks_math_test.jsonl")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    extract_answer = build_answer_extractor(repo_root)

    train_rows = make_hendrycks_rows(args.hendrycks_dir, "train", extract_answer)
    test_rows = make_hendrycks_rows(args.hendrycks_dir, "test", extract_answer)

    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.test_output, test_rows)

    print(f"hendrycks_math train: {len(train_rows)} -> {args.train_output}")
    print(f"hendrycks_math test: {len(test_rows)} -> {args.test_output}")


if __name__ == "__main__":
    main()
