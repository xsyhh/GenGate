#!/usr/bin/env python3
"""Rewrite MATH-500 into unified jsonl schema with collision-safe unique_id."""

from __future__ import annotations

import argparse
from pathlib import Path

from prepare_math_data import make_math500_rows, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--math500-dir", type=Path, default=Path("data/MATH-500"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/math/math500_test.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = make_math500_rows(args.math500_dir)
    write_jsonl(args.output, rows)

    print(f"MATH-500 test: {len(rows)} -> {args.output}")


if __name__ == "__main__":
    main()
