from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--lines", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_json(args.input_file, lines=args.lines)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_file, index=False, encoding="utf-8-sig")
    print(f"转换完成，已保存至 {args.output_file}")


if __name__ == "__main__":
    main()
