from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_pairs import build_chat_s0_context, build_chat_s1_context, build_pair_records
from io_utils import iter_completed_jsonl, read_mmlu_rows


def _write_records(f, records: list[dict[str, Any]]) -> int:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MMLU same-state BCE preference pairs from completed rollouts.")
    parser.add_argument("--evaluated", required=True, help="JSONL with completed MMLU rollouts or sliced drafts.")
    parser.add_argument("--data_jsonl", required=True, help="Original MMLU train/eval JSONL.")
    parser.add_argument("--output", required=True, help="Output pair JSONL path.")
    parser.add_argument("--model_path", default=None, help="Optional tokenizer path. If set, wrap contexts with chat template.")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--task_id_key", default="unique_id")
    parser.add_argument("--problem_key", default="problem")
    parser.add_argument("--answer_key", default="answer")
    args = parser.parse_args()

    raw_map = read_mmlu_rows(
        args.data_jsonl,
        task_id_key=args.task_id_key,
        problem_key=args.problem_key,
        answer_key=args.answer_key,
    )
    tokenizer = None
    if args.model_path is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    current_task_id: str | None = None
    current_rows: list[dict[str, Any]] = []

    def flush_task(f) -> None:
        nonlocal total_records, current_rows
        if not current_rows:
            return
        records = build_pair_records(raw_map, current_rows)
        for record in records:
            if tokenizer is None:
                continue
            raw = raw_map[str(record["task_id"])]
            if str(record["state_type"]) == "s0":
                record["context"] = build_chat_s0_context(tokenizer, raw["problem"])
                continue
            record["context"] = build_chat_s1_context(
                tokenizer,
                raw["problem"],
                str(record.get("reasoning", "")),
            )
        total_records += _write_records(f, records)
        current_rows = []

    with output_path.open("w", encoding="utf-8") as f:
        for row in iter_completed_jsonl(args.evaluated):
            task_id = str(row.get("task_id", ""))
            if current_task_id is None:
                current_task_id = task_id
            if task_id != current_task_id:
                flush_task(f)
                current_task_id = task_id
            current_rows.append(row)
        flush_task(f)

    print(f"Saved {total_records} MMLU pair records to: {args.output}")


if __name__ == "__main__":
    main()
