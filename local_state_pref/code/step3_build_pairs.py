from __future__ import annotations

import argparse
from pathlib import Path

from build_pairs import build_pair_records
from common import build_chat_s0_context, build_chat_s1_context, dump_jsonl, load_jsonl, read_code_rows


def _parse_sample_index(row: dict) -> int:
    value = row.get("sample_index", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _build_first_rollout_success_map(evaluated_rows: list[dict]) -> dict[str, float]:
    first_map: dict[str, tuple[int, float]] = {}
    for row in evaluated_rows:
        task_id = str(row.get("task_id", "")).strip()
        extracted_code = str(row.get("extracted_code", "")).strip()
        if not task_id or not extracted_code:
            continue
        sample_index = _parse_sample_index(row)
        self_passed = 1.0 if bool(row.get("self_passed")) else 0.0
        prev = first_map.get(task_id)
        if prev is None or sample_index < prev[0]:
            first_map[task_id] = (sample_index, self_passed)
    return {task_id: value for task_id, (_, value) in first_map.items()}


def _apply_ablation_options(
    records: list[dict],
    *,
    include_states: str,
    s0_target_mode: str,
    first_rollout_success: dict[str, float],
) -> list[dict]:
    if include_states == "s0_only":
        filtered = [record for record in records if str(record.get("state_type")) == "s0"]
    elif include_states == "s1_only":
        filtered = [record for record in records if str(record.get("state_type")) == "s1"]
    else:
        filtered = list(records)

    if s0_target_mode == "first_rollout":
        for record in filtered:
            if str(record.get("state_type")) != "s0":
                continue
            task_id = str(record.get("task_id", ""))
            if task_id in first_rollout_success:
                record["target_prob"] = float(first_rollout_success[task_id])
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Build same-state preference pairs from evaluated attempt rollouts.")
    parser.add_argument("--evaluated", required=True, help="JSONL output from the existing step2_evaluate.py")
    parser.add_argument("--data_csv", required=True, help="Original code train/eval CSV")
    parser.add_argument("--output", required=True, help="Output pair JSONL path")
    parser.add_argument("--model_path", default=None, help="Optional model path. If set, wrap contexts with tokenizer chat template.")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--include_states",
        choices=["both", "s0_only", "s1_only"],
        default="both",
        help="Keep both states (default), or only s0/s1 records.",
    )
    parser.add_argument(
        "--s0_target_mode",
        choices=["p_hat", "first_rollout"],
        default="p_hat",
        help="How to set s0 target_prob: rollout mean p_hat (default) or first rollout outcome.",
    )
    args = parser.parse_args()

    rows = read_code_rows(args.data_csv)
    raw_map = {
        row.task_id: {
            "problem": row.problem,
            "starter_code": row.starter_code,
        }
        for row in rows
    }

    evaluated = load_jsonl(args.evaluated)
    records = build_pair_records(raw_map, evaluated)
    first_rollout_success = _build_first_rollout_success_map(evaluated)
    records = _apply_ablation_options(
        records,
        include_states=args.include_states,
        s0_target_mode=args.s0_target_mode,
        first_rollout_success=first_rollout_success,
    )

    if args.model_path is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
        for record in records:
            raw = raw_map[str(record["task_id"])]
            if str(record["state_type"]) == "s0":
                record["context"] = build_chat_s0_context(tokenizer, raw["problem"], raw["starter_code"])
            else:
                record["context"] = build_chat_s1_context(
                    tokenizer,
                    raw["problem"],
                    raw["starter_code"],
                    str(record.get("code", "")),
                )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    dump_jsonl(args.output, records)
    s0_count = sum(1 for record in records if str(record.get("state_type")) == "s0")
    s1_count = sum(1 for record in records if str(record.get("state_type")) == "s1")
    print(f"Saved {len(records)} pair records to: {args.output}")
    print(
        f"Options: include_states={args.include_states}, s0_target_mode={args.s0_target_mode}, "
        f"s0_count={s0_count}, s1_count={s1_count}"
    )


if __name__ == "__main__":
    main()
