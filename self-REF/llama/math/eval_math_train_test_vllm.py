from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from math_eval_setup import ensure_math_eval_on_path
from tokens import strip_trailing_decision_token

ensure_math_eval_on_path()

from parser import extract_answer, strip_string  # type: ignore
from grader import math_equal  # type: ignore


MATH_PROMPT_TEMPLATE = """You are a math reasoning agent.
### Question:
{problem}
### INSTRUCION:
1. First, solve the problem step by step.
2. Put your final answer within \\boxed{{}}.
3. Output only your reasoning and the final boxed answer.
### FORMAT:
[your reasoning and final boxed answer]
"""

@dataclass
class EvalStats:
    split: str
    total: int
    correct: int
    accuracy: float


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _load_jsonl(path: str, limit: Optional[int] = None) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _strip_decision_token_if_any(text: str) -> str:
    cleaned, _ = strip_trailing_decision_token(text)
    return cleaned


def _math_equal_passed(
    pred_text: str,
    answer: str,
) -> Tuple[bool, str, str]:
    reference = str(answer or "").strip()
    if not reference:
        return False, "", ""

    prediction = str(extract_answer(str(pred_text), "math"))
    reference_norm = str(strip_string(reference))
    passed = bool(math_equal(prediction, reference_norm))
    return passed, prediction, reference_norm


def _build_prompts(
    rows: List[Dict],
    tokenizer,
    task_id_key: str,
    problem_key: str,
) -> Tuple[List[str], List[Dict]]:
    prompts: List[str] = []
    metas: List[Dict] = []
    for i, row in enumerate(rows):
        task_id = str(row.get(task_id_key, row.get("id", i)))
        problem = str(row.get(problem_key, ""))
        answer = str(row.get("answer", "")).strip()

        user_prompt = MATH_PROMPT_TEMPLATE.format(problem=problem)
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)
        metas.append(
            {
                "id": task_id,
                "answer": answer,
                "problem": problem,
            }
        )
    return prompts, metas


def _evaluate_split(
    split_name: str,
    rows: List[Dict],
    llm: LLM,
    tokenizer,
    sampling_params: SamplingParams,
    task_id_key: str,
    problem_key: str,
    batch_size: int,
    save_path: Optional[str],
) -> EvalStats:
    prompts, metas = _build_prompts(rows, tokenizer, task_id_key=task_id_key, problem_key=problem_key)

    total = 0
    correct = 0

    out_f = None
    if save_path:
        _ensure_dir(Path(save_path).parent.as_posix())
        out_f = open(save_path, "w", encoding="utf-8")

    try:
        for st in tqdm(range(0, len(prompts), batch_size), desc=f"Infer {split_name}"):
            batch_prompts = prompts[st : st + batch_size]
            batch_metas = metas[st : st + batch_size]

            outputs = llm.generate(batch_prompts, sampling_params)

            for meta, out in zip(batch_metas, outputs):
                total += 1
                raw_output = str(out.outputs[0].text) if out.outputs else ""
                reasoning = _strip_decision_token_if_any(raw_output)

                passed, pred, ref_norm = _math_equal_passed(
                    pred_text=reasoning,
                    answer=str(meta["answer"]),
                )
                if passed:
                    correct += 1

                if out_f is not None:
                    out_f.write(
                        json.dumps(
                            {
                                "split": split_name,
                                "id": meta["id"],
                                "answer": str(meta["answer"]),
                                "reference_normalized": ref_norm,
                                "prediction": pred,
                                "passed": bool(passed),
                                "raw_output": raw_output,
                                "extracted_reasoning": reasoning,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    finally:
        if out_f is not None:
            out_f.close()

    acc = (correct / total) if total > 0 else 0.0
    return EvalStats(split=split_name, total=total, correct=correct, accuracy=acc)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="llama/math/ckpt/llama_math_only_decision")
    p.add_argument(
        "--train_jsonl",
        type=str,
        default="data/MATH/hendrycks_math_train.jsonl",
    )
    p.add_argument(
        "--test_jsonl",
        type=str,
        default="data/MATH/hendrycks_math_test.jsonl",
    )
    p.add_argument("--trust_remote_code", action="store_true")

    p.add_argument("--task_id_key", type=str, default="unique_id")
    p.add_argument("--problem_key", type=str, default="problem")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit_train", type=int, default=None)
    p.add_argument("--limit_test", type=int, default=None)

    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=256)

    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    p.add_argument(
        "--output_dir",
        type=str,
        default="llama/math/eval_output/llama_math_only_decision",
    )
    p.add_argument("--save_predictions", action="store_true")
    args = p.parse_args()

    random.seed(args.seed)
    train_rows = _load_jsonl(args.train_jsonl, limit=args.limit_train)
    test_rows = _load_jsonl(args.test_jsonl, limit=args.limit_test)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    llm = LLM(
        model=args.model_path,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
    )

    sampling_params = SamplingParams(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
        n=1,
        seed=int(args.seed),
    )

    train_pred_path = None
    test_pred_path = None
    if args.save_predictions:
        _ensure_dir(args.output_dir)
        train_pred_path = str(Path(args.output_dir) / "train_predictions.jsonl")
        test_pred_path = str(Path(args.output_dir) / "test_predictions.jsonl")

    train_stats = _evaluate_split(
        split_name="train",
        rows=train_rows,
        llm=llm,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        task_id_key=args.task_id_key,
        problem_key=args.problem_key,
        batch_size=int(args.batch_size),
        save_path=train_pred_path,
    )
    test_stats = _evaluate_split(
        split_name="test",
        rows=test_rows,
        llm=llm,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        task_id_key=args.task_id_key,
        problem_key=args.problem_key,
        batch_size=int(args.batch_size),
        save_path=test_pred_path,
    )

    print("=" * 64)
    print(f"Model: {args.model_path}")
    print(
        f"Train pass rate: {train_stats.correct}/{train_stats.total} "
        f"= {train_stats.accuracy:.4%}"
    )
    print(
        f"Test  pass rate: {test_stats.correct}/{test_stats.total} "
        f"= {test_stats.accuracy:.4%}"
    )
    print("=" * 64)

    _ensure_dir(args.output_dir)
    summary_path = Path(args.output_dir) / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": args.model_path,
                "train": train_stats.__dict__,
                "test": test_stats.__dict__,
                "args": vars(args),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[done] summary written to {summary_path}")


if __name__ == "__main__":
    main()
