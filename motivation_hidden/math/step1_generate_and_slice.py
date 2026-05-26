"""Math step1: stream rollouts, judge final answer, slice by generation ratio."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motivation_hidden.common.generation_utils import (  # noqa: E402
    build_chat_prompts,
    iter_chunks,
    load_completed_task_ids,
    parse_ratios,
    set_seed,
    write_pass1_summary,
    write_sliced_records,
)
from motivation_hidden.common.math_judge import answers_match, extract_final_answer, load_qwen_math_tools  # noqa: E402


DOMAIN = "math"

PROMPT_TEMPLATE = """You are a math reasoning agent.
### Question:
{problem}
### INSTRUCTION:
1. Solve the problem step by step.
2. Conclude your response with exactly "Final answer: " followed IMMEDIATELY by the bare mathematical expression or value.
3. DO NOT output a full sentence, summary, or extra words after "Final answer: ". Just the raw answer.
"""


def make_prompt(problem: str) -> str:
    return PROMPT_TEMPLATE.format(problem=problem)


def iter_math_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            raw = json.loads(line)
            problem = str(raw.get("problem", ""))
            row = {
                "domain": DOMAIN,
                "task_id": str(raw.get("unique_id") or raw.get("task_id") or idx),
                "problem": problem,
                "answer": str(raw.get("answer", "")),
                "solution": str(raw.get("solution", "")),
                "subject": str(raw.get("subject", "")),
                "level": str(raw.get("level", "")),
            }
            row["prompt_text"] = make_prompt(problem)
            yield row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data_jsonl", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--out_name", default=None)
    parser.add_argument("--summary_name", default=None)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--ratios", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.4)
    parser.add_argument("--max_model_len", type=int, default=1024)
    parser.add_argument("--max_num_seqs", type=int, default=None)
    parser.add_argument("--swap_space", type=int, default=None)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--qwen_math_eval_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    from tqdm import tqdm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.out_dir / (args.out_name or f"math_{args.data_jsonl.stem}_sliced_drafts.jsonl")
    summary_path = args.out_dir / (args.summary_name or f"{out_jsonl.stem}_pass1_summary.json")
    ratios = parse_ratios(args.ratios)
    completed = load_completed_task_ids(out_jsonl, args.K) if args.resume else set()
    strip_string, math_equal = load_qwen_math_tools(args.qwen_math_eval_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "trust_remote_code": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "enforce_eager": args.enforce_eager,
    }
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.swap_space is not None:
        llm_kwargs["swap_space"] = args.swap_space
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        n=args.K,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["<|endoftext|>"],
    )

    mode = "a" if args.resume and out_jsonl.exists() else "w"
    total_saved = 0
    with out_jsonl.open(mode, encoding="utf-8") as f_out:
        chunks = iter_chunks(iter_math_rows(args.data_jsonl), args.chunk_size, args.start, args.limit, completed)
        for chunk in tqdm(chunks, desc="Math generation chunks"):
            outputs = llm.generate(build_chat_prompts(chunk, tokenizer), sampling_params)
            for row, output in tqdm(
                zip(chunk, outputs),
                total=len(chunk),
                desc="Math validation/judging",
                leave=False,
            ):
                for sample_idx, out in enumerate(output.outputs):
                    full_raw = getattr(out, "text", "")
                    if not isinstance(full_raw, str) or not full_raw.strip():
                        continue
                    pred = extract_final_answer(full_raw)
                    y_final = int(answers_match(full_raw, row["answer"], strip_string, math_equal))
                    total_saved += write_sliced_records(
                        f_out,
                        row,
                        sample_idx,
                        full_raw,
                        ratios,
                        tokenizer,
                        y_final=y_final,
                        pred_answer=pred,
                    )
            f_out.flush()

    summary = write_pass1_summary(out_jsonl, summary_path, DOMAIN)
    print(f"Saved {total_saved} new sliced records to {out_jsonl}")
    print(f"Saved raw pass@1 summary to {summary_path}")
    print(f"raw_pass1={summary['n_ratio1_pass']}/{summary['n_ratio1_rollouts']}={summary['raw_pass1']:.4f}")


if __name__ == "__main__":
    main()
