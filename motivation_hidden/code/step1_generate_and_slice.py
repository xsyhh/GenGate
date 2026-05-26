"""Code step1: stream rollouts, run tests, slice raw generations by ratio."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motivation_hidden.common.code_judge import clean_markdown_fences, run_code_tests  # noqa: E402
from motivation_hidden.common.generation_utils import (  # noqa: E402
    build_chat_prompts,
    iter_chunks,
    load_completed_task_ids,
    parse_ratios,
    set_seed,
    write_pass1_summary,
    write_sliced_records,
)


DOMAIN = "code"

PROMPT_TEMPLATE = """You are a code agent.
### Question:
{problem}
### Starter Code:
```python
{starter_code}
```
### INSTRUCTION: 
You will use the following starter code to write the solution to the problem. 
Output the solution enclosed in ```python ... ``` blocks.
Do NOT include any explanations, comments, or extra text outside the code block.
Do NOT generate any test cases, assertions, usage examples, or `if __name__ == "__main__":` blocks.
"""


def make_prompt(problem: str, starter_code: str) -> str:
    return PROMPT_TEMPLATE.format(problem=problem, starter_code=starter_code)


def iter_code_rows(path: Path):
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, raw in enumerate(reader):
            problem = str(raw.get("problem", ""))
            starter_code = str(raw.get("starter_code", ""))
            row = {
                "domain": DOMAIN,
                "task_id": str(raw.get("id") or raw.get("task_id") or idx),
                "problem": problem,
                "starter_code": starter_code,
                "tests": str(raw.get("test") or raw.get("tests") or ""),
                "entry_point": str(raw.get("entry_point", "")),
                "answer": "",
                "subject": "",
                "level": "",
            }
            row["prompt_text"] = make_prompt(problem, starter_code)
            yield row


def judge_one(row, output, tokenizer, code_timeout):
    records = []
    for sample_idx, out in enumerate(output.outputs):
        full_raw = getattr(out, "text", "")
        if not isinstance(full_raw, str) or not full_raw.strip():
            continue
        clean_code = clean_markdown_fences(full_raw)
        y_final = int(
            bool(clean_code)
            and run_code_tests(
                clean_code,
                row["tests"],
                row["entry_point"],
                row["task_id"],
                timeout=code_timeout,
            )
        )
        records.append((row, sample_idx, full_raw, y_final, clean_code))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--out_name", type=str, default=None)
    parser.add_argument("--summary_name", type=str, default=None)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--ratios", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
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
    parser.add_argument("--code_timeout", type=int, default=10)
    parser.add_argument("--max_workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    from tqdm import tqdm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.out_dir / (args.out_name or f"code_{args.data_csv.stem}_sliced_drafts.jsonl")
    summary_path = args.out_dir / (args.summary_name or f"{out_jsonl.stem}_pass1_summary.json")
    ratios = parse_ratios(args.ratios)
    completed = load_completed_task_ids(out_jsonl, args.K) if args.resume else set()

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
        chunks = iter_chunks(iter_code_rows(args.data_csv), args.chunk_size, args.start, args.limit, completed)
        for chunk in tqdm(chunks, desc="Code generation chunks"):
            outputs = llm.generate(build_chat_prompts(chunk, tokenizer), sampling_params)
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = [
                    executor.submit(judge_one, row, output, tokenizer, args.code_timeout)
                    for row, output in zip(chunk, outputs)
                ]
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Code validation/tests",
                    leave=False,
                ):
                    for row, sample_idx, full_raw, y_final, clean_code in future.result():
                        total_saved += write_sliced_records(
                            f_out,
                            row,
                            sample_idx,
                            full_raw,
                            ratios,
                            tokenizer,
                            y_final=y_final,
                            pred_answer="",
                            extra={
                                "full_code_clean": clean_code,
                                "tests": row["tests"],
                                "entry_point": row["entry_point"],
                            },
                        )
            f_out.flush()

    summary = write_pass1_summary(out_jsonl, summary_path, DOMAIN)
    print(f"Saved {total_saved} new sliced records to {out_jsonl}")
    print(f"Saved raw pass@1 summary to {summary_path}")
    print(f"raw_pass1={summary['n_ratio1_pass']}/{summary['n_ratio1_rollouts']}={summary['raw_pass1']:.4f}")


if __name__ == "__main__":
    main()
