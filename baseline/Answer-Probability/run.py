from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.common.domain_data import apply_chat_template, build_plain_prompt, evaluate_completion, read_domain_examples
from baseline.common.inference import (
    generate_texts,
    generate_texts_vllm,
    load_model_and_tokenizer,
    logprob_to_score,
    mean_token_logprobs_batch,
)
from baseline.common.metrics import attach_expert, load_expert_map, row_from_score, write_metadata, write_rows_csv
from baseline.common.paths import make_run_paths
from baseline.common.progress import progress_iter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Answer Probability baseline.")
    p.add_argument("--domain", required=True, choices=["code", "math", "mmlu"])
    p.add_argument("--model", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--run_name", default="default")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--generation_backend", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--device_map", default="auto")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.45)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--max_num_seqs", type=int, default=None)
    p.add_argument("--swap_space", type=int, default=None)
    p.add_argument("--enforce_eager", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--expert_csv", default=None)
    p.add_argument("--assume_expert_correct", action="store_true")
    p.add_argument("--eval_timeout", type=int, default=10)
    return p.parse_args()


def build_generation_prompts(tokenizer, domain: str, examples) -> list[str]:
    return [apply_chat_template(tokenizer, build_plain_prompt(domain, ex)) for ex in examples]


def generate_answers(args, prompts: list[str], *, hf_model, tokenizer) -> list[str]:
    if args.generation_backend == "vllm":
        return generate_texts_vllm(
            prompts,
            model_path=args.model,
            trust_remote_code=args.trust_remote_code,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            swap_space=args.swap_space,
            enforce_eager=args.enforce_eager,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    return generate_texts(
        hf_model,
        tokenizer,
        prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


def main() -> None:
    args = parse_args()
    paths = make_run_paths(method="Answer-Probability", domain=args.domain, model=args.model, run_name=args.run_name)
    from transformers import AutoTokenizer

    examples = read_domain_examples(args.domain, args.data_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    prompts = build_generation_prompts(tokenizer, args.domain, examples)
    hf_model = None
    if args.generation_backend == "hf":
        hf_model, tokenizer = load_model_and_tokenizer(
            args.model,
            dtype=args.dtype,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
        )
    responses = generate_answers(args, prompts, hf_model=hf_model, tokenizer=tokenizer)
    if hf_model is None:
        hf_model, tokenizer = load_model_and_tokenizer(
            args.model,
            dtype=args.dtype,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
        )
    logprob_rows = mean_token_logprobs_batch(
        hf_model,
        tokenizer,
        prompts,
        responses,
        batch_size=args.batch_size,
    )
    records = []
    iterator = zip(examples, responses, logprob_rows)
    for ex, response, (mean_lp, token_count) in progress_iter(iterator, desc="answer probability eval", total=len(examples)):
        score = logprob_to_score(mean_lp)
        self_passed, extracted = evaluate_completion(args.domain, ex, response, timeout=args.eval_timeout)
        records.append(
            {
                "task_id": ex.task_id,
                "score": score,
                "mean_logprob": mean_lp,
                "token_count": token_count,
                "raw_response": response,
                "extracted": extracted,
                "self_passed": self_passed,
            }
        )
    expert_map = load_expert_map(args.expert_csv, assume_correct=args.assume_expert_correct)
    records = attach_expert(records, expert_map, assume_correct=args.assume_expert_correct or args.expert_csv is None)
    metadata_rows = [
        row_from_score(
            task_id=row["task_id"],
            score=row["score"],
            self_passed=row["self_passed"],
            threshold=args.threshold,
            method="Answer-Probability",
            domain=args.domain,
            model_slug=paths.model_slug,
            dataset_slug=paths.dataset_slug,
            expert_passed=row.get("expert_passed", ""),
        )
        for row in records
    ]
    write_metadata(paths.eval_output / "metadata.csv", metadata_rows)
    write_rows_csv(paths.outputs / "scores.csv", records)
    with (paths.eval_output / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "domain": args.domain,
                "model": args.model,
                "data_path": args.data_path,
                "threshold": args.threshold,
                "generation_backend": args.generation_backend,
                "hf_score_batch_size": args.batch_size,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved metadata to {paths.eval_output / 'metadata.csv'}")


if __name__ == "__main__":
    main()
