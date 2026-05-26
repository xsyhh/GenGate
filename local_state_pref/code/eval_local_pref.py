from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from common import (
    ACTIONS,
    action_probs_from_logps,
    build_attempt_generation_context,
    build_chat_attempt_generation_context,
    build_chat_s0_context,
    build_chat_s1_context,
    build_s0_context,
    build_s1_context,
    count_tokens,
    extract_attempt_code,
    load_model_and_tokenizer,
    model_input_device,
    read_code_rows,
    run_humaneval_tests,
)
from text_actions import extract_action_suffix_ids


def _sequence_logprob(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    picked = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    return picked.sum()


@torch.inference_mode()
def compute_action_logprob(model, tokenizer, context: str, action_text: str, device: torch.device) -> float:
    context_ids = tokenizer(context, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
    suffix_ids = extract_action_suffix_ids(tokenizer, context, action_text)
    suffix_tensor = torch.tensor(suffix_ids, device=device, dtype=torch.long)
    full_ids = torch.cat([context_ids, suffix_tensor], dim=0).unsqueeze(0)
    logits = model(input_ids=full_ids).logits[0]
    start = context_ids.shape[0] - 1
    end = start + suffix_tensor.shape[0]
    return float(_sequence_logprob(logits[start:end], suffix_tensor).item())


@torch.inference_mode()
def compute_pair_distribution(model, tokenizer, context: str, positive_action: str, defer_action: str, device: torch.device):
    logp_other = compute_action_logprob(model, tokenizer, context, positive_action, device)
    logp_defer = compute_action_logprob(model, tokenizer, context, defer_action, device)
    p_defer, p_other, margin = action_probs_from_logps(logp_defer, logp_other)
    return p_defer, p_other, margin


@torch.inference_mode()
def generate_attempt_drafts(model, tokenizer, prompts: list[str], batch_size: int, max_new_tokens: int, temperature: float, top_p: float) -> list[str]:
    device = model_input_device(model)
    tokenizer.padding_side = "left"
    drafts: list[str] = []

    do_sample = temperature > 0.0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "top_p": top_p if do_sample else None,
        "temperature": temperature if do_sample else None,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

    for start_idx in tqdm(range(0, len(prompts), batch_size), desc="Generating drafts"):
        batch_prompts = prompts[start_idx : start_idx + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)
        prompt_width = encoded["input_ids"].shape[1]
        outputs = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            **generation_kwargs,
        )
        for row_idx in range(outputs.shape[0]):
            continuation_ids = outputs[row_idx, prompt_width:]
            continuation_text = tokenizer.decode(continuation_ids, skip_special_tokens=True)
            drafts.append(extract_attempt_code(continuation_text))

    return drafts


def generate_attempt_drafts_vllm(
    prompts: list[str],
    *,
    model_path: str,
    trust_remote_code: bool,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        trust_remote_code=trust_remote_code,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        n=1,
    )
    outputs = llm.generate(prompts, sampling_params)
    drafts = [extract_attempt_code(output.outputs[0].text) for output in outputs]
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return drafts


def main():
    parser = argparse.ArgumentParser(description="Evaluate code local same-state preference policy.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--generation_backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.45)
    parser.add_argument("--test_timeout", type=float, default=10.0)
    parser.add_argument("--save_completions", action="store_true")
    parser.add_argument(
        "--force_route",
        choices=["none", "post_only", "early_only", "always_self"],
        default="none",
        help=(
            "Ablation eval switch: "
            "'post_only' disables early defer (equivalent to w/o initial decision); "
            "'early_only' disables post decision (equivalent to w/o posterior decision); "
            "'always_self' forces all attempted answers to be submitted locally."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = read_code_rows(args.data_csv)
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    device = model_input_device(model)

    first_stage = []
    generation_prompts = []
    for row in tqdm(dataset, desc="Scoring s0"):
        s0_context = build_chat_s0_context(tokenizer, row.problem, row.starter_code)
        first_p_defer, first_p_attempt, first_margin = compute_pair_distribution(
            model,
            tokenizer,
            s0_context,
            positive_action=ACTIONS["attempt"],
            defer_action=ACTIONS["defer"],
            device=device,
        )
        first_stage.append(
            {
                "task_id": row.task_id,
                "first_p_defer": first_p_defer,
                "first_p_self": first_p_attempt,
                "first_margin": first_margin,
            }
        )
        generation_prompts.append(build_chat_attempt_generation_context(tokenizer, row.problem, row.starter_code))

    if args.generation_backend == "vllm":
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        drafts = generate_attempt_drafts_vllm(
            generation_prompts,
            model_path=args.model_path,
            trust_remote_code=args.trust_remote_code,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        model, tokenizer = load_model_and_tokenizer(
            args.model_path,
            dtype=args.dtype,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
        )
        device = model_input_device(model)
    else:
        drafts = generate_attempt_drafts(
            model,
            tokenizer,
            generation_prompts,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    rows = []
    completions = []
    stats = {"early_defer": 0, "post_self": 0, "post_defer": 0}

    for row, first_record, code in tqdm(list(zip(dataset, first_stage, drafts)), total=len(dataset), desc="Scoring s1 and testing"):
        s1_context = build_chat_s1_context(tokenizer, row.problem, row.starter_code, code)
        post_p_defer, post_p_self, post_margin = compute_pair_distribution(
            model,
            tokenizer,
            s1_context,
            positive_action=ACTIONS["self"],
            defer_action=ACTIONS["defer"],
            device=device,
        )
        passed = int(
            bool(
                run_humaneval_tests(
                    code=code,
                    tests=row.tests,
                    entry_point=row.entry_point,
                    task_id=row.task_id,
                    timeout=args.test_timeout,
                )
            )
        )

        if args.force_route == "post_only":
            if post_p_defer > args.threshold:
                route = "post_defer"
                model_decision = "defer"
                unified_p_defer = post_p_defer
                unified_p_self = post_p_self
                unified_margin = post_margin
            else:
                route = "post_self"
                model_decision = "self"
                unified_p_defer = post_p_defer
                unified_p_self = post_p_self
                unified_margin = post_margin
        elif args.force_route == "early_only":
            if first_record["first_p_defer"] > args.threshold:
                route = "early_defer"
                model_decision = "defer"
                unified_p_defer = first_record["first_p_defer"]
                unified_p_self = first_record["first_p_self"]
                unified_margin = first_record["first_margin"]
            else:
                route = "post_self"
                model_decision = "self"
                unified_p_defer = first_record["first_p_defer"]
                unified_p_self = first_record["first_p_self"]
                unified_margin = first_record["first_margin"]
        elif args.force_route == "always_self":
            route = "post_self"
            model_decision = "self"
            unified_p_defer = post_p_defer
            unified_p_self = post_p_self
            unified_margin = post_margin
        elif first_record["first_p_defer"] > args.threshold:
            route = "early_defer"
            model_decision = "defer"
            unified_p_defer = first_record["first_p_defer"]
            unified_p_self = first_record["first_p_self"]
            unified_margin = first_record["first_margin"]
        elif post_p_defer > args.threshold:
            route = "post_defer"
            model_decision = "defer"
            unified_p_defer = post_p_defer
            unified_p_self = post_p_self
            unified_margin = post_margin
        else:
            route = "post_self"
            model_decision = "self"
            unified_p_defer = post_p_defer
            unified_p_self = post_p_self
            unified_margin = post_margin

        stats[route] += 1
        answer_len = count_tokens(tokenizer, code)
        rows.append(
            {
                "task_id": row.task_id,
                "route": route,
                "model_decision": model_decision,
                "first_p_defer": round(float(first_record["first_p_defer"]), 6),
                "first_p_self": round(float(first_record["first_p_self"]), 6),
                "first_margin": round(float(first_record["first_margin"]), 4),
                "post_p_defer": round(float(post_p_defer), 6),
                "post_p_self": round(float(post_p_self), 6),
                "post_margin": round(float(post_margin), 4),
                "p_defer": round(float(unified_p_defer), 6),
                "p_self": round(float(unified_p_self), 6),
                "margin": round(float(unified_margin), 4),
                "self_passed": passed,
                "code_len": len(code or ""),
                "answer_len": answer_len,
                "actual_local_tokens": answer_len,
            }
        )

        if args.save_completions:
            completions.append(
                {
                    "task_id": row.task_id,
                    "route": route,
                    "code": code,
                }
            )

    fieldnames = [
        "task_id",
        "route",
        "model_decision",
        "first_p_defer",
        "first_p_self",
        "first_margin",
        "post_p_defer",
        "post_p_self",
        "post_margin",
        "p_defer",
        "p_self",
        "margin",
        "self_passed",
        "code_len",
        "answer_len",
        "actual_local_tokens",
    ]
    with open(out_dir / "metadata.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if args.save_completions:
        from common import dump_jsonl

        dump_jsonl(str(out_dir / "completions.jsonl"), completions)

    total = len(rows)
    passed = sum(int(row["self_passed"]) for row in rows)
    print(f"Saved metadata to: {out_dir / 'metadata.csv'}")
    print(f"Total tasks: {total}")
    print(
        f"Routes: early_defer={stats['early_defer']}, "
        f"post_self={stats['post_self']}, post_defer={stats['post_defer']}"
    )
    print(f"Counterfactual self pass rate: {passed / max(total, 1) * 100:.2f}%")


if __name__ == "__main__":
    main()
