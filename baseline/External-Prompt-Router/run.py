from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.common.domain_data import (
    apply_chat_template,
    build_external_router_prompt,
    evaluate_completion,
    external_router_attempt_close,
    external_router_attempt_open,
    read_domain_examples,
)
from baseline.common.inference import generate_texts, generate_texts_vllm, load_model_and_tokenizer, model_input_device
from baseline.common.metrics import attach_expert, load_expert_map, row_from_score, write_metadata, write_rows_csv
from baseline.common.paths import make_run_paths
from baseline.common.progress import progress_iter


DECISION_RE = re.compile(r"\b(?:yes|no)\b", flags=re.IGNORECASE)
ACTIONS = {"self": "yes", "defer": "no"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="External Prompt Router baseline.")
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
    return [apply_chat_template(tokenizer, build_external_router_prompt(domain, ex)) for ex in examples]


def build_attempt_generation_prompts(prompts: list[str], domain: str) -> list[str]:
    open_text = external_router_attempt_open(domain)
    return [prompt + open_text for prompt in prompts]


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


def truncate_after_final_answer(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""

    marker_match = None
    for match in re.finditer(r"final\s+answer\s*[:：]?", text, flags=re.IGNORECASE):
        marker_match = match
    if marker_match is None:
        decision_match = DECISION_RE.search(text)
        if decision_match:
            return text[: decision_match.start()].strip() + "\n"
        return text.strip() + "\n"

    newline_idx = text.find("\n", marker_match.end())
    if newline_idx != -1:
        target_line = text[marker_match.start() : newline_idx]
    else:
        target_line = text[marker_match.start() :]

    decision_match = DECISION_RE.search(target_line)
    if decision_match:
        target_line = target_line[: decision_match.start()]

    return text[: marker_match.start()] + target_line.strip() + "\n"


def extract_attempt_content(text: str, domain: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    if domain == "code":
        marker = "\n```"
        idx = text.find(marker)
        if idx == -1:
            return text.strip()
        return text[:idx].strip()
    return truncate_after_final_answer(text).strip()


def format_answer_for_eval(domain: str, answer_text: str) -> str:
    domain = domain.lower()
    text = str(answer_text or "").strip()
    if domain == "code":
        if not text:
            return ""
        return f"```python\n{text}\n```"
    if domain in {"math", "mmlu"}:
        return text
    raise ValueError(f"Unknown domain: {domain}")


def count_output_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def extract_action_suffix_ids(tokenizer, context: str, action_text: str) -> list[int]:
    context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(context + action_text, add_special_tokens=False)["input_ids"]
    if full_ids[: len(context_ids)] != context_ids:
        # Some tokenizers may retokenize at the concatenation boundary.
        # Try one hard separator before action to stabilize tokenization.
        context_fallback = context + "\n"
        context_ids_fb = tokenizer(context_fallback, add_special_tokens=False)["input_ids"]
        full_ids_fb = tokenizer(context_fallback + action_text, add_special_tokens=False)["input_ids"]
        if full_ids_fb[: len(context_ids_fb)] != context_ids_fb:
            raise ValueError("Context tokenization is not a prefix of context+action tokenization")
        return list(full_ids_fb[len(context_ids_fb) :])
    return list(full_ids[len(context_ids) :])


def _sequence_logprob(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    picked = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    return picked.sum()


@torch.inference_mode()
def compute_action_logprob(model, tokenizer, context: str, action_text: str, device: torch.device) -> float:
    context_ids = tokenizer(context, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
    suffix_ids = extract_action_suffix_ids(tokenizer, context, action_text)
    if not suffix_ids:
        return float("-inf")
    suffix_tensor = torch.tensor(suffix_ids, device=device, dtype=torch.long)
    full_ids = torch.cat([context_ids, suffix_tensor], dim=0).unsqueeze(0)
    logits = model(input_ids=full_ids).logits[0]
    start = context_ids.shape[0] - 1
    end = start + suffix_tensor.shape[0]
    return float(_sequence_logprob(logits[start:end], suffix_tensor).item())


def action_probs_from_logps(logp_defer: float, logp_self: float) -> tuple[float, float, float]:
    max_logp = max(logp_defer, logp_self)
    denom = math.exp(logp_defer - max_logp) + math.exp(logp_self - max_logp)
    p_defer = math.exp(logp_defer - max_logp) / denom
    return p_defer, 1.0 - p_defer, logp_defer - logp_self


@torch.inference_mode()
def compute_pair_distribution(model, tokenizer, context: str, self_action: str, defer_action: str, device: torch.device):
    logp_self = compute_action_logprob(model, tokenizer, context, self_action, device)
    logp_defer = compute_action_logprob(model, tokenizer, context, defer_action, device)
    p_defer, p_self, margin = action_probs_from_logps(logp_defer, logp_self)
    return p_defer, p_self, margin


def main() -> None:
    args = parse_args()
    paths = make_run_paths(method="External-Prompt-Router", domain=args.domain, model=args.model, run_name=args.run_name)
    from transformers import AutoTokenizer

    examples = read_domain_examples(args.domain, args.data_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    prompts = build_generation_prompts(tokenizer, args.domain, examples)
    generation_prompts = build_attempt_generation_prompts(prompts, args.domain)
    hf_model = None
    if args.generation_backend == "hf":
        hf_model, tokenizer = load_model_and_tokenizer(
            args.model,
            dtype=args.dtype,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
        )
    responses = generate_answers(args, generation_prompts, hf_model=hf_model, tokenizer=tokenizer)
    if hf_model is None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        hf_model, tokenizer = load_model_and_tokenizer(
            args.model,
            dtype=args.dtype,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
        )
    device = model_input_device(hf_model)
    records = []
    iterator = zip(examples, prompts, responses)
    attempt_open = external_router_attempt_open(args.domain)
    attempt_close = external_router_attempt_close(args.domain)
    for ex, prompt, response in progress_iter(iterator, desc="external router eval", total=len(examples)):
        attempt_text = extract_attempt_content(response, args.domain)
        answer_response = format_answer_for_eval(args.domain, attempt_text)
        s1_context = prompt + attempt_open + attempt_text + attempt_close
        p_defer, p_self, margin = compute_pair_distribution(
            hf_model,
            tokenizer,
            s1_context,
            self_action=ACTIONS["self"],
            defer_action=ACTIONS["defer"],
            device=device,
        )
        self_passed, extracted = evaluate_completion(args.domain, ex, answer_response, timeout=args.eval_timeout)
        answer_len = count_output_tokens(tokenizer, attempt_text)
        records.append(
            {
                "task_id": ex.task_id,
                "score": p_self,
                "parsed": 1,
                "raw_response": answer_response,
                "prompt": prompt,
                "self_passed": self_passed,
                "p_defer": p_defer,
                "p_self": p_self,
                "margin": margin,
                "extracted": extracted,
                "answer_len": answer_len,
                "actual_local_tokens": answer_len,
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
            method="External-Prompt-Router",
            domain=args.domain,
            model_slug=paths.model_slug,
            dataset_slug=paths.dataset_slug,
            expert_passed=row.get("expert_passed", ""),
            answer_len=row.get("answer_len", 0),
            actual_local_tokens=row.get("actual_local_tokens", 0),
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
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved metadata to {paths.eval_output / 'metadata.csv'}")


if __name__ == "__main__":
    main()
