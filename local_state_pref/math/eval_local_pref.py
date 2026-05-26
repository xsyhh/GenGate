from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from build_pairs import ACTIONS, ATTEMPT_CLOSE, ATTEMPT_OPEN, build_chat_s0_context, build_chat_s1_context
from text_actions import extract_action_suffix_ids


NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
DECISION_RE = re.compile(r"<(?:\|\s*)?(self|defer)(?:\s*\|)?>", flags=re.IGNORECASE)


class MathRow(NamedTuple):
    task_id: str
    problem: str
    answer: str
    solution: str


def parse_dtype(dtype: str):
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return dtype_map[dtype]


def ensure_local_model_path_exists(model_path: str) -> str:
    expanded = os.path.expanduser(model_path)
    is_local_path = os.path.isabs(expanded) or expanded.startswith("./") or expanded.startswith("../")
    if is_local_path and not os.path.isdir(expanded):
        raise FileNotFoundError(f"Local model path does not exist: {expanded}")
    return expanded


def _latest_checkpoint_in_output_dir(output_dir: str) -> str | None:
    if not os.path.isdir(output_dir):
        return None

    latest_step = -1
    latest_path = None
    for name in os.listdir(output_dir):
        if not name.startswith("checkpoint-"):
            continue
        step_text = name[len("checkpoint-") :]
        if not step_text.isdigit():
            continue
        path = os.path.join(output_dir, name)
        if not os.path.isdir(path):
            continue
        step = int(step_text)
        if step > latest_step:
            latest_step = step
            latest_path = path
    return latest_path


def _has_tokenizer_files(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "tokenizer.json")) or os.path.isfile(os.path.join(path, "vocab.json"))


def _read_adapter_base_model_path(model_path: str) -> str | None:
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if not os.path.isfile(adapter_config_path):
        return None
    try:
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    base_path = str(config.get("base_model_name_or_path") or "").strip()
    return base_path or None


def _resolve_model_and_tokenizer_paths(model_path: str, tokenizer_path: str | None = None) -> tuple[str, str]:
    original_model_path = ensure_local_model_path_exists(model_path)
    resolved_model_path = original_model_path

    has_loadable_model_files = any(
        os.path.isfile(os.path.join(resolved_model_path, filename))
        for filename in ("adapter_config.json", "config.json")
    )
    if not has_loadable_model_files:
        latest_checkpoint = _latest_checkpoint_in_output_dir(resolved_model_path)
        if latest_checkpoint is not None:
            resolved_model_path = latest_checkpoint

    if tokenizer_path:
        return resolved_model_path, ensure_local_model_path_exists(tokenizer_path)

    if _has_tokenizer_files(resolved_model_path):
        return resolved_model_path, resolved_model_path

    if _has_tokenizer_files(original_model_path):
        return resolved_model_path, original_model_path

    base_model_path = _read_adapter_base_model_path(resolved_model_path)
    if base_model_path:
        return resolved_model_path, ensure_local_model_path_exists(base_model_path)

    return resolved_model_path, resolved_model_path


def load_tokenizer(tokenizer_path: str, trust_remote_code: bool):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        ensure_local_model_path_exists(tokenizer_path),
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model_and_tokenizer(
    model_path: str,
    dtype: str,
    device_map: str | None,
    trust_remote_code: bool,
    tokenizer_path: str | None = None,
):
    from transformers import AutoModelForCausalLM

    model_path, tokenizer_path = _resolve_model_and_tokenizer_paths(model_path, tokenizer_path=tokenizer_path)
    print(f"Using model path: {model_path}")
    print(f"Using tokenizer path: {tokenizer_path}")
    tokenizer = load_tokenizer(tokenizer_path, trust_remote_code=trust_remote_code)
    kwargs: dict[str, Any] = {
        "torch_dtype": parse_dtype(dtype),
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if device_map is not None and device_map != "none":
        kwargs["device_map"] = device_map

    try:
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(model_path, **kwargs)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    model.eval()
    return model, tokenizer


def model_input_device(model, fallback: torch.device | None = None) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        if fallback is not None:
            return fallback
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_math_dataset(
    jsonl_path: str,
    *,
    task_id_key: str,
    problem_key: str,
    answer_key: str,
    solution_key: str,
    limit: int | None,
) -> list[MathRow]:
    rows: list[MathRow] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rows.append(
                MathRow(
                    task_id=str(record.get(task_id_key, "")).strip() or str(idx),
                    problem=str(record.get(problem_key, "")).strip(),
                    answer=str(record.get(answer_key, "")).strip(),
                    solution=str(record.get(solution_key, "")).strip(),
                )
            )
    return rows


def normalize_answer(raw: str) -> str:
    text = str(raw or "").strip()
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r"\s+", "", text)
    text = text.rstrip(".。;；,，")
    return text


def truncate_after_final_answer(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""

    marker = "Final answer:"
    idx = text.rfind(marker)
    if idx == -1:
        match = DECISION_RE.search(text)
        if match:
            return text[: match.start()].strip() + "\n"
        return text.strip() + "\n"

    newline_idx = text.find("\n", idx)
    if newline_idx != -1:
        target_line = text[idx:newline_idx]
    else:
        target_line = text[idx:]

    match = DECISION_RE.search(target_line)
    if match:
        target_line = target_line[: match.start()]

    return text[:idx] + target_line.strip() + "\n"


def extract_final_answer(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""

    parts = text.rsplit("Final answer:", 1)
    if len(parts) > 1:
        answer = parts[1].strip()
    else:
        lower_parts = text.lower().rsplit("final answer:", 1)
        answer = lower_parts[1].strip() if len(lower_parts) > 1 else text

    decision_match = DECISION_RE.search(answer)
    if decision_match:
        answer = answer[: decision_match.start()].strip()

    answer = answer.strip().strip("*").strip().strip("$").strip()
    if answer.startswith("\\(") and answer.endswith("\\)"):
        answer = answer[2:-2].strip()
    elif answer.startswith("\\[") and answer.endswith("\\]"):
        answer = answer[2:-2].strip()
    answer = answer.strip(" \n.。;；")

    boxed_match = re.search(r"\\boxed\s*\{", answer)
    if boxed_match:
        i = boxed_match.end()
        depth = 1
        chars: list[str] = []
        while i < len(answer):
            ch = answer[i]
            if ch == "{":
                depth += 1
                chars.append(ch)
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
                chars.append(ch)
            else:
                chars.append(ch)
            i += 1
        answer = "".join(chars).strip()

    if not answer:
        numbers = NUMBER_RE.findall(text.replace(" ", ""))
        if numbers:
            return normalize_answer(numbers[-1].replace(",", ""))
        return ""

    return normalize_answer(answer)


def load_qwen_math_tools(eval_dir: str | None):
    if eval_dir:
        qwen_eval_dir = Path(eval_dir)
    else:
        qwen_eval_dir = Path("third_party/Qwen2.5-Math/evaluation")

    parser_path = qwen_eval_dir / "parser.py"
    grader_path = qwen_eval_dir / "grader.py"
    if not parser_path.exists() or not grader_path.exists():
        print(f"[WARN] Qwen math evaluator not found at {qwen_eval_dir}; using normalized string match.")
        return None, None

    if str(qwen_eval_dir) not in sys.path:
        sys.path.insert(0, str(qwen_eval_dir))

    parser_spec = importlib.util.spec_from_file_location("qwen_math_parser", str(parser_path))
    grader_spec = importlib.util.spec_from_file_location("qwen_math_grader", str(grader_path))
    if parser_spec is None or parser_spec.loader is None or grader_spec is None or grader_spec.loader is None:
        print(f"[WARN] Failed to load Qwen math evaluator from {qwen_eval_dir}; using normalized string match.")
        return None, None

    parser_mod = importlib.util.module_from_spec(parser_spec)
    grader_mod = importlib.util.module_from_spec(grader_spec)
    parser_spec.loader.exec_module(parser_mod)
    grader_spec.loader.exec_module(grader_mod)
    return getattr(parser_mod, "strip_string", None), getattr(grader_mod, "math_equal", None)


def answers_match(prediction_text: str, gold_answer: str, strip_string=None, math_equal=None) -> bool:
    prediction = extract_final_answer(prediction_text)
    gold = normalize_answer(gold_answer)
    if not prediction or not gold:
        return False

    if math_equal is None:
        return prediction == gold

    try:
        prediction_stripped = strip_string(prediction) if strip_string else prediction
        gold_stripped = strip_string(gold) if strip_string else gold
        return bool(
            math_equal(
                prediction_stripped,
                gold_stripped,
                include_percentage=True,
                is_close=True,
                timeout=True,
            )
        )
    except TypeError:
        return bool(math_equal(prediction, gold))
    except Exception:
        return False


def evaluate_answer_passes(
    dataset: list[MathRow],
    reasonings: list[str],
    strip_string=None,
    math_equal=None,
    max_workers: int = 64,
) -> list[bool]:
    examples = list(zip(dataset, reasonings))
    if max_workers <= 1:
        return [
            answers_match(reasoning, row.answer, strip_string=strip_string, math_equal=math_equal)
            for row, reasoning in tqdm(examples, total=len(examples), desc="Evaluating answers")
        ]

    results = [False] * len(examples)

    def _match_one(index_and_example: tuple[int, tuple[MathRow, str]]) -> tuple[int, bool]:
        index, (row, reasoning) = index_and_example
        passed = answers_match(reasoning, row.answer, strip_string=strip_string, math_equal=math_equal)
        return index, passed

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_match_one, item) for item in enumerate(examples)]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Evaluating answers ({max_workers} workers)",
        ):
            index, passed = future.result()
            results[index] = passed

    return results


def count_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _position_ids_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    return position_ids.clamp_min_(0)


def _infer_pad_id(*tensors: torch.Tensor) -> int:
    for tensor in tensors:
        if tensor.numel() > 0:
            return int(tensor.reshape(-1)[0].item())
    return 0


def _pad_sequences(seqs: list[list[int]], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if not seqs:
        empty = torch.empty((0, 0), dtype=torch.long, device=device)
        return empty, empty.bool()

    max_len = max(len(seq) for seq in seqs)
    ids = []
    mask = []
    for seq in seqs:
        pad_len = max_len - len(seq)
        ids.append(seq + [pad_id] * pad_len)
        mask.append([1] * len(seq) + [0] * pad_len)
    return torch.tensor(ids, dtype=torch.long, device=device), torch.tensor(mask, dtype=torch.bool, device=device)


def _pad_actions_to_length(
    action_ids: torch.Tensor,
    action_mask: torch.Tensor,
    max_action_len: int,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    current_len = action_ids.size(1)
    if current_len == max_action_len:
        return action_ids, action_mask

    pad_len = max_action_len - current_len
    padded_ids = torch.cat(
        [
            action_ids,
            action_ids.new_full((action_ids.size(0), pad_len), pad_id),
        ],
        dim=1,
    )
    padded_mask = torch.cat(
        [
            action_mask,
            action_mask.new_zeros((action_mask.size(0), pad_len)),
        ],
        dim=1,
    )
    return padded_ids, padded_mask


def _forward_selected_logits(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    keep_positions: torch.Tensor,
) -> torch.Tensor:
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": _position_ids_from_attention_mask(attention_mask),
    }
    try:
        return model(**kwargs, logits_to_keep=keep_positions).logits
    except TypeError:
        logits = model(**kwargs).logits
        return logits.index_select(1, keep_positions)


@torch.inference_mode()
def _batch_action_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    action_ids: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size = input_ids.size(0)
    context_lens = attention_mask.sum(dim=1).long()
    action_lens = action_mask.sum(dim=1).long()
    max_context_len = int(context_lens.max().item())
    max_action_len = int(action_lens.max().item())
    if max_action_len == 0:
        return input_ids.new_zeros(batch_size, dtype=torch.float32)

    pad_id = _infer_pad_id(
        input_ids[attention_mask == 0],
        action_ids[action_mask == 0],
        input_ids,
    )
    full_len = max_context_len + max_action_len
    full_ids = input_ids.new_full((batch_size, full_len), pad_id)
    full_mask = attention_mask.new_zeros((batch_size, full_len))

    for row_idx in range(batch_size):
        context_len = int(context_lens[row_idx].item())
        action_len = int(action_lens[row_idx].item())
        context_start = max_context_len - context_len
        full_ids[row_idx, context_start:max_context_len] = input_ids[row_idx, :context_len]
        full_mask[row_idx, context_start:max_context_len] = 1
        full_ids[row_idx, max_context_len : max_context_len + action_len] = action_ids[row_idx, :action_len]
        full_mask[row_idx, max_context_len : max_context_len + action_len] = 1

    keep_positions = torch.arange(
        max_context_len - 1,
        max_context_len - 1 + max_action_len,
        device=input_ids.device,
        dtype=torch.long,
    )
    logits = _forward_selected_logits(
        model=model,
        input_ids=full_ids,
        attention_mask=full_mask,
        keep_positions=keep_positions,
    )
    log_probs = F.log_softmax(logits, dim=-1)
    picked = log_probs.gather(dim=-1, index=action_ids[:, :max_action_len].unsqueeze(-1)).squeeze(-1)
    return (picked * action_mask[:, :max_action_len].to(dtype=picked.dtype)).sum(dim=-1)


@torch.inference_mode()
def _batch_action_pair_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    positive_action_ids: torch.Tensor,
    positive_action_mask: torch.Tensor,
    defer_action_ids: torch.Tensor,
    defer_action_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = input_ids.size(0)
    max_action_len = max(positive_action_ids.size(1), defer_action_ids.size(1))
    pad_id = _infer_pad_id(
        positive_action_ids[positive_action_mask == 0],
        defer_action_ids[defer_action_mask == 0],
        input_ids[attention_mask == 0],
        input_ids,
    )
    positive_action_ids, positive_action_mask = _pad_actions_to_length(
        action_ids=positive_action_ids,
        action_mask=positive_action_mask,
        max_action_len=max_action_len,
        pad_id=pad_id,
    )
    defer_action_ids, defer_action_mask = _pad_actions_to_length(
        action_ids=defer_action_ids,
        action_mask=defer_action_mask,
        max_action_len=max_action_len,
        pad_id=pad_id,
    )
    combined_logprobs = _batch_action_logprobs(
        model=model,
        input_ids=torch.cat([input_ids, input_ids], dim=0),
        attention_mask=torch.cat([attention_mask, attention_mask], dim=0),
        action_ids=torch.cat([positive_action_ids, defer_action_ids], dim=0),
        action_mask=torch.cat([positive_action_mask, defer_action_mask], dim=0),
    )
    return combined_logprobs[:batch_size], combined_logprobs[batch_size:]


def compute_pair_distribution_from_logps(logp_positive: float, logp_defer: float) -> dict[str, float]:
    max_logp = max(logp_defer, logp_positive)
    denom = math.exp(logp_defer - max_logp) + math.exp(logp_positive - max_logp)
    p_defer = math.exp(logp_defer - max_logp) / denom
    return {
        "p_defer": float(p_defer),
        "p_self": float(1.0 - p_defer),
        "margin": float(logp_defer - logp_positive),
    }


@torch.inference_mode()
def score_action_pairs(
    model,
    tokenizer,
    contexts: list[str],
    *,
    positive_action: str,
    defer_action: str,
    batch_size: int,
    desc: str,
) -> list[dict[str, float]]:
    device = model_input_device(model)
    pad_id = int(tokenizer.pad_token_id)
    prepared = []
    for context in tqdm(contexts, desc=f"Tokenizing {desc} actions"):
        context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
        positive_ids = extract_action_suffix_ids(tokenizer, context, positive_action)
        defer_ids = extract_action_suffix_ids(tokenizer, context, defer_action)
        if not positive_ids or not defer_ids:
            raise ValueError(f"Empty action ids for context with action {positive_action!r}/{defer_action!r}")
        prepared.append((context_ids, positive_ids, defer_ids))

    distributions: list[dict[str, float]] = []
    for start_idx in tqdm(range(0, len(prepared), batch_size), desc=desc):
        batch = prepared[start_idx : start_idx + batch_size]
        context_ids, context_mask = _pad_sequences([item[0] for item in batch], pad_id, device)
        positive_ids, positive_mask = _pad_sequences([item[1] for item in batch], pad_id, device)
        defer_ids, defer_mask = _pad_sequences([item[2] for item in batch], pad_id, device)
        positive_logps, defer_logps = _batch_action_pair_logprobs(
            model=model,
            input_ids=context_ids,
            attention_mask=context_mask.long(),
            positive_action_ids=positive_ids,
            positive_action_mask=positive_mask,
            defer_action_ids=defer_ids,
            defer_action_mask=defer_mask,
        )
        for positive_logp, defer_logp in zip(positive_logps.tolist(), defer_logps.tolist()):
            distributions.append(
                compute_pair_distribution_from_logps(
                    logp_positive=float(positive_logp),
                    logp_defer=float(defer_logp),
                )
            )
    return distributions


@torch.inference_mode()
def generate_reasoning_drafts_hf(
    model,
    tokenizer,
    prompts: list[str],
    *,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    device = model_input_device(model)
    tokenizer.padding_side = "left"
    do_sample = temperature > 0.0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "top_p": top_p if do_sample else None,
        "temperature": temperature if do_sample else None,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}

    drafts: list[str] = []
    for start_idx in tqdm(range(0, len(prompts), batch_size), desc="Generating reasoning (HF)"):
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
            continuation_text = tokenizer.decode(continuation_ids, skip_special_tokens=False)
            drafts.append(truncate_after_final_answer(continuation_text))
    return drafts


def generate_reasoning_drafts_vllm(
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
    drafts = [truncate_after_final_answer(output.outputs[0].text) for output in outputs]
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return drafts


def decide_route(first_p_defer: float, post_p_defer: float, threshold: float) -> tuple[str, str]:
    if first_p_defer > threshold:
        return "early_defer", "defer"
    if post_p_defer > threshold:
        return "post_defer", "defer"
    return "post_self", "self"


def build_metadata_row(
    *,
    task_id: str,
    first: dict[str, float],
    post: dict[str, float],
    self_passed: bool,
    reasoning: str,
    answer_len: int,
    threshold: float,
    force_route: str = "none",
) -> dict[str, Any]:
    if force_route == "post_only":
        if float(post["p_defer"]) > threshold:
            route, model_decision = "post_defer", "defer"
        else:
            route, model_decision = "post_self", "self"
        unified = post
    elif force_route == "early_only":
        if float(first["p_defer"]) > threshold:
            route, model_decision = "early_defer", "defer"
        else:
            route, model_decision = "post_self", "self"
        unified = first
    elif force_route == "always_self":
        route, model_decision = "post_self", "self"
        unified = post
    else:
        route, model_decision = decide_route(
            first_p_defer=float(first["p_defer"]),
            post_p_defer=float(post["p_defer"]),
            threshold=threshold,
        )
        if route == "early_defer":
            unified = first
        else:
            unified = post

    return {
        "task_id": task_id,
        "route": route,
        "model_decision": model_decision,
        "first_p_defer": round(float(first["p_defer"]), 6),
        "first_p_self": round(float(first["p_self"]), 6),
        "first_margin": round(float(first["margin"]), 4),
        "post_p_defer": round(float(post["p_defer"]), 6),
        "post_p_self": round(float(post["p_self"]), 6),
        "post_margin": round(float(post["margin"]), 4),
        "p_defer": round(float(unified["p_defer"]), 6),
        "p_self": round(float(unified["p_self"]), 6),
        "margin": round(float(unified["margin"]), 4),
        "self_passed": int(bool(self_passed)),
        "reasoning_len": len(reasoning or ""),
        "answer_len": int(answer_len),
        "actual_local_tokens": int(answer_len),
    }


def build_chat_attempt_generation_context(tokenizer, problem: str) -> str:
    return build_chat_s0_context(tokenizer, problem, add_generation_prompt=True) + ATTEMPT_OPEN


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MATH local same-state preference policy.")
    parser.add_argument("--model_path", "--model", dest="model_path", required=True)
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--task_id_key", "--task_id_col", dest="task_id_key", default="unique_id")
    parser.add_argument("--problem_key", default="problem")
    parser.add_argument("--answer_key", default="answer")
    parser.add_argument("--solution_key", default="solution")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4, help="HF generation batch size.")
    parser.add_argument("--score_batch_size", type=int, default=8, help="Batched action scoring size.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--generation_backend", choices=["hf", "vllm"], default="vllm")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.45)
    parser.add_argument("--qwen_math_eval_dir", default=None)
    parser.add_argument(
        "--answer_eval_workers",
        type=int,
        default=64,
        help="Parallel workers for CPU MATH answer equivalence checks. Use 1 for serial evaluation.",
    )
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

    dataset = read_math_dataset(
        args.data_jsonl,
        task_id_key=args.task_id_key,
        problem_key=args.problem_key,
        answer_key=args.answer_key,
        solution_key=args.solution_key,
        limit=args.limit,
    )
    print(f"Loaded {len(dataset)} MATH examples")

    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    s0_contexts = [build_chat_s0_context(tokenizer, row.problem, add_generation_prompt=True) for row in dataset]
    generation_prompts = [context + ATTEMPT_OPEN for context in s0_contexts]

    first_results = score_action_pairs(
        model,
        tokenizer,
        s0_contexts,
        positive_action=ACTIONS["attempt"],
        defer_action=ACTIONS["defer"],
        batch_size=args.score_batch_size,
        desc="Scoring s0",
    )

    if args.generation_backend == "vllm":
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        reasonings = generate_reasoning_drafts_vllm(
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
    else:
        reasonings = generate_reasoning_drafts_hf(
            model,
            tokenizer,
            generation_prompts,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    s1_contexts = [build_chat_s1_context(tokenizer, row.problem, reasoning) for row, reasoning in zip(dataset, reasonings)]
    post_results = score_action_pairs(
        model,
        tokenizer,
        s1_contexts,
        positive_action=ACTIONS["self"],
        defer_action=ACTIONS["defer"],
        batch_size=args.score_batch_size,
        desc="Scoring s1",
    )

    strip_string, math_equal = load_qwen_math_tools(args.qwen_math_eval_dir)
    answer_passes = evaluate_answer_passes(
        dataset,
        reasonings,
        strip_string=strip_string,
        math_equal=math_equal,
        max_workers=args.answer_eval_workers,
    )
    rows = []
    completions = []
    stats = {"early_defer": 0, "post_self": 0, "post_defer": 0}
    for row, first, post, reasoning, self_passed in tqdm(
        list(zip(dataset, first_results, post_results, reasonings, answer_passes)),
        total=len(dataset),
        desc="Writing metadata",
    ):
        answer_len = count_tokens(tokenizer, reasoning)
        metadata_row = build_metadata_row(
            task_id=row.task_id,
            first=first,
            post=post,
            self_passed=self_passed,
            reasoning=reasoning,
            answer_len=answer_len,
            threshold=args.threshold,
            force_route=args.force_route,
        )
        stats[metadata_row["route"]] += 1
        rows.append(metadata_row)

        if args.save_completions:
            completions.append(
                {
                    "task_id": row.task_id,
                    "route": metadata_row["route"],
                    "model_decision": metadata_row["model_decision"],
                    "reasoning": reasoning,
                    "extracted_answer": extract_final_answer(reasoning),
                    "gold_answer": row.answer,
                    "self_passed": int(bool(self_passed)),
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
        "reasoning_len",
        "answer_len",
        "actual_local_tokens",
    ]
    metadata_path = out_dir / "metadata.csv"
    with open(metadata_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if args.save_completions:
        from io_utils import dump_jsonl

        dump_jsonl(str(out_dir / "completions.jsonl"), completions)

    total = len(rows)
    passed = sum(int(row["self_passed"]) for row in rows)
    print(f"Saved metadata to: {metadata_path}")
    print(f"Total tasks: {total}")
    print(
        f"Routes: early_defer={stats['early_defer']}, "
        f"post_self={stats['post_self']}, post_defer={stats['post_defer']}"
    )
    print(f"Counterfactual self pass rate: {passed / max(total, 1) * 100:.2f}%")


if __name__ == "__main__":
    main()
