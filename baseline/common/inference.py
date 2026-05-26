from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from .progress import progress_iter


def load_model_and_tokenizer(model_path: str, *, dtype: str = "bf16", device_map: str | None = "auto", trust_remote_code: bool = True):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype_map[dtype],
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if device_map and str(device_map).lower() not in {"none", "null", ""}:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    return model, tokenizer


def model_input_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def generate_texts(
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
    out: list[str] = []
    do_sample = temperature > 0
    starts = range(0, len(prompts), batch_size)
    total_batches = (len(prompts) + max(batch_size, 1) - 1) // max(batch_size, 1)
    for start in progress_iter(starts, desc="generate", total=total_batches):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p
        generated = model.generate(**encoded, **kwargs)
        prompt_width = encoded["input_ids"].shape[1]
        for row_idx in range(generated.shape[0]):
            out.append(tokenizer.decode(generated[row_idx, prompt_width:], skip_special_tokens=True).strip())
    return out


def generate_texts_vllm(
    prompts: list[str],
    *,
    model_path: str,
    trust_remote_code: bool,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int | None,
    max_num_seqs: int | None,
    swap_space: int | None,
    enforce_eager: bool,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    from vllm import LLM, SamplingParams

    llm_kwargs: dict[str, Any] = {
        "model": model_path,
        "trust_remote_code": trust_remote_code,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": enforce_eager,
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    if max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = max_num_seqs
    if swap_space is not None:
        llm_kwargs["swap_space"] = swap_space

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        n=1,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=False,
    )
    outputs = llm.generate(prompts, sampling_params)
    return [str(output.outputs[0].text or "").strip() for output in outputs]


@torch.inference_mode()
def mean_token_logprob(model, tokenizer, prompt: str, response: str, *, max_length: int = 8192) -> tuple[float, int]:
    device = model_input_device(model)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + response, add_special_tokens=False, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"].to(device)
    prompt_len = min(len(prompt_ids), full_ids.shape[1])
    if full_ids.shape[1] <= prompt_len:
        return float("-inf"), 0
    logits = model(input_ids=full_ids).logits[0]
    total = 0.0
    count = 0
    for pos in range(prompt_len, full_ids.shape[1]):
        token_id = int(full_ids[0, pos].item())
        total += float(F.log_softmax(logits[pos - 1], dim=-1)[token_id].item())
        count += 1
    return total / max(count, 1), count


@torch.inference_mode()
def mean_token_logprobs_batch(
    model,
    tokenizer,
    prompts: list[str],
    responses: list[str],
    *,
    batch_size: int,
    max_length: int = 8192,
) -> list[tuple[float, int]]:
    device = model_input_device(model)
    out: list[tuple[float, int]] = []
    starts = range(0, len(prompts), batch_size)
    total_batches = (len(prompts) + max(batch_size, 1) - 1) // max(batch_size, 1)
    for start in progress_iter(starts, desc="answer logprob", total=total_batches):
        prompt_batch = prompts[start : start + batch_size]
        response_batch = responses[start : start + batch_size]
        full_texts = [prompt + response for prompt, response in zip(prompt_batch, response_batch)]
        prompt_lens = [
            len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            for prompt in prompt_batch
        ]
        encoded = tokenizer(
            full_texts,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)
        logits = model(input_ids=encoded["input_ids"], attention_mask=encoded.get("attention_mask")).logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        labels = encoded["input_ids"][:, 1:]
        attention = encoded["attention_mask"][:, 1:] if "attention_mask" in encoded else torch.ones_like(labels)
        gathered = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)

        for row_idx, prompt_len in enumerate(prompt_lens):
            if "attention_mask" in encoded:
                real_len = int(encoded["attention_mask"][row_idx].sum().item())
                pad_len = int(encoded["attention_mask"].shape[1] - real_len)
            else:
                real_len = int(encoded["input_ids"].shape[1])
                pad_len = 0
            seq_len = pad_len + real_len
            start_pos = min(max(pad_len + prompt_len - 1, 0), max(seq_len - 1, 0))
            end_pos = max(seq_len - 1, start_pos)
            mask = attention[row_idx, start_pos:end_pos].bool()
            values = gathered[row_idx, start_pos:end_pos][mask]
            if values.numel() == 0:
                out.append((float("-inf"), 0))
            else:
                out.append((float(values.mean().item()), int(values.numel())))
    return out


def logprob_to_score(mean_logprob: float) -> float:
    if math.isinf(mean_logprob) and mean_logprob < 0:
        return 0.0
    return max(0.0, min(1.0, math.exp(float(mean_logprob))))


def first_number_0_1(text: str) -> tuple[float, bool]:
    import re

    match = re.search(r"(?<![\d.])(?:0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])", str(text))
    if not match:
        return 0.5, False
    value = float(match.group(0))
    return max(0.0, min(1.0, value)), True


def parse_confidence_token(
    text: str,
    *,
    positive_token: str = "<CN>",
    negative_token: str = "<UN>",
) -> tuple[float, bool]:
    import re

    pos = positive_token.strip().strip("<>").upper()
    neg = negative_token.strip().strip("<>").upper()
    pattern = re.compile(rf"<\s*({re.escape(pos)}|{re.escape(neg)})\s*>", flags=re.IGNORECASE)
    matches = list(pattern.finditer(str(text or "")))
    if not matches:
        return 0.5, False
    token = matches[-1].group(1).upper()
    if token == pos:
        return 1.0, True
    if token == neg:
        return 0.0, True
    return 0.5, False
