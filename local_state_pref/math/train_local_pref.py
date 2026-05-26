from __future__ import annotations

import argparse
import os
from typing import Any

import torch
import torch.nn.functional as F

from build_pairs import ATTEMPT_CLOSE
from text_actions import extract_action_suffix_ids_from_context_ids


KL_START_BUCKET_SIZE = 256


def _extract_kl_targets(
    tokenizer,
    context: str,
    reasoning_text: str,
    max_seq_length: int,
) -> tuple[list[int], list[int], int]:
    full_context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    if max_seq_length and len(full_context_ids) > max_seq_length:
        offset = len(full_context_ids) - max_seq_length
        context_ids = full_context_ids[offset:]
    else:
        offset = 0
        context_ids = full_context_ids

    reasoning_text = str(reasoning_text or "").strip()
    if not reasoning_text:
        return context_ids, [], 0

    suffix = reasoning_text + ATTEMPT_CLOSE
    if not context.endswith(suffix):
        raise ValueError("S1 context does not end with reasoning text followed by attempt close marker")

    prefix_text = context[: -len(suffix)]
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    prefix_plus_reasoning_ids = tokenizer(prefix_text + reasoning_text, add_special_tokens=False)["input_ids"]
    if prefix_plus_reasoning_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("Prefix tokenization is not a prefix of prefix+reasoning tokenization")

    full_kl_target_ids = prefix_plus_reasoning_ids[len(prefix_ids) :]
    if not full_kl_target_ids:
        return context_ids, [], 0

    full_kl_start = len(prefix_ids)
    full_kl_end = full_kl_start + len(full_kl_target_ids)
    kept_start = max(full_kl_start, offset)
    kept_end = min(full_kl_end, offset + len(context_ids))
    if kept_start >= kept_end:
        return context_ids, [], 0

    keep_from = kept_start - full_kl_start
    keep_to = kept_end - full_kl_start
    kl_target_ids = full_kl_target_ids[keep_from:keep_to]
    start_in_context = kept_start - offset

    if start_in_context == 0:
        kl_target_ids = kl_target_ids[1:]
        start_in_context = 1

    if not kl_target_ids:
        return context_ids, [], 0

    return context_ids, kl_target_ids, start_in_context - 1


def prepare_pair_example(tokenizer, record: dict[str, Any], max_seq_length: int) -> dict[str, Any]:
    context = str(record["context"])
    context_ids, kl_target_ids, kl_target_start = _extract_kl_targets(
        tokenizer=tokenizer,
        context=context,
        reasoning_text=str(record.get("reasoning", "")),
        max_seq_length=max_seq_length,
    )

    action_a_ids = extract_action_suffix_ids_from_context_ids(
        tokenizer,
        context,
        context_ids,
        str(record["action_a"]),
    )
    action_b_ids = extract_action_suffix_ids_from_context_ids(
        tokenizer,
        context,
        context_ids,
        str(record["action_b"]),
    )
    if not action_a_ids or not action_b_ids:
        raise ValueError("Action text produced an empty suffix id sequence")

    return {
        "input_ids": context_ids,
        "input_length": len(context_ids),
        "attention_mask": [1] * len(context_ids),
        "action_a_ids": action_a_ids,
        "action_b_ids": action_b_ids,
        "target_prob": float(record["target_prob"]),
        "state_weight": float(record.get("state_weight", 1.0)),
        "is_s1": 1 if str(record.get("state_type", "")) == "s1" else 0,
        "kl_target_ids": kl_target_ids,
        "kl_target_start": int(kl_target_start),
    }


class PairCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        max_len = max(len(feature["input_ids"]) for feature in features)
        batch_input_ids = []
        batch_attention_mask = []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch_input_ids.append(feature["input_ids"] + [self.pad_id] * pad_len)
            batch_attention_mask.append(feature["attention_mask"] + [0] * pad_len)

        max_action_a_len = max(len(feature["action_a_ids"]) for feature in features)
        max_action_b_len = max(len(feature["action_b_ids"]) for feature in features)
        max_kl_target_len = max(len(feature["kl_target_ids"]) for feature in features)

        def pad_actions(key: str, max_action_len: int) -> tuple[torch.Tensor, torch.Tensor]:
            ids = []
            mask = []
            for feature in features:
                action_ids = feature[key]
                pad_len = max_action_len - len(action_ids)
                ids.append(action_ids + [self.pad_id] * pad_len)
                mask.append([1] * len(action_ids) + [0] * pad_len)
            return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)

        action_a_ids, action_a_mask = pad_actions("action_a_ids", max_action_a_len)
        action_b_ids, action_b_mask = pad_actions("action_b_ids", max_action_b_len)
        kl_target_ids, kl_target_mask = pad_actions("kl_target_ids", max_kl_target_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "target_prob": torch.tensor([feature["target_prob"] for feature in features], dtype=torch.float32),
            "state_weight": torch.tensor([feature["state_weight"] for feature in features], dtype=torch.float32),
            "is_s1": torch.tensor([feature["is_s1"] for feature in features], dtype=torch.float32),
            "action_a_ids": action_a_ids,
            "action_a_mask": action_a_mask,
            "action_b_ids": action_b_ids,
            "action_b_mask": action_b_mask,
            "kl_target_ids": kl_target_ids,
            "kl_target_mask": kl_target_mask,
            "kl_target_start": torch.tensor([feature["kl_target_start"] for feature in features], dtype=torch.long),
        }


def _infer_pad_id(*tensors: torch.Tensor) -> int:
    for tensor in tensors:
        if tensor.numel() > 0:
            return int(tensor.reshape(-1)[0].item())
    return 0


def _position_ids_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    return position_ids.clamp_min_(0)


def _sequence_logprob(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    picked = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    return picked.sum()


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


def _batch_action_pair_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    action_a_ids: torch.Tensor,
    action_a_mask: torch.Tensor,
    action_b_ids: torch.Tensor,
    action_b_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = input_ids.size(0)
    max_action_len = max(action_a_ids.size(1), action_b_ids.size(1))
    if max_action_len == 0:
        zeros = input_ids.new_zeros(batch_size, dtype=torch.float32)
        return zeros, zeros

    pad_id = _infer_pad_id(
        action_a_ids[action_a_mask == 0],
        action_b_ids[action_b_mask == 0],
        input_ids[attention_mask == 0],
        input_ids,
    )
    action_a_ids, action_a_mask = _pad_actions_to_length(
        action_ids=action_a_ids,
        action_mask=action_a_mask,
        max_action_len=max_action_len,
        pad_id=pad_id,
    )
    action_b_ids, action_b_mask = _pad_actions_to_length(
        action_ids=action_b_ids,
        action_mask=action_b_mask,
        max_action_len=max_action_len,
        pad_id=pad_id,
    )

    combined_logprobs = _batch_action_logprob(
        model=model,
        input_ids=torch.cat([input_ids, input_ids], dim=0),
        attention_mask=torch.cat([attention_mask, attention_mask], dim=0),
        action_ids=torch.cat([action_a_ids, action_b_ids], dim=0),
        action_mask=torch.cat([action_a_mask, action_b_mask], dim=0),
    )
    return combined_logprobs[:batch_size], combined_logprobs[batch_size:]


def _packed_action_pair_logprobs_and_s1_kl(
    model,
    ref_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    action_a_ids: torch.Tensor,
    action_a_mask: torch.Tensor,
    action_b_ids: torch.Tensor,
    action_b_mask: torch.Tensor,
    kl_target_mask: torch.Tensor,
    kl_target_start: torch.Tensor,
    is_s1: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    batch_size = input_ids.size(0)
    context_lens = attention_mask.sum(dim=1).long()
    action_a_lens = action_a_mask.sum(dim=1).long()
    action_b_lens = action_b_mask.sum(dim=1).long()
    max_context_len = int(context_lens.max().item())
    max_action_len = max(int(action_a_lens.max().item()), int(action_b_lens.max().item()))
    if max_action_len == 0:
        zeros = input_ids.new_zeros(batch_size, dtype=torch.float32)
        return zeros, zeros, None

    pad_id = _infer_pad_id(
        action_a_ids[action_a_mask == 0],
        action_b_ids[action_b_mask == 0],
        input_ids[attention_mask == 0],
        input_ids,
    )
    action_a_ids, action_a_mask = _pad_actions_to_length(
        action_ids=action_a_ids,
        action_mask=action_a_mask,
        max_action_len=max_action_len,
        pad_id=pad_id,
    )
    action_b_ids, action_b_mask = _pad_actions_to_length(
        action_ids=action_b_ids,
        action_mask=action_b_mask,
        max_action_len=max_action_len,
        pad_id=pad_id,
    )

    full_len = max_context_len + max_action_len
    full_ids = input_ids.new_full((batch_size * 2, full_len), pad_id)
    full_mask = attention_mask.new_zeros((batch_size * 2, full_len))
    combined_action_ids = torch.cat([action_a_ids, action_b_ids], dim=0)
    combined_action_mask = torch.cat([action_a_mask, action_b_mask], dim=0)

    context_starts = max_context_len - context_lens
    for row_idx in range(batch_size):
        context_len = int(context_lens[row_idx].item())
        context_start = int(context_starts[row_idx].item())
        full_ids[row_idx, context_start:max_context_len] = input_ids[row_idx, :context_len]
        full_mask[row_idx, context_start:max_context_len] = 1
        full_ids[batch_size + row_idx, context_start:max_context_len] = input_ids[row_idx, :context_len]
        full_mask[batch_size + row_idx, context_start:max_context_len] = 1

        action_a_len = int(action_a_lens[row_idx].item())
        action_b_len = int(action_b_lens[row_idx].item())
        full_ids[row_idx, max_context_len : max_context_len + action_a_len] = action_a_ids[row_idx, :action_a_len]
        full_mask[row_idx, max_context_len : max_context_len + action_a_len] = 1
        full_ids[batch_size + row_idx, max_context_len : max_context_len + action_b_len] = action_b_ids[
            row_idx, :action_b_len
        ]
        full_mask[batch_size + row_idx, max_context_len : max_context_len + action_b_len] = 1

    action_positions = torch.arange(
        max_context_len - 1,
        max_context_len - 1 + max_action_len,
        device=input_ids.device,
        dtype=torch.long,
    )
    selected_position_chunks = [action_positions]
    valid_kl_entries = []
    if ref_model is not None:
        s1_rows = torch.nonzero(is_s1 > 0.5, as_tuple=False).flatten()
        for row_idx in s1_rows.tolist():
            target_len = int(kl_target_mask[row_idx].sum().item())
            if target_len <= 0:
                continue
            start = int(kl_target_start[row_idx].item()) + int(context_starts[row_idx].item())
            positions = torch.arange(
                start,
                start + target_len,
                device=input_ids.device,
                dtype=torch.long,
            )
            selected_position_chunks.append(positions)
            valid_kl_entries.append((row_idx, positions))

    selected_positions = torch.unique(torch.cat(selected_position_chunks, dim=0), sorted=True)
    student_logits = model(
        input_ids=full_ids,
        attention_mask=full_mask,
        position_ids=_position_ids_from_attention_mask(full_mask),
        logits_to_keep=selected_positions,
    ).logits

    action_indices = torch.searchsorted(selected_positions, action_positions)
    action_logits = student_logits[:, action_indices, :]
    log_probs = F.log_softmax(action_logits, dim=-1)
    picked = log_probs.gather(dim=-1, index=combined_action_ids[:, :max_action_len].unsqueeze(-1)).squeeze(-1)
    combined_logprobs = (picked * combined_action_mask[:, :max_action_len].to(dtype=picked.dtype)).sum(dim=-1)
    policy_a = combined_logprobs[:batch_size]
    policy_b = combined_logprobs[batch_size:]

    if not valid_kl_entries:
        return policy_a, policy_b, None

    kl_positions = torch.unique(torch.cat([positions for _, positions in valid_kl_entries], dim=0), sorted=True)
    kl_row_indices = torch.tensor([row_idx for row_idx, _ in valid_kl_entries], device=input_ids.device, dtype=torch.long)
    context_only_ids = full_ids[:batch_size, :max_context_len]
    context_only_mask = full_mask[:batch_size, :max_context_len]
    with torch.no_grad():
        ref_logits = ref_model(
            input_ids=context_only_ids.index_select(0, kl_row_indices),
            attention_mask=context_only_mask.index_select(0, kl_row_indices),
            position_ids=_position_ids_from_attention_mask(context_only_mask.index_select(0, kl_row_indices)),
            logits_to_keep=kl_positions,
        ).logits

    tau = float(temperature)
    row_kl_losses = []
    for ref_row_idx, (student_row_idx, positions) in enumerate(valid_kl_entries):
        student_indices = torch.searchsorted(selected_positions, positions)
        ref_indices = torch.searchsorted(kl_positions, positions)
        student_sel = student_logits[student_row_idx, student_indices, :] / tau
        ref_sel = ref_logits[ref_row_idx, ref_indices, :] / tau
        per_token_kl = F.kl_div(
            F.log_softmax(student_sel, dim=-1),
            F.softmax(ref_sel, dim=-1),
            reduction="none",
            log_target=False,
        ).sum(dim=-1) * (tau ** 2)
        row_kl_losses.append(per_token_kl.sum())

    return policy_a, policy_b, torch.stack(row_kl_losses).mean()


def _batch_action_logprob(
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

    pad_id = _infer_pad_id(input_ids[attention_mask == 0], action_ids[action_mask == 0], input_ids)
    full_len = max_context_len + max_action_len
    full_ids = input_ids.new_full((batch_size, full_len), pad_id)
    full_mask = attention_mask.new_zeros((batch_size, full_len))

    for row_idx in range(batch_size):
        context_len = int(context_lens[row_idx].item())
        action_len = int(action_lens[row_idx].item())
        context_ids = input_ids[row_idx, :context_len]
        target_ids = action_ids[row_idx, :action_len]
        context_start = max_context_len - context_len
        full_ids[row_idx, context_start:max_context_len] = context_ids
        full_mask[row_idx, context_start:max_context_len] = 1
        full_ids[row_idx, max_context_len : max_context_len + action_len] = target_ids
        full_mask[row_idx, max_context_len : max_context_len + action_len] = 1

    keep_positions = torch.arange(
        max_context_len - 1,
        max_context_len - 1 + max_action_len,
        device=full_ids.device,
        dtype=torch.long,
    )
    logits = model(
        input_ids=full_ids,
        attention_mask=full_mask,
        position_ids=_position_ids_from_attention_mask(full_mask),
        logits_to_keep=keep_positions,
    ).logits
    log_probs = F.log_softmax(logits, dim=-1)
    target_ids = action_ids[:, :max_action_len]
    picked = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    return (picked * action_mask[:, :max_action_len].to(dtype=picked.dtype)).sum(dim=-1)


def _compute_s1_reasoning_kl(
    model,
    ref_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    kl_target_ids: torch.Tensor,
    kl_target_mask: torch.Tensor,
    kl_target_start: torch.Tensor,
    is_s1: torch.Tensor,
    temperature: float,
) -> torch.Tensor | None:
    s1_rows = torch.nonzero(is_s1 > 0.5, as_tuple=False).flatten()
    if s1_rows.numel() == 0:
        return None

    valid_entries = []
    for row_idx in s1_rows.tolist():
        row_mask = kl_target_mask[row_idx]
        if row_mask.numel() == 0 or not row_mask.any():
            continue
        context_len = int(attention_mask[row_idx].sum().item())
        target_len = int(row_mask.sum().item())
        if target_len == 0:
            continue
        valid_entries.append(
            {
                "row_idx": row_idx,
                "context_len": context_len,
                "target_len": target_len,
                "start": int(kl_target_start[row_idx].item()),
            }
        )

    if not valid_entries:
        return None

    valid_entries.sort(key=lambda entry: entry["start"])
    buckets = []
    current_bucket = []
    current_min_start = None
    for entry in valid_entries:
        if current_min_start is None or entry["start"] - current_min_start <= KL_START_BUCKET_SIZE:
            current_bucket.append(entry)
            if current_min_start is None:
                current_min_start = entry["start"]
        else:
            buckets.append(current_bucket)
            current_bucket = [entry]
            current_min_start = entry["start"]
    if current_bucket:
        buckets.append(current_bucket)

    row_kl_losses = []
    pad_id = _infer_pad_id(input_ids[attention_mask == 0], input_ids)
    tau = float(temperature)
    for bucket in buckets:
        common_start = max(entry["start"] for entry in bucket)
        max_target_len = max(entry["target_len"] for entry in bucket)
        max_full_len = max(
            max(common_start - entry["start"], 0) + entry["context_len"]
            for entry in bucket
        )
        max_full_len = max(max_full_len, common_start + max_target_len)
        bucket_size = len(bucket)

        full_ids = input_ids.new_full((bucket_size, max_full_len), pad_id)
        full_mask = attention_mask.new_zeros((bucket_size, max_full_len))
        target_mask = torch.zeros((bucket_size, max_target_len), dtype=torch.bool, device=input_ids.device)
        for bucket_idx, entry in enumerate(bucket):
            row_idx = entry["row_idx"]
            context_len = entry["context_len"]
            target_len = entry["target_len"]
            left_pad = common_start - entry["start"]
            full_ids[bucket_idx, left_pad : left_pad + context_len] = input_ids[row_idx, :context_len]
            full_mask[bucket_idx, left_pad : left_pad + context_len] = 1
            target_mask[bucket_idx, :target_len] = True

        keep_positions = torch.arange(
            common_start,
            common_start + max_target_len,
            device=full_ids.device,
            dtype=torch.long,
        )
        student_logits = model(
            input_ids=full_ids,
            attention_mask=full_mask,
            position_ids=_position_ids_from_attention_mask(full_mask),
            logits_to_keep=keep_positions,
        ).logits
        with torch.no_grad():
            ref_logits = ref_model(
                input_ids=full_ids,
                attention_mask=full_mask,
                position_ids=_position_ids_from_attention_mask(full_mask),
                logits_to_keep=keep_positions,
            ).logits

        student_log_probs = F.log_softmax(student_logits / tau, dim=-1)
        ref_probs = F.softmax(ref_logits / tau, dim=-1)
        per_token_kl = F.kl_div(
            student_log_probs,
            ref_probs,
            reduction="none",
            log_target=False,
        ).sum(dim=-1) * (tau ** 2)
        row_kl_losses.append((per_token_kl * target_mask.to(dtype=per_token_kl.dtype)).sum(dim=-1))

    return torch.cat(row_kl_losses, dim=0).mean()


def load_policy_model(model_path: str, bf16: bool, trust_remote_code: bool, device_map):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


def build_training_args_kwargs(
    *,
    output_dir: str,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    num_train_epochs: int,
    logging_steps: int,
    save_strategy: str,
    save_total_limit: int,
    bf16: bool,
    report_to: str,
    gradient_checkpointing: bool,
    world_size: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "logging_steps": logging_steps,
        "save_strategy": save_strategy,
        "save_total_limit": save_total_limit,
        "lr_scheduler_type": "cosine",
        "bf16": bf16,
        "report_to": report_to,
        "remove_unused_columns": False,
        "gradient_checkpointing": gradient_checkpointing,
        "logging_first_step": True,
        "weight_decay": 0.0,
    }
    if gradient_checkpointing:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if world_size > 1:
        kwargs["ddp_find_unused_parameters"] = False
    return kwargs


def _latest_checkpoint_in_output_dir(output_dir: str | None) -> str | None:
    if not output_dir or not os.path.isdir(output_dir):
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


def build_trainer_train_kwargs(
    resume_from_checkpoint: str | None,
    output_dir: str | None = None,
) -> dict[str, str]:
    checkpoint = resume_from_checkpoint or _latest_checkpoint_in_output_dir(output_dir)
    if not checkpoint:
        return {}
    return {"resume_from_checkpoint": checkpoint}


def main():
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    class MathLocalPrefTrainer(Trainer):
        def __init__(
            self,
            s1_alpha: float,
            kl_weight: float,
            kl_temperature: float,
            ref_model=None,
            *args,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.s1_alpha = float(s1_alpha)
            self.kl_weight = float(kl_weight)
            self.kl_temperature = float(kl_temperature)
            self.ref_model = ref_model

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            target_prob = inputs["target_prob"]
            state_weight = inputs["state_weight"]
            is_s1 = inputs["is_s1"]

            policy_a, policy_b, kl_loss = _packed_action_pair_logprobs_and_s1_kl(
                model=model,
                ref_model=self.ref_model if self.kl_weight > 0 else None,
                input_ids=input_ids,
                attention_mask=attention_mask,
                action_a_ids=inputs["action_a_ids"],
                action_a_mask=inputs["action_a_mask"],
                action_b_ids=inputs["action_b_ids"],
                action_b_mask=inputs["action_b_mask"],
                kl_target_mask=inputs["kl_target_mask"],
                kl_target_start=inputs["kl_target_start"],
                is_s1=is_s1,
                temperature=self.kl_temperature,
            )

            margin = policy_a - policy_b
            per_state_loss = -target_prob * F.logsigmoid(margin) - (1.0 - target_prob) * F.logsigmoid(-margin)
            alpha = torch.where(is_s1 > 0.5, torch.full_like(is_s1, self.s1_alpha), torch.ones_like(is_s1))
            weight = state_weight * alpha
            loss = (per_state_loss * weight).sum() / weight.sum().clamp_min(1e-8)

            if self.kl_weight > 0:
                if kl_loss is not None:
                    loss = loss + self.kl_weight * kl_loss

            return (loss, None) if return_outputs else loss

    parser = argparse.ArgumentParser(description="Train MATH same-state BCE local preference policy with optional KL.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ref_model_path", default=None)
    parser.add_argument("--s1_alpha", type=float, default=1.0)
    parser.add_argument("--kl_weight", type=float, default=0.0)
    parser.add_argument("--kl_temperature", type=float, default=1.0)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_strategy", default="epoch")
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--tokenize_num_proc", type=int, default=8)
    parser.add_argument("--tokenize_batch_size", type=int, default=1000)
    parser.add_argument("--resume_from_checkpoint", default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    raw_dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    def map_record(record):
        return prepare_pair_example(tokenizer, record, args.max_seq_length)

    tokenized = raw_dataset.map(
        map_record,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing MATH pair records",
        num_proc=args.tokenize_num_proc if args.tokenize_num_proc > 1 else None,
        writer_batch_size=args.tokenize_batch_size,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device_map = {"": local_rank} if torch.cuda.is_available() and world_size > 1 else None
    ref_device_map = {"": local_rank} if torch.cuda.is_available() else None

    model = load_policy_model(args.model_path, args.bf16, args.trust_remote_code, device_map=device_map)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    ref_model = None
    if args.kl_weight > 0:
        ref_path = args.ref_model_path or args.model_path
        ref_model = load_policy_model(ref_path, args.bf16, args.trust_remote_code, device_map=ref_device_map)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

    training_args = TrainingArguments(
        **build_training_args_kwargs(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            logging_steps=args.logging_steps,
            save_strategy=args.save_strategy,
            save_total_limit=args.save_total_limit,
            bf16=args.bf16,
            report_to=args.report_to,
            gradient_checkpointing=args.gradient_checkpointing,
            world_size=world_size,
        )
    )

    trainer = MathLocalPrefTrainer(
        s1_alpha=args.s1_alpha,
        kl_weight=args.kl_weight,
        kl_temperature=args.kl_temperature,
        ref_model=ref_model,
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=PairCollator(tokenizer),
    )
    trainer.train(**build_trainer_train_kwargs(args.resume_from_checkpoint, output_dir=args.output_dir))
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved MATH local preference model to: {args.output_dir}")


if __name__ == "__main__":
    main()
