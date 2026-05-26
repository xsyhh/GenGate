from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

SELF_REF_ROOT = Path(__file__).resolve().parents[2]
if str(SELF_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(SELF_REF_ROOT))

from decision_embedding_mask import (  # noqa: E402
    DecisionEmbeddingRowsCallback,
    build_lora_config_with_embedding_mask,
    inspect_embedding_tying,
    save_decision_embedding_rows,
    setup_decision_embedding_row_training,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SFTMathDataset(Dataset):
    def __init__(self, path: str):
        self.rows: List[Dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "text" not in obj:
                    continue
                self.rows.append(obj)
        if not self.rows:
            raise ValueError(f"No rows loaded from {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        return self.rows[idx]


def _find_subsequence(haystack: List[int], needle: List[int]) -> int:
    if not needle or len(needle) > len(haystack):
        return -1
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return -1


def _find_last_subsequence(haystack: List[int], needle: List[int], start: int = 0) -> int:
    if not needle or len(needle) > len(haystack):
        return -1
    n = len(needle)
    lo = max(0, start)
    for i in range(len(haystack) - n, lo - 1, -1):
        if haystack[i : i + n] == needle:
            return i
    return -1


@dataclass
class ConfidenceTokenCollator:
    tokenizer: AutoTokenizer
    max_length: int
    train_on_prompt: bool = False

    def __post_init__(self):
        self.self_token_ids = self.tokenizer.encode("<CN>", add_special_tokens=False)
        self.defer_token_ids = self.tokenizer.encode("<UN>", add_special_tokens=False)
        if not self.self_token_ids or not self.defer_token_ids:
            raise ValueError("Failed to tokenize <CN> or <UN>.")

    def _encode_text(self, text: str) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=True,
        )
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]
        labels = input_ids.copy()
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _assistant_response_start(self, input_ids: List[int]) -> int:
        assistant_tok = self.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
        pos = _find_last_subsequence(input_ids, assistant_tok, start=0)
        if pos >= 0:
            return pos + len(assistant_tok)
        return 0

    def _mask_prompt_tokens(self, input_ids: List[int], labels: List[int], response_start: int) -> None:
        for i in range(min(response_start, len(labels))):
            labels[i] = -100

    def _mask_wrong_answer_keep_decision(self, input_ids: List[int], labels: List[int], response_start: int) -> None:
        pos_self = _find_last_subsequence(input_ids, self.self_token_ids, start=response_start)
        pos_defer = _find_last_subsequence(input_ids, self.defer_token_ids, start=response_start)
        
        if pos_self < 0 and pos_defer < 0:
            for i in range(len(labels)):
                labels[i] = -100
            return

        if pos_self >= 0 and (pos_defer < 0 or pos_self > pos_defer):
            start = pos_self
            dec_len = len(self.self_token_ids)
        else:
            start = pos_defer
            dec_len = len(self.defer_token_ids)
        end = start + dec_len

        for i in range(len(labels)):
            if not (start <= i < end):
                labels[i] = -100

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_input_ids: List[torch.Tensor] = []
        batch_attn: List[torch.Tensor] = []
        batch_labels: List[torch.Tensor] = []

        for feat in features:
            text = str(feat["text"])
            self_passed = bool(int(feat.get("_self_passed", 0)))
            enc = self._encode_text(text)

            input_ids = enc["input_ids"].tolist()
            labels = enc["labels"].tolist()

            response_start = self._assistant_response_start(input_ids)
            if not self.train_on_prompt:
                self._mask_prompt_tokens(input_ids, labels, response_start=response_start)

            if not self_passed:
                self._mask_wrong_answer_keep_decision(input_ids, labels, response_start=response_start)

            batch_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            batch_attn.append(enc["attention_mask"])
            batch_labels.append(torch.tensor(labels, dtype=torch.long))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            batch_attn, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            batch_labels, batch_first=True, padding_value=-100
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def split_train_eval(rows: List[Dict], eval_ratio: float, seed: int) -> tuple[List[Dict], List[Dict]]:
    idx = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_eval = int(math.floor(len(rows) * eval_ratio))
    eval_idx = set(idx[:n_eval])
    train_rows, eval_rows = [], []
    for i, r in enumerate(rows):
        if i in eval_idx:
            eval_rows.append(r)
        else:
            train_rows.append(r)
    return train_rows, eval_rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", type=str, required=True, help="step3 output jsonl")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument(
        "--fix_mistral_regex",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--num_train_epochs", type=float, default=5.0)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--weight_decay", type=float, default=0.0) # 注意：如果启用 weight_decay，哪怕梯度为0权重也会缓慢衰减。建议保持默认0.0
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    p.add_argument("--eval_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=None)
    p.add_argument("--eval_ratio", type=float, default=0.02)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--train_on_prompt", action="store_true")

    p.add_argument("--disable_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj",
    )
    args = p.parse_args()

    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tok_kwargs: Dict[str, object] = {"trust_remote_code": args.trust_remote_code}
    if args.fix_mistral_regex:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, fix_mistral_regex=True, **tok_kwargs)
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, **tok_kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, **tok_kwargs)
    
    decision_tokens = ["<CN>", "<UN>"]
    decision_ids = tokenizer.convert_tokens_to_ids(decision_tokens)
    print("Decision tokens:")
    for tok, tok_id in zip(decision_tokens, decision_ids):
        print(f"  {tok} -> {tok_id}")
    if any(tok_id is None for tok_id in decision_ids):
        raise ValueError("Decision tokens must already exist in the tokenizer; refusing to add them again.")
    if any(tok_id == tokenizer.unk_token_id for tok_id in decision_ids if tokenizer.unk_token_id is not None):
        raise ValueError("Decision tokens map to unk_token; use a model/tokenizer that already contains them.")
    num_added_toks = 0
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if args.dtype == "bf16":
        load_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        load_dtype = torch.float16
    else:
        load_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        dtype=load_dtype,
    )
    model.config.use_cache = False
    inspect_embedding_tying(model)

    # Decision tokens are expected to be present in the model/tokenizer already.

    if not args.disable_lora:
        try:
            from peft import get_peft_model
        except ImportError as e:
            raise RuntimeError("peft is required for LoRA. Please install `peft`.") from e

        peft_cfg = build_lora_config_with_embedding_mask(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )
        model = get_peft_model(model, peft_cfg)
        setup_decision_embedding_row_training(model, decision_ids)
        model.print_trainable_parameters()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    full_ds = SFTMathDataset(args.train_jsonl)
    train_rows, eval_rows = split_train_eval(full_ds.rows, eval_ratio=args.eval_ratio, seed=args.seed)
    train_ds = SFTMathDataset.__new__(SFTMathDataset)
    train_ds.rows = train_rows
    eval_ds = SFTMathDataset.__new__(SFTMathDataset)
    eval_ds.rows = eval_rows

    collator = ConfidenceTokenCollator(
        tokenizer=tokenizer,
        max_length=args.max_length,
        train_on_prompt=bool(args.train_on_prompt),
    )

    use_bf16 = args.dtype == "bf16"
    use_fp16 = args.dtype == "fp16"

    training_kwargs: Dict[str, object] = {
        "output_dir": args.output_dir,
        "overwrite_output_dir": True,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,  # 参数默认是 0.0，非常完美
        "lr_scheduler_type": args.lr_scheduler_type,
        "logging_steps": args.logging_steps,
        "save_strategy": args.save_strategy,
        "save_total_limit": args.save_total_limit,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "seed": args.seed,
        "report_to": ["none"],
        "dataloader_num_workers": 2,
        "remove_unused_columns": False,
    }
    
    eval_strategy = args.eval_strategy if len(eval_rows) > 0 else "no"
    if args.save_strategy == "steps":
        training_kwargs["save_steps"] = args.save_steps
    if eval_strategy == "steps":
        training_kwargs["eval_steps"] = args.eval_steps
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in ta_params:
        training_kwargs["evaluation_strategy"] = eval_strategy
    elif "eval_strategy" in ta_params:
        training_kwargs["eval_strategy"] = eval_strategy

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if len(eval_rows) > 0 else None,
        data_collator=collator,
        callbacks=[DecisionEmbeddingRowsCallback(decision_ids)],
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    save_decision_embedding_rows(model, args.output_dir, decision_ids)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[done] model & tokenizer saved to {args.output_dir}")


if __name__ == "__main__":
    main()
