"""
MMLU Decision-Only SFT Training (TRL 2-way version)

Uses TRL's SFTTrainer with LoRA. Only trains on the last decision token
(<CN> or <UN>) via DecisionOnlyCollator.

Usage:
    torchrun --nproc_per_node=N train_trl_2way.py \
        --train_jsonl  <step3 output> \
        --model_path   <base model> \
        --output_dir   <save dir>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from trl import SFTTrainer

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


class DecisionOnlyCollator(DataCollatorForLanguageModeling):
    """Correct rows train answer+decision; incorrect rows train only decision."""

    def __init__(self, tokenizer, decision_token_ids, *args, **kwargs):
        super().__init__(tokenizer=tokenizer, *args, **kwargs)
        self.decision_token_ids = torch.tensor(decision_token_ids, dtype=torch.long)
        self.assistant_markers = [
            tokenizer.encode("<|im_start|>assistant", add_special_tokens=False),
            tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False),
        ]

    @staticmethod
    def _find_last_subsequence(haystack, needle):
        if not needle or len(needle) > len(haystack):
            return -1
        n = len(needle)
        for idx in range(len(haystack) - n, -1, -1):
            if haystack[idx : idx + n] == needle:
                return idx
        return -1

    def _assistant_response_start(self, input_ids):
        seq = input_ids.tolist()
        for marker in self.assistant_markers:
            pos = self._find_last_subsequence(seq, marker)
            if pos >= 0:
                return pos + len(marker)
        return 0

    def __call__(self, features, return_tensors=None):
        batch = super().__call__(features, return_tensors=return_tensors)
        labels = batch["labels"]
        target_ids = self.decision_token_ids.to(labels.device)

        for i in range(labels.size(0)):
            original = labels[i].clone()
            target_mask = torch.isin(original, target_ids)
            target_positions = target_mask.nonzero(as_tuple=True)[0]

            labels[i].fill_(-100)

            if len(target_positions) > 0:
                decision_pos = target_positions[-1].item()
                self_passed = bool(int(features[i].get("_self_passed", 0)))
                if self_passed:
                    start_pos = self._assistant_response_start(batch["input_ids"][i])
                    labels[i, start_pos:] = original[start_pos:]
                else:
                    labels[i, decision_pos] = original[decision_pos]

        return batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", type=str, required=True, help="step3 output jsonl with 'text' field")
    p.add_argument("--model_path", type=str, required=True, help="base model (with decision tokens in vocab)")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--max_length", type=int, default=4096)

    # training hypers
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--num_train_epochs", type=float, default=5.0)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_strategy", type=str, default="epoch")
    p.add_argument("--save_total_limit", type=int, default=5)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--report_to", type=str, default="none")

    # LoRA hypers
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj",
    )

    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = p.parse_args()

    print("启动 MMLU 路由模型训练 (TRL 2-way, 只训练决策 Token)...")

    # --- 1. 加载 tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- 2. 获取决策 token ID ---
    decision_tokens = ["<UN>", "<CN>"]
    decision_ids = tokenizer.convert_tokens_to_ids(decision_tokens)

    print("决策 Token 与对应 ID：")
    for tok, tok_id in zip(decision_tokens, decision_ids):
        print(f"  {tok} -> {tok_id}")

    if any(tok_id is None for tok_id in decision_ids):
        raise ValueError("存在决策 Token 未被 tokenizer 识别。")
    if any(
        tok_id == tokenizer.unk_token_id
        for tok_id in decision_ids
        if tokenizer.unk_token_id is not None
    ):
        raise ValueError("存在决策 Token 被映射成 unk_token，请检查 tokenizer 词表。")

    # --- 3. 加载并 tokenize 数据集 ---
    dataset = load_dataset("json", data_files=args.train_jsonl, split="train")

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
        )
        tokenized["_self_passed"] = [int(x) for x in examples["_self_passed"]]
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    # --- 4. 加载模型 ---
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_map = {"": local_rank}

    print("正在加载模型权重...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=args.trust_remote_code,
    )
    inspect_embedding_tying(model)

    # --- 5. LoRA 配置 ---
    lora_config = build_lora_config_with_embedding_mask(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --- 6. Decision-only collator ---
    collator = DecisionOnlyCollator(
        tokenizer=tokenizer,
        decision_token_ids=decision_ids,
        mlm=False,
    )

    # --- 7. 训练参数 ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        save_strategy=args.save_strategy,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=True,
        report_to=args.report_to,
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        save_total_limit=args.save_total_limit,
    )

    # --- 8. 构建 SFTTrainer ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset,
        peft_config=lora_config,
        data_collator=collator,
        args=training_args,
    )
    setup_decision_embedding_row_training(trainer.model, decision_ids)
    trainer.add_callback(DecisionEmbeddingRowsCallback(decision_ids))

    # --- 9. 开始训练 ---
    print("\n" + "=" * 60)
    print("训练即将开始！")
    print("正确轨迹训练答案+决策；错误轨迹只训练决策 Token。")
    print("=" * 60 + "\n")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # --- 10. 保存模型 ---
    trainer.save_model(args.output_dir)
    save_decision_embedding_rows(trainer.model, args.output_dir, decision_ids)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n训练完成，MMLU 二元路由模型已保存至: {args.output_dir}")


if __name__ == "__main__":
    main()
