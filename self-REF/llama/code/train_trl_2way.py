import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer
import random
import numpy as np

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
from tokens import TOKEN_DEFER, TOKEN_SELF
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DecisionOnlyCollator(DataCollatorForLanguageModeling):
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

            # 找到序列里所有 decision token 的位置
            target_mask = torch.isin(original, target_ids)
            target_positions = target_mask.nonzero(as_tuple=True)[0]

            # 默认整条样本全 mask
            labels[i].fill_(-100)

            if len(target_positions) > 0:
                decision_pos = target_positions[-1].item()
                self_passed = bool(int(features[i].get("_self_passed", 0)))
                if self_passed:
                    start_pos = self._assistant_response_start(batch["input_ids"][i])
                    labels[i, start_pos:] = original[start_pos:]
                else:
                    labels[i, decision_pos] = original[decision_pos]
            # 如果没有 decision token，就整条作废

        return batch

def main():
    print("🚀 启动物理极限测试：纯净版 TRL 路由模型训练（二元两路版，只训练决策 Token）...")

    # --- 1. 路径配置 ---
    model_path = "Meta-Llama-3-8B-Instruct_Expert_Direct"
    dataset_path = "llama/code/output/trl_sft_data_2way.jsonl"
    output_dir = "llama/code/ckpt/llama_code_only_decision_r0.8"

    # --- 2. 加载 tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- 3. 自动获取决策 token 的 ID ---
    decision_tokens = [TOKEN_DEFER, TOKEN_SELF]
    decision_ids = tokenizer.convert_tokens_to_ids(decision_tokens)

    print("决策 Token 与对应 ID：")
    for tok, tok_id in zip(decision_tokens, decision_ids):
        print(f"  {tok} -> {tok_id}")

    if any(tok_id is None for tok_id in decision_ids):
        raise ValueError("存在决策 Token 未被 tokenizer 识别。")

    if any(tok_id == tokenizer.unk_token_id for tok_id in decision_ids if tokenizer.unk_token_id is not None):
        raise ValueError("存在决策 Token 被映射成 unk_token，请检查 tokenizer 词表。")

    # --- 4. 加载数据集 ---
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    print("正在对数据集进行 Tokenization...")

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=4096,
        )
        if "_self_passed" in examples:
            tokenized["_self_passed"] = [int(x) for x in examples["_self_passed"]]
        else:
            tokenized["_self_passed"] = [
                1 if str(x).strip() == TOKEN_SELF else 0
                for x in examples.get("_decision", [""] * len(examples["text"]))
            ]
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    # --- 5. 设备映射 ---
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_map = {"": local_rank}

    # --- 6. 加载模型 ---
    print("正在加载模型权重...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    inspect_embedding_tying(model)

    # 如果你真的新增过 special tokens，需要解除下面两行注释
    # model.resize_token_embeddings(len(tokenizer))

    # --- 7. LoRA 配置 ---
    lora_config = build_lora_config_with_embedding_mask(
        r=8,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --- 8. 自定义碰撞器 ---
    collator = DecisionOnlyCollator(
        tokenizer=tokenizer,
        decision_token_ids=decision_ids,
        mlm=False,
    )

    # --- 9. 训练参数 ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        logging_steps=5,
        num_train_epochs=5,
        warmup_steps=200,
        save_strategy="epoch",
        lr_scheduler_type="cosine",
        bf16=True,
        report_to="tensorboard",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        save_total_limit=5,
    )

    # --- 10. 构建 Trainer ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset,
        peft_config=lora_config,
        data_collator=collator,
        args=training_args,
    )
    setup_decision_embedding_row_training(trainer.model, decision_ids)
    trainer.add_callback(DecisionEmbeddingRowsCallback(decision_ids))

    # --- 11. 开始训练 ---
    print("\n" + "=" * 60)
    print("🔥 训练即将开始！")
    print("当前设置为：每条样本只对最后一个决策 Token 计算 Loss。")
    print("不会再训练后续的 <|im_end|>、<|endoftext|> 或 EOS。")
    print("=" * 60 + "\n")

    trainer.train()

    # --- 12. 保存模型 ---
    trainer.save_model(output_dir)
    save_decision_embedding_rows(trainer.model, output_dir, decision_ids)
    tokenizer.save_pretrained(output_dir)

    print(f"\n🎉 审判实验结束，二元路由模型已保存至: {output_dir}")


if __name__ == "__main__":
    set_seed(42)
    main()
