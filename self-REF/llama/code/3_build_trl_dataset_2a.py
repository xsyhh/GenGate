import argparse
import json
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer

from path_setup import ensure_self_ref_root_on_path
from tokens import decision_token_from_passed

ensure_self_ref_root_on_path()
from build_sampling import add_sampling_args, balance_decision_records, describe_decision_records

# 你的无历史对话模板
HUMAN_EVAL_AGENT_TEMPLATE_NO_HIS = """You are a code agent with a clear understanding of your own capabilities, you can actively seek help from expert.
### Question:
{problem}
### Starter Code:
```python
{current_code}
```
### INSTRUCION:
1. First, output the solution code enclosed in ```python ... ``` block.
2. Then, output a decision token: <CN> or <UN>.
    - <CN>: You believe your code is complete and correct. It passes all tests without expert help.
    - <UN>: You believe your code is uncertain or incorrect, and therefore need to request expert help.
3. Do NOT include any explanations, comments, or extra text.
4. Do NOT generate any test cases, assertions, usage examples, or `if __name__ == "__main__":` blocks.
5. DO NOT put decision token inside code block.
### FORMAT:
```python
Your code
```
Your decision token
"""

def get_decision_token(self_passed, expert_passed=None):

    return decision_token_from_passed(bool(self_passed))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--raw_csv_path",
        type=str,
        default="data/code_benchmarks/mbppplus_leetcode_humanevalplus_v2/train_with_id.csv",
    )
    p.add_argument(
        "--validation_jsonl",
        type=str,
        default="llama/code/output/llama3_8b_validation_results.jsonl",
    )
    p.add_argument(
        "--output_jsonl",
        type=str,
        default="llama/code/output/trl_sft_data_2way.jsonl",
    )
    p.add_argument(
        "--model_path",
        type=str,
        default="Meta-Llama-3-8B-Instruct_Expert_Direct",
    )
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    add_sampling_args(p)
    return p.parse_args()

def main():
    args = parse_args()
    print("🚀 开始构造二元版 TRL SFT 训练数据 (无 Reject)...")

    # --- 1. 初始化 Tokenizer ---
    print(f"正在加载 Tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)

    # --- 2. 读取原始 CSV ---
    print(f"正在读取原始数据: {args.raw_csv_path}")
    df_raw = pd.read_csv(args.raw_csv_path)
    raw_dict = {}
    for _, row in df_raw.iterrows():
        tid = str(row['id']).strip()
        raw_dict[tid] = {
            'problem': row['problem'],
            'starter_code': row['starter_code']
        }

    # --- 3. 读取验证结果并拼接数据 ---
    print(f"正在读取验证结果: {args.validation_jsonl}")
    trl_dataset = []
    missing_ids = 0

    with open(args.validation_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            task_id = str(item["id"]).strip()

            if task_id not in raw_dict:
                missing_ids += 1
                continue

            problem_text = raw_dict[task_id]['problem']
            starter_code = raw_dict[task_id]['starter_code']
            actor_code = item["extracted_code"]
            decision_token = get_decision_token(item["self_passed"])

            # 4.1 构造 User 侧的内容
            user_content = HUMAN_EVAL_AGENT_TEMPLATE_NO_HIS.format(
                problem=problem_text,
                current_code=starter_code
            )

            # 4.2 使用 apply_chat_template 构造标准的对话前缀
            messages = [{"role": "user", "content": user_content}]
            prompt_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True 
            )

            # 4.3 构造 Assistant 的完整回复
            assistant_response = f"```python\n{actor_code}\n```\n{decision_token}"
            
            # 加上 EOS token 闭合整个序列
            full_text = prompt_text + assistant_response + tokenizer.eos_token

            trl_dataset.append({
                "text": full_text,
                "_task_id": task_id, 
                "_decision": decision_token,
                "_self_passed": int(bool(item["self_passed"])),
            })

    if missing_ids > 0:
        print(f"⚠️ 警告：有 {missing_ids} 条数据在 CSV 中找不到 ID。")

    print(f"[step3] before sampling: {describe_decision_records(trl_dataset)}")
    trl_dataset = balance_decision_records(
        trl_dataset,
        max_total_samples=args.max_total_samples,
        self_ratio=args.self_ratio,
        seed=args.seed,
    )
    print(f"[step3] after sampling: {describe_decision_records(trl_dataset)}")

    # --- 4. 导出文件 ---
    print(f"正在写入数据，共 {len(trl_dataset)} 条样本...")
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for entry in trl_dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ 二元 TRL 数据集准备完毕！文件保存在: {args.output_jsonl}")
    
    # 打印一条样例，确保格式正确
    if trl_dataset:
        print("\n--- 📝 随机抽取一条数据样例检查 ---")
        print(trl_dataset[0]["text"]) 
        print("----------------------------------\n")

if __name__ == "__main__":
    main()
