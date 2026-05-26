import pandas as pd
import json
import re
import argparse
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# 定义你的模板 (请根据你实际的 Prompt 替换此处内容)
HUMAN_EVAL_EXPERT_TEMPLATE = """You are a code agent.
### Question:
{problem}
### Starter Code:
```python
{current_code}
```
### INSTRUCION: 
You will use the following starter code to write the solution to the problem. 
Output the solution enclosed in ```python ... ``` blocks.
Do NOT include any explanations, comments, or extra text outside the code block.
Do NOT generate any test cases, assertions, usage examples, or `if __name__ == "__main__":` blocks.
"""

def extract_python_code(text):
    """从模型输出中提取并清洗 Python 代码。"""
    text = str(text or "").strip()

    # 兼容 ```python、```python 空格、无闭合 fence 等常见格式。
    match = re.search(r"```(?:python|py)?\s*\n?(.*?)(?:\n?```|$)", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    text = re.sub(r"^\s*```(?:python|py)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    text = re.sub(r"^\s*(?:python|py)\s*\n", "", text, flags=re.IGNORECASE)
    return text.strip()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=str, default="data/code_benchmarks/mbppplus_leetcode_humanevalplus_v2/train_with_id.csv")
    p.add_argument("--model_path", type=str, default="Meta-Llama-3-8B-Instruct_Expert_Direct")
    p.add_argument("--output_path", type=str, default="llama/code/output/llama3_8b_actor_samples.jsonl")
    p.add_argument("--num_samples_per_task", type=int, default=3)
    p.add_argument("--trust_remote_code", action="store_true")
    args = p.parse_args()

    print("🚀 Step 1: 开始构造数据并进行 Actor 模型多次采样...")

    # 2. 初始化 Tokenizer 和 vLLM
    print(f"加载模型和分词器: {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    # 开启 vLLM，可根据显存调整 tensor_parallel_size 和 gpu_memory_utilization
    llm = LLM(model=args.model_path, trust_remote_code=args.trust_remote_code) 
    
    # 设置采样参数，n=num_samples_per_task 实现对同一 prompt 的多次采样
    sampling_params = SamplingParams(
        temperature=0.7, 
        top_p=0.95, 
        max_tokens=1024, 
        n=args.num_samples_per_task 
    )
    
    # 3. 读取 CSV 并构造 Prompts
    df = pd.read_csv(args.csv_path)
    prompts = []
    task_metadata = [] # 用于记录每条 prompt 对应的原信息
    
    print("正在通过 Tokenizer 应用对话模板...")
    for _, row in df.iterrows():
        task_id = row['id']
        problem = row['problem']
        starter_code = row['starter_code']
        entry_point = row['entry_point']
        
        # 将参数传入模板
        user_content = HUMAN_EVAL_EXPERT_TEMPLATE.format(
            problem=problem, 
            current_code=starter_code
        )
        
        # 组装 Chat 格式
        messages = [
            {"role": "user", "content": user_content}
        ]
        
        # 应用 apply_chat_template 构造最终输入字符串
        prompt_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        prompts.append(prompt_text)
        task_metadata.append({
            "id": task_id,
            "entry_point": entry_point,
            "starter_code": starter_code # 备用，有时评测需要
        })
        
    # 4. 批量生成 (vLLM 会自动处理批处理和并发)
    print(f"开始推理生成，共 {len(prompts)} 个问题，每个问题采样 {args.num_samples_per_task} 次...")
    outputs = llm.generate(prompts, sampling_params)
    
    # 5. 提取代码并保存
    results = []
    for metadata, output in zip(task_metadata, outputs):
        task_id = metadata["id"]
        entry_point = metadata["entry_point"]
        
        # output.outputs 包含了 n 次采样的结果
        for i, response in enumerate(output.outputs):
            raw_text = response.text
            code_only = extract_python_code(raw_text)
            
            results.append({
                "id": task_id,
                "sample_index": i,
                "entry_point": entry_point,
                "raw_output": raw_text,
                "extracted_code": code_only
            })
            
    # 6. 写入 JSONL 文件
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
    print(f"✅ Step 1 完成！共生成 {len(results)} 条采样数据，已保存至 {args.output_path}")

if __name__ == "__main__":
    main()
