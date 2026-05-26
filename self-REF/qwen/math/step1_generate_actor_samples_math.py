from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from tokens import strip_trailing_decision_token

# ==============================================================================
# 1. 统一 Prompt：强制模型在最后输出 Final answer:
# ==============================================================================
MATH_PROMPT_TEMPLATE = """You are a math reasoning agent.
### Question:
{problem}
### INSTRUCTION:
1. Solve the problem step by step.
2. Conclude your response with exactly "Final answer: " followed IMMEDIATELY by the bare mathematical expression or value.
3. Do NOT include any extra text after the final answer.
"""

NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")

def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def _load_jsonl(path: str, limit: Optional[int] = None) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def _strip_decision_tokens(text: str) -> str:
    """剥离可能意外生成的决策 token，保留纯推理文本"""
    cleaned, _ = strip_trailing_decision_token(text)
    return cleaned

# ==============================================================================
# 2. 防弹版答案提取逻辑
# ==============================================================================
def normalize_answer(raw: str) -> str:
    text = str(raw or "").strip()
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r"\s+", "", text)
    text = text.rstrip(".。;；,，")
    return text

def extract_final_answer(text: str) -> str:
    """定位 Final answer: 并剥离废话、LaTeX定界符、Markdown及 \boxed"""
    text = str(text or "").strip()
    if not text:
        return ""

    # 1. 截取 Final answer: 之后的内容
    if "Final answer:" in text:
        ans = text.rsplit("Final answer:", 1)[1].strip()
    elif "final answer:" in text.lower():
        ans = re.split(r"final answer:", text, flags=re.IGNORECASE)[-1].strip()
    else:
        ans = text

    # 2. 去除 Markdown 和 LaTeX 定界符
    ans = ans.strip("*").strip("$").strip()
    
    if ans.startswith("\\(") and ans.endswith("\\)"):
        ans = ans[2:-2].strip()
    elif ans.startswith("\\[") and ans.endswith("\\]"):
        ans = ans[2:-2].strip()

    # 3. 对抗“肌肉记忆”：剥离 \boxed{}
    match = re.search(r"\\boxed\s*\{", ans)
    if match:
        i = match.end()
        depth = 1
        chars = []
        while i < len(ans):
            ch = ans[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            chars.append(ch)
            i += 1
        ans = "".join(chars).strip()

    # 4. 兜底逻辑：尝试找纯数字
    if not ans:
        numbers = NUMBER_RE.findall(text.replace(" ", ""))
        if numbers:
            return normalize_answer(numbers[-1].replace(",", ""))
        return ""
        
    return normalize_answer(ans)

# ==============================================================================
# 3. 主干推理逻辑 (vLLM)
# ==============================================================================
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_jsonl", type=str, required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--output_jsonl", type=str, required=True)
    p.add_argument("--num_samples_per_task", type=int, default=3)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--task_id_key", type=str, default="unique_id")
    p.add_argument("--problem_key", type=str, default="problem")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    _ensure_dir(Path(args.output_jsonl).parent.as_posix())

    rows = _load_jsonl(args.data_jsonl, limit=args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    
    # 初始化 vLLM
    llm = LLM(model=args.model_path, trust_remote_code=args.trust_remote_code)

    sampling_params = SamplingParams(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
        n=int(args.num_samples_per_task),
        seed=int(args.seed),
    )

    prompts: List[str] = []
    metas: List[Dict] = []
    for i, row in enumerate(rows):
        task_id = str(row.get(args.task_id_key, row.get("id", i)))
        problem = str(row.get(args.problem_key, ""))
        prompt = MATH_PROMPT_TEMPLATE.format(problem=problem)
        
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)
        metas.append({"id": task_id})

    # 批量推理
    outputs = llm.generate(prompts, sampling_params)

    # 结果写入
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for meta, out in zip(metas, outputs):
            for sample_idx, resp in enumerate(out.outputs):
                raw = str(resp.text)
                # 剥离最后的 decision token (如果有的话)
                cleaned_reasoning = _strip_decision_tokens(raw)
                # 提取出最终的标准答案
                final_answer = extract_final_answer(cleaned_reasoning)
                
                record = {
                    "id": meta["id"],
                    "sample_index": sample_idx,
                    "raw_output": raw,
                    "extracted_reasoning": cleaned_reasoning,
                    "extracted_answer": final_answer,  # 新增答案字段，方便后续直接评测
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[step1] wrote samples to {args.output_jsonl}")

if __name__ == "__main__":
    main()
