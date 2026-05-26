from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from math_eval_setup import ensure_math_eval_on_path

ensure_math_eval_on_path()

# 弃用 extract_answer，只保留清洗和等价判断
from parser import strip_string
from grader import math_equal

NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")

# ==============================================================================
# 1. 防弹版提取逻辑
# ==============================================================================
def normalize_answer(raw: str) -> str:
    text = str(raw or "").strip()
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r"\s+", "", text)
    text = text.rstrip(".。;；,，")
    return text

def extract_final_answer(text: str) -> str:
    """防弹版抽取：定位 Final answer: 并剥离废话、LaTeX定界符、Markdown及 \boxed 肌肉记忆"""
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

    if not ans:
        return ""

    # 2. 去除 Markdown 和 LaTeX 定界符
    ans = ans.strip()
    ans = ans.strip("*").strip()
    ans = ans.strip("$").strip()
    
    if ans.startswith("\\(") and ans.endswith("\\)"):
        ans = ans[2:-2].strip()
    elif ans.startswith("\\[") and ans.endswith("\\]"):
        ans = ans[2:-2].strip()

    ans = ans.strip(" \n.。;；")

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
                chars.append(ch)
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
                chars.append(ch)
            else:
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
# 2. 评测主逻辑
# ==============================================================================
def _load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _build_answer_map(path: str, task_id_key: str, answer_key: str) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for i, row in enumerate(_load_jsonl(path)):
        task_id = str(row.get(task_id_key, row.get("id", i)))
        answer = str(row.get(answer_key, "")).strip()
        m[task_id] = answer
    return m


def _eval_self_passed(pred_text: str, answer: str) -> Tuple[bool, str]:
    if answer is None or str(answer).strip() == "":
        return False, ""

    # 使用我们自己的提取逻辑拿到纯净的答案表达式
    prediction = extract_final_answer(str(pred_text))
    reference = str(answer).strip()

    if not prediction:
        return False, ""

    # 1. 先做基础的字符串严格匹配
    if prediction == reference:
        return True, prediction

    # 2. 动用 Qwen 的数学等价判断神器
    try:
        p_stripped = strip_string(prediction)
        g_stripped = strip_string(reference)
        # include_percentage=True 允许处理 50% = 0.5 等情况
        passed = bool(math_equal(p_stripped, g_stripped, include_percentage=True, is_close=True, timeout=True))
    except Exception as e:
        # 极少数极端公式可能会导致 sympy 超时或报错，做个兜底
        try:
            passed = bool(math_equal(prediction, reference, include_percentage=True, is_close=True, timeout=True))
        except Exception:
            passed = False

    return passed, prediction


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--actor_samples_jsonl", type=str, default="llama/math/output/step1.jsonl")
    p.add_argument("--data_jsonl", type=str, default="data/MATH/hendrycks_math_train.jsonl")
    p.add_argument("--output_jsonl", type=str, default="llama/math/output/step2.jsonl")
    p.add_argument("--task_id_key", type=str, default="unique_id")
    p.add_argument("--answer_key", type=str, default="answer")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    answer_map = _build_answer_map(args.data_jsonl, task_id_key=args.task_id_key, answer_key=args.answer_key)
    samples = _load_jsonl(args.actor_samples_jsonl)

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for item in tqdm(samples, desc="Validate"):
            task_id = str(item["id"])
            # 取出 Step 1 中截取好的 reasoning
            reasoning = str(item.get("extracted_reasoning", item.get("raw_output", "")))
            answer = answer_map.get(task_id, "")

            self_passed, prediction = _eval_self_passed(
                pred_text=reasoning,
                answer=answer
            )

            out = {
                "id": task_id,
                "sample_index": int(item.get("sample_index", 0)),
                "raw_output": item.get("raw_output", ""),
                "extracted_reasoning": reasoning,
                "prediction": prediction,       # 模型提取后的答案
                "gold_answer": answer,          # 方便你人工 Debug 的标准答案
                "self_passed": bool(self_passed),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[step2] wrote validation to {args.output_jsonl}")


if __name__ == "__main__":
    main()
