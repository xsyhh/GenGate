from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
from transformers import AutoTokenizer

from path_setup import ensure_self_ref_root_on_path
from tokens import decision_token_from_passed

ensure_self_ref_root_on_path()
from build_sampling import add_sampling_args, balance_decision_records, describe_decision_records

# ==============================================================================
# 统一 Prompt：使用 Final answer: 并追加 decision token
# ==============================================================================
MATH_PROMPT_TEMPLATE = """You are a math reasoning agent.
### Question:
{problem}
### INSTRUCTION:
1. Solve the problem step by step.
2. Conclude your response with exactly "Final answer: " followed IMMEDIATELY by the bare mathematical expression or value.
3. Finally, output a decision token: <CN> or <UN>.
   - <CN>: You believe your solution and final answer are correct.
   - <UN>: You believe your solution is uncertain.
4. Do NOT include any extra text after the decision token.
### FORMAT:
Your reasoning
Final answer: value
Your decision token
"""


def _load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _decision_from_self_passed(self_passed: bool) -> str:
    return decision_token_from_passed(bool(self_passed))


def _build_problem_map(data_jsonl: str, task_id_key: str, problem_key: str) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for i, row in enumerate(_load_jsonl(data_jsonl)):
        task_id = str(row.get(task_id_key, row.get("id", i)))
        m[task_id] = str(row.get(problem_key, ""))
    return m


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_jsonl", type=str, default="data/MATH/hendrycks_math_train.jsonl")
    p.add_argument("--validation_jsonl", type=str, default="llama/math/output/step2.jsonl")
    p.add_argument("--model_path", type=str, default="Meta-Llama-3-8B-Instruct_Expert_Direct")
    p.add_argument("--output_jsonl", type=str, default="llama/math/output/step3.jsonl")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--task_id_key", type=str, default="unique_id")
    p.add_argument("--problem_key", type=str, default="problem")
    p.add_argument("--seed", type=int, default=42)
    add_sampling_args(p)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    _ensure_dir(Path(args.output_jsonl).parent.as_posix())
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    problem_map = _build_problem_map(args.data_jsonl, task_id_key=args.task_id_key, problem_key=args.problem_key)
    val_rows = _load_jsonl(args.validation_jsonl)

    out_rows: List[Dict] = []
    for item in val_rows:
        task_id = str(item["id"])
        if task_id not in problem_map:
            continue

        problem = problem_map[task_id]
            
            # 这里的 extracted_reasoning 已经包含了 " xxx"
        reasoning = str(item.get("extracted_reasoning", "")).strip()
            
            # 根据 Step 2 验证结果，生成对应的高质量训练标签
        decision = _decision_from_self_passed(bool(item.get("self_passed", False)))

        user_content = MATH_PROMPT_TEMPLATE.format(problem=problem)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )

        # 将推理过程和 decision token 拼接在一起作为 assistant 的标准输出
        assistant_response = f"{reasoning}\n{decision}"
        full_text = prompt + assistant_response + (tokenizer.eos_token or "")

        out_rows.append({
            "text": full_text,
            "_task_id": task_id,
            "_sample_index": int(item.get("sample_index", 0)),
            "_decision": decision,
            "_self_passed": int(bool(item.get("self_passed", False))),
        })

    print(f"[step3] before sampling: {describe_decision_records(out_rows)}")
    out_rows = balance_decision_records(
        out_rows,
        max_total_samples=args.max_total_samples,
        self_ratio=args.self_ratio,
        seed=args.seed,
    )
    print(f"[step3] after sampling: {describe_decision_records(out_rows)}")

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for out in out_rows:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[step3] wrote sft data: {args.output_jsonl}, rows={len(out_rows)}")


if __name__ == "__main__":
    main()
