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


MMLU_PROMPT_TEMPLATE = """You are a general-domain multiple-choice reasoning agent.
### Question:
{problem}
### INSTRUCTION:
1. Do NOT output reasoning steps.
2. Output final answer in exact format: #### [OPTION]. [ANSWER_CONTENT]
   Example: #### B. photosynthesis
3. Output only one final answer line.
### FORMAT:
#### [OPTION]. [ANSWER_CONTENT]
"""


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


def _extract_reasoning_only(text: str) -> str:
    cleaned, _ = strip_trailing_decision_token(text)
    return cleaned


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
        prompt = MMLU_PROMPT_TEMPLATE.format(problem=problem)
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)
        metas.append({"id": task_id})

    outputs = llm.generate(prompts, sampling_params)

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for meta, out in zip(metas, outputs):
            for sample_idx, resp in enumerate(out.outputs):
                raw = str(resp.text)
                reasoning = _extract_reasoning_only(raw)
                record = {
                    "id": meta["id"],
                    "sample_index": sample_idx,
                    "raw_output": raw,
                    "extracted_reasoning": reasoning,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[step1] wrote samples to {args.output_jsonl}")


if __name__ == "__main__":
    main()
