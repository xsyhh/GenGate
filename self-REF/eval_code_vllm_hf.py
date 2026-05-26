from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from tokens import (
    TOKEN_DEFER,
    TOKEN_SELF,
    label_from_decision_token,
)


PROMPT_TEMPLATE = """You are a code agent with a clear understanding of your own capabilities, you can actively seek help from expert.
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


STRICT_PATTERN = re.compile(
    rf"^\s*```python\s*(.*?)\s*```\s*({re.escape(TOKEN_SELF)}|{re.escape(TOKEN_DEFER)})?\s*$",
    flags=re.DOTALL | re.IGNORECASE,
)


CODE_BLOCK_PATTERN = re.compile(r"```(?:python)?\s*(.*?)\s*```", flags=re.DOTALL | re.IGNORECASE)
DECISION_PATTERN = re.compile(rf"{re.escape(TOKEN_SELF)}|{re.escape(TOKEN_DEFER)}", flags=re.IGNORECASE)


GLOBAL_HEADER = """
import sys
import math
import datetime
import random
import string
import operator
import collections
import itertools
import heapq
import bisect
from typing import List, Optional, Dict, Tuple, Set, Any, Union
from collections import deque, defaultdict, Counter
from heapq import heapify, heappush, heappop, heappushpop, heapreplace
from bisect import bisect_left, bisect_right, insort_left, insort_right
from functools import lru_cache, cache, reduce, cmp_to_key
from itertools import combinations, permutations, accumulate, product
try:
    from sortedcontainers import SortedList, SortedDict, SortedSet
except Exception:
    pass
inf = float('inf')

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def list_node(values: list):
    if not values:
        return None
    head = ListNode(values[0])
    p = head
    for val in values[1:]:
        p.next = ListNode(val)
        p = p.next
    return head

def is_same_list(p1, p2):
    if p1 is None and p2 is None:
        return True
    if not p1 or not p2:
        return False
    return p1.val == p2.val and is_same_list(p1.next, p2.next)

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def tree_node(values: list):
    if not values:
        return None
    root = TreeNode(values[0])
    i = 1
    queue = deque([root])
    while queue:
        node = queue.popleft()
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    return root

def is_same_tree(p, q):
    if not p and not q:
        return True
    if not p or not q:
        return False
    if p.val != q.val:
        return False
    return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)
"""


@dataclass(frozen=True)
class CodeTask:
    task_id: str
    problem: str
    starter_code: str
    tests: str
    entry_point: str


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def read_code_csv(path: str, limit: int | None = None) -> list[CodeTask]:
    csv.field_size_limit(sys.maxsize)
    rows: list[CodeTask] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            task_id = str(row.get("id") or row.get("task_id") or i).strip()
            rows.append(
                CodeTask(
                    task_id=task_id,
                    problem=str(row.get("problem", "")),
                    starter_code=str(row.get("starter_code", "")),
                    tests=str(row.get("test", "")),
                    entry_point=str(row.get("entry_point", "")).strip(),
                )
            )
    return rows


def load_expert_map(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    csv.field_size_limit(sys.maxsize)
    out: dict[str, float] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = str(row.get("task_id") or row.get("id") or "").strip()
            if not task_id:
                continue
            raw = row.get("expert_passed", row.get("self_passed", "0"))
            try:
                out[task_id] = float(raw)
            except (TypeError, ValueError):
                out[task_id] = 0.0
    return out


def make_prompt(tokenizer: Any, task: CodeTask) -> tuple[str, str]:
    raw = PROMPT_TEMPLATE.format(problem=task.problem, current_code=task.starter_code)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": raw}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return raw, prompt


def clean_generated_text(tokenizer: Any, text: str) -> str:
    out = str(text or "")
    for tok in [
        getattr(tokenizer, "eos_token", None),
        getattr(tokenizer, "pad_token", None),
        getattr(tokenizer, "bos_token", None),
    ]:
        if tok:
            out = out.replace(tok, "")
    for tok in ["<|assistant|>", "<|user|>", "<|system|>", "<|im_start|>", "<|im_end|>"]:
        out = out.replace(tok, "")
    return out.strip()


def parse_code_and_decision(text: str) -> tuple[int, str, str]:
    value = str(text or "").strip()
    strict = STRICT_PATTERN.match(value)
    if strict:
        code = (strict.group(1) or "").strip()
        decision = label_from_decision_token((strict.group(2) or "").strip())
        return int(bool(code and decision)), code, decision

    code = ""
    block_matches = list(CODE_BLOCK_PATTERN.finditer(value))
    if block_matches:
        code = (block_matches[-1].group(1) or "").strip()
    else:
        decision_first = DECISION_PATTERN.search(value)
        code = value[: decision_first.start()].strip() if decision_first else value.strip()

    decisions = list(DECISION_PATTERN.finditer(value))
    decision = label_from_decision_token(decisions[-1].group(0).strip()) if decisions else ""
    return 0, code, decision


def code_prefix(prompt: str, code: str) -> str:
    return f"{prompt}```python\n{str(code or '').strip()}\n```\n"


def run_tests(code: str, task: CodeTask, timeout: int) -> bool:
    if not code.strip() or not task.entry_point:
        return False
    from human_eval.execution import check_correctness  # type: ignore

    problem = {
        "task_id": task.task_id,
        "prompt": "",
        "test": task.tests,
        "entry_point": task.entry_point,
    }
    try:
        result = check_correctness(problem, GLOBAL_HEADER + "\n" + code, timeout=timeout)
    except Exception:
        return False
    return bool(result.get("passed", False)) if isinstance(result, dict) else False


def generate_stage(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    tasks = read_code_csv(args.data_csv, limit=args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    prompts_raw: list[str] = []
    prompts: list[str] = []
    for task in tasks:
        raw, prompt = make_prompt(tokenizer, task)
        prompts_raw.append(raw)
        prompts.append(prompt)

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "trust_remote_code": bool(args.trust_remote_code),
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
    }
    if args.max_model_len:
        llm_kwargs["max_model_len"] = int(args.max_model_len)
    if args.max_num_seqs:
        llm_kwargs["max_num_seqs"] = int(args.max_num_seqs)
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True

    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        n=1,
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
        skip_special_tokens=False,
    )
    outputs = llm.generate(prompts, sampling)

    completions_path = out_dir / "completions.jsonl"
    with completions_path.open("w", encoding="utf-8") as f:
        for task, prompt_raw, prompt, output in zip(tasks, prompts_raw, prompts, outputs):
            response = str(output.outputs[0].text or "")
            response_clean = clean_generated_text(tokenizer, response)
            valid_format, code, decision = parse_code_and_decision(response_clean)
            f.write(
                json.dumps(
                    {
                        "task_id": task.task_id,
                        "prompt_raw": prompt_raw,
                        "prompt": prompt,
                        "response_raw": response,
                        "response_clean": response_clean,
                        "valid_format": int(valid_format),
                        "model_decision_raw": decision,
                        "code": code,
                        "code_len": len(code),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"[generate] wrote {completions_path}")

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def read_completions(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def continuation_logprobs_batch(
    model: Any,
    tokenizer: Any,
    prefixes: list[str],
    continuations: list[str],
    max_length: int,
) -> list[float]:
    device = next(model.parameters()).device
    full_texts = [p + c for p, c in zip(prefixes, continuations)]
    prefix_lens = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prefixes]
    encoded = tokenizer(
        full_texts,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    ).to(device)
    logits = model(input_ids=encoded["input_ids"], attention_mask=encoded.get("attention_mask")).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    labels = encoded["input_ids"][:, 1:]
    gathered = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
    attention = encoded["attention_mask"][:, 1:].bool()

    out: list[float] = []
    for row_idx, prefix_len in enumerate(prefix_lens):
        real_len = int(encoded["attention_mask"][row_idx].sum().item())
        pad_len = int(encoded["attention_mask"].shape[1] - real_len)
        seq_end = pad_len + real_len - 1
        start = min(max(pad_len + prefix_len - 1, 0), max(seq_end, 0))
        mask = attention[row_idx, start:seq_end]
        vals = gathered[row_idx, start:seq_end][mask]
        out.append(float(vals.sum().item()) if vals.numel() else float("-inf"))
    return out


@torch.inference_mode()
def score_stage(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    completions_path = Path(args.completions_jsonl or out_dir / "completions.jsonl")
    completions = read_completions(completions_path)
    task_map = {task.task_id: task for task in read_code_csv(args.data_csv, limit=args.limit)}
    expert_map = load_expert_map(args.expert_csv)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype_map[str(args.dtype)],
        device_map=args.hf_device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()

    prefixes = [code_prefix(str(row["prompt"]), str(row.get("code", ""))) for row in completions]
    batch_size = max(1, int(args.score_batch_size))
    logp_self: list[float] = []
    logp_defer: list[float] = []
    for start in tqdm(range(0, len(prefixes), batch_size), desc="HF score decision tokens"):
        batch_prefixes = prefixes[start : start + batch_size]
        logp_self.extend(
            continuation_logprobs_batch(
                model,
                tokenizer,
                batch_prefixes,
                [TOKEN_SELF] * len(batch_prefixes),
                max_length=int(args.max_score_length),
            )
        )
        logp_defer.extend(
            continuation_logprobs_batch(
                model,
                tokenizer,
                batch_prefixes,
                [TOKEN_DEFER] * len(batch_prefixes),
                max_length=int(args.max_score_length),
            )
        )

    metadata_path = out_dir / "metadata.csv"
    summary = {
        "n": 0,
        "valid_format": 0,
        "natural_defer": 0,
        "self_passed": 0,
    }

    with metadata_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "task_id",
            "route",
            "model_decision",
            "model_decision_raw",
            "valid_format",
            "p_defer",
            "p_self",
            "score",
            "margin",
            "logp_self",
            "logp_defer",
            "self_passed",
            "expert_passed",
            "expert_available",
            "code_len",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(tqdm(completions, desc="Run tests + write metadata")):
            task_id = str(row["task_id"])
            task = task_map[task_id]
            lp_s = float(logp_self[idx])
            lp_d = float(logp_defer[idx])
            mx = max(lp_s, lp_d)
            denom = math.exp(lp_s - mx) + math.exp(lp_d - mx)
            p_defer = math.exp(lp_d - mx) / denom if denom else 0.5
            p_self = 1.0 - p_defer
            model_decision = "defer" if p_defer >= p_self else "self"
            raw_decision = label_from_decision_token(str(row.get("model_decision_raw", "")).strip())
            route = "post_defer" if model_decision == "defer" else "post_self"
            self_passed = run_tests(str(row.get("code", "")), task, timeout=int(args.test_timeout))
            expert_available = task_id in expert_map
            expert_passed = expert_map.get(task_id, "")

            summary["n"] += 1
            summary["valid_format"] += int(row.get("valid_format", 0))
            summary["natural_defer"] += int(model_decision == "defer")
            summary["self_passed"] += int(self_passed)

            writer.writerow(
                {
                    "task_id": task_id,
                    "route": route,
                    "model_decision": model_decision,
                    "model_decision_raw": raw_decision,
                    "valid_format": int(row.get("valid_format", 0)),
                    "p_defer": float(p_defer),
                    "p_self": float(p_self),
                    "score": float(p_self),
                    "margin": float(lp_d - lp_s),
                    "logp_self": lp_s,
                    "logp_defer": lp_d,
                    "self_passed": int(bool(self_passed)),
                    "expert_passed": expert_passed,
                    "expert_available": int(bool(expert_available)),
                    "code_len": int(row.get("code_len", len(str(row.get("code", ""))))),
                }
            )

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[score] wrote {metadata_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["generate", "score", "all"], default="all")
    parser.add_argument("--model", required=True, help="Full merged model path used by both vLLM and HF scoring.")
    parser.add_argument("--data_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--expert_csv", default=None)
    parser.add_argument("--completions_jsonl", default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--max_num_seqs", type=int, default=None)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--hf_device_map", default="auto")
    parser.add_argument("--score_batch_size", type=int, default=4)
    parser.add_argument("--max_score_length", type=int, default=4096)
    parser.add_argument("--test_timeout", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in {"generate", "all"}:
        generate_stage(args)
    if args.stage in {"score", "all"}:
        score_stage(args)


if __name__ == "__main__":
    main()
