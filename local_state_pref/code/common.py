from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

import torch


csv.field_size_limit(sys.maxsize)

ACTIONS = {
    "attempt": "<Attempt>",
    "defer": "<defer>",
    "self": "<self>",
}

ATTEMPT_OPEN = ACTIONS["attempt"] + "\n```python\n"
ATTEMPT_CLOSE = "\n```\n"

CODE_PROMPT = """You are a code agent with a clear understanding of your own capabilities, and you can actively seek help from an expert.
### Question:
{problem}
### Starter Code:
```python
{starter_code}
```
### INSTRUCTION:
1. At the beginning, choose either <Attempt> or <defer>.
2. If you choose <Attempt>, write a bounded local draft inside a ```python ... ``` block.
3. After the draft, choose either <self> or <defer>.
4. Do NOT include extra explanations outside the required format.
### FORMAT:
<defer>
or
<Attempt>
```python
Your code
```
<self> or <defer>
"""

PYTHON_BLOCK_PATTERN = re.compile(r"```python\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)

GLOBAL_HEADER = """
import sys
import math
import datetime
import collections
import functools
import itertools
import random
import heapq
import bisect
import string
import operator
from typing import *
from functools import *
from collections import *
from itertools import *
from heapq import *
from bisect import *
from string import *
from operator import *
from math import *

from sortedcontainers import SortedList


inf = float('inf')

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def list_node(values: list):
    if not values: return None
    head = ListNode(values[0])
    p = head
    for val in values[1:]:
        node = ListNode(val)
        p.next = node
        p = node
    return head

def is_same_list(p1, p2):
    if p1 is None and p2 is None: return True
    if not p1 or not p2: return False
    return p1.val == p2.val and is_same_list(p1.next, p2.next)

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def tree_node(values: list):
    if not values: return None
    root = TreeNode(values[0])
    i = 1
    queue = collections.deque([root])
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
    if not p and not q: return True
    if not p or not q: return False
    if p.val != q.val: return False
    return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)
"""


@dataclass(frozen=True)
class CodeRow:
    task_id: str
    problem: str
    starter_code: str
    entry_point: str
    tests: str


def read_code_rows(csv_path: str) -> list[CodeRow]:
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                CodeRow(
                    task_id=str(row["id"]).strip(),
                    problem=str(row["problem"]),
                    starter_code=str(row["starter_code"]),
                    entry_point=str(row["entry_point"]),
                    tests=str(row["test"]),
                )
            )
    return rows


def load_jsonl(path: str) -> list[dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def dump_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_s0_context(problem: str, starter_code: str) -> str:
    return CODE_PROMPT.format(problem=problem, starter_code=starter_code)


def build_attempt_generation_context(problem: str, starter_code: str) -> str:
    return build_s0_context(problem, starter_code) + ATTEMPT_OPEN


def build_s1_context(problem: str, starter_code: str, code: str) -> str:
    return build_s0_context(problem, starter_code) + ATTEMPT_OPEN + code.strip() + ATTEMPT_CLOSE


def build_chat_s0_context(tokenizer, problem: str, starter_code: str, add_generation_prompt: bool = True) -> str:
    return apply_chat_template(
        tokenizer,
        build_s0_context(problem, starter_code),
        add_generation_prompt=add_generation_prompt,
    )


def build_chat_attempt_generation_context(tokenizer, problem: str, starter_code: str) -> str:
    return build_chat_s0_context(tokenizer, problem, starter_code, add_generation_prompt=True) + ATTEMPT_OPEN


def build_chat_s1_context(tokenizer, problem: str, starter_code: str, code: str) -> str:
    return (
        build_chat_s0_context(tokenizer, problem, starter_code, add_generation_prompt=True)
        + ATTEMPT_OPEN
        + code.strip()
        + ATTEMPT_CLOSE
    )


def apply_chat_template(tokenizer, user_text: str, add_generation_prompt: bool = True) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    return user_text


def count_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def extract_attempt_code(continuation: str) -> str:
    if not continuation:
        return ""
    text = continuation.strip()
    match = re.search(r"\n```", text)
    if match:
        return text[: match.start()].strip()
    return text.strip()


def extract_last_python_block(text: str) -> str:
    matches = list(PYTHON_BLOCK_PATTERN.finditer(text or ""))
    if matches:
        return (matches[-1].group(1) or "").strip()
    return (text or "").strip()


def action_probs_from_logps(logp_defer: float, logp_other: float) -> tuple[float, float, float]:
    max_logp = max(logp_defer, logp_other)
    denom = math.exp(logp_defer - max_logp) + math.exp(logp_other - max_logp)
    p_defer = math.exp(logp_defer - max_logp) / denom
    return p_defer, 1.0 - p_defer, logp_defer - logp_other


def parse_dtype(dtype: str):
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return dtype_map[dtype]


def ensure_local_model_path_exists(model_path: str) -> str:
    expanded = os.path.expanduser(model_path)
    is_local_path = os.path.isabs(expanded) or expanded.startswith("./") or expanded.startswith("../")
    if is_local_path and not os.path.isdir(expanded):
        raise FileNotFoundError(f"Local model path does not exist: {expanded}")
    return expanded


def _latest_checkpoint_in_output_dir(output_dir: str) -> str | None:
    if not os.path.isdir(output_dir):
        return None

    latest_step = -1
    latest_path = None
    for name in os.listdir(output_dir):
        if not name.startswith("checkpoint-"):
            continue
        step_text = name[len("checkpoint-") :]
        if not step_text.isdigit():
            continue
        path = os.path.join(output_dir, name)
        if not os.path.isdir(path):
            continue
        step = int(step_text)
        if step > latest_step:
            latest_step = step
            latest_path = path
    return latest_path


def _has_tokenizer_files(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "tokenizer.json")) or os.path.isfile(os.path.join(path, "vocab.json"))


def _read_adapter_base_model_path(model_path: str) -> str | None:
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if not os.path.isfile(adapter_config_path):
        return None
    try:
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    base_path = str(config.get("base_model_name_or_path") or "").strip()
    return base_path or None


def _resolve_model_and_tokenizer_paths(model_path: str, tokenizer_path: str | None = None) -> tuple[str, str]:
    original_model_path = ensure_local_model_path_exists(model_path)
    resolved_model_path = original_model_path

    has_loadable_model_files = any(
        os.path.isfile(os.path.join(resolved_model_path, filename))
        for filename in ("adapter_config.json", "config.json")
    )
    if not has_loadable_model_files:
        latest_checkpoint = _latest_checkpoint_in_output_dir(resolved_model_path)
        if latest_checkpoint is not None:
            resolved_model_path = latest_checkpoint

    if tokenizer_path:
        return resolved_model_path, ensure_local_model_path_exists(tokenizer_path)

    if _has_tokenizer_files(resolved_model_path):
        return resolved_model_path, resolved_model_path

    if _has_tokenizer_files(original_model_path):
        return resolved_model_path, original_model_path

    base_model_path = _read_adapter_base_model_path(resolved_model_path)
    if base_model_path:
        return resolved_model_path, ensure_local_model_path_exists(base_model_path)

    return resolved_model_path, resolved_model_path


def load_model_and_tokenizer(
    model_path: str,
    dtype: str = "bf16",
    device_map: str | None = "auto",
    trust_remote_code: bool = True,
    tokenizer_path: str | None = None,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path, tokenizer_path = _resolve_model_and_tokenizer_paths(model_path, tokenizer_path=tokenizer_path)
    print(f"Using model path: {model_path}")
    print(f"Using tokenizer path: {tokenizer_path}")
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    kwargs = {
        "torch_dtype": parse_dtype(dtype),
        "trust_remote_code": trust_remote_code,
    }
    if device_map is not None and device_map != "none":
        kwargs["device_map"] = device_map

    model = None
    try:
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(model_path, **kwargs)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    model.eval()
    return model, tok


def model_input_device(model, fallback: torch.device | None = None) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        if fallback is not None:
            return fallback
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_humaneval_tests(code: str, tests: str, entry_point: str, task_id: str, timeout: float = 5.0) -> bool:
    from human_eval.execution import check_correctness

    if not entry_point or not code.strip():
        return False

    problem = {
        "task_id": str(task_id),
        "prompt": "",
        "test": tests,
        "entry_point": entry_point,
    }
    full_code = GLOBAL_HEADER + "\n" + code
    try:
        result = check_correctness(problem, full_code, timeout=timeout)
        return bool(result.get("passed", False))
    except Exception:
        return False
