from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import REPO_ROOT


_MODULE_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class DomainExample:
    task_id: str
    problem: str
    gold_answer: str = ""
    starter_code: str = ""
    entry_point: str = ""
    tests: str = ""
    subject: str = ""


def _load_module(path: Path, module_name: str):
    key = str(path.resolve())
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]

    old_common = sys.modules.pop("common", None)
    added_repo_root = False
    added_module_dir = False
    module_dir = path.parent
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
        added_repo_root = True
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
        added_module_dir = True
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop("common", None)
        if old_common is not None:
            sys.modules["common"] = old_common
        if added_module_dir:
            try:
                sys.path.remove(str(module_dir))
            except ValueError:
                pass
        if added_repo_root:
            try:
                sys.path.remove(str(REPO_ROOT))
            except ValueError:
                pass

    _MODULE_CACHE[key] = module
    return module


def read_domain_examples(domain: str, data_path: str, *, limit: int | None = None) -> list[DomainExample]:
    domain = domain.lower()
    if domain == "code":
        common = _load_module(REPO_ROOT / "coder" / "eval" / "code" / "common.py", "baseline_code_common")
        rows = common.read_csv_rows(data_path, limit=limit)
        return [
            DomainExample(
                task_id=row.task_id,
                problem=row.problem,
                starter_code=row.starter_code,
                entry_point=row.entry_point,
                tests=row.tests,
            )
            for row in rows
        ]
    if domain == "math":
        common = _load_module(REPO_ROOT / "coder" / "eval" / "math" / "common.py", "baseline_math_common")
        rows = common.read_jsonl_rows(data_path, limit=limit)
        return [DomainExample(task_id=row.task_id, problem=row.problem, gold_answer=row.answer) for row in rows]
    if domain == "mmlu":
        common = _load_module(REPO_ROOT / "coder" / "eval" / "MMLU" / "common.py", "baseline_mmlu_common")
        rows = common.read_jsonl_rows(data_path, limit=limit)
        return [
            DomainExample(
                task_id=row.task_id,
                problem=row.problem,
                gold_answer=row.answer,
                subject=row.subject,
            )
            for row in rows
        ]
    raise ValueError(f"Unknown domain: {domain}")


def evaluate_completion(domain: str, example: DomainExample, raw_response: str, *, timeout: int = 10) -> tuple[int, str]:
    domain = domain.lower()
    if domain == "code":
        common = _load_module(REPO_ROOT / "coder" / "eval" / "code" / "common.py", "baseline_code_common")
        code = common.extract_code_block(raw_response)
        passed = int(bool(code) and common.run_humaneval_tests(code, example.tests, example.entry_point, example.task_id, timeout=timeout))
        return passed, code
    if domain == "math":
        common = _load_module(REPO_ROOT / "coder" / "eval" / "math" / "common.py", "baseline_math_common")
        extractor = common.build_answer_extractor()
        checker = common.answers_match
        answer = extractor(raw_response)
        passed = int(bool(answer) and checker(answer, example.gold_answer))
        return passed, answer
    if domain == "mmlu":
        common = _load_module(REPO_ROOT / "coder" / "eval" / "MMLU" / "common.py", "baseline_mmlu_common")
        answer = common.extract_final_choice(raw_response)
        passed = int(answer == common.normalize_choice(example.gold_answer))
        return passed, answer
    raise ValueError(f"Unknown domain: {domain}")


def build_plain_prompt(domain: str, example: DomainExample) -> str:
    domain = domain.lower()
    if domain == "code":
        common = _load_module(REPO_ROOT / "coder" / "eval" / "code" / "common.py", "baseline_code_common")
        return common.EXPERT_PROMPT.format(problem=example.problem, current_code=example.starter_code)
    if domain == "math":
        generate_expert = _load_module(
            REPO_ROOT / "coder" / "eval" / "math" / "generate_expert.py",
            "baseline_math_generate_expert",
        )
        return generate_expert.EXPERT_PROMPT.format(problem=example.problem)
    if domain == "mmlu":
        generate_expert = _load_module(
            REPO_ROOT / "coder" / "eval" / "MMLU" / "generate_expert.py",
            "baseline_mmlu_generate_expert",
        )
        return generate_expert.EXPERT_PROMPT.format(problem=example.problem)
    raise ValueError(f"Unknown domain: {domain}")


def build_external_router_prompt(domain: str, example: DomainExample) -> str:
    domain = domain.lower()
    if domain == "code":
        return (
            "You are a code agent with a clear understanding of your own capabilities, you can actively seek help from expert.\n"
            "### Question:\n"
            f"{example.problem}\n"
            "### Starter Code:\n"
            "```python\n"
            f"{example.starter_code}\n"
            "```\n"
            "### INSTRUCTION:\n"
            "1. First, output the solution code enclosed in ```python ... ``` block.\n"
            "2. Then, depending on your solution, output one final decision token: yes or no.\n"
            "   - yes means you are confident your solution is correct.\n"
            "   - no means you are not confident and would defer.\n"
            "3. Do NOT include any explanations, comments, or extra text.\n"
            "4. Do NOT generate any test cases, assertions, usage examples, or `if __name__ == \"__main__\":` blocks.\n"
            "5. DO NOT put decision token inside code block.\n"
            "### FORMAT SKELETON:\n"
            "```python\n"
            "Your code\n"
            "```\n"
            "yes or no\n"
        )
    if domain == "math":
        return (
            "You are a math reasoning agent with a clear understanding of your own capabilities, you can actively seek help from expert.\n"
            "### Question:\n"
            f"{example.problem}\n"
            "### INSTRUCTION:\n"
            "1. First, solve the problem step by step.\n"
            "2. Then, conclude your response with exactly \"Final answer: \" followed IMMEDIATELY by the bare mathematical expression or value.\n"
            "3. Finally, assess your confidence. If you are confident in your answer, output yes. If you are unsure, output no.\n"
            "4. Do NOT include any extra text after the decision token.\n"
            "### FORMAT SKELETON:\n"
            "reasoning\n"
            "Final answer: ans\n"
            "yes or no\n"
        )
    if domain == "mmlu":
        return (
            "You are a multiple-choice reasoning agent with a clear understanding of your own capabilities, you can actively seek help from expert.\n"
            "### Question:\n"
            f"{example.problem}\n"
            "### INSTRUCTION:\n"
            "1. First solve the problem step by step.\n"
            "2. Then conclude with exactly \"Final answer: \" followed by exactly one option letter among A, B, C, D.\n"
            "3. Finally output one final decision token: yes or no.\n"
            "   - yes means you are confident in your final answer.\n"
            "   - no means you are not confident and would defer.\n"
            "4. Do NOT include any extra text after the decision token.\n"
            "### FORMAT SKELETON:\n"
            "reasoning\n"
            "Final answer: B\n"
            "yes or no\n"
        )
    raise ValueError(f"Unknown domain: {domain}")


def external_router_attempt_open(domain: str) -> str:
    domain = domain.lower()
    if domain == "code":
        return "```python\n"
    if domain in {"math", "mmlu"}:
        return ""
    raise ValueError(f"Unknown domain: {domain}")


def external_router_attempt_close(domain: str) -> str:
    domain = domain.lower()
    if domain == "code":
        return "\n```\n"
    if domain in {"math", "mmlu"}:
        # Keep a hard separator before scoring yes/no to avoid boundary retokenization.
        return "\n"
    raise ValueError(f"Unknown domain: {domain}")


def apply_chat_template(tokenizer: Any, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": str(prompt)}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return str(prompt)
