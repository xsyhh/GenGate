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

from math_eval_setup import ensure_math_eval_on_path
from tokens import (
    TOKEN_DEFER,
    TOKEN_SELF,
    DECISION_TOKEN_AT_END_PATTERN,
    DECISION_TOKEN_PATTERN,
    label_from_decision_token,
    strip_all_decision_tokens,
)

ensure_math_eval_on_path()

from grader import math_equal  # type: ignore
from parser import extract_answer, strip_string  # type: ignore


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

MMLU_PROMPT_TEMPLATE = """You are a general-domain multiple-choice reasoning agent.
### Question:
{problem}
### INSTRUCTION:
1. Do NOT output reasoning steps.
2. Output final answer in exact format: #### [OPTION]. [ANSWER_CONTENT]
   Example: #### B. photosynthesis
3. Then, output one decision token: <CN> or <UN>.
   - <CN>: You believe your solution and final answer are correct.
   - <UN>: You believe your solution is uncertain or likely wrong.
4. Do NOT include any extra text after the decision token.
### FORMAT:
#### [OPTION]. [ANSWER_CONTENT] <CN> or <UN>
"""

OPTION_LINE_RE = re.compile(r"^\s*([A-D])\s*[\).:-]\s*(.+?)\s*$", flags=re.IGNORECASE)
HASH_ANSWER_RE = re.compile(r"####\s*([A-D])\s*[\).:-]\s*([^\n\r]+)", flags=re.IGNORECASE)
FINAL_ANSWER_RE = re.compile(r"Final answer:\s*(.+?)(?:\n|$)", flags=re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class QATask:
    task_id: str
    problem: str
    answer: str


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def read_jsonl(path: str, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_tasks(args: argparse.Namespace) -> list[QATask]:
    out: list[QATask] = []
    for i, row in enumerate(read_jsonl(args.data_jsonl, limit=args.limit)):
        out.append(
            QATask(
                task_id=str(row.get(args.task_id_key, row.get("id", i))).strip(),
                problem=str(row.get(args.problem_key, "")),
                answer=str(row.get(args.answer_key, "")).strip(),
            )
        )
    return out


def clean_generated_text(tokenizer: Any, text: str) -> str:
    out = str(text or "")
    for tok in [
        getattr(tokenizer, "eos_token", None),
        getattr(tokenizer, "pad_token", None),
        getattr(tokenizer, "bos_token", None),
    ]:
        if tok:
            out = out.replace(tok, "")
    for tok in ["<|assistant|>", "<|user|>", "<|system|>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
        out = out.replace(tok, "")
    return out.strip()


def make_prompt(tokenizer: Any, task: QATask, domain: str) -> tuple[str, str]:
    template = MATH_PROMPT_TEMPLATE if domain == "math" else MMLU_PROMPT_TEMPLATE
    raw = template.format(problem=task.problem)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": raw}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return raw, prompt


def split_answer_and_decision(text: str) -> tuple[int, str, str]:
    value = str(text or "").strip()
    strict = DECISION_TOKEN_AT_END_PATTERN.match(value)
    decisions = list(DECISION_TOKEN_PATTERN.finditer(value))
    if decisions:
        last = decisions[-1]
        decision = label_from_decision_token(str(last.group(0) or "").strip())
        answer_text = value[: last.start()].strip()
        return int(bool(strict)), answer_text, decision
    return 0, value, ""


def normalize_mmlu_text(raw: str) -> str:
    text = str(raw or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .。;；,，:：!?！？\"'`[](){}")


def parse_mmlu_options(problem: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for line in str(problem or "").splitlines():
        match = OPTION_LINE_RE.match(line)
        if match:
            options[str(match.group(1)).upper()] = str(match.group(2)).strip()
    return options


def reference_mmlu_option(answer: str, options: dict[str, str]) -> str:
    raw = str(answer or "").strip().upper()
    if raw in {"A", "B", "C", "D"}:
        return raw
    norm = normalize_mmlu_text(answer)
    for letter, text in options.items():
        if normalize_mmlu_text(text) == norm:
            return str(letter).upper()
    return ""


def extract_mmlu_prediction(text: str) -> tuple[str, str]:
    value = str(text or "").strip()
    matches = HASH_ANSWER_RE.findall(value)
    if matches:
        letter, content = matches[-1]
        return str(letter).upper(), str(content).strip()
    for line in reversed(value.splitlines()):
        match = OPTION_LINE_RE.match(line)
        if match:
            return str(match.group(1)).upper(), str(match.group(2)).strip()
    return "", ""


def extract_math_final_answer(text: str) -> str:
    value = str(text or "")
    matches = list(FINAL_ANSWER_RE.finditer(value))
    if not matches:
        return ""
    answer = str(matches[-1].group(1) or "").strip()
    answer = strip_all_decision_tokens(answer)
    answer = answer.strip("$").strip()
    return str(strip_string(answer)) if answer else ""


def math_passed(answer_text: str, answer: str) -> tuple[bool, str, str]:
    reference = str(answer or "").strip()
    if not reference:
        return False, "", ""
    reference_norm = str(strip_string(reference))
    prediction = extract_math_final_answer(str(answer_text))
    if not prediction:
        prediction = str(extract_answer(str(answer_text), "math"))
    return bool(math_equal(prediction, reference_norm, timeout=True)), prediction, reference_norm


def mmlu_passed(answer_text: str, answer: str, problem: str) -> tuple[bool, str, str]:
    options = parse_mmlu_options(problem)
    pred_letter, pred_content = extract_mmlu_prediction(answer_text)
    ref_letter = reference_mmlu_option(answer, options)
    return bool(pred_letter and ref_letter and pred_letter == ref_letter), pred_letter, ref_letter


def evaluate_answer(domain: str, answer_text: str, task: QATask) -> tuple[bool, str, str]:
    if domain == "math":
        return math_passed(answer_text, task.answer)
    return mmlu_passed(answer_text, task.answer, task.problem)


def expert_passed_from_raw(domain: str, expert_raw: str, task: QATask) -> bool:
    if not str(expert_raw or "").strip():
        return False
    if domain == "mmlu":
        direct = str(expert_raw).strip().upper()
        if direct in {"A", "B", "C", "D"}:
            ref = reference_mmlu_option(task.answer, parse_mmlu_options(task.problem))
            return direct == ref
    passed, _, _ = evaluate_answer(domain, str(expert_raw), task)
    return bool(passed)


def read_expert_map(path: str | None, task_id_key: str, answer_key: str) -> dict[str, str]:
    if not path:
        return {}
    out: dict[str, str] = {}
    for row in read_jsonl(path):
        task_id = str(row.get(task_id_key, row.get("task_id", row.get("id", "")))).strip()
        if task_id:
            out[task_id] = str(row.get(answer_key, row.get("raw_output", row.get("prediction", ""))))
    return out


def generate_stage(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    tasks = read_tasks(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    prompts_raw: list[str] = []
    prompts: list[str] = []
    for task in tasks:
        raw, prompt = make_prompt(tokenizer, task, args.domain)
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
        seed=int(args.seed),
    )
    outputs = llm.generate(prompts, sampling)

    completions_path = out_dir / "completions.jsonl"
    with completions_path.open("w", encoding="utf-8") as f:
        for task, prompt_raw, prompt, output in zip(tasks, prompts_raw, prompts, outputs):
            response = str(output.outputs[0].text or "")
            response_clean = clean_generated_text(tokenizer, response)
            valid_format, answer_text, decision = split_answer_and_decision(response_clean)
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
                        "answer_text": answer_text,
                        "answer_len": len(answer_text),
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


def truncate_text_by_tokens(tokenizer: Any, text: str, max_tokens: int) -> str:
    value = str(text or "")
    if max_tokens <= 0:
        return value
    ids = tokenizer(value, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return value
    return tokenizer.decode(ids[-max_tokens:], skip_special_tokens=False)


def decision_prefix(prompt: str, answer_text: str) -> str:
    value = str(answer_text or "").strip()
    if value and not value.endswith("\n"):
        value += "\n"
    return prompt + value


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
    task_map = {task.task_id: task for task in read_tasks(args)}
    expert_map = read_expert_map(args.expert_jsonl, args.expert_task_id_key, args.expert_answer_key)

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

    prefixes = [
        decision_prefix(
            str(row["prompt"]),
            truncate_text_by_tokens(tokenizer, str(row.get("answer_text", "")), int(args.max_score_prefix_tokens)),
        )
        for row in completions
    ]

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
    summary = {"n": 0, "valid_format": 0, "natural_defer": 0, "self_passed": 0}
    debug_left = int(args.debug_samples)

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
            "prediction",
            "reference",
            "expert_passed",
            "expert_available",
            "expert_has_answer",
            "answer_len",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(tqdm(completions, desc="Evaluate + write metadata")):
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

            answer_text = str(row.get("answer_text", ""))
            self_passed, prediction, reference = evaluate_answer(args.domain, answer_text, task)

            expert_available = task_id in expert_map
            expert_has_answer = bool(str(expert_map.get(task_id, "")).strip())
            expert_passed = expert_passed_from_raw(args.domain, expert_map.get(task_id, ""), task) if expert_available else ""

            summary["n"] += 1
            summary["valid_format"] += int(row.get("valid_format", 0))
            summary["natural_defer"] += int(model_decision == "defer")
            summary["self_passed"] += int(bool(self_passed))

            if debug_left > 0:
                debug_left -= 1
                preview = str(row.get("response_clean", ""))[:500]
                print(
                    f"[debug] task_id={task_id} valid={int(row.get('valid_format', 0))} "
                    f"decision={model_decision} p_defer={p_defer:.4f} self_passed={int(bool(self_passed))}"
                )
                print(preview)

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
                    "prediction": prediction,
                    "reference": reference,
                    "expert_passed": expert_passed,
                    "expert_available": int(bool(expert_available)),
                    "expert_has_answer": int(bool(expert_has_answer)),
                    "answer_len": int(row.get("answer_len", len(answer_text))),
                }
            )

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[score] wrote {metadata_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["math", "mmlu"], required=True)
    parser.add_argument("--stage", choices=["generate", "score", "all"], default="all")
    parser.add_argument("--model", required=True, help="Full merged model path used by both vLLM and HF scoring.")
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--task_id_key", default="unique_id")
    parser.add_argument("--problem_key", default="problem")
    parser.add_argument("--answer_key", default="answer")
    parser.add_argument("--expert_jsonl", default=None)
    parser.add_argument("--expert_task_id_key", default="id")
    parser.add_argument("--expert_answer_key", default=None)
    parser.add_argument("--completions_jsonl", default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--debug_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

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
    parser.add_argument("--max_score_prefix_tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.expert_answer_key is None:
        args.expert_answer_key = "raw_output" if args.domain == "math" else "prediction"
    return args


def main() -> None:
    args = parse_args()
    if args.stage in {"generate", "all"}:
        generate_stage(args)
    if args.stage in {"score", "all"}:
        score_stage(args)


if __name__ == "__main__":
    main()
