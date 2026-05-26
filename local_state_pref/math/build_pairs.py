from __future__ import annotations

from collections import defaultdict
from typing import Any

# ACTIONS = {
#     "attempt": "<|Attempt|>",
#     "defer": "<|defer|>",
#     "self": "<|self|>",
# }

# ATTEMPT_OPEN = ACTIONS["attempt"] + "\n"
# ATTEMPT_CLOSE = "\n"

# MATH_AGENT_TEMPLATE = """You are a math reasoning agent with a clear understanding of your own capabilities, you can actively seek help from expert.
# ### Question:
# {problem}
# ### INSTRUCTION:
# 1. First decide whether to try solving by yourself or defer immediately.
# 2. If you defer immediately, output ONLY: <|defer|>
# 3. If you try solving by yourself:
#    - First output <|Attempt|> to indicate you will try.
#    - Solve the problem step by step.
#    - Conclude your response with exactly "Final answer: " followed IMMEDIATELY by the bare mathematical expression or value.
#    - Finally, assess your confidence. If you are confident in your answer, output <|self|>. If you are unsure, output <|defer|>.
# 4. Do NOT include any extra text after the decision token.
# ### FORMAT SKELETON:
# Immediate Defer: <|defer|>
# Try Solving: <|Attempt|>\nreasoning\nFinal answer: ans\n<|self|> or <|defer|>
# """


ACTIONS = {
    "attempt": "<Attempt>",
    "defer": "<defer>",
    "self": "<self>",
}

ATTEMPT_OPEN = ACTIONS["attempt"] + "\n"
ATTEMPT_CLOSE = "\n"

MATH_AGENT_TEMPLATE = """You are a math reasoning agent with a clear understanding of your own capabilities, you can actively seek help from expert.
### Question:
{problem}
### INSTRUCTION:
1. First decide whether to try solving by yourself or defer immediately.
2. If you defer immediately, output ONLY: <defer>
3. If you try solving by yourself:
   - First output <Attempt> to indicate you will try.
   - Solve the problem step by step.
   - Conclude your response with exactly "Final answer: " followed IMMEDIATELY by the bare mathematical expression or value.
   - Finally, assess your confidence. If you are confident in your answer, output <self>. If you are unsure, output <defer>.
4. Do NOT include any extra text after the decision token.
### FORMAT SKELETON:
Immediate Defer: <defer>
Try Solving: <Attempt>\nreasoning\nFinal answer: ans\n<self> or <defer>
"""


def build_s0_context(problem: str) -> str:
    return MATH_AGENT_TEMPLATE.format(problem=problem)


def build_s1_context(problem: str, reasoning: str) -> str:
    return build_s0_context(problem) + ATTEMPT_OPEN + reasoning.strip() + ATTEMPT_CLOSE


def apply_chat_template(tokenizer, user_text: str, add_generation_prompt: bool = True) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    return user_text


def build_chat_s0_context(tokenizer, problem: str, add_generation_prompt: bool = True) -> str:
    return apply_chat_template(tokenizer, build_s0_context(problem), add_generation_prompt=add_generation_prompt)


def build_chat_s1_context(tokenizer, problem: str, reasoning: str) -> str:
    return build_chat_s0_context(tokenizer, problem, add_generation_prompt=True) + ATTEMPT_OPEN + reasoning.strip() + ATTEMPT_CLOSE


def build_pair_records(raw_map: dict[str, dict[str, Any]], evaluated_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated_rows:
        grouped[str(row["task_id"])].append(row)

    records: list[dict[str, Any]] = []
    for task_id, rows in grouped.items():
        if task_id not in raw_map:
            continue

        valid_rows = [
            row
            for row in rows
            if str(row.get("raw_output", "")).strip() and str(row.get("extracted_answer", "")).strip()
        ]
        if not valid_rows:
            continue

        valid_rows.sort(key=lambda row: int(row.get("sample_index", 0)))
        raw = raw_map[task_id]
        rollout_count = len(valid_rows)
        p_hat = sum(1.0 if row.get("self_passed") else 0.0 for row in valid_rows) / rollout_count

        records.append(
            {
                "state_type": "s0",
                "task_id": task_id,
                "sample_index": None,
                "context": build_s0_context(str(raw["problem"])),
                "reasoning": "",
                "action_a": ACTIONS["attempt"],
                "action_b": ACTIONS["defer"],
                "target_prob": float(p_hat),
                "state_weight": 1.0,
                "rollout_count": rollout_count,
            }
        )

        per_rollout_weight = 1.0 / rollout_count
        for row in valid_rows:
            reasoning = str(row["raw_output"]).strip()
            records.append(
                {
                    "state_type": "s1",
                    "task_id": task_id,
                    "sample_index": row.get("sample_index"),
                    "context": build_s1_context(str(raw["problem"]), reasoning),
                    "reasoning": reasoning,
                    "action_a": ACTIONS["self"],
                    "action_b": ACTIONS["defer"],
                    "target_prob": 1.0 if row.get("self_passed") else 0.0,
                    "state_weight": per_rollout_weight,
                    "rollout_count": rollout_count,
                }
            )

    return records
