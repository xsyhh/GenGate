from __future__ import annotations


def extract_action_suffix_ids(tokenizer, context: str, action_text: str) -> list[int]:
    context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    return extract_action_suffix_ids_from_context_ids(tokenizer, context, context_ids, action_text)


def extract_action_suffix_ids_from_context_ids(
    tokenizer,
    context: str,
    context_ids: list[int],
    action_text: str,
) -> list[int]:
    full_ids = tokenizer(context + action_text, add_special_tokens=False)["input_ids"]

    if full_ids[: len(context_ids)] != context_ids:
        raise ValueError("Context tokenization is not a prefix of context+action tokenization")

    return list(full_ids[len(context_ids) :])
