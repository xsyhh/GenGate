from __future__ import annotations

import re

TOKEN_SELF = "<CN>"
TOKEN_DEFER = "<UN>"

LABEL_SELF = "self"
LABEL_DEFER = "defer"

DECISION_TOKENS = (TOKEN_SELF, TOKEN_DEFER)
DECISION_TOKEN_PATTERN = re.compile(
    rf"(?:{re.escape(TOKEN_SELF)}|{re.escape(TOKEN_DEFER)})"
)
DECISION_TOKEN_AT_END_PATTERN = re.compile(
    rf"(?s)^(.*?)(?:\s*({re.escape(TOKEN_SELF)}|{re.escape(TOKEN_DEFER)}))\s*$"
)


def decision_token_from_passed(self_passed: bool) -> str:
    return TOKEN_SELF if bool(self_passed) else TOKEN_DEFER


def label_from_decision_token(token: str) -> str:
    text = str(token or "").strip()
    if text == TOKEN_SELF:
        return LABEL_SELF
    if text == TOKEN_DEFER:
        return LABEL_DEFER
    return ""


def strip_trailing_decision_token(text: str) -> tuple[str, str]:
    value = str(text or "").strip()
    match = DECISION_TOKEN_AT_END_PATTERN.match(value)
    if not match:
        return value, ""
    return str(match.group(1) or "").strip(), str(match.group(2) or "").strip()


def strip_all_decision_tokens(text: str) -> str:
    return DECISION_TOKEN_PATTERN.sub("", str(text or "")).strip()
