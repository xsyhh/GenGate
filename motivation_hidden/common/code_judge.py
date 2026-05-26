"""Standalone code answer cleaning and HumanEval-style judging."""

from __future__ import annotations

import re


GLOBAL_CODE_HEADER = """
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

try:
    from sortedcontainers import SortedList
except Exception:
    SortedList = list

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

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def tree_node(values: list):
    if not values: return None
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
"""


def clean_markdown_fences(text: str) -> str:
    if not isinstance(text, str):
        return ""
    match = re.search(r"```[a-zA-Z0-9]*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1)
    text = re.sub(r"```[a-zA-Z0-9]*\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text)
    return text.strip()


def run_code_tests(code: str, tests: str, entry_point: str, task_id: str, timeout: int) -> bool:
    try:
        from human_eval.execution import check_correctness
    except Exception as exc:
        print(f"[WARN] human_eval.execution unavailable; marking code task {task_id} as failed: {exc}")
        return False
    problem = {
        "task_id": str(task_id),
        "prompt": "",
        "test": str(tests or ""),
        "entry_point": str(entry_point or ""),
    }
    result = check_correctness(problem, GLOBAL_CODE_HEADER + "\n" + code, timeout=int(timeout))
    return bool(result.get("passed")) if isinstance(result, dict) else False
