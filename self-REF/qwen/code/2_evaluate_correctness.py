import json
import pandas as pd
import concurrent.futures
import argparse
from tqdm import tqdm
import threading
import time
from pathlib import Path
import re

# ==========================================
# 核心评测逻辑（如果没装 human_eval，这里提供一个简易实现）
# ==========================================
from human_eval.execution import check_correctness

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

# 明确引入常用数据结构，避免使用 import * 污染全局空间
from collections import deque, defaultdict, Counter
from heapq import heapify, heappush, heappop, heappushpop, heapreplace
from bisect import bisect_left, bisect_right, insort_left, insort_right
from functools import lru_cache, cache, reduce, cmp_to_key
from itertools import combinations, permutations, accumulate, product

# 如果本地装了 sortedcontainers，这句保留
from sortedcontainers import SortedList, SortedDict, SortedSet

# 定义常用常量（不依赖 from math import *）
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
        p.next = ListNode(val)
        p = p.next
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
    if not p and not q: return True
    if not p or not q: return False
    if p.val != q.val: return False
    return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)
"""

expert_results_cache = {}
cache_lock = threading.Lock()


def clean_python_code(text):
    text = str(text or "").strip()
    match = re.search(r"```(?:python|py)?\s*\n?(.*?)(?:\n?```|$)", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    text = re.sub(r"^\s*```(?:python|py)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    text = re.sub(r"^\s*(?:python|py)\s*\n", "", text, flags=re.IGNORECASE)
    return text.strip()

def _optional_bool(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return bool(value)

def load_tests_by_id(csv_paths):
    test_dict = {}
    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            continue
        if "id" not in df.columns or "test" not in df.columns:
            continue
        for _, row in df.iterrows():
            test_dict[str(row["id"]).strip()] = row["test"]
    return test_dict

def build_expert_dict(expert_df, test_dict):
    if "id" in expert_df.columns:
        id_col = "id"
    elif "task_id" in expert_df.columns:
        id_col = "task_id"
    else:
        raise KeyError(f"Expert CSV 缺少 id/task_id 列，现有列: {list(expert_df.columns)}")

    if "expert_code" not in expert_df.columns and "expert_passed" not in expert_df.columns:
        raise KeyError(f"Expert CSV 缺少 expert_code/expert_passed 列，现有列: {list(expert_df.columns)}")

    expert_dict = {}
    for _, row in expert_df.iterrows():
        eid = str(row[id_col]).strip()
        if "test" in expert_df.columns:
            tests = row["test"]
        else:
            tests = test_dict.get(eid)
        expert_code = row["expert_code"] if "expert_code" in expert_df.columns and not pd.isna(row["expert_code"]) else ""
        expert_passed = None
        if "expert_passed" in expert_df.columns:
            expert_passed = _optional_bool(row["expert_passed"])
        expert_dict[eid] = {
            "test": tests,
            "expert_code": expert_code,
            "expert_passed": expert_passed,
        }
    return expert_dict

def evaluate_code(task_id, code, tests, entry_point, timeout=5.0):
    """
    真正的评测函数
    """
    if not entry_point:
        return False

    problem = {
        "task_id": str(task_id),
        "prompt": "",
        "test": tests,
        "entry_point": entry_point,
    }
    
    full_code = GLOBAL_HEADER + "\n" + code
    
    try:
        # 使用 human_eval 的官方评测函数
        result = check_correctness(problem, full_code, timeout=timeout)
        return result.get('passed', False)
    except Exception as e:
        # 如果你想调试为什么全是 False，可以取消下面这行的注释
        # print(f"DEBUG: Task {task_id} failed with error: {e}")
        return False

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--actor_samples_path", type=str, default="qwen/code/output/3B_actor_samples.jsonl")
    p.add_argument("--expert_csv_path", type=str, default="expert_output/code/expert_results.csv")
    p.add_argument("--output_path", type=str, default="qwen/code/output/3B_validation_results.jsonl")
    p.add_argument(
        "--raw_csv_paths",
        nargs="+",
        default=[
            "data/code_benchmarks/mbppplus_leetcode_humanevalplus_v2/train_with_id.csv",
            "data/code_benchmarks/mbppplus_leetcode_humanevalplus_v2/val_with_id.csv",
            "data/code_benchmarks/mbppplus_leetcode_humanevalplus/RL/val_with_id.csv",
        ],
    )
    p.add_argument("--max_threads", type=int, default=32)
    args = p.parse_args()

    print("🔬 Step 2: 启动多线程代码验证 (SEU Server Mode)...")
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载 Expert 数据
    print(f"正在读取专家数据: {args.expert_csv_path}")
    expert_df = pd.read_csv(args.expert_csv_path)
    test_dict = load_tests_by_id(args.raw_csv_paths)
    expert_dict = build_expert_dict(expert_df, test_dict)
    print(f"已加载测试用例: {len(test_dict)} 个 task")
    print(f"已加载专家结果: {len(expert_dict)} 个 task")

    # 2. 读取 Actor 采样数据
    print(f"正在读取采样数据: {args.actor_samples_path}")
    actor_data = []
    with open(args.actor_samples_path, "r", encoding="utf-8") as f:
        for line in f:
            actor_data.append(json.loads(line))
    actor_ids = [str(x.get("id")) for x in actor_data]
    print(f"采样轨迹: {len(actor_data)} 条，唯一 task: {len(set(actor_ids))}")
    print(f"actor/expert ID 重合轨迹数: {sum(1 for tid in actor_ids if tid in expert_dict)}")
    print(f"actor 可找到测试用例轨迹数: {sum(1 for tid in actor_ids if tid in test_dict)}")

    # 3. 定义工作线程
    def worker(item):
        task_id = str(item["id"])
        # 直接使用 item 里的 entry_point，确保准确性
        entry_point = item.get("entry_point") 
        
        expert_info = expert_dict.get(task_id)
        tests = None
        expert_code = ""
        exp_passed_from_csv = None

        if expert_info is not None:
            tests = expert_info.get("test")
            expert_code = str(expert_info.get("expert_code") or "")
            exp_passed_from_csv = expert_info.get("expert_passed")

        if tests is None or pd.isna(tests):
            tests = test_dict.get(task_id)

        if tests is None or pd.isna(tests):
            return None 

        actor_code = clean_python_code(item["extracted_code"])
        
        # --- 评测 Expert ---
        if exp_passed_from_csv is not None:
            exp_passed = bool(exp_passed_from_csv)
        elif expert_code.strip():
            with cache_lock:
                if task_id not in expert_results_cache:
                    expert_results_cache[task_id] = "pending"
                    need_eval_expert = True
                else:
                    need_eval_expert = False
                    
            if need_eval_expert:
                exp_passed = evaluate_code(task_id, expert_code, tests, entry_point)
                with cache_lock:
                    expert_results_cache[task_id] = exp_passed
            else:
                while expert_results_cache.get(task_id) == "pending":
                    time.sleep(0.02)
                exp_passed = expert_results_cache.get(task_id, False)
        else:
            exp_passed = False
            
        # --- 评测 Actor (Self) ---
        self_passed = evaluate_code(task_id, actor_code, tests, entry_point)
        
        return {
            "id": task_id,
            "sample_index": item["sample_index"],
            "self_passed": self_passed,
            "expert_passed": exp_passed,
            "expert_available": expert_info is not None,
            "extracted_code": actor_code # 保留代码供第三步使用
        }

    # 4. 并发执行
    print(f"开始评测... 线程数: {args.max_threads}")
    
    final_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_threads) as executor:
        futures = {executor.submit(worker, it): it for it in actor_data}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(actor_data)):
            res = future.result()
            if res:
                final_results.append(res)

    # 5. 保存
    final_results.sort(key=lambda x: (x['id'], x['sample_index']))
    with open(args.output_path, "w", encoding="utf-8") as f:
        for res in final_results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
    # 统计成功率，帮你快速定位问题
    self_ok = sum(1 for x in final_results if x['self_passed'])
    exp_ok = sum(1 for x in final_results if x['expert_passed'])
    print(f"📊 统计结果: Actor通过率 {self_ok}/{len(final_results)} | Expert通过率 {exp_ok}/{len(final_results)}")
    print(f"✅ Step 2 完成！结果保存至 {args.output_path}")

if __name__ == "__main__":
    main()
