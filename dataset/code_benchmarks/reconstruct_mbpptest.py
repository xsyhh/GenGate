from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# ==========================================
# 2. 核心处理函数：Wrap Existing Test
# ==========================================
def wrap_existing_test(row):
    """
    逻辑：
    1. 生成 def check(candidate):
    2. 生成映射: entry_point = candidate
    3. 将原始 row['test'] 的每一行都加上 4 个空格的缩进，拼接到后面。
    """
    entry_point = row.get('entry_point')      # 函数名
    original_test_code = row.get('test')      # 原始的测试代码
    
    lines = []
    
    # --- 1. 函数头 ---
    # 标准评测框架通常调用 check(candidate)，所以这里固定写 candidate
    lines.append("def check(candidate):")

    # --- 2. 关键映射 ---
    # 只要加上这句，下面原始代码里调用的 similar_elements 就会自动指向 candidate
    if pd.notna(entry_point):
        lines.append(f"    {entry_point} = candidate")

    # --- 3. 嵌入原始 Test 代码 (核心步骤) ---
    if pd.notna(original_test_code):
        # 按行分割原始代码
        code_lines = str(original_test_code).split('\n')
        
        for line in code_lines:
            # 这里的关键是：直接给每一行加缩进，不要随意 strip 掉它内部的缩进
            # 因为原始 test 代码里可能也有 for 循环或 if 判断
            lines.append(f"    {line}") 
            # 注意：如果 original_test_code 本身首尾有空行，这也会缩进空行，问题不大

    return "\n".join(lines)

# ==========================================
# 3. 执行
# ==========================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    print("正在读取 CSV...")
    df = pd.read_csv(args.input_csv)

    # ==========================================
    # [新增逻辑] 删除 entry_point 为 'check' 的数据
    # ==========================================
    print("正在检查并删除冲突数据 (entry_point == 'check')...")
    initial_count = len(df)

    # 过滤条件：只保留 entry_point 不等于 'check' 的行
    df = df[df['entry_point'] != 'check']

    deleted_count = initial_count - len(df)
    print(f"-> 原始数量: {initial_count}")
    print(f"-> 删除数量: {deleted_count}")
    print(f"-> 剩余数量: {len(df)}")
    print("-" * 30)

    print("正在直接封装 test 列...")
    df['test'] = df.apply(wrap_existing_test, axis=1)

    print("-" * 50)
    print("处理后的结果预览 (Task 1):")
    print("-" * 50)
    if not df.empty:
        print(f"Entry Point: {df.iloc[0].get('entry_point')}")
        print("New Check Function:")
        print(df.iloc[0]['test'])
    else:
        print("警告：处理后 DataFrame 为空！")
    print("-" * 50)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False, encoding='utf-8-sig')
    print(f"文件已保存至: {args.output_csv}")


if __name__ == "__main__":
    main()
