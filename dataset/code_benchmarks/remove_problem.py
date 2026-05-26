from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re

import pandas as pd

# ==========================================
# 工具函数 1: 提取 Docstring (你原本的逻辑)
# ==========================================
def extract_docstring(code_text):
    """提取函数的 Docstring"""
    try:
        tree = ast.parse(code_text)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                docstring = ast.get_docstring(node)
                if docstring:
                    return docstring.strip()
    except (SyntaxError, ValueError):
        pass

    # 正则回退
    match = re.search(r'("""|\'\'\')(.*?)(\1)', code_text, re.DOTALL)
    if match:
        return match.group(2).strip()
    return ""

# ==========================================
# 工具函数 2: 移除 Docstring (生成 current_code)
# ==========================================
def remove_docstring(code_text):
    """
    从代码中移除 Docstring，保留函数定义和具体实现。
    优先使用 AST 获取 Docstring 的起止行号进行删除。
    """
    if not isinstance(code_text, str) or not code_text.strip():
        return ""

    # 方法 1: AST 解析 (精确定位行号删除)
    try:
        tree = ast.parse(code_text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # 获取 docstring 节点
                if not node.body: continue
                
                # 检查函数体第一条语句是否为字符串（即 Docstring）
                first_node = node.body[0]
                if isinstance(first_node, ast.Expr) and isinstance(first_node.value, ast.Constant) and isinstance(first_node.value.value, str):
                    
                    # 获取 Docstring 的起止行号 (AST行号从1开始)
                    # 注意：end_lineno 在 Python 3.8+ 才支持
                    start_lineno = first_node.lineno - 1
                    end_lineno = first_node.end_lineno 
                    
                    lines = code_text.splitlines()
                    
                    # 只有当行号有效时才操作
                    if end_lineno <= len(lines):
                        # 保留 docstring 之前的部分 + docstring 之后的部分
                        # 注意：这里会把整行删掉。如果 docstring 和代码混在一行（极少见），可能会有问题，但在 MBPP 中一般没问题。
                        new_lines = lines[:start_lineno] + lines[end_lineno:]
                        
                        # 重新组合，并去除首尾可能产生的多余空行
                        return "\n".join(new_lines).strip()
                        
    except (SyntaxError, AttributeError, ValueError):
        pass # AST 失败，回退到正则

    # 方法 2: 正则表达式回退
    # 替换掉第一个匹配到的三引号内容为空字符串
    # count=1 确保只替换函数开头的 docstring，不替换代码中间定义的变量字符串
    cleaned_code = re.sub(r'(\s*)("""|\'\'\')(.*?)(\3)', '', code_text, count=1, flags=re.DOTALL)
    
    # 清理可能残留的空行
    return cleaned_code.strip()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--problem_source_column", type=str, default="problem_description")
    parser.add_argument("--code_source_column", type=str, default="starter_code")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"正在读取文件: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    print("正在提取 problem (Docstring)...")
    df['problem'] = df[args.problem_source_column].apply(extract_docstring)
    print("正在生成 current_code (Code without Docstring)...")
    df['current_code'] = df[args.code_source_column].apply(remove_docstring)
    print("-" * 50)
    print("提取的 Problem 示例:")
    print(df.iloc[0]['problem'])
    print("-" * 50)
    print("生成的 Current_Code 示例 (不应包含上面的注释):")
    print(df.iloc[0]['current_code'])
    print("-" * 50)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False, encoding='utf-8-sig')
    print(f"处理完成，已保存为 {args.output_csv}")


if __name__ == "__main__":
    main()

