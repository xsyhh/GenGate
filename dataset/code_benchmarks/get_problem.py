from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re

import pandas as pd

def extract_docstring(code_text):
    """
    尝试从代码字符串中提取函数的 Docstring。
    优先使用 AST 解析（能自动处理缩进），如果解析失败（因代码片段不完整），则回退到正则提取。
    """
    # 方法 1: 使用 AST 解析 (最准确，自动去除缩进)
    try:
        # 解析代码成抽象语法树
        tree = ast.parse(code_text)
        # 遍历树找到第一个函数定义
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                docstring = ast.get_docstring(node)
                if docstring:
                    return docstring.strip()
    except (SyntaxError, ValueError):
        pass # 如果代码片段有语法错误，进入正则回退

    # 方法 2: 正则表达式回退 (用于 AST 失败的情况)
    # 匹配 """ 内容 """ 或 ''' 内容 '''，re.DOTALL 让 . 能匹配换行符
    match = re.search(r'("""|\'\'\')(.*?)(\1)', code_text, re.DOTALL)
    if match:
        # group(2) 是引号中间的内容
        return match.group(2).strip()
    
    return ""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--source_column", type=str, default="complete_prompt")
    parser.add_argument("--target_column", type=str, default="problem")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    print("正在提取 docstring...")
    df[args.target_column] = df[args.source_column].apply(extract_docstring)
    print("-" * 30)
    print("提取结果示例 (First 300 chars):")
    print(df.iloc[0][args.target_column][:300])
    print("-" * 30)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False, encoding='utf-8-sig')
    print(f"处理完成，已保存为 {args.output_csv}")


if __name__ == "__main__":
    main()
