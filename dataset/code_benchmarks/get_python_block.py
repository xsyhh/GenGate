from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd

# ==========================================
# 2. 定义提取函数
# ==========================================
def extract_python_code(text):
    """
    使用正则从混合文本中提取 ```python ... ``` 之间的代码。
    """
    if not isinstance(text, str):
        return ""
    
    # 正则逻辑：
    # ```python : 匹配开头标记
    # \s* : 匹配可能存在的换行或空格
    # (.*?)     : 非贪婪匹配中间的所有内容 (即代码本体)
    # ```       : 匹配结尾标记
    # re.DOTALL : 让 '.' 也能匹配换行符，因为代码是多行的
    pattern = r"```python\s*(.*?)```"
    
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        # group(1) 是我们要的代码部分，strip() 去除首尾多余空白
        return match.group(1).strip()
    else:
        # 如果没找到代码块，视情况返回空字符串或原始文本
        # 这里默认返回空字符串，表示提取失败
        return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    return parser.parse_args()

# ==========================================
# 3. 执行处理
# ==========================================
def main():
    args = parse_args()
    print(f"正在读取文件: {args.input_csv} ...")
    try:
        df = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print("❌ 错误：找不到输入文件，请检查路径。")
        return

    # 检查是否存在 response 列
    if 'response' not in df.columns:
        print("❌ 错误：CSV 中缺少 'response' 列。")
        return

    print("正在提取 Python 代码...")
    # 核心操作：应用提取函数
    df['solution'] = df['response'].apply(extract_python_code)

    # 打印简报
    total = len(df)
    extracted = len(df[df['solution'] != ""])
    print("-" * 30)
    print(f"总行数: {total}")
    print(f"成功提取代码: {extracted}")
    print(f"未找到代码块: {total - extracted}")
    print("-" * 30)

    # 保存结果
    # 使用 utf-8-sig 以防止 Excel 打开中文乱码
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 处理完成，已保存至: {args.output_csv}")

if __name__ == "__main__":
    main()
