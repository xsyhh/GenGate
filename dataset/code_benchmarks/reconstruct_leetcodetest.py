from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    output_file = args.output_file or args.input_file.with_name(args.input_file.stem + "-processed.csv")
    print(f"--- 正在读取文件: {args.input_file} ---")
    
    if not os.path.exists(args.input_file):
        print(f"错误: 找不到文件 {args.input_file}")
        print("请检查路径是否正确。")
        return

    try:
        df = pd.read_csv(args.input_file)
        
        # 检查必要的列
        required_cols = ['test', 'prompt']
        for col in required_cols:
            if col not in df.columns:
                print(f"错误: CSV 中缺少列 '{col}'")
                print(f"现有列: {list(df.columns)}")
                return

        print(f"--- 正在处理 {len(df)} 条数据 ---")
        
        # 核心逻辑: new_test = prompt + 换行 + test
        # 使用 fillna('') 避免因为空值导致拼接失败
        df['new_test'] = df['prompt'].fillna('') + "\n" + df['test'].fillna('')
        
        # 保存结果
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        
        print(f"--- 成功！结果已保存至: {output_file} ---")
        
        # 打印示例预览
        print("\n[预览第一条数据的 new_test]:")
        print("-" * 50)
        print(df['new_test'].iloc[0][:300] + "...\n(后面省略)")
        print("-" * 50)

    except Exception as e:
        print(f"处理过程中发生错误: {e}")

if __name__ == "__main__":
    main()
