from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_train_path", type=Path, required=True)
    parser.add_argument("--input_val_path", type=Path, required=True)
    parser.add_argument("--output_merged_path", type=Path, required=True)
    parser.add_argument("--output_train_new_path", type=Path, required=True)
    parser.add_argument("--output_val_new_path", type=Path, required=True)
    parser.add_argument("--id_col_name", type=str, default="id")
    return parser.parse_args()


def process_and_reindex_datasets(args: argparse.Namespace):

    print(f"{'='*20} 开始处理数据 {'='*20}")

    # --- 1. 读取数据 ---
    if not os.path.exists(args.input_train_path) or not os.path.exists(args.input_val_path):
        print("❌ 错误：找不到输入文件，请检查路径。")
        return

    print(f"正在读取 Train: {args.input_train_path}")
    df_train = pd.read_csv(args.input_train_path)
    print(f"正在读取 Val:   {args.input_val_path}")
    df_val = pd.read_csv(args.input_val_path)

    len_train = len(df_train)
    len_val = len(df_val)
    print(f"原始行数 -> Train: {len_train}, Val: {len_val}")

    # --- 2. 合并数据 ---
    # ignore_index=True 重置 pandas 的索引，但这还不是我们要的 task_id
    df_all = pd.concat([df_train, df_val], ignore_index=True)
    print(f"合并后总行数: {len(df_all)}")

    # --- 3. 构造全局唯一 ID ---
    # 这里生成从 0 到 N-1 的 ID，并转为字符串格式（兼容性更好）
    print(f"正在重新构造全局唯一 ID 列: '{args.id_col_name}' ...")
    df_all[args.id_col_name] = [str(i) for i in range(len(df_all))]

    # 确保 ID 列排在第一位 (可选，为了好看)
    cols = [args.id_col_name] + [c for c in df_all.columns if c != args.id_col_name]
    df_all = df_all[cols]

    # --- 4. 拆分回 Train 和 Val ---
    # 利用之前的长度切片，保证数据内容归属不变，但 ID 已经更新
    df_train_new = df_all.iloc[:len_train].copy()
    df_val_new = df_all.iloc[len_train:].copy()

    # --- 5. 保存文件 ---
    # 辅助函数：确保目录存在
    def ensure_dir_and_save(df, path, desc):
        output_dir = os.path.dirname(str(path))
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建目录: {output_dir}")
        
        df.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"✅ 已保存 {desc}: {path} (行数: {len(df)})")

    print(f"\n{'-'*10} 正在保存文件 {'-'*10}")
    ensure_dir_and_save(df_all, args.output_merged_path, "合并全量数据")
    ensure_dir_and_save(df_train_new, args.output_train_new_path, "新 Train 数据")
    ensure_dir_and_save(df_val_new, args.output_val_new_path, "新 Val 数据")

    print(f"\n所有任务完成。")

if __name__ == "__main__":
    process_and_reindex_datasets(parse_args())
