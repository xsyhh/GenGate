from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# 核心配置：定义要提取的列以及重命名规则
# 格式：{'原列名': '新列名'}
# 注意：只有出现在这个字典里的键会被提取，其他的列都会被丢弃
# columns_mapping = {
#     'task_id': 'task_id',             # 示例：把 task_id 改名为 id
#     'problem': 'problem',     # 示例：把 prompt 改名为 instruction
#     'current_code': 'starter_code',     # 示例：把 current_code 改名为 input
#     'test': 'test', # 示例：保留原名也可以，或者改名
#     'entry_point': 'entry_point',
#     'completion': 'solution',
# }

columns_mapping = {
    'task_id': 'task_id',             # 示例：把 task_id 改名为 id
    'problem': 'problem',     # 示例：把 prompt 改名为 instruction
    'current_code': 'starter_code',     # 示例：把 current_code 改名为 input
    'test': 'test', # 示例：保留原名也可以，或者改名
    'entry_point': 'entry_point',
    'code': 'solution',
}

# ==========================================
# 2. 执行逻辑
# ==========================================
def extract_and_rename(in_path, out_path, mapping):
    print(f"正在读取: {in_path}")
    try:
        df = pd.read_csv(in_path)
    except FileNotFoundError:
        print("错误：找不到输入文件，请检查路径。")
        return

    # 检查有哪些列是真实存在的
    available_cols = [col for col in mapping.keys() if col in df.columns]
    missing_cols = [col for col in mapping.keys() if col not in df.columns]

    if missing_cols:
        print(f"⚠️ 警告：以下列在原文件中未找到，将被跳过: {missing_cols}")

    if not available_cols:
        print("错误：没有找到任何有效的列，脚本终止。")
        return

    # 核心操作：筛选列 -> 重命名
    # df[available_cols] 先只取出需要的列
    # .rename(columns=mapping) 再将这些列改名
    df_new = df[available_cols].rename(columns=mapping)

    # 打印预览
    print("-" * 30)
    print(f"处理成功！提取了 {len(available_cols)} 列。")
    print("前 3 行预览:")
    print(df_new.head(3))
    print("-" * 30)

    # 保存
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df_new.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"文件已保存至: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    return parser.parse_args()

# ==========================================
# 3. 运行
# ==========================================
if __name__ == "__main__":
    args = parse_args()
    extract_and_rename(args.input_file, args.output_file, columns_mapping)
