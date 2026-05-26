from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

# ==========================================
# 2. 执行合并逻辑
# ==========================================
def merge_csv_files(file_list, output_path):
    print(f"准备合并 {len(file_list)} 个文件...")
    
    dataframes = []
    
    for file_path in file_list: 
        print(f"正在读取: {file_path}")
        if not os.path.exists(file_path):
            print(f"⚠️ 警告: 文件不存在，已跳过 -> {file_path}")
            continue
            
        try:
            # 读取 CSV
            df = pd.read_csv(file_path)
            dataframes.append(df)
            print(f"  -> 成功加载 {len(df)} 行数据")
        except Exception as e:
            print(f"❌ 错误: 无法读取文件 {file_path}. 原因: {e}")

    if not dataframes:
        print("没有可合并的数据，脚本结束。")
        return

    # 合并所有 DataFrame
    print("-" * 30)
    print("正在拼接所有数据...")
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # ==========================================
    # 核心修改：ID 重编码与重命名
    # ==========================================
    print("正在处理 ID 列...")


    # ----------------------------------------------------
    # [新增] 检查 task_id 是否重复
    # ----------------------------------------------------
    if 'task_id' in merged_df.columns:
        dup_count = merged_df['task_id'].duplicated().sum()
        if dup_count > 0:
            print(f"⚠️ 警告: 发现 {dup_count} 个重复的 task_id！")
            
            # (可选) 如果你想看具体的重复项，可以取消下面这行的注释
            # print(merged_df[merged_df['task_id'].duplicated(keep=False)]['task_id'].head(10))
            
            # (可选) 如果你想直接删除重复项，取消下面这两行的注释
            # print("  -> 正在自动去重 (保留第一条)...")
            # merged_df = merged_df.drop_duplicates(subset=['task_id'], keep='first').reset_index(drop=True)
        else:
            print("  ->task_id 唯一，无重复。")
    else:
        print("  -> 未找到 'task_id' 列，跳过重复检查。")
    # ----------------------------------------------------

    # # 2. 生成新的唯一 'id' (从 0 开始的整数索引)
    # new_ids = range(len(merged_df))
    # merged_df['id'] = new_ids
    # print(f"  -> 已生成 {len(merged_df)} 个新的唯一 ID")

    # # 3. 调整列顺序，确保 'id' 在第一列
    # cols = merged_df.columns.tolist()
    # if 'id' in cols:
    #     cols.insert(0, cols.pop(cols.index('id')))
    # merged_df = merged_df[cols]

    # ==========================================
    # 3. 保存结果
    # ==========================================
    print(f"合并完成！总数据量: {len(merged_df)} 行")
    
    output_dir = os.path.dirname(str(output_path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"文件已保存至: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", action="append", dest="csv_file_list", required=True)
    parser.add_argument("--output_csv_path", type=Path, required=True)
    return parser.parse_args()

# ==========================================
# 运行
# ==========================================
if __name__ == "__main__":
    args = parse_args()
    merge_csv_files(args.csv_file_list, str(args.output_csv_path))
