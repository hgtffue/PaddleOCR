#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
將分類資料夾格式的數據轉換為 PaddleOCR 訓練格式
"""

import os
import json
import csv
from pathlib import Path
import random

def convert_data(
    binarized_data_dir,
    samples_csv_path,
    id2char_json_path,
    output_dir,
    train_ratio=0.9,
    use_relative_path=True
):
    """
    轉換數據格式

    Args:
        binarized_data_dir: binarized_data 資料夾路徑
        samples_csv_path: samples.csv 文件路徑
        id2char_json_path: id2char.json 文件路徑
        output_dir: 輸出目錄
        train_ratio: 訓練集比例（默認 0.9，即 90% 訓練，10% 驗證）
        use_relative_path: 是否使用相對路徑（默認 True）
    """

    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)

    # 讀取 id2char 映射
    with open(id2char_json_path, 'r', encoding='utf-8') as f:
        id2char = json.load(f)

    print(f"載入 {len(id2char)} 個字符映射")

    # 讀取 samples.csv
    data_list = []
    with open(samples_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 將路徑轉換為實際的本地路徑
            # 原路徑格式: /work/hgtffue/dataset/cleaned_data/5378/紝_26.png
            # 轉換為: binarized_data/5378/紝_26.png

            old_path = row['path']
            char_id = row['char_id']
            char = row['char']

            # 提取文件名部分
            filename = os.path.basename(old_path)

            # 構建新路徑
            if use_relative_path:
                new_path = f"binarized_data/{char_id}/{filename}"
            else:
                new_path = os.path.join(binarized_data_dir, char_id, filename)

            data_list.append({
                'path': new_path,
                'char': char,
                'char_id': char_id
            })

    print(f"載入 {len(data_list)} 筆訓練樣本")

    # 隨機打亂數據
    random.shuffle(data_list)

    # 按比例分割訓練集和驗證集
    split_idx = int(len(data_list) * train_ratio)
    train_data = data_list[:split_idx]
    val_data = data_list[split_idx:]

    print(f"訓練集: {len(train_data)} 筆")
    print(f"驗證集: {len(val_data)} 筆")

    # 寫入訓練集標註文件
    train_txt_path = os.path.join(output_dir, 'train_list.txt')
    with open(train_txt_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(f"{item['path']}\t{item['char']}\n")

    print(f"訓練集標註文件已保存: {train_txt_path}")

    # 寫入驗證集標註文件
    val_txt_path = os.path.join(output_dir, 'val_list.txt')
    with open(val_txt_path, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(f"{item['path']}\t{item['char']}\n")

    print(f"驗證集標註文件已保存: {val_txt_path}")

    # 生成字典文件
    # 提取所有唯一字符並排序
    unique_chars = sorted(set(id2char.values()))

    dict_path = os.path.join(output_dir, 'custom_dict.txt')
    with open(dict_path, 'w', encoding='utf-8') as f:
        for char in unique_chars:
            f.write(f"{char}\n")

    print(f"字典文件已生成: {dict_path}")
    print(f"字典包含 {len(unique_chars)} 個字符")

    # 生成統計信息
    stats = {
        'total_samples': len(data_list),
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'unique_chars': len(unique_chars),
        'char_list': unique_chars[:20]  # 僅保存前20個字符作為示例
    }

    stats_path = os.path.join(output_dir, 'dataset_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"統計信息已保存: {stats_path}")

    return {
        'train_txt': train_txt_path,
        'val_txt': val_txt_path,
        'dict_txt': dict_path,
        'stats': stats
    }


def main():
    """主函數 - 根據你的實際路徑修改這裡"""

    # ========== 請根據實際情況修改以下路徑 ==========

    # binarized_data 資料夾的路徑
    binarized_data_dir = "/path/to/binarized_data"

    # samples.csv 文件路徑
    samples_csv_path = "/path/to/finetune_data/samples.csv"

    # id2char.json 文件路徑
    id2char_json_path = "/path/to/finetune_data/id2char.json"

    # 輸出目錄（生成的標註文件將保存在這裡）
    output_dir = "./train_data"

    # 訓練集比例（0.9 表示 90% 訓練，10% 驗證）
    train_ratio = 0.9

    # 是否使用相對路徑（如果選 True，則圖片路徑相對於 train_data/；如果選 False，則使用絕對路徑）
    use_relative_path = True

    # ================================================

    print("=" * 60)
    print("PaddleOCR 數據格式轉換工具")
    print("=" * 60)

    result = convert_data(
        binarized_data_dir=binarized_data_dir,
        samples_csv_path=samples_csv_path,
        id2char_json_path=id2char_json_path,
        output_dir=output_dir,
        train_ratio=train_ratio,
        use_relative_path=use_relative_path
    )

    print("\n" + "=" * 60)
    print("轉換完成！")
    print("=" * 60)
    print(f"\n生成的文件：")
    print(f"  - 訓練集標註: {result['train_txt']}")
    print(f"  - 驗證集標註: {result['val_txt']}")
    print(f"  - 字典文件: {result['dict_txt']}")
    print(f"\n數據統計：")
    print(f"  - 總樣本數: {result['stats']['total_samples']}")
    print(f"  - 訓練樣本: {result['stats']['train_samples']}")
    print(f"  - 驗證樣本: {result['stats']['val_samples']}")
    print(f"  - 字符數量: {result['stats']['unique_chars']}")
    print("\n下一步：")
    print("  1. 檢查生成的 train_list.txt 和 val_list.txt")
    print("  2. 修改訓練配置文件，指定字典路徑和數據路徑")
    print("  3. 開始訓練！")
    print("=" * 60)


if __name__ == '__main__':
    main()
