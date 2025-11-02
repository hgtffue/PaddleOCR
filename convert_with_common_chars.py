#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用常用字表過濾數據並轉換為 PaddleOCR 訓練格式
"""

import os
import json
import csv
from pathlib import Path
import random

def read_common_chars_from_excel(excel_path):
    """
    從 Excel 文件讀取常用字列表

    Args:
        excel_path: Excel 文件路徑

    Returns:
        set: 常用字集合
    """
    common_chars = set()

    try:
        import pandas as pd
        print(f"正在讀取常用字表: {excel_path}")

        # 讀取 Excel，支持 .xls 和 .xlsx
        df = pd.read_excel(excel_path)

        print(f"Excel 列名: {df.columns.tolist()}")
        print(f"前 5 行數據:\n{df.head()}")

        # 嘗試從各種可能的列名中提取字符
        possible_columns = ['字', '常用字', '字符', '漢字', 'char', 'character']

        char_column = None
        for col in df.columns:
            if any(name in str(col) for name in possible_columns):
                char_column = col
                break

        # 如果找不到明顯的列名，使用第一列
        if char_column is None:
            char_column = df.columns[0]
            print(f"未找到明確的字符列，使用第一列: {char_column}")

        print(f"使用列: {char_column}")

        # 提取字符
        for idx, row in df.iterrows():
            char = str(row[char_column]).strip()
            # 只取第一個字符（如果有多個字）
            if char and len(char) > 0 and char != 'nan':
                common_chars.add(char[0])

        print(f"成功載入 {len(common_chars)} 個常用字")

        # 顯示前 50 個字符作為驗證
        sample_chars = list(common_chars)[:50]
        print(f"前 50 個常用字: {''.join(sample_chars)}")

        return common_chars

    except ImportError:
        print("錯誤: 需要安裝 pandas 和 openpyxl")
        print("請運行: pip install pandas openpyxl")
        return None
    except Exception as e:
        print(f"讀取 Excel 文件時出錯: {e}")
        return None


def read_common_chars_from_txt(txt_path):
    """
    從文本文件讀取常用字列表（備用方案）

    格式：每行一個字符，或者一行多個字符

    Args:
        txt_path: 文本文件路徑

    Returns:
        set: 常用字集合
    """
    common_chars = set()

    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # 支持每行一個字或每行多個字
                    for char in line:
                        if char and not char.isspace():
                            common_chars.add(char)

        print(f"從文本文件載入 {len(common_chars)} 個常用字")
        return common_chars

    except Exception as e:
        print(f"讀取文本文件時出錯: {e}")
        return None


def convert_data_with_filter(
    binarized_data_dir,
    samples_csv_path,
    id2char_json_path,
    output_dir,
    common_chars_path=None,
    train_ratio=0.9,
    use_relative_path=True,
    min_samples_per_char=5,
    max_samples_per_char=None
):
    """
    轉換數據格式（帶常用字過濾）

    Args:
        binarized_data_dir: binarized_data 資料夾路徑
        samples_csv_path: samples.csv 文件路徑
        id2char_json_path: id2char.json 文件路徑
        output_dir: 輸出目錄
        common_chars_path: 常用字表路徑（.xls/.xlsx/.txt），如果為 None 則不過濾
        train_ratio: 訓練集比例（默認 0.9）
        use_relative_path: 是否使用相對路徑（默認 True）
        min_samples_per_char: 每個字符最少樣本數（默認 5）
        max_samples_per_char: 每個字符最多樣本數（None 表示不限制）
    """

    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)

    # 讀取常用字表（如果提供）
    common_chars = None
    if common_chars_path:
        if common_chars_path.endswith(('.xls', '.xlsx')):
            common_chars = read_common_chars_from_excel(common_chars_path)
        elif common_chars_path.endswith('.txt'):
            common_chars = read_common_chars_from_txt(common_chars_path)
        else:
            print(f"警告: 不支持的文件格式 {common_chars_path}")

        if common_chars is None or len(common_chars) == 0:
            print("錯誤: 無法載入常用字表")
            return None

    # 讀取 id2char 映射
    with open(id2char_json_path, 'r', encoding='utf-8') as f:
        id2char = json.load(f)

    print(f"載入 {len(id2char)} 個字符映射")

    # 讀取 samples.csv
    print("正在讀取 samples.csv...")
    data_list = []
    filtered_count = 0
    char_sample_count = {}  # 統計每個字符的樣本數

    with open(samples_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            old_path = row['path']
            char_id = row['char_id']
            char = row['char']

            # 如果提供了常用字表，進行過濾
            if common_chars and char not in common_chars:
                filtered_count += 1
                continue

            # 統計每個字符的樣本數
            if char not in char_sample_count:
                char_sample_count[char] = 0

            # 如果設置了最大樣本數限制
            if max_samples_per_char and char_sample_count[char] >= max_samples_per_char:
                continue

            char_sample_count[char] += 1

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
    if common_chars:
        print(f"過濾掉 {filtered_count} 筆非常用字樣本")

    # 過濾掉樣本數過少的字符
    if min_samples_per_char > 1:
        chars_to_remove = set()
        for char, count in char_sample_count.items():
            if count < min_samples_per_char:
                chars_to_remove.add(char)

        if chars_to_remove:
            original_len = len(data_list)
            data_list = [item for item in data_list if item['char'] not in chars_to_remove]
            print(f"移除了 {len(chars_to_remove)} 個樣本不足的字符（< {min_samples_per_char} 個樣本）")
            print(f"移除了 {original_len - len(data_list)} 筆樣本")

    if len(data_list) == 0:
        print("錯誤: 沒有可用的訓練數據！")
        return None

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

    # 生成字典文件（只包含實際使用的字符）
    actual_chars = sorted(set(item['char'] for item in data_list))

    dict_path = os.path.join(output_dir, 'custom_dict.txt')
    with open(dict_path, 'w', encoding='utf-8') as f:
        for char in actual_chars:
            f.write(f"{char}\n")

    print(f"字典文件已生成: {dict_path}")
    print(f"字典包含 {len(actual_chars)} 個字符")

    # 統計每個字符的樣本數（最終版本）
    final_char_count = {}
    for item in data_list:
        char = item['char']
        final_char_count[char] = final_char_count.get(char, 0) + 1

    # 找出樣本數最多和最少的字符
    sorted_chars = sorted(final_char_count.items(), key=lambda x: x[1], reverse=True)

    print("\n樣本數統計:")
    print(f"  平均每個字符: {len(data_list) / len(actual_chars):.1f} 個樣本")
    print(f"  樣本數最多的 5 個字符:")
    for char, count in sorted_chars[:5]:
        print(f"    {char}: {count} 個")
    print(f"  樣本數最少的 5 個字符:")
    for char, count in sorted_chars[-5:]:
        print(f"    {char}: {count} 個")

    # 生成統計信息
    stats = {
        'total_samples': len(data_list),
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'unique_chars': len(actual_chars),
        'filtered_samples': filtered_count if common_chars else 0,
        'avg_samples_per_char': len(data_list) / len(actual_chars),
        'char_sample_count': final_char_count,
        'char_list_sample': actual_chars[:50]  # 保存前50個字符
    }

    stats_path = os.path.join(output_dir, 'dataset_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"統計信息已保存: {stats_path}")

    # 如果使用了常用字過濾，額外保存常用字列表
    if common_chars:
        common_chars_path_out = os.path.join(output_dir, 'common_chars_list.txt')
        with open(common_chars_path_out, 'w', encoding='utf-8') as f:
            for char in sorted(common_chars):
                f.write(f"{char}\n")
        print(f"常用字列表已保存: {common_chars_path_out}")

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

    # 常用字表路徑（支持 .xls, .xlsx, .txt）
    # 如果不需要過濾，設置為 None
    common_chars_path = "/path/to/finetune_data/教育部4808個常用字.xls"

    # 輸出目錄（生成的標註文件將保存在這裡）
    output_dir = "./train_data"

    # 訓練集比例（0.9 表示 90% 訓練，10% 驗證）
    train_ratio = 0.9

    # 是否使用相對路徑
    use_relative_path = True

    # 每個字符最少樣本數（樣本數少於此值的字符會被過濾掉）
    min_samples_per_char = 5

    # 每個字符最多樣本數（限制每個字符的樣本數，避免數據不平衡）
    # 設置為 None 表示不限制
    max_samples_per_char = 200  # 可以根據需要調整

    # ================================================

    print("=" * 60)
    print("PaddleOCR 數據格式轉換工具（常用字過濾版）")
    print("=" * 60)

    result = convert_data_with_filter(
        binarized_data_dir=binarized_data_dir,
        samples_csv_path=samples_csv_path,
        id2char_json_path=id2char_json_path,
        output_dir=output_dir,
        common_chars_path=common_chars_path,
        train_ratio=train_ratio,
        use_relative_path=use_relative_path,
        min_samples_per_char=min_samples_per_char,
        max_samples_per_char=max_samples_per_char
    )

    if result is None:
        print("\n轉換失敗！請檢查錯誤信息。")
        return

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
    if result['stats']['filtered_samples'] > 0:
        print(f"  - 過濾樣本: {result['stats']['filtered_samples']}")
    print(f"  - 平均每字符: {result['stats']['avg_samples_per_char']:.1f} 個樣本")
    print("\n下一步：")
    print("  1. 檢查生成的 train_list.txt 和 val_list.txt")
    print("  2. 確認字典文件 custom_dict.txt 包含所有需要的字符")
    print("  3. 修改訓練配置文件，指定字典路徑和數據路徑")
    print("  4. 開始訓練！")
    print("=" * 60)


if __name__ == '__main__':
    main()
