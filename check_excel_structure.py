#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
檢查教育部常用字表 Excel 文件的結構
"""

import sys

def check_excel_structure(excel_path):
    """
    檢查 Excel 文件結構並顯示內容

    Args:
        excel_path: Excel 文件路徑
    """
    try:
        import pandas as pd
    except ImportError:
        print("錯誤: 需要安裝 pandas")
        print("請運行: pip install pandas openpyxl")
        return

    try:
        print("=" * 60)
        print(f"正在檢查文件: {excel_path}")
        print("=" * 60)

        # 讀取 Excel
        df = pd.read_excel(excel_path)

        print(f"\n✓ 文件讀取成功！")
        print(f"\n基本信息:")
        print(f"  - 總行數: {len(df)}")
        print(f"  - 總列數: {len(df.columns)}")
        print(f"  - 列名: {df.columns.tolist()}")

        print(f"\n前 20 行數據:")
        print(df.head(20))

        print(f"\n數據類型:")
        print(df.dtypes)

        # 嘗試提取字符
        print(f"\n正在嘗試提取字符...")

        # 嘗試各種可能的列
        possible_columns = []
        for col in df.columns:
            # 檢查第一列
            if df[col].dtype == 'object':  # 文本類型
                sample = df[col].head(10).tolist()
                # 檢查是否是單個字符
                if all(isinstance(x, str) and len(x) == 1 for x in sample if pd.notna(x)):
                    possible_columns.append(col)

        if possible_columns:
            print(f"\n找到可能包含字符的列: {possible_columns}")

            for col in possible_columns:
                chars = df[col].dropna().tolist()
                unique_chars = set(chars)

                print(f"\n列 '{col}' 的統計:")
                print(f"  - 總字符數: {len(chars)}")
                print(f"  - 唯一字符數: {len(unique_chars)}")
                print(f"  - 前 50 個字符: {''.join(list(unique_chars)[:50])}")

                # 保存到文本文件
                output_file = f"common_chars_from_{col}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    for char in sorted(unique_chars):
                        f.write(f"{char}\n")

                print(f"  - 已保存到: {output_file}")
        else:
            print("\n未找到明顯的字符列")
            print("將使用第一列作為字符列")

            first_col = df.columns[0]
            chars = df[first_col].dropna().astype(str).tolist()

            # 提取每個單元格的第一個字符
            extracted_chars = set()
            for item in chars:
                if item and item != 'nan':
                    extracted_chars.add(item[0])

            print(f"\n從第一列提取的字符:")
            print(f"  - 唯一字符數: {len(extracted_chars)}")
            print(f"  - 前 50 個字符: {''.join(list(extracted_chars)[:50])}")

            # 保存到文本文件
            output_file = "common_chars_extracted.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                for char in sorted(extracted_chars):
                    f.write(f"{char}\n")

            print(f"  - 已保存到: {output_file}")

        print("\n" + "=" * 60)
        print("檢查完成！")
        print("=" * 60)
        print("\n下一步:")
        print("  1. 如果生成了 .txt 文件，可以在轉換腳本中使用它")
        print("  2. 或者直接使用 Excel 文件（腳本會自動處理）")

    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # 默認路徑
    excel_path = "finetune_data/教育部4808個常用字.xls"

    # 如果命令行提供了路徑，使用命令行參數
    if len(sys.argv) > 1:
        excel_path = sys.argv[1]

    check_excel_structure(excel_path)
