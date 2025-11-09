#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
從 PPOCRLabel 標註數據準備 PaddleOCR 訓練數據（修復版）
使用簡單的邊界框裁剪，避免透視變換導致的變形
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import random


def load_label_file(label_path):
    """載入 PPOCRLabel 格式的標註文件"""
    data = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 分割圖片路徑和標註
            parts = line.split('\t')
            if len(parts) != 2:
                print(f"警告: 跳過格式錯誤的行: {line[:50]}")
                continue

            img_path = parts[0]
            try:
                annotations = json.loads(parts[1])
                data.append({
                    'img_path': img_path,
                    'annotations': annotations
                })
            except json.JSONDecodeError as e:
                print(f"警告: JSON 解析錯誤: {line[:50]}, 錯誤: {e}")
                continue

    return data


def crop_text_region_simple(image, points):
    """
    使用旋轉矩形裁剪，保持原始形狀不變形
    points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    """
    points = np.array(points, dtype=np.float32)

    # 使用 cv2.minAreaRect 獲取最小外接矩形
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # 獲取矩形的寬高
    width = int(rect[1][0])
    height = int(rect[1][1])

    # 如果寬高為0，跳過
    if width == 0 or height == 0:
        return None

    # 獲取旋轉矩陣
    center = rect[0]
    angle = rect[2]

    # 調整角度
    if width < height:
        angle = angle + 90
        width, height = height, width

    # 獲取旋轉矩陣
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 旋轉整個圖片
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # 裁剪旋轉後的矩形區域
    cropped = cv2.getRectSubPix(rotated, (width, height), center)

    return cropped


def prepare_recognition_data(data, base_dir, output_dir, train_ratio=0.9):
    """
    準備識別模型訓練數據
    從原始圖片中裁剪出文字區域
    """
    print("\n========== 準備識別模型數據 ==========")

    # 創建輸出目錄
    rec_img_dir = os.path.join(output_dir, 'rec_images')
    os.makedirs(rec_img_dir, exist_ok=True)

    train_samples = []
    val_samples = []
    char_set = set()

    total_regions = sum(len(item['annotations']) for item in data)
    print(f"總共需要裁剪 {total_regions} 個文字區域")

    region_idx = 0
    success_count = 0
    fail_count = 0

    # 遍歷每張圖片
    for item in tqdm(data, desc="處理圖片"):
        img_path = os.path.join(base_dir, item['img_path'])

        if not os.path.exists(img_path):
            print(f"警告: 圖片不存在 {img_path}")
            continue

        # 讀取圖片
        image = cv2.imread(img_path)
        if image is None:
            print(f"警告: 無法讀取圖片 {img_path}")
            continue

        # 處理每個標註區域
        for ann in item['annotations']:
            transcription = ann.get('transcription', '')
            points = ann.get('points', [])
            difficult = ann.get('difficult', False)

            # 跳過困難樣本或空文本
            if difficult or not transcription or transcription == '###':
                continue

            # 跳過非四邊形
            if len(points) != 4:
                continue

            # 裁剪文字區域
            try:
                cropped = crop_text_region_simple(image, points)

                if cropped is None:
                    fail_count += 1
                    continue

                # 過濾太小的圖片
                if cropped.shape[0] < 8 or cropped.shape[1] < 8:
                    fail_count += 1
                    continue

                # 保存裁剪後的圖片
                cropped_filename = f"crop_{region_idx:06d}.jpg"
                cropped_path = os.path.join(rec_img_dir, cropped_filename)
                cv2.imwrite(cropped_path, cropped)

                # 記錄樣本
                sample = f"rec_images/{cropped_filename}\t{transcription}\n"

                # 隨機分配到訓練集或驗證集
                if random.random() < train_ratio:
                    train_samples.append(sample)
                else:
                    val_samples.append(sample)

                # 收集字符集
                for char in transcription:
                    char_set.add(char)

                region_idx += 1
                success_count += 1

            except Exception as e:
                print(f"警告: 裁剪失敗 {img_path}, 區域 {region_idx}, 錯誤: {e}")
                fail_count += 1
                continue

    print(f"\n裁剪完成:")
    print(f"  - 成功: {success_count}")
    print(f"  - 失敗: {fail_count}")
    print(f"  - 訓練樣本: {len(train_samples)}")
    print(f"  - 驗證樣本: {len(val_samples)}")
    print(f"  - 字符集大小: {len(char_set)}")

    # 保存訓練和驗證標註文件
    train_label_path = os.path.join(output_dir, 'rec_train.txt')
    val_label_path = os.path.join(output_dir, 'rec_val.txt')

    with open(train_label_path, 'w', encoding='utf-8') as f:
        f.writelines(train_samples)

    with open(val_label_path, 'w', encoding='utf-8') as f:
        f.writelines(val_samples)

    # 保存字典文件
    dict_path = os.path.join(output_dir, 'rec_dict.txt')
    sorted_chars = sorted(list(char_set))
    with open(dict_path, 'w', encoding='utf-8') as f:
        for char in sorted_chars:
            f.write(char + '\n')

    print(f"\n識別模型數據已保存:")
    print(f"  - 訓練標註: {train_label_path}")
    print(f"  - 驗證標註: {val_label_path}")
    print(f"  - 字典文件: {dict_path}")

    return len(train_samples), len(val_samples), len(char_set)


def prepare_detection_data(data, base_dir, output_dir, train_ratio=0.9):
    """
    準備檢測模型訓練數據
    將標註轉換為 PaddleOCR 檢測格式
    """
    print("\n========== 準備檢測模型數據 ==========")

    # 創建輸出目錄
    det_img_dir = os.path.join(output_dir, 'det_images')
    os.makedirs(det_img_dir, exist_ok=True)

    train_samples = []
    val_samples = []

    # 隨機打亂數據
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"訓練圖片: {len(train_data)}, 驗證圖片: {len(val_data)}")

    def process_split(split_data, split_name):
        samples = []
        for item in tqdm(split_data, desc=f"處理{split_name}數據"):
            img_path = os.path.join(base_dir, item['img_path'])

            if not os.path.exists(img_path):
                print(f"警告: 圖片不存在 {img_path}")
                continue

            # 複製圖片到輸出目錄
            img_filename = os.path.basename(item['img_path'])
            dst_img_path = os.path.join(det_img_dir, img_filename)
            shutil.copy2(img_path, dst_img_path)

            # 構建標註
            label_list = []
            for ann in item['annotations']:
                points = ann.get('points', [])
                transcription = ann.get('transcription', '')

                if len(points) != 4:
                    continue

                # 轉換為檢測格式
                label_item = {
                    'transcription': transcription,
                    'points': points
                }
                label_list.append(label_item)

            # 保存標註
            sample = f"det_images/{img_filename}\t{json.dumps(label_list, ensure_ascii=False)}\n"
            samples.append(sample)

        return samples

    train_samples = process_split(train_data, "訓練")
    val_samples = process_split(val_data, "驗證")

    # 保存標註文件
    train_label_path = os.path.join(output_dir, 'det_train.txt')
    val_label_path = os.path.join(output_dir, 'det_val.txt')

    with open(train_label_path, 'w', encoding='utf-8') as f:
        f.writelines(train_samples)

    with open(val_label_path, 'w', encoding='utf-8') as f:
        f.writelines(val_samples)

    print(f"\n檢測模型數據已保存:")
    print(f"  - 訓練標註: {train_label_path}")
    print(f"  - 驗證標註: {val_label_path}")

    return len(train_samples), len(val_samples)


def main():
    # ========== 配置區 ==========
    # 標註文件路徑
    label_file = "ppocrlabel/dataset/label.txt"

    # 圖片所在目錄（label.txt 中路徑的根目錄）
    base_dir = "ppocrlabel/dataset"

    # 輸出目錄
    output_dir = "train_data"

    # 訓練集比例
    train_ratio = 0.9

    # 隨機種子
    random.seed(42)
    # ============================

    print("PaddleOCR 數據準備工具 (修復版)")
    print("=" * 50)
    print(f"標註文件: {label_file}")
    print(f"圖片目錄: {base_dir}")
    print(f"輸出目錄: {output_dir}")
    print(f"訓練集比例: {train_ratio:.1%}")
    print("=" * 50)

    # 檢查文件是否存在
    if not os.path.exists(label_file):
        print(f"錯誤: 標註文件不存在 {label_file}")
        return

    if not os.path.exists(base_dir):
        print(f"錯誤: 圖片目錄不存在 {base_dir}")
        return

    # 清空舊的輸出目錄
    if os.path.exists(output_dir):
        print(f"\n清空舊的輸出目錄...")
        shutil.rmtree(output_dir)

    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)

    # 載入標註數據
    print("\n載入標註數據...")
    data = load_label_file(label_file)
    print(f"成功載入 {len(data)} 張圖片的標註")

    # 準備識別數據
    rec_train_count, rec_val_count, char_count = prepare_recognition_data(
        data, base_dir, output_dir, train_ratio
    )

    # 準備檢測數據
    det_train_count, det_val_count = prepare_detection_data(
        data, base_dir, output_dir, train_ratio
    )

    # 保存統計信息
    stats = {
        'recognition': {
            'train_samples': rec_train_count,
            'val_samples': rec_val_count,
            'total_samples': rec_train_count + rec_val_count,
            'char_count': char_count
        },
        'detection': {
            'train_images': det_train_count,
            'val_images': det_val_count,
            'total_images': det_train_count + det_val_count
        }
    }

    stats_path = os.path.join(output_dir, 'dataset_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print("數據準備完成！")
    print("=" * 50)
    print("\n識別模型:")
    print(f"  - 訓練樣本: {rec_train_count}")
    print(f"  - 驗證樣本: {rec_val_count}")
    print(f"  - 字符種類: {char_count}")
    print("\n檢測模型:")
    print(f"  - 訓練圖片: {det_train_count}")
    print(f"  - 驗證圖片: {det_val_count}")
    print(f"\n統計信息已保存: {stats_path}")
    print("\n下一步:")
    print("  1. 檢查生成的數據是否正確")
    print("  2. 下載預訓練模型")
    print("  3. 開始訓練")


if __name__ == '__main__':
    main()
