# 使用常用字表進行 Fine-tune 指南

## 📋 方案概述

只使用教育部 4808 個常用字進行訓練，可以：
- ✅ **減少訓練時間**：數據量更少，訓練更快
- ✅ **提高準確率**：專注於常用字，避免生僻字干擾
- ✅ **更實用**：4808 個常用字已經能覆蓋 99% 的日常使用場景

---

## 🚀 快速開始

### 步驟 1：安裝依賴

首先需要安裝 pandas 和 openpyxl 來讀取 Excel 文件：

```bash
pip install pandas openpyxl
```

### 步驟 2：修改轉換腳本

編輯 `convert_with_common_chars.py`，修改 `main()` 函數中的路徑：

```python
def main():
    # 修改這些路徑
    binarized_data_dir = "/your/path/to/binarized_data"
    samples_csv_path = "/your/path/to/finetune_data/samples.csv"
    id2char_json_path = "/your/path/to/finetune_data/id2char.json"

    # 常用字表路徑
    common_chars_path = "/your/path/to/finetune_data/教育部4808個常用字.xls"

    output_dir = "./train_data"
    train_ratio = 0.9
    min_samples_per_char = 5  # 每個字符至少 5 個樣本
    max_samples_per_char = 200  # 每個字符最多 200 個樣本（避免數據不平衡）
```

### 步驟 3：運行轉換

```bash
python convert_with_common_chars.py
```

### 步驟 4：檢查結果

```bash
# 查看生成的文件
ls train_data/

# 查看訓練樣本
head train_data/train_list.txt

# 查看字典（應該只包含常用字）
head train_data/custom_dict.txt
wc -l train_data/custom_dict.txt  # 應該不超過 4808 個字符

# 查看統計信息
cat train_data/dataset_stats.json
```

### 步驟 5：開始訓練

```bash
python tools/train.py \
    -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
    -o Global.pretrained_model=./pretrained/PP-OCRv5_mobile_rec_pretrained \
       Global.character_dict_path=./train_data/custom_dict.txt \
       Train.dataset.label_file_list=['./train_data/train_list.txt'] \
       Eval.dataset.label_file_list=['./train_data/val_list.txt'] \
       Optimizer.lr.learning_rate=0.00005
```

---

## 🔧 進階配置

### 1. 調整每個字符的樣本數

如果你的數據分佈不均（有些字很多樣本，有些字很少），可以調整：

```python
# 在 convert_with_common_chars.py 中修改
min_samples_per_char = 10   # 提高最低要求，過濾掉樣本太少的字
max_samples_per_char = 100  # 降低上限，讓數據更平衡
```

**效果：**
- `min_samples_per_char` 越高：字符數越少，但每個字符的質量更好
- `max_samples_per_char` 越低：數據越平衡，但可能浪費一些優質數據

**建議配置：**
- 數據充足（總樣本 > 100萬）：`min=10, max=100`
- 數據中等（總樣本 10-100萬）：`min=5, max=200`
- 數據較少（總樣本 < 10萬）：`min=3, max=None`（不限制）

### 2. 調整訓練/驗證集比例

```python
train_ratio = 0.95  # 95% 訓練，5% 驗證（數據多時）
# 或
train_ratio = 0.85  # 85% 訓練，15% 驗證（數據少時，需要更多驗證集）
```

---

## 🆘 備用方案：Excel 讀取失敗

如果腳本無法讀取 Excel 文件，可以手動轉換為 TXT 格式：

### 方法 1：使用 Excel 手動轉換

1. 打開 `教育部4808個常用字.xls`
2. 複製所有常用字
3. 粘貼到新的文本文件 `common_chars.txt`
4. 每行一個字，或者一行全部字符都可以

範例 `common_chars.txt`：
```
一
二
三
...
```

或者：
```
一二三四五六七八九十百千萬億...
```

### 方法 2：使用 Python 腳本轉換

創建一個簡單的轉換腳本 `convert_excel_to_txt.py`：

```python
import pandas as pd

# 讀取 Excel
df = pd.read_excel('finetune_data/教育部4808個常用字.xls')

# 提取字符（假設在第一列）
chars = df.iloc[:, 0].tolist()

# 保存為文本文件
with open('common_chars.txt', 'w', encoding='utf-8') as f:
    for char in chars:
        if str(char) != 'nan' and len(str(char)) > 0:
            f.write(str(char)[0] + '\n')

print(f"已保存 {len(chars)} 個常用字到 common_chars.txt")
```

運行：
```bash
pip install pandas openpyxl
python convert_excel_to_txt.py
```

### 方法 3：直接提供常用字列表

如果以上方法都不行，我可以幫你準備一個常用字文本文件。教育部常用字通常包括：

```python
# 創建 common_chars.txt
common_chars = """
的一是不了人我在有他這為之大來以個中上們到說國和地也子時道出而要於就下得可你年生自會那後能對著事其裡所去行過家十用發天如然作方成者多日都三小軍二無同麼經法當起與好看學進種將還分此心前面又定見只主沒公從...
"""

with open('common_chars.txt', 'w', encoding='utf-8') as f:
    for char in common_chars:
        if not char.isspace():
            f.write(char + '\n')
```

然後修改腳本使用 TXT 文件：
```python
common_chars_path = "/your/path/to/common_chars.txt"
```

---

## 📊 預期效果

### 數據統計範例

假設你的原始數據有：
- 總樣本數：1,000,000
- 字符種類：13,000+ 個（包含很多生僻字）

使用常用字過濾後：
- 總樣本數：~800,000（80%）
- 字符種類：~4,000 個（教育部常用字）
- 每字符平均：200 個樣本

### 訓練時間估算

- **原始數據**（13,000 字符）：~6-8 小時（單卡）
- **常用字數據**（4,000 字符）：~2-3 小時（單卡）

### 準確率預期

- 常用字準確率：**95%+**
- 訓練速度：**快 2-3 倍**
- 推理速度：**略快**（字典更小）

---

## 🎯 常見問題

### Q1: 如果某些常用字在我的數據集中沒有怎麼辦？

**A:** 腳本會自動過濾，最終字典只包含**實際有數據的常用字**。比如教育部列表有 4808 個字，但你的數據只有 3500 個常用字有樣本，那字典就只會包含這 3500 個。

### Q2: 我應該過濾掉樣本數太少的字符嗎？

**A:** 建議過濾。如果某個字符只有 1-2 個樣本，模型很難學好。推薦設置：
```python
min_samples_per_char = 5  # 至少 5 個樣本
```

### Q3: 我的數據不平衡，有些字有 1000 個樣本，有些只有 10 個？

**A:** 使用 `max_samples_per_char` 限制：
```python
max_samples_per_char = 200  # 每個字符最多 200 個樣本
```

這會讓模型對每個字符的學習更均衡。

### Q4: 我能同時使用常用字 + 部分生僻字嗎？

**A:** 可以！修改腳本中的常用字列表，添加你需要的生僻字：

```python
# 讀取常用字
common_chars = read_common_chars_from_excel(common_chars_path)

# 添加額外的生僻字
extra_chars = {'㗊', '㐀', '龘'}  # 你需要的生僻字
common_chars.update(extra_chars)
```

---

## 📝 完整流程檢查清單

- [ ] 已安裝 `pandas` 和 `openpyxl`
- [ ] 已準備好教育部常用字表（.xls 或 .txt）
- [ ] 已修改 `convert_with_common_chars.py` 中的路徑
- [ ] 已設置合適的 `min_samples_per_char` 和 `max_samples_per_char`
- [ ] 運行轉換腳本成功
- [ ] 檢查生成的 `custom_dict.txt` 字符數合理（~3000-4500）
- [ ] 檢查 `dataset_stats.json` 中的統計信息
- [ ] 查看 `train_list.txt` 前幾行確認格式正確
- [ ] 下載 PP-OCRv5 預訓練模型
- [ ] 開始訓練！

---

## 💡 快速命令（複製即用）

```bash
# 1. 安裝依賴
pip install pandas openpyxl

# 2. 修改腳本後運行轉換
python convert_with_common_chars.py

# 3. 檢查結果
head train_data/train_list.txt
wc -l train_data/custom_dict.txt

# 4. 下載預訓練模型（如果還沒下載）
mkdir -p pretrained && cd pretrained
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams
cd ..

# 5. 開始訓練
python tools/train.py \
    -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
    -o Global.pretrained_model=./pretrained/PP-OCRv5_mobile_rec_pretrained \
       Global.character_dict_path=./train_data/custom_dict.txt \
       Global.save_model_dir=./output/common_chars_finetune \
       Train.dataset.label_file_list=['./train_data/train_list.txt'] \
       Eval.dataset.label_file_list=['./train_data/val_list.txt'] \
       Optimizer.lr.learning_rate=0.00005 \
       Train.loader.batch_size_per_card=64

# 6. 查看訓練日誌
tail -f output/common_chars_finetune/train.log
```

---

Good luck! 🚀
