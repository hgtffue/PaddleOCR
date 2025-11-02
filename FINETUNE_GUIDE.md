# PaddleOCR PP-OCRv5 Fine-tune 完整指南

## 📚 數據結構說明

### 你的原始數據結構：
```
binarized_data/
├── 1/
│   ├── 字_0.png
│   ├── 字_1.png
│   └── ...
├── 2/
├── 3/
...
└── 10000/

finetune_data/
├── samples.csv        # 包含所有樣本的路徑、字符、ID
└── id2char.json       # ID 到字符的映射
```

### PaddleOCR 需要的格式：
```
train_data/
├── train_list.txt     # 訓練集標註文件
├── val_list.txt       # 驗證集標註文件
└── custom_dict.txt    # 字典文件

binarized_data/        # 圖片數據（保持不變）
├── 1/
├── 2/
...
```

---

## 🚀 完整訓練流程

### 步驟 1：數據格式轉換

#### 1.1 修改轉換腳本中的路徑

編輯 `convert_to_paddleocr_format.py`，修改 `main()` 函數中的路徑：

```python
def main():
    # 修改這些路徑為你服務器上的實際路徑
    binarized_data_dir = "/your/path/to/binarized_data"
    samples_csv_path = "/your/path/to/finetune_data/samples.csv"
    id2char_json_path = "/your/path/to/finetune_data/id2char.json"
    output_dir = "./train_data"

    train_ratio = 0.9  # 90% 訓練，10% 驗證
    use_relative_path = True
```

#### 1.2 運行轉換腳本

```bash
python convert_to_paddleocr_format.py
```

#### 1.3 檢查生成的文件

轉換完成後會生成：

- `train_data/train_list.txt` - 訓練集標註文件
- `train_data/val_list.txt` - 驗證集標註文件
- `train_data/custom_dict.txt` - 字典文件
- `train_data/dataset_stats.json` - 數據統計

**檢查 train_list.txt 的格式：**
```bash
head train_data/train_list.txt
```

應該看到類似的內容：
```
binarized_data/5378/紝_26.png	紝
binarized_data/10474/鰈_12.png	鰈
binarized_data/1597/棼_8.png	棼
```

---

### 步驟 2：下載預訓練模型

```bash
# 創建目錄
mkdir -p pretrained

# 下載 PP-OCRv5 Mobile 識別模型
cd pretrained
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams
cd ..

# 或者，如果需要高精度，下載 Server 版本
# wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams
```

---

### 步驟 3：創建或修改訓練配置

#### 選項 A：使用自動化腳本（推薦）

修改 `prepare_and_train.sh` 中的配置區域，然後運行：

```bash
chmod +x prepare_and_train.sh
./prepare_and_train.sh
```

#### 選項 B：手動創建配置文件

複製一份配置文件並修改：

```bash
cp configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml config_finetune.yml
```

修改以下關鍵參數：

```yaml
Global:
  pretrained_model: ./pretrained/PP-OCRv5_mobile_rec_pretrained
  character_dict_path: ./train_data/custom_dict.txt
  epoch_num: 50
  save_model_dir: ./output/PP-OCRv5_finetune
  use_space_char: false  # 如果你的數據不需要識別空格

Optimizer:
  lr:
    learning_rate: 0.00005  # 單卡訓練建議降低學習率

Train:
  dataset:
    data_dir: ./
    label_file_list:
      - ./train_data/train_list.txt
  loader:
    batch_size_per_card: 64

Eval:
  dataset:
    data_dir: ./
    label_file_list:
      - ./train_data/val_list.txt
  loader:
    batch_size_per_card: 64
```

---

### 步驟 4：開始訓練

#### 單卡訓練
```bash
python tools/train.py -c config_finetune.yml
```

#### 多卡訓練
```bash
python -m paddle.distributed.launch --gpus '0,1,2,3' \
    tools/train.py -c config_finetune.yml \
    -o Optimizer.lr.learning_rate=0.0002  # 多卡時可適當提高學習率
```

#### 斷點續訓
```bash
python tools/train.py -c config_finetune.yml \
    -o Global.checkpoints=./output/PP-OCRv5_finetune/latest
```

---

### 步驟 5：監控訓練

查看訓練日誌：
```bash
tail -f output/PP-OCRv5_finetune/train.log
```

關鍵指標：
- `acc`: 識別準確率（越高越好）
- `norm_edit_dis`: 標準化編輯距離（越低越好）
- `loss`: 損失值（應該逐漸下降）

---

### 步驟 6：評估模型

```bash
python tools/eval.py -c config_finetune.yml \
    -o Global.checkpoints=./output/PP-OCRv5_finetune/best_accuracy
```

---

### 步驟 7：導出推理模型

```bash
python tools/export_model.py \
    -c config_finetune.yml \
    -o Global.pretrained_model=./output/PP-OCRv5_finetune/best_accuracy \
       Global.save_inference_dir=./inference/rec_model
```

---

### 步驟 8：測試模型

```bash
python tools/infer_rec.py \
    -c config_finetune.yml \
    -o Global.pretrained_model=./output/PP-OCRv5_finetune/best_accuracy \
       Global.infer_img=./test_images/
```

---

## ⚙️ 超參數調整建議

根據你的數據量和硬件配置：

### 如果顯存不足：
```yaml
Train:
  loader:
    batch_size_per_card: 32  # 減小 batch size

Architecture:
  Backbone:
    scale: 0.5  # 使用更小的模型（如果 0.95 太大）
```

### 如果數據量較少（< 10000）：
```yaml
Optimizer:
  lr:
    learning_rate: 0.00002  # 進一步降低學習率
    warmup_epoch: 5  # 增加 warmup

Global:
  epoch_num: 100  # 增加訓練輪數
```

### 如果數據量很大（> 100000）：
```yaml
Optimizer:
  lr:
    learning_rate: 0.0001  # 可以使用較高學習率

Train:
  loader:
    batch_size_per_card: 128  # 增大 batch size
```

---

## 🔍 常見問題

### Q1: 訓練時提示找不到圖片？
**A:** 檢查以下幾點：
1. `train_list.txt` 中的路徑是否正確
2. 配置文件中的 `data_dir` 是否設置正確
3. 如果使用相對路徑，確保路徑相對於 `data_dir`

### Q2: 訓練 loss 不下降或 acc 為 0？
**A:** 可能原因：
1. 學習率過高或過低，嘗試調整為 `5e-5` 或 `1e-5`
2. 字典文件不匹配，確保 `custom_dict.txt` 包含所有訓練數據中的字符
3. 數據標註有誤，檢查 `train_list.txt` 格式是否正確

### Q3: 訓練很慢？
**A:** 優化建議：
1. 增大 `batch_size_per_card`
2. 增大 `num_workers`（數據載入線程數）
3. 使用多卡訓練
4. 使用混合精度訓練（AMP）

### Q4: 如何恢復訓練？
**A:** 使用 checkpoints：
```bash
python tools/train.py -c config_finetune.yml \
    -o Global.checkpoints=./output/PP-OCRv5_finetune/iter_epoch_10
```

---

## 📊 數據增強建議

如果數據量較少，可以在配置文件中啟用更多增強：

```yaml
Train:
  dataset:
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - RecAug:  # 自動增強
          prob: 0.5  # 增強概率
      - MultiLabelEncode:
          gtc_encode: NRTRLabelEncode
      - RecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys: [image, label_ctc, label_gtc, length, valid_ratio]
```

---

## 🎯 預期效果

根據數據質量和數量：

- **數據量 > 50000，質量好**：準確率可達 95%+
- **數據量 10000-50000**：準確率 85-95%
- **數據量 < 10000**：準確率 70-85%

如果效果不理想：
1. 增加數據量
2. 檢查數據質量（標註準確性、圖片清晰度）
3. 調整超參數
4. 增加訓練輪數

---

## 📝 快速檢查清單

訓練前確認：

- [ ] 已安裝 PaddlePaddle 和 PaddleOCR
- [ ] 已下載預訓練模型
- [ ] 已生成 train_list.txt 和 val_list.txt
- [ ] 已生成 custom_dict.txt
- [ ] 配置文件中的路徑都正確
- [ ] 檢查了幾個樣本的標註是否正確
- [ ] 確定了合適的 batch_size 和 learning_rate

---

## 💡 快速開始命令（複製粘貼）

```bash
# 1. 轉換數據
python convert_to_paddleocr_format.py

# 2. 下載預訓練模型
mkdir -p pretrained && cd pretrained
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams
cd ..

# 3. 開始訓練（單卡）
python tools/train.py \
    -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
    -o Global.pretrained_model=./pretrained/PP-OCRv5_mobile_rec_pretrained \
       Global.character_dict_path=./train_data/custom_dict.txt \
       Global.save_model_dir=./output/finetune \
       Train.dataset.data_dir=./ \
       Train.dataset.label_file_list=['./train_data/train_list.txt'] \
       Eval.dataset.data_dir=./ \
       Eval.dataset.label_file_list=['./train_data/val_list.txt'] \
       Optimizer.lr.learning_rate=0.00005 \
       Train.loader.batch_size_per_card=64
```

祝訓練順利！🚀
