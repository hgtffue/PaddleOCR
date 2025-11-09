# PaddleOCR Fine-tune 完整指南（PPOCRLabel 數據）

## 📚 數據說明

你的數據集來自 PPOCRLabel 標註工具，格式如下：

```
ppocrlabel/dataset/
├── images/               # 圖片文件
└── label.txt            # 標註文件（每行：圖片路徑\t標註JSON）
```

標註格式示例：
```
images/class01_10_001.jpg	[{"transcription": "問題（一）：", "points": [[3566, 482], ...], "difficult": false}, ...]
```

## 🚀 完整訓練流程

### 步驟 1：數據準備

#### 1.1 運行數據準備腳本

```bash
python prepare_ppocrlabel_data.py
```

這個腳本會：
- 從原始圖片中裁剪出文字區域（用於識別模型）
- 複製和轉換檢測數據（用於檢測模型）
- 自動分割訓練集和驗證集（90/10）
- 生成字典文件

#### 1.2 檢查生成的數據

運行完成後會生成：

```
train_data/
├── rec_images/           # 裁剪後的文字區域圖片
│   ├── crop_000000.jpg
│   ├── crop_000001.jpg
│   └── ...
├── det_images/           # 檢測用的完整圖片
│   ├── class01_10_001.jpg
│   └── ...
├── rec_train.txt        # 識別訓練標註
├── rec_val.txt          # 識別驗證標註
├── rec_dict.txt         # 識別字典
├── det_train.txt        # 檢測訓練標註
├── det_val.txt          # 檢測驗證標註
└── dataset_stats.json   # 數據統計
```

**檢查生成的文件：**

```bash
# 查看統計信息
cat train_data/dataset_stats.json

# 查看識別標註格式
head -5 train_data/rec_train.txt

# 查看檢測標註格式
head -2 train_data/det_train.txt

# 查看字典
head -20 train_data/rec_dict.txt
```

---

### 步驟 2：下載預訓練模型

#### 2.1 創建目錄

```bash
mkdir -p pretrained
cd pretrained
```

#### 2.2 下載識別模型

```bash
# PP-OCRv5 Mobile 識別模型（推薦）
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams

# 或使用 Server 版本（更高精度，更慢）
# wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams
```

#### 2.3 下載檢測模型

```bash
# PP-OCRv4 Mobile 檢測模型（推薦）
wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_train.tar
tar -xf ch_PP-OCRv4_det_train.tar

# 或使用 Server 版本
# wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_server_train.tar
# tar -xf ch_PP-OCRv4_det_server_train.tar
```

```bash
cd ..
```

---

### 步驟 3：訓練識別模型

#### 3.1 單卡訓練（推薦先從識別開始）

```bash
python tools/train.py \
    -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
    -o Global.pretrained_model=./pretrained/PP-OCRv5_mobile_rec_pretrained \
       Global.character_dict_path=./train_data/rec_dict.txt \
       Global.save_model_dir=./output/rec_ppocr_mobile \
       Global.epoch_num=100 \
       Train.dataset.data_dir=./ \
       Train.dataset.label_file_list=['./train_data/rec_train.txt'] \
       Train.loader.batch_size_per_card=128 \
       Eval.dataset.data_dir=./ \
       Eval.dataset.label_file_list=['./train_data/rec_val.txt'] \
       Eval.loader.batch_size_per_card=128 \
       Optimizer.lr.learning_rate=0.0001
```

#### 3.2 多卡訓練（如果有多張 GPU）

```bash
python -m paddle.distributed.launch --gpus '0,1' \
    tools/train.py \
    -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
    -o Global.pretrained_model=./pretrained/PP-OCRv5_mobile_rec_pretrained \
       Global.character_dict_path=./train_data/rec_dict.txt \
       Global.save_model_dir=./output/rec_ppocr_mobile \
       Global.epoch_num=100 \
       Train.dataset.data_dir=./ \
       Train.dataset.label_file_list=['./train_data/rec_train.txt'] \
       Train.loader.batch_size_per_card=256 \
       Eval.dataset.data_dir=./ \
       Eval.dataset.label_file_list=['./train_data/rec_val.txt'] \
       Optimizer.lr.learning_rate=0.0002
```

#### 3.3 監控訓練

```bash
# 查看訓練日誌
tail -f output/rec_ppocr_mobile/train.log

# 或使用 VisualDL（如果啟用）
visualdl --logdir output/rec_ppocr_mobile/vdl/ --port 8080
```

關鍵指標：
- `acc`: 識別準確率（越高越好，目標 > 85%）
- `norm_edit_dis`: 標準化編輯距離（越低越好）
- `loss`: 損失值（應該逐漸下降）

---

### 步驟 4：訓練檢測模型

#### 4.1 單卡訓練

```bash
python tools/train.py \
    -c configs/det/PP-OCRv4/ch_PP-OCRv4_det_student.yml \
    -o Global.pretrained_model=./pretrained/ch_PP-OCRv4_det_train/best_accuracy \
       Global.save_model_dir=./output/det_ppocr_v4 \
       Global.epoch_num=200 \
       Train.dataset.data_dir=./ \
       Train.dataset.label_file_list=['./train_data/det_train.txt'] \
       Train.loader.batch_size_per_card=8 \
       Eval.dataset.data_dir=./ \
       Eval.dataset.label_file_list=['./train_data/det_val.txt'] \
       Optimizer.lr.learning_rate=0.001
```

#### 4.2 多卡訓練

```bash
python -m paddle.distributed.launch --gpus '0,1' \
    tools/train.py \
    -c configs/det/PP-OCRv4/ch_PP-OCRv4_det_student.yml \
    -o Global.pretrained_model=./pretrained/ch_PP-OCRv4_det_train/best_accuracy \
       Global.save_model_dir=./output/det_ppocr_v4 \
       Global.epoch_num=200 \
       Train.dataset.data_dir=./ \
       Train.dataset.label_file_list=['./train_data/det_train.txt'] \
       Train.loader.batch_size_per_card=16 \
       Eval.dataset.data_dir=./ \
       Eval.dataset.label_file_list=['./train_data/det_val.txt'] \
       Optimizer.lr.learning_rate=0.002
```

#### 4.3 監控訓練

```bash
tail -f output/det_ppocr_v4/train.log
```

關鍵指標：
- `hmean`: F1 分數（精確率和召回率的調和平均，目標 > 85%）
- `precision`: 精確率
- `recall`: 召回率

---

### 步驟 5：導出推理模型

#### 5.1 導出識別模型

```bash
python tools/export_model.py \
    -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
    -o Global.pretrained_model=./output/rec_ppocr_mobile/best_accuracy \
       Global.character_dict_path=./train_data/rec_dict.txt \
       Global.save_inference_dir=./inference/rec_model
```

#### 5.2 導出檢測模型

```bash
python tools/export_model.py \
    -c configs/det/PP-OCRv4/ch_PP-OCRv4_det_student.yml \
    -o Global.pretrained_model=./output/det_ppocr_v4/best_accuracy \
       Global.save_inference_dir=./inference/det_model
```

---

### 步驟 6：測試完整的 OCR 系統

#### 6.1 使用 PaddleOCR 進行推理

創建測試腳本 `test_ocr.py`：

```python
from paddleocr import PaddleOCR

# 初始化 OCR，使用你 fine-tune 的模型
ocr = PaddleOCR(
    det_model_dir='./inference/det_model',
    rec_model_dir='./inference/rec_model',
    rec_char_dict_path='./train_data/rec_dict.txt',
    use_angle_cls=True,
    lang='ch'
)

# 測試圖片
img_path = './ppocrlabel/dataset/images/class01_10_001.jpg'
result = ocr.ocr(img_path, cls=True)

# 顯示結果
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(f"文字: {line[1][0]}, 置信度: {line[1][1]:.4f}")
```

運行測試：

```bash
python test_ocr.py
```

#### 6.2 批量測試

```bash
python tools/infer/predict_system.py \
    --image_dir="./ppocrlabel/dataset/images/" \
    --det_model_dir="./inference/det_model" \
    --rec_model_dir="./inference/rec_model" \
    --rec_char_dict_path="./train_data/rec_dict.txt" \
    --use_angle_cls=True \
    --use_gpu=True
```

---

## ⚙️ 超參數調整建議

### 識別模型調整

#### 如果數據量較少（< 2000 樣本）：

```yaml
Global:
  epoch_num: 150           # 增加訓練輪數

Optimizer:
  lr:
    learning_rate: 0.00005  # 降低學習率
    warmup_epoch: 5         # 增加 warmup

Train:
  loader:
    batch_size_per_card: 64  # 減小 batch size
```

#### 如果顯存不足：

```yaml
Train:
  loader:
    batch_size_per_card: 32   # 進一步減小
    num_workers: 2            # 減少 worker
```

### 檢測模型調整

#### 如果圖片較大或文字較小：

```yaml
Train:
  dataset:
    transforms:
      - DetResizeForTest:
          limit_side_len: 1600  # 增加到 1600 或 2000
          limit_type: 'max'
```

#### 如果檢測效果不好：

```yaml
Global:
  epoch_num: 300            # 增加訓練輪數

Optimizer:
  lr:
    learning_rate: 0.0005    # 降低學習率
```

---

## 🔍 常見問題

### Q1: 訓練數據量是否足夠？

**A:** 根據你的 200 張圖片：
- 假設每張圖片平均有 10-15 個文字區域
- 識別模型大約有 2000-3000 個訓練樣本
- 檢測模型有 180 張訓練圖片（90% 訓練集）

**建議**：
- **識別模型**：數據量基本夠用，但建議增加到 5000+ 樣本以獲得更好效果
- **檢測模型**：數據量偏少，建議增加到 500+ 張圖片

### Q2: 訓練時出現 CUDA Out of Memory？

**A:** 解決方案：
1. 減小 batch_size
2. 減小輸入圖片尺寸
3. 使用梯度累積
4. 使用混合精度訓練

```bash
# 啟用混合精度
python tools/train.py \
    -c config.yml \
    -o Global.use_amp=True
```

### Q3: 如何評估模型效果？

**A:** 使用評估命令：

```bash
# 評估識別模型
python tools/eval.py \
    -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
    -o Global.checkpoints=./output/rec_ppocr_mobile/best_accuracy \
       Global.character_dict_path=./train_data/rec_dict.txt \
       Eval.dataset.data_dir=./ \
       Eval.dataset.label_file_list=['./train_data/rec_val.txt']

# 評估檢測模型
python tools/eval.py \
    -c configs/det/PP-OCRv4/ch_PP-OCRv4_det_student.yml \
    -o Global.checkpoints=./output/det_ppocr_v4/best_accuracy \
       Eval.dataset.data_dir=./ \
       Eval.dataset.label_file_list=['./train_data/det_val.txt']
```

### Q4: 訓練中斷如何恢復？

**A:** 使用 checkpoints 恢復：

```bash
python tools/train.py \
    -c config.yml \
    -o Global.checkpoints=./output/rec_ppocr_mobile/latest
```

### Q5: 如何增加數據量？

**A:** 數據增強建議：

1. **手動標註更多數據**（最有效）
2. **數據增強**（在配置文件中已啟用）
3. **使用現有模型生成偽標籤**

---

## 📊 預期效果

根據你的數據量（200 張圖片，約 2000-3000 個文字區域）：

### 識別模型
- **預期準確率**：75-85%
- **訓練時間**：2-4 小時（單卡 GPU）

### 檢測模型
- **預期 F1**：70-80%
- **訓練時間**：4-8 小時（單卡 GPU）

### 如何提升效果：
1. 增加數據量到 500+ 張圖片
2. 確保標註質量（無錯誤標註）
3. 調整超參數（學習率、epoch 數）
4. 使用更大的模型（Server 版本）

---

## 📝 快速開始檢查清單

訓練前確認：

- [ ] 已安裝 PaddlePaddle >= 2.6.0 和 PaddleOCR
- [ ] 已運行 `prepare_ppocrlabel_data.py` 生成訓練數據
- [ ] 檢查生成的標註文件格式正確
- [ ] 已下載預訓練模型
- [ ] 確定了合適的 batch_size（根據 GPU 顯存）
- [ ] 準備好測試圖片

---

## 💡 快速開始命令（複製粘貼）

```bash
# ============ 第 1 步：準備數據 ============
python prepare_ppocrlabel_data.py

# ============ 第 2 步：下載預訓練模型 ============
mkdir -p pretrained && cd pretrained

# 下載識別模型
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams

# 下載檢測模型
wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_train.tar
tar -xf ch_PP-OCRv4_det_train.tar

cd ..

# ============ 第 3 步：訓練識別模型 ============
python tools/train.py \
    -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
    -o Global.pretrained_model=./pretrained/PP-OCRv5_mobile_rec_pretrained \
       Global.character_dict_path=./train_data/rec_dict.txt \
       Global.save_model_dir=./output/rec_ppocr_mobile \
       Global.epoch_num=100 \
       Train.dataset.data_dir=./ \
       Train.dataset.label_file_list=['./train_data/rec_train.txt'] \
       Train.loader.batch_size_per_card=128 \
       Eval.dataset.data_dir=./ \
       Eval.dataset.label_file_list=['./train_data/rec_val.txt'] \
       Optimizer.lr.learning_rate=0.0001

# ============ 第 4 步：訓練檢測模型 ============
python tools/train.py \
    -c configs/det/PP-OCRv4/ch_PP-OCRv4_det_student.yml \
    -o Global.pretrained_model=./pretrained/ch_PP-OCRv4_det_train/best_accuracy \
       Global.save_model_dir=./output/det_ppocr_v4 \
       Global.epoch_num=200 \
       Train.dataset.data_dir=./ \
       Train.dataset.label_file_list=['./train_data/det_train.txt'] \
       Train.loader.batch_size_per_card=8 \
       Eval.dataset.data_dir=./ \
       Eval.dataset.label_file_list=['./train_data/det_val.txt'] \
       Optimizer.lr.learning_rate=0.001

# ============ 第 5 步：導出模型 ============
# 導出識別模型
python tools/export_model.py \
    -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
    -o Global.pretrained_model=./output/rec_ppocr_mobile/best_accuracy \
       Global.character_dict_path=./train_data/rec_dict.txt \
       Global.save_inference_dir=./inference/rec_model

# 導出檢測模型
python tools/export_model.py \
    -c configs/det/PP-OCRv4/ch_PP-OCRv4_det_student.yml \
    -o Global.pretrained_model=./output/det_ppocr_v4/best_accuracy \
       Global.save_inference_dir=./inference/det_model
```

---

## 🎯 訓練順序建議

建議按以下順序進行：

1. **先訓練識別模型**（2-4 小時）
   - 數據量較大，容易看到效果
   - 可以快速驗證數據準備是否正確

2. **再訓練檢測模型**（4-8 小時）
   - 訓練時間較長
   - 需要更多調參經驗

3. **聯合測試和調優**
   - 使用完整的檢測+識別流程測試
   - 根據結果調整超參數

---

祝訓練順利！🚀

如有問題，可以查看：
- [PaddleOCR 官方文檔](https://github.com/PaddlePaddle/PaddleOCR/blob/main/README_ch.md)
- [PaddleOCR FAQ](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_ch/FAQ.md)
