#!/bin/bash
# PaddleOCR PP-OCRv5 Fine-tune 完整流程腳本

set -e  # 遇到錯誤立即退出

echo "========================================"
echo "PaddleOCR PP-OCRv5 Fine-tune 流程"
echo "========================================"

# ========== 配置區 - 請根據實際情況修改 ==========

# 數據路徑
BINARIZED_DATA_DIR="/path/to/binarized_data"
SAMPLES_CSV="/path/to/finetune_data/samples.csv"
ID2CHAR_JSON="/path/to/finetune_data/id2char.json"

# 輸出目錄
TRAIN_DATA_DIR="./train_data"
OUTPUT_MODEL_DIR="./output/PP-OCRv5_mobile_rec_finetune"

# 預訓練模型路徑
PRETRAINED_MODEL="./pretrained/PP-OCRv5_mobile_rec_pretrained"

# 訓練參數
BATCH_SIZE=64
LEARNING_RATE=0.00005
EPOCH_NUM=50

# GPU 設置
USE_GPU=true
GPU_ID="0"

# =================================================

echo ""
echo "步驟 1: 數據格式轉換"
echo "========================================"

python convert_to_paddleocr_format.py

if [ ! -f "$TRAIN_DATA_DIR/train_list.txt" ]; then
    echo "錯誤: 未生成 train_list.txt"
    exit 1
fi

echo "✓ 數據轉換完成"

echo ""
echo "步驟 2: 下載預訓練模型（如果尚未下載）"
echo "========================================"

if [ ! -f "$PRETRAINED_MODEL.pdparams" ]; then
    echo "下載 PP-OCRv5 預訓練模型..."
    mkdir -p pretrained
    cd pretrained
    wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams
    cd ..
    echo "✓ 預訓練模型下載完成"
else
    echo "✓ 預訓練模型已存在"
fi

echo ""
echo "步驟 3: 創建訓練配置文件"
echo "========================================"

cat > config_finetune.yml <<EOF
Global:
  model_name: PP-OCRv5_mobile_rec
  debug: false
  use_gpu: $USE_GPU
  epoch_num: $EPOCH_NUM
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: $OUTPUT_MODEL_DIR
  save_epoch_step: 5
  eval_batch_step: [0, 500]
  cal_metric_during_train: true
  pretrained_model: $PRETRAINED_MODEL
  checkpoints:
  save_inference_dir:
  use_visualdl: false
  infer_img: doc/imgs_words/ch/word_1.jpg
  character_dict_path: $TRAIN_DATA_DIR/custom_dict.txt
  max_text_length: 25
  infer_mode: false
  use_space_char: false
  distributed: false
  save_res_path: ./output/rec/predicts.txt

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: $LEARNING_RATE
    warmup_epoch: 2
  regularizer:
    name: L2
    factor: 3.0e-05

Architecture:
  model_type: rec
  algorithm: SVTR_LCNet
  Transform:
  Backbone:
    name: PPLCNetV3
    scale: 0.95
  Head:
    name: MultiHead
    head_list:
      - CTCHead:
          Neck:
            name: svtr
            dims: 120
            depth: 2
            hidden_dims: 120
            kernel_size: [1, 3]
            use_guide: True
          Head:
            fc_decay: 0.00001
      - NRTRHead:
          nrtr_dim: 384
          max_text_length: 25

Loss:
  name: MultiLoss
  loss_config_list:
    - CTCLoss:
    - NRTRLoss:

PostProcess:
  name: CTCLabelDecode

Metric:
  name: RecMetric
  main_indicator: acc

Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./
    label_file_list:
      - $TRAIN_DATA_DIR/train_list.txt
    ratio_list: [1.0]
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - RecAug:
      - MultiLabelEncode:
          gtc_encode: NRTRLabelEncode
      - RecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys:
            - image
            - label_ctc
            - label_gtc
            - length
            - valid_ratio
  loader:
    shuffle: true
    batch_size_per_card: $BATCH_SIZE
    drop_last: true
    num_workers: 4

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ./
    label_file_list:
      - $TRAIN_DATA_DIR/val_list.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - MultiLabelEncode:
          gtc_encode: NRTRLabelEncode
      - RecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys:
            - image
            - label_ctc
            - label_gtc
            - length
            - valid_ratio
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: $BATCH_SIZE
    num_workers: 4
EOF

echo "✓ 配置文件已生成: config_finetune.yml"

echo ""
echo "步驟 4: 開始訓練"
echo "========================================"

if [ "$USE_GPU" = true ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    echo "使用 GPU: $GPU_ID"
else
    echo "使用 CPU 訓練"
fi

python tools/train.py -c config_finetune.yml

echo ""
echo "========================================"
echo "訓練完成！"
echo "========================================"
echo "模型保存在: $OUTPUT_MODEL_DIR"
echo ""
echo "下一步："
echo "  1. 查看訓練日誌: $OUTPUT_MODEL_DIR/train.log"
echo "  2. 導出推理模型: python tools/export_model.py ..."
echo "  3. 測試模型效果"
echo "========================================"
