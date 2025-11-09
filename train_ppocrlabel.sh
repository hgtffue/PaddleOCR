#!/bin/bash
# PaddleOCR PPOCRLabel 數據 Fine-tune 一鍵訓練腳本

set -e  # 遇到錯誤立即退出

echo "========================================"
echo "PaddleOCR Fine-tune 訓練流程"
echo "========================================"

# ========== 配置區 - 請根據實際情況修改 ==========

# 訓練模式選擇
TRAIN_DETECTION=true    # 是否訓練檢測模型
TRAIN_RECOGNITION=true  # 是否訓練識別模型

# 設備配置
USE_GPU=true
GPU_ID="0"              # 單卡：GPU_ID="0"，多卡：GPU_ID="0,1,2,3"

# 識別模型配置
REC_BATCH_SIZE=128      # 根據顯存調整（V100: 256, 2080Ti: 128, 1080Ti: 64）
REC_LEARNING_RATE=0.0001
REC_EPOCH_NUM=100

# 檢測模型配置
DET_BATCH_SIZE=8        # 檢測模型顯存占用較大
DET_LEARNING_RATE=0.001
DET_EPOCH_NUM=200

# 輸出目錄
OUTPUT_REC_DIR="./output/rec_ppocr_mobile"
OUTPUT_DET_DIR="./output/det_ppocr_v4"

# 預訓練模型路徑
PRETRAINED_REC="./pretrained/PP-OCRv5_mobile_rec_pretrained"
PRETRAINED_DET="./pretrained/ch_PP-OCRv4_det_train/best_accuracy"

# =================================================

# 顏色輸出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_step() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}"
}

print_warning() {
    echo -e "${YELLOW}警告: $1${NC}"
}

print_error() {
    echo -e "${RED}錯誤: $1${NC}"
}

# 檢查依賴
check_dependencies() {
    print_step "檢查依賴"

    # 檢查 Python
    if ! command -v python &> /dev/null; then
        print_error "未找到 Python"
        exit 1
    fi
    echo "✓ Python: $(python --version)"

    # 檢查 PaddlePaddle
    if ! python -c "import paddle; print(paddle.__version__)" &> /dev/null; then
        print_error "未安裝 PaddlePaddle"
        exit 1
    fi
    echo "✓ PaddlePaddle: $(python -c 'import paddle; print(paddle.__version__)')"

    # 檢查必要的 Python 包
    python -c "import cv2, numpy, tqdm" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_error "缺少必要的 Python 包（cv2, numpy, tqdm）"
        exit 1
    fi
    echo "✓ 依賴包已安裝"
}

# 步驟 1：數據準備
prepare_data() {
    print_step "步驟 1/5: 數據準備"

    if [ ! -f "train_data/dataset_stats.json" ]; then
        echo "運行數據準備腳本..."
        python prepare_ppocrlabel_data.py

        if [ $? -ne 0 ]; then
            print_error "數據準備失敗"
            exit 1
        fi
    else
        print_warning "數據已準備好，跳過此步驟"
        echo "如需重新準備數據，請刪除 train_data 目錄"
    fi

    # 顯示數據統計
    if [ -f "train_data/dataset_stats.json" ]; then
        echo ""
        echo "數據統計:"
        cat train_data/dataset_stats.json
    fi

    echo "✓ 數據準備完成"
}

# 步驟 2：下載預訓練模型
download_pretrained() {
    print_step "步驟 2/5: 下載預訓練模型"

    mkdir -p pretrained

    # 下載識別模型
    if [ "$TRAIN_RECOGNITION" = true ]; then
        if [ ! -f "${PRETRAINED_REC}.pdparams" ]; then
            echo "下載 PP-OCRv5 識別模型..."
            cd pretrained
            wget -q --show-progress https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams
            cd ..
            echo "✓ 識別模型下載完成"
        else
            echo "✓ 識別預訓練模型已存在"
        fi
    fi

    # 下載檢測模型
    if [ "$TRAIN_DETECTION" = true ]; then
        if [ ! -d "pretrained/ch_PP-OCRv4_det_train" ]; then
            echo "下載 PP-OCRv4 檢測模型..."
            cd pretrained
            wget -q --show-progress https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_train.tar
            tar -xf ch_PP-OCRv4_det_train.tar
            rm ch_PP-OCRv4_det_train.tar
            cd ..
            echo "✓ 檢測模型下載完成"
        else
            echo "✓ 檢測預訓練模型已存在"
        fi
    fi
}

# 步驟 3：訓練識別模型
train_recognition() {
    if [ "$TRAIN_RECOGNITION" != true ]; then
        return
    fi

    print_step "步驟 3/5: 訓練識別模型"

    # 設置 GPU
    if [ "$USE_GPU" = true ]; then
        export CUDA_VISIBLE_DEVICES=$GPU_ID
        echo "使用 GPU: $GPU_ID"
    else
        echo "使用 CPU 訓練（速度較慢）"
    fi

    # 計算卡數
    IFS=',' read -ra GPUS <<< "$GPU_ID"
    NUM_GPUS=${#GPUS[@]}

    # 調整多卡訓練的學習率
    ADJUSTED_REC_LR=$REC_LEARNING_RATE
    if [ $NUM_GPUS -gt 1 ]; then
        ADJUSTED_REC_LR=$(python -c "print($REC_LEARNING_RATE * $NUM_GPUS)")
        echo "多卡訓練，學習率調整為: $ADJUSTED_REC_LR"
    fi

    # 訓練命令
    TRAIN_CMD="python"
    if [ $NUM_GPUS -gt 1 ]; then
        TRAIN_CMD="python -m paddle.distributed.launch --gpus $GPU_ID"
    fi

    echo "開始訓練識別模型..."
    echo "配置: Batch Size=$REC_BATCH_SIZE, LR=$ADJUSTED_REC_LR, Epochs=$REC_EPOCH_NUM"

    $TRAIN_CMD tools/train.py \
        -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
        -o Global.pretrained_model=$PRETRAINED_REC \
           Global.character_dict_path=./train_data/rec_dict.txt \
           Global.save_model_dir=$OUTPUT_REC_DIR \
           Global.epoch_num=$REC_EPOCH_NUM \
           Train.dataset.data_dir=./ \
           Train.dataset.label_file_list=['./train_data/rec_train.txt'] \
           Train.loader.batch_size_per_card=$REC_BATCH_SIZE \
           Eval.dataset.data_dir=./ \
           Eval.dataset.label_file_list=['./train_data/rec_val.txt'] \
           Eval.loader.batch_size_per_card=$REC_BATCH_SIZE \
           Optimizer.lr.learning_rate=$ADJUSTED_REC_LR

    if [ $? -eq 0 ]; then
        echo "✓ 識別模型訓練完成"
    else
        print_error "識別模型訓練失敗"
        exit 1
    fi
}

# 步驟 4：訓練檢測模型
train_detection() {
    if [ "$TRAIN_DETECTION" != true ]; then
        return
    fi

    print_step "步驟 4/5: 訓練檢測模型"

    # 設置 GPU
    if [ "$USE_GPU" = true ]; then
        export CUDA_VISIBLE_DEVICES=$GPU_ID
        echo "使用 GPU: $GPU_ID"
    fi

    # 計算卡數
    IFS=',' read -ra GPUS <<< "$GPU_ID"
    NUM_GPUS=${#GPUS[@]}

    # 調整多卡訓練的學習率和 batch size
    ADJUSTED_DET_LR=$DET_LEARNING_RATE
    ADJUSTED_DET_BS=$DET_BATCH_SIZE
    if [ $NUM_GPUS -gt 1 ]; then
        ADJUSTED_DET_LR=$(python -c "print($DET_LEARNING_RATE * $NUM_GPUS)")
        ADJUSTED_DET_BS=$(python -c "print(int($DET_BATCH_SIZE * 1.5))")
        echo "多卡訓練，學習率調整為: $ADJUSTED_DET_LR, Batch Size 調整為: $ADJUSTED_DET_BS"
    fi

    # 訓練命令
    TRAIN_CMD="python"
    if [ $NUM_GPUS -gt 1 ]; then
        TRAIN_CMD="python -m paddle.distributed.launch --gpus $GPU_ID"
    fi

    echo "開始訓練檢測模型..."
    echo "配置: Batch Size=$ADJUSTED_DET_BS, LR=$ADJUSTED_DET_LR, Epochs=$DET_EPOCH_NUM"

    $TRAIN_CMD tools/train.py \
        -c configs/det/PP-OCRv4/ch_PP-OCRv4_det_student.yml \
        -o Global.pretrained_model=$PRETRAINED_DET \
           Global.save_model_dir=$OUTPUT_DET_DIR \
           Global.epoch_num=$DET_EPOCH_NUM \
           Train.dataset.data_dir=./ \
           Train.dataset.label_file_list=['./train_data/det_train.txt'] \
           Train.loader.batch_size_per_card=$ADJUSTED_DET_BS \
           Eval.dataset.data_dir=./ \
           Eval.dataset.label_file_list=['./train_data/det_val.txt'] \
           Optimizer.lr.learning_rate=$ADJUSTED_DET_LR

    if [ $? -eq 0 ]; then
        echo "✓ 檢測模型訓練完成"
    else
        print_error "檢測模型訓練失敗"
        exit 1
    fi
}

# 步驟 5：導出推理模型
export_models() {
    print_step "步驟 5/5: 導出推理模型"

    mkdir -p inference

    # 導出識別模型
    if [ "$TRAIN_RECOGNITION" = true ]; then
        if [ -f "$OUTPUT_REC_DIR/best_accuracy.pdparams" ]; then
            echo "導出識別模型..."
            python tools/export_model.py \
                -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
                -o Global.pretrained_model=$OUTPUT_REC_DIR/best_accuracy \
                   Global.character_dict_path=./train_data/rec_dict.txt \
                   Global.save_inference_dir=./inference/rec_model

            if [ $? -eq 0 ]; then
                echo "✓ 識別模型已導出到 ./inference/rec_model"
            else
                print_warning "識別模型導出失敗"
            fi
        else
            print_warning "未找到訓練好的識別模型"
        fi
    fi

    # 導出檢測模型
    if [ "$TRAIN_DETECTION" = true ]; then
        if [ -f "$OUTPUT_DET_DIR/best_accuracy.pdparams" ]; then
            echo "導出檢測模型..."
            python tools/export_model.py \
                -c configs/det/PP-OCRv4/ch_PP-OCRv4_det_student.yml \
                -o Global.pretrained_model=$OUTPUT_DET_DIR/best_accuracy \
                   Global.save_inference_dir=./inference/det_model

            if [ $? -eq 0 ]; then
                echo "✓ 檢測模型已導出到 ./inference/det_model"
            else
                print_warning "檢測模型導出失敗"
            fi
        else
            print_warning "未找到訓練好的檢測模型"
        fi
    fi
}

# 顯示總結
show_summary() {
    print_step "訓練完成！"

    echo ""
    echo "模型保存位置:"
    if [ "$TRAIN_RECOGNITION" = true ]; then
        echo "  - 識別模型: $OUTPUT_REC_DIR"
        echo "  - 識別推理模型: ./inference/rec_model"
    fi
    if [ "$TRAIN_DETECTION" = true ]; then
        echo "  - 檢測模型: $OUTPUT_DET_DIR"
        echo "  - 檢測推理模型: ./inference/det_model"
    fi

    echo ""
    echo "下一步："
    echo "  1. 查看訓練日誌和指標"
    if [ "$TRAIN_RECOGNITION" = true ]; then
        echo "     tail -100 $OUTPUT_REC_DIR/train.log"
    fi
    if [ "$TRAIN_DETECTION" = true ]; then
        echo "     tail -100 $OUTPUT_DET_DIR/train.log"
    fi

    echo ""
    echo "  2. 測試模型效果"
    echo "     創建 test_ocr.py 並運行（參考 PPOCRLABEL_FINETUNE_GUIDE.md）"

    echo ""
    echo "  3. 如果效果不理想："
    echo "     - 調整超參數（學習率、epoch 數）"
    echo "     - 增加數據量"
    echo "     - 檢查數據標註質量"
}

# 主流程
main() {
    echo "配置信息:"
    echo "  - 訓練識別模型: $TRAIN_RECOGNITION"
    echo "  - 訓練檢測模型: $TRAIN_DETECTION"
    echo "  - GPU: $USE_GPU (設備: $GPU_ID)"
    echo ""

    check_dependencies
    prepare_data
    download_pretrained
    train_recognition
    train_detection
    export_models
    show_summary
}

# 捕獲 Ctrl+C
trap 'echo -e "\n${RED}訓練已中斷${NC}"; exit 1' INT

# 運行主流程
main
