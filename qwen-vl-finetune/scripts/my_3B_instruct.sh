#!/bin/bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation

# ======================
# Distributed Configuration
# ======================
# MASTER_ADDR="127.0.0.1"                     # [Required] Master node IP for multi-GPU training
# MASTER_PORT=$(shuf -i 20000-29999 -n 1)     # Random port to avoid conflicts
# NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Automatically detects available GPUs

# ======================
# Path Configuration
# ======================
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"  # [ModelArguments] Pretrained model path
OUTPUT_DIR="./checkpoints"                   # Directory for saving checkpoints
CACHE_DIR="./cache"                          # [TrainingArguments] Cache directory for models

# ======================
# Model Configuration
# ======================
DATASETS="test_dataset%100"                  # [DataArguments] Dataset with sampling rate

# ======================
# Image and Video Size Configuration
# ======================
IMAGE_MAX_PIXELS=451584 # $((576*28*28))  # 计算结果：451584
IMAGE_MIN_PIXELS=12544 # $((16*28*28))   # 计算结果：12544
VIDEO_MAX_PIXELS=1284608 # $((1664*28*28)) # 计算结果：1284608
VIDEO_MIN_PIXELS=197120 # $((256*28*28))  # 计算结果：197120

# ======================
# Training Hyperparameters
# ======================
torchrun --nproc_per_node=$NPROC_PER_NODE \
         qwen-vl-finetune/qwenvl/train/train_qwen.py \
         --model_name_or_path $MODEL_PATH \
         --tune_mm_llm True \
         --tune_mm_vision False \
         --tune_mm_mlp False \
         --dataset_use $DATASETS \
         --output_dir $OUTPUT_DIR \
         --cache_dir $CACHE_DIR \
         --bf16 \
         --per_device_train_batch_size 4 \
         --gradient_accumulation_steps 4 \
         --learning_rate 2e-7 \
         --mm_projector_lr 1e-5 \
         --vision_tower_lr 1e-6 \
         --optim adamw_torch \
         --model_max_length 4096 \
         --data_flatten True \
         --data_packing True \
         --max_pixels $IMAGE_MAX_PIXELS \
         --min_pixels $IMAGE_MIN_PIXELS \
         --video_fps 2 \
         --video_max_frames 8 \
         --video_min_frames 4 \
         --video_max_pixels $VIDEO_MAX_PIXELS \
         --video_min_pixels $VIDEO_MIN_PIXELS \
         --num_train_epochs 3 \
         --warmup_ratio 0.03 \
         --lr_scheduler_type "cosine" \
         --weight_decay 0.01 \
         --logging_steps 10 \
         --save_steps 500 \
         --save_total_limit 3 \
         --lora_enable True \
         --lora_r 8 \
         --lora_alpha 16 \
         --lora_dropout 0.0 \
         --deepspeed /root/autodl-tmp/Qwen3-VL/qwen-vl-finetune/scripts/zero2.json