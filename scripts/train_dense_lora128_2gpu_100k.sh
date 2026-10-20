#!/bin/bash
# Dense-LRC-LoRA SFT: 2x 7900XTX, ultrachat_200k 前10万条, 1 epoch
# 策略: dense分支 + 全层 LoRA128 (lora_mode=dense, 不设 lora_layers)
# 根因修复: TWYD_LORA_RANK=128 (之前=0导致零可训参数->优化器空组崩溃)

export CUDA_VISIBLE_DEVICES=0,1
export TWYD_TEACHER=/home/chenwh/models/Llama-3.2-3B-Instruct
export TWYD_R=512
export TWYD_STAGE=sft
export TWYD_WARM_START=slice
export TWYD_T=40.0
export TWYD_LAM=0.2
export TWYD_TARGET_MODULES='gate_proj,up_proj,down_proj'
export TWYD_LORA_RANK=128
export TWYD_LORA_ALPHA=16.0
export TWYD_LORA_MODE=dense
export TWYD_LORA_LAYERS=
export PYTORCH_ALLOC_CONF=expandable_segments:True

OUTPUT_DIR=/home/chenwh/Papers/twyd/output/dense-lora128-2gpu-100k

mkdir -p $OUTPUT_DIR

echo "=============================="
echo "Dense-LRC-LoRA SFT (2x 7900XTX)"
echo "分支: dense + 全层LoRA128"
echo "数据: swift/ultrachat_200k#100000, 1 epoch, ctx1024"
echo "模型: Llama-3.2-1B student(r=512) + Llama-3.2-3B teacher"
echo "输出: $OUTPUT_DIR"
echo "=============================="

/home/chenwh/miniconda3/envs/twyd/bin/swift sft \
    --model /home/chenwh/models/Llama-3.2-1B-Instruct \
    --dataset swift/ultrachat_200k#100000 \
    --tuner_type dense_lrc \
    --loss_type lrc_distill \
    --external_plugins /home/chenwh/Papers/twyd/twyd_plugin.py \
    --torch_dtype bfloat16 \
    --max_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --num_train_epochs 1 \
    --save_strategy epoch \
    --logging_steps 10 \
    --optim adamw_torch \
    --gradient_checkpointing true \
    --output_dir $OUTPUT_DIR \
    --dataloader_num_workers 0 \
    --remove_unused_columns false \
    2>&1 | tee $OUTPUT_DIR/training.log

echo "=============================="
echo "训练完成，检查输出目录"
echo "=============================="
ls -lh $OUTPUT_DIR/
find $OUTPUT_DIR -name "checkpoint-*" -type d 2>/dev/null