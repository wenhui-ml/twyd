#!/bin/bash
# TWYD 2.0 轻量验证：2x 7900 XTX (48G) 可跑
# 方向1: 只在浅层 L0-8 开 LoRA，深层冻结 plain（砍掉 ~65% 无效参数）
# 方向3: 补空间 LoRA 用 PT 级别 lr=1e-4 + cosine + T=40 KL + lam=0.2 aux
# 方向4: lora_mode=core -> 低秩版 CORE-LRC（正交无竞争，推理零开销合并）
set -e
export CUDA_VISIBLE_DEVICES=0,1
export TWYD_TEACHER=/home/chenwh/models/Llama-3.2-3B-Instruct
export TWYD_R=512
export TWYD_STAGE=sft
export TWYD_WARM_START=slice
export TWYD_T=40.0
export TWYD_LAM=0.2
export TWYD_TARGET_MODULES='gate_proj,up_proj,down_proj'
# --- TWYD 2.0 LoRA 开关 ---
export TWYD_LORA_RANK=128
export TWYD_LORA_ALPHA=16.0
export TWYD_LORA_MODE=core
export TWYD_LORA_LAYERS='0-8'
export PYTORCH_ALLOC_CONF=expandable_segments:True

OUTPUT_DIR=/home/chenwh/Papers/twyd/output/core-lora-2gpu-100k
mkdir -p $OUTPUT_DIR
echo "=============================="
echo "CORE-LoRA 2GPU SFT: 10万条 ultrachat_200k, 1 epoch"
echo "branch=core + lora_mode=core rank=128 layers=0-8 lr=1e-4 cosine"
echo "输出: $OUTPUT_DIR"
echo "=============================="
ATTN_IMPL=${ATTN_IMPL:-sdpa}
MAX_LENGTH=${MAX_LENGTH:-1024}
SWIFT=${SWIFT:-/home/chenwh/miniconda3/envs/twyd/bin/swift}

"$SWIFT" sft \
    --model /home/chenwh/models/Llama-3.2-1B-Instruct \
    --dataset swift/ultrachat_200k#100000 \
    --tuner_type core_lrc \
    --loss_type lrc_distill \
    --attn_impl "$ATTN_IMPL" \
    --external_plugins /home/chenwh/Papers/twyd/twyd_plugin.py \
    --torch_dtype bfloat16 \
    --max_length "$MAX_LENGTH" \
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
