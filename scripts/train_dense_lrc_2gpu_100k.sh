#!/bin/bash
# Dense-LRC 分布式训练脚本：2张卡 7900 XTX，10万条数据，1个epoch
# 使用 Llama-3.2-1B-Instruct 作为学生，Llama-3.2-3B-Instruct 作为教师

export CUDA_VISIBLE_DEVICES=0,1
export TWYD_TEACHER=/home/chenwh/models/Llama-3.2-3B-Instruct
export TWYD_R=1024
export TWYD_STAGE=sft
export TWYD_WARM_START=slice
export TWYD_T=40.0
export TWYD_LAM=0.2
export TWYD_TARGET_MODULES='gate_proj,up_proj,down_proj'
export TWYD_LORA_RANK=128
export TWYD_LORA_ALPHA=16.0
export TWYD_LORA_MODE=dense
export TWYD_OPTIM=adafactor
export PYTORCH_ALLOC_CONF=expandable_segments:True
ATTN_IMPL=${ATTN_IMPL:-sdpa}
MAX_LENGTH=${MAX_LENGTH:-1024}
SWIFT=${SWIFT:-/home/chenwh/miniconda3/envs/twyd/bin/swift}

OUTPUT_DIR=/home/chenwh/Papers/twyd/output/dense-lora128-r1024-2gpu-100k

mkdir -p $OUTPUT_DIR

echo "=============================="
echo "开始 LoRA-Dense-LRC 分布式 SFT 训练 (rank=128)"
echo "GPU: 0, 1"
echo "数据: 10万条 (ultrachat_200k + alpaca-cleaned + alpaca-gpt4-data-zh)"
echo "模型: Llama-3.2-1B-Instruct (student) + Llama-3.2-3B-Instruct (teacher)"
echo "输出: $OUTPUT_DIR"
echo "=============================="

"$SWIFT" sft \
    --model /home/chenwh/models/Llama-3.2-1B-Instruct \
    --dataset swift/ultrachat_200k#100000 \
    --tuner_type dense_lrc \
    --loss_type lrc_distill \
    --attn_impl "$ATTN_IMPL" \
    --external_plugins /home/chenwh/Papers/twyd/twyd_plugin.py \
    --torch_dtype bfloat16 \
    --max_length "$MAX_LENGTH" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --num_train_epochs 1 \
    --save_strategy epoch \
    --save_steps 3125 \
    --logging_steps 10 \
    --optim adafactor \
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
