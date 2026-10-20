#!/usr/bin/env bash
# CORE-LR pilot: TWYD-native parameter-efficient variant.
#
#   W = T Z_col^T + U_perp (A B^T)^T,  Z_col frozen, only A/B train.
#   All trainable coordinates live in col(T)^perp (the stranded capacity).
#   At init B=0 => exactly plain LRC; after merge => one dense d_ff x r weight,
#   zero inference cost.
#
# The step-0 gate is enforced: if the warm start does not track the teacher,
# this script exits before spending GPU hours.
#
# Usage:
#   K=256 bash scripts/train_core_lr_pilot.sh
#   K=128 LR=2e-4 MAX_STEPS=3000 bash scripts/train_core_lr_pilot.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
TEACHER=${TEACHER:-/home/chenwh/models/Llama-3.2-3B-Instruct}
MODEL=${MODEL:-/home/chenwh/models/Llama-3.2-3B-Instruct}
R=${R:-1536}
K=${K:-256}
ALPHA=${ALPHA:-$((2 * K))}
LR=${LR:-1e-4}
ATTN_IMPL=${ATTN_IMPL:-sdpa}
MAX_LENGTH=${MAX_LENGTH:-1024}
MAX_STEPS=${MAX_STEPS:-2000}
DATASET=${DATASET:-swift/ultrachat_200k#100000}
SWIFT=${SWIFT:-/home/chenwh/miniconda3/envs/twyd/bin/swift}
OUT=${OUT:-/home/chenwh/Papers/twyd/output/core-lr-r${R}-k${K}}

export TWYD_TEACHER="$TEACHER"
export TWYD_R="$R"
export TWYD_STAGE=sft
export TWYD_WARM_START=slice
export TWYD_T=40.0
export TWYD_LAM=0.2
export TWYD_USE_AUX=true
export TWYD_TARGET_MODULES='gate_proj,up_proj,down_proj'
export TWYD_LORA_RANK="$K"
export TWYD_LORA_ALPHA="$ALPHA"
export TWYD_LORA_MODE=core
export TWYD_FREEZE_COL=true
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUT"
echo "=============================="
echo "CORE-LR pilot: r=$R k=$K alpha=$ALPHA lr=$LR steps=$MAX_STEPS"
echo "teacher=$TEACHER model=$MODEL dataset=$DATASET"
echo "out=$OUT"
echo "=============================="

# Step-0 gate: do not train on a random warm start.
/home/chenwh/miniconda3/envs/twyd/bin/python -u scripts/check_init.py \
  --teacher "$TEACHER" --model "$MODEL" --r "$R" --branch core --seq-len 64 \
  || { echo "[gate] FAILED: fix the LRC warm start before training"; exit 1; }

"$SWIFT" sft \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --tuner_type core_lrc \
  --loss_type lrc_distill \
  --attn_impl "$ATTN_IMPL" \
  --external_plugins /home/chenwh/Papers/twyd/twyd_plugin.py \
  --torch_dtype bfloat16 \
  --max_length "$MAX_LENGTH" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate "$LR" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --max_steps "$MAX_STEPS" \
  --save_steps "$MAX_STEPS" \
  --logging_steps 10 \
  --optim adamw_torch \
  --gradient_checkpointing true \
  --output_dir "$OUT" \
  --dataloader_num_workers 0 \
  --remove_unused_columns false \
  2>&1 | tee "$OUT/training.log"

echo "=============================="
echo "done: $OUT"
echo "next: compare CE/acc + Avg9 against plain LRC and dense at matched tokens"
echo "=============================="
