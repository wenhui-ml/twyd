#!/usr/bin/env bash
# ============================================================================
# TWYD: LoRA-Dense-LRC single-GPU PT on 7900 XTX (24GB)
# Teacher: Llama-3.2-3B-Instruct (frozen, KL + SVD basis)
# Student: width-r LRC student (r=1024), d_ff inherited from teacher
# Tuner:   dense_lrc with LoRA (lora_rank=128) -> only LoRA A/B trained
# Optim:   Adafactor (no full momentum, ~1/8 optimizer VRAM vs AdamW)
# ============================================================================
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export TWYD_TEACHER=/home/chenwh/models/Llama-3.2-3B-Instruct
export TWYD_R=1024
export TWYD_LORA_RANK=128
export TWYD_LORA_ALPHA=16
export TWYD_STAGE=pt
export TWYD_WARM_START=slice
export TWYD_KL_TEMPERATURE=40.0
export TWYD_AUX_WEIGHT=0.2
export TWYD_TARGET_MODULES="gate_proj,up_proj,down_proj"

DATASET=${1:-"/home/chenwh/Papers/twyd/data/smoke_pt.jsonl"}
OUTPUT_DIR=${2:-"./output/llama3-3b-dense-lora-pt"}

/home/chenwh/miniconda3/envs/twyd/bin/swift pt \
  --model /home/chenwh/models/Llama-3.2-3B-Instruct \
  --dataset "$DATASET" \
  --tuner_type dense_lrc \
  --loss_type lrc_distill \
  --external_plugins /home/chenwh/Papers/twyd/twyd_plugin.py \
  --optim adafactor \
  --torch_dtype bfloat16 \
  --max_length 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --gradient_checkpointing true \
  --learning_rate 1e-3 \
  --warmup_ratio 0.005 \
  --max_steps 500 \
  --save_steps 500 \
  --logging_steps 10 \
  --output_dir "$OUTPUT_DIR" \
  --dataloader_num_workers 2 \
  --remove_unused_columns false

echo "=== Done. Checkpoint at: $OUTPUT_DIR ==="