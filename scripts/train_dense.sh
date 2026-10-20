#!/usr/bin/env bash
# Dense-LRC on ms-swift (pretrain distillation).
# Usage: STUDENT=<hf-id> TEACHER=<hf-id> DATASET=<id-or-path> bash scripts/train_dense.sh
set -euo pipefail
STUDENT="${STUDENT:-Qwen/Qwen2.5-3B-Instruct}"
TEACHER="${TEACHER:-Qwen/Qwen2.5-3B-Instruct}"
DATASET="${DATASET:-AI-ModelScope/alpaca-gpt4-data-zh#5000}"
R="${STUDENT_HIDDEN:-1200}"
ATTN_IMPL=${ATTN_IMPL:-sdpa}

swift pt \
  --model "${STUDENT}" \
  --dataset "${DATASET}" \
  --tuner_type dense_lrc \
  --loss_type lrc_distill \
  --attn_impl "$ATTN_IMPL" \
  --external_plugins twyd.plugin \
  --tuner_kwargs "{\"teacher_model\":\"${TEACHER}\",\"student_hidden\":${R},\"kl_temperature\":40.0,\"aux_weight\":0.2}" \
  --train_type full \
  --torch_dtype bfloat16 \
  --max_length 2048 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.005 \
  --max_steps 20000 \
  --save_steps 2000 \
  --logging_steps 50 \
  "$@"
