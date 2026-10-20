#!/usr/bin/env bash
# Dense-LRC-SFT on ms-swift.
# Student init: merged PT ckpt (--model PT_CKPT or warm_start pt_ckpt).
# Teacher: standard open-source teacher (same id as PT teacher).
# Data: SFT chat dataset (prompt masked to -100 by template); KL masked too.
# Usage: PT_CKPT=<merged-pt> TEACHER=<hf-id> SFT_DATA=<id> bash scripts/train_dense_sft.sh
set -euo pipefail
PT_CKPT="${PT_CKPT:-./outputs/dense-pt}"
TEACHER="${TEACHER:-Qwen/Qwen2.5-3B-Instruct}"
SFT_DATA="${SFT_DATA:-AI-ModelScope/alpaca-gpt4-data-zh#5000}"
R="${STUDENT_HIDDEN:-1200}"
ATTN_IMPL=${ATTN_IMPL:-sdpa}

swift sft \
  --model "${PT_CKPT}" \
  --dataset "${SFT_DATA}" \
  --tuner_type dense_lrc \
  --loss_type lrc_distill \
  --attn_impl "$ATTN_IMPL" \
  --external_plugins twyd.plugin \
  --tuner_kwargs "{\"teacher_model\":\"${TEACHER}\",\"student_hidden\":${R},\"stage\":\"sft\",\"warm_start\":\"pt_ckpt\",\"pt_ckpt\":\"${PT_CKPT}\",\"kl_temperature\":40.0,\"aux_weight\":0.2}" \
  --train_type full \
  --torch_dtype bfloat16 \
  --max_length 2048 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-5 \
  --warmup_ratio 0.03 \
  --max_steps 3000 \
  --save_steps 500 \
  --logging_steps 20 \
  "$@"
