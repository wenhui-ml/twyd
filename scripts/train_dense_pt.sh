#!/usr/bin/env bash
# Dense-LRC-PT on ms-swift (LRC Mixed pretraining distillation).
# Data: LRC-PT packed chunks (labels==input_ids, see twyd/data_pt.py).
# Usage: STUDENT=<hf-id> TEACHER=<hf-id> PT_DATA=<packed-jsonl> bash scripts/train_dense_pt.sh
set -euo pipefail
STUDENT="${STUDENT:-Qwen/Qwen2.5-3B-Instruct}"
TEACHER="${TEACHER:-Qwen/Qwen2.5-3B-Instruct}"
PT_DATA="${PT_DATA:-./data/pt_chunks.jsonl}"
R="${STUDENT_HIDDEN:-1200}"
ATTN_IMPL=${ATTN_IMPL:-sdpa}

swift pt \
  --model "${STUDENT}" \
  --dataset "${PT_DATA}" \
  --tuner_type dense_lrc \
  --loss_type lrc_distill \
  --attn_impl "$ATTN_IMPL" \
  --external_plugins twyd.plugin \
  --tuner_kwargs "{\"teacher_model\":\"${TEACHER}\",\"student_hidden\":${R},\"stage\":\"pt\",\"kl_temperature\":40.0,\"aux_weight\":0.2}" \
  --train_type full \
  --torch_dtype bfloat16 \
  --max_length 1024 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.005 \
  --max_steps 20000 \
  --save_steps 2000 \
  --logging_steps 50 \
  "$@"
