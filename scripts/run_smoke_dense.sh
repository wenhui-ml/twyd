#!/usr/bin/env bash
# Dense-LRC smoke test on ms-swift (single card, 3 steps).
# Student=Teacher=Llama-3.2-1B-Instruct (teacher resolved via deepcopy to save VRAM).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export TWYD_TEACHER=/home/chenwh/models/Llama-3.2-1B-Instruct
export TWYD_R=1024
export TWYD_STAGE=pt
export TWYD_WARM_START=slice
export TWYD_KL_TEMPERATURE=40.0
export TWYD_KL_WEIGHT=1.0
export TWYD_NTP_WEIGHT=1.0
export TWYD_AUX_WEIGHT=0.2
export TWYD_USE_AUX=true
export TWYD_TARGET_MODULES=gate_proj,up_proj,down_proj

/home/chenwh/miniconda3/envs/twyd/bin/swift pt \
  --model /home/chenwh/models/Llama-3.2-1B-Instruct \
  --dataset /home/chenwh/Papers/twyd/data/smoke_pt.jsonl \
  --tuner_type dense_lrc \
  --loss_type lrc_distill \
  --external_plugins twyd.plugin \
  --train_type full \
  --torch_dtype bfloat16 \
  --max_length 256 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.0 \
  --max_steps 3 \
  --save_steps 3 \
  --logging_steps 1 \
  --dataloader_num_workers 0 \
  "$@"
