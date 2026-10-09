#!/usr/bin/env bash
# Smoke: Qwen2.5-0.5B teacher -> dense student (ModelScope source, avoids HF xet CDN).
set -euo pipefail
TEACHER="${TEACHER:-Qwen/Qwen2.5-0.5B}"
OUT="${OUT:-/home/chenwh/Papers/twyd/outputs/smoke-qwen05b}"
PY="${PY:-/home/chenwh/miniconda3/envs/swift/bin/python}"
export HF_HUB_OFFLINE=0
exec "$PY" /home/chenwh/Papers/twyd/scripts/train_dense_standalone.py \
  --teacher "$TEACHER" --output "$OUT" \
  --student-hidden 448 --seq-len 256 --batch 1 --grad-accum 4 \
  --steps 100 --lr 1e-4 --temperature 40.0 --device cuda:0
