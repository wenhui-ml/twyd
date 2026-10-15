#!/usr/bin/env bash
# Smoke: local Qwen3-VL text tower -> dense student. Zero download, runs tonight.
set -euo pipefail
TEACHER="${TEACHER:-/home/chenwh/models/Qwen3-VL-2B-Instruct}"
OUT="${OUT:-/home/chenwh/Papers/twyd/outputs/smoke-qwen3vl-text}"
PY="${PY:-/home/chenwh/miniconda3/envs/swift/bin/python}"
exec "$PY" /home/chenwh/Papers/twyd/scripts/train_dense_standalone.py \
  --teacher "$TEACHER" --output "$OUT" \
  --student-hidden 1024 --seq-len 256 --batch 1 --grad-accum 4 \
  --steps 200 --lr 1e-4 --temperature 40.0 --device cuda:0
