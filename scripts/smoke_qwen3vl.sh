#!/usr/bin/env bash
# Smoke: local Qwen3-VL text tower -> dense student. Zero download, runs tonight.
set -euo pipefail
TEACHER="${TEACHER:-${MODELS_DIR:-/home/chenwh/models}/Qwen3-VL-2B-Instruct}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$HERE/outputs/smoke-qwen3vl-text}"
PY="${PY:-python}"
exec "$PY" "$HERE/scripts/train_dense_standalone.py" \
  --teacher "$TEACHER" --output "$OUT" \
  --student-hidden 1024 --seq-len 256 --batch 1 --grad-accum 4 \
  --steps 200 --lr 1e-4 --temperature 40.0 --device cuda:0
