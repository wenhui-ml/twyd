#!/usr/bin/env bash
# Smoke: Qwen2.5-0.5B teacher -> dense student (ModelScope source, avoids HF xet CDN).
set -euo pipefail
TEACHER="${TEACHER:-Qwen/Qwen2.5-0.5B}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$HERE/outputs/smoke-qwen05b}"
PY="${PY:-python}"
export HF_HUB_OFFLINE=0
exec "$PY" "$HERE/scripts/train_dense_standalone.py" \
  --teacher "$TEACHER" --output "$OUT" \
  --student-hidden 448 --seq-len 256 --batch 1 --grad-accum 4 \
  --steps 100 --lr 1e-4 --temperature 40.0 --device cuda:0
