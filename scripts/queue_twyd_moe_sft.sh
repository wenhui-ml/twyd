#!/usr/bin/env bash
# Phase 2 Plan A: Short 0.5-epoch pure-CE SFT wrap-up on top of TWYD-MoE v1 (checkpoint-1525).
#
# Motivation:
#   - pt-moe-v1-200M broke the 0.504-0.508 MMLU wall (MMLU 0.5138, +0.96 pt, p=3.85e-7;
#     Pool 0.6008, new all-time high), while raw PT caused a minor -0.47 pt format drift
#     on WinoGrande/BoolQ relative to its SFT'd mother (SFTdense2.1).
#   - Freezing the already-SFT'd 1.61B shared mother and gently SFT'ing only the 906M
#     routed experts (0.5 epoch = 1585 steps, LR=3e-5 cosine -> 0, pure CE) aligns the
#     new MoE knowledge with instruction/Q&A formatting without erasing MMLU facts.
#
# Usage:
#   bash scripts/queue_twyd_moe_sft.sh
set -u
cd "$(dirname "$0")/.."
ROOT=$PWD

PT_INIT=${PT_INIT:-$ROOT/output/product/pt-moe-v1-200M/v0-20261119-161728/checkpoint-1525}
SFT_DATA=${SFT_DATA:-$ROOT/data/sft_paper8.jsonl}
OUT=${OUT:-$ROOT/output/product/sft-moe-v1-halfep}
EVAL_TAG=${EVAL_TAG:-paper_SFTmoe_v1}

LR=${LR:-3e-5}
CLIP=${CLIP:-50.0}
DS=${DS:-zero2}
SEQ=${SEQ:-2048}
BS=${BS:-1}
GAS=${GAS:-32}
NPROC=${NPROC:-2}
MAX_STEPS=${MAX_STEPS:-1585}
SAVE_STEPS=${SAVE_STEPS:-$MAX_STEPS}

if [ ! -d "$PT_INIT" ]; then
  echo "[sft-moe-v1] ERROR: missing PT_INIT $PT_INIT" >&2
  exit 1
fi
if [ ! -f "$SFT_DATA" ]; then
  echo "[sft-moe-v1] ERROR: missing SFT_DATA $SFT_DATA" >&2
  exit 1
fi

echo "[sft-moe-v1] start $(date '+%F %H:%M') pt_init=$PT_INIT steps=$MAX_STEPS lr=$LR bs=$BS gas=$GAS"
rm -rf "$OUT"

env PYTORCH_HIP_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=${GPU:-0,1} NPROC="$NPROC" DS="$DS" \
  BRANCH=moe STAGE=sft LOSS_TYPE= WARM_START=pt_ckpt PT_CKPT="$PT_INIT" \
  MODEL=/home/chenwh/models/LRC-1.5B-Base MODEL_TYPE=llama TEMPLATE=llama3_2 \
  TEACHER=/home/chenwh/models/Llama-3.2-3B-Instruct \
  DATASET="$SFT_DATA" TRUNC=delete DATA_CACHE=true DATA_PROC=12 DATALOADER_WORKERS=4 \
  MAX_LENGTH="$SEQ" BS="$BS" GRAD_ACCUM="$GAS" \
  MAX_STEPS="$MAX_STEPS" SAVE_STEPS="$SAVE_STEPS" \
  LR="$LR" CLIP="$CLIP" SEED=42 GATE=0 USE_AUX=0 \
  TWYD_MOE_LB_COEF=0.02 TWYD_MOE_Z_COEF=0.001 \
  TWYD_MOE_ROUTER_LR_SCALE=1.0 TWYD_MOE_UNFREEZE_SHARED=0 \
  OUT="$OUT" \
  bash "$ROOT/scripts/train_core_lr_pilot.sh" > "$ROOT/output/sft_moe_v1.log" 2>&1
RC=$?
echo "[sft-moe-v1] SFT rc=$RC $(date '+%H:%M')"

CK=$(ls -d "$OUT"/v*/checkpoint-* 2>/dev/null | sort -V | tail -1)
echo "[sft-moe-v1] ckpt=$CK"
if [ "$RC" -eq 0 ] && [ -n "$CK" ]; then
  find "$OUT" -maxdepth 3 -type d -name "global_step*" -exec rm -rf {} + 2>/dev/null || true
  env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PRESET=paper GPU=0 BATCH=16 \
    bash "$ROOT/scripts/run_lm_eval.sh" "$CK" "$EVAL_TAG" > "$ROOT/output/ev_${EVAL_TAG}.log" 2>&1
  echo "[sft-moe-v1] eval rc=$? $(date '+%H:%M')"
fi
echo "[sft-moe-v1] DONE $(date '+%F %H:%M')"
