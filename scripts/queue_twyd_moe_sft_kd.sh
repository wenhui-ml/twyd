#!/usr/bin/env bash
# Phase 2 Option 1: KD-Anchored 0.5-epoch SFT on top of TWYD-MoE v1 (checkpoint-1525, MMLU 0.5138).
#
# Why KD-Anchored SFT vs pure-CE SFT (SFT-MoE-v1):
#   - Pure-CE SFT lifted Beh_pool +0.32 pt (Pool 0.6012 new high), but let MMLU drift
#     -0.20 pt (0.5138 -> 0.5118) and only recovered half of WinoGrande.
#   - Adding lightweight Top-256 KL anchoring against Llama-3.2-3B-Instruct (T=1.0, K=256,
#     wKL=2.0 -> ~1:2 KL:CE ratio on assistant tokens) + gentler LR=2e-5 preserves the
#     0.5138 OpenStax/Stanford textbook knowledge while aligning instruction/Q&A format.
#   - Uses Fused Chunked Loss + PyTorch CPU ZeRO-2 Offload (BS=2, GAS=16 -> ~16.7 GiB
#     peak VRAM, zero GPU0 GTT spill, 100% dual-GPU TDP).
#
# Usage:
#   bash scripts/queue_twyd_moe_sft_kd.sh
set -u
cd "$(dirname "$0")/.."
ROOT=$PWD

PT_INIT=${PT_INIT:-$ROOT/output/product/pt-moe-v1-200M/v0-20261119-161728/checkpoint-1525}
SFT_DATA=${SFT_DATA:-$ROOT/data/sft_paper8.jsonl}
OUT=${OUT:-$ROOT/output/product/sft-moe-v1-kd-halfep}
EVAL_TAG=${EVAL_TAG:-paper_SFTmoe_v1_kd}

LR=${LR:-2e-5}
CLIP=${CLIP:-50.0}
DS=${DS:-$ROOT/configs/ds_zero2_offload_torch.json}
SEQ=${SEQ:-2048}
BS=${BS:-1}
GAS=${GAS:-32}
NPROC=${NPROC:-2}
OMP_THREADS=${OMP_THREADS:-8}
MAX_STEPS=${MAX_STEPS:-1585}
SAVE_STEPS=${SAVE_STEPS:-$MAX_STEPS}
WKL=${TWYD_WKL:-2.0}

if [ ! -d "$PT_INIT" ]; then
  echo "[sft-moe-v1-kd] ERROR: missing PT_INIT $PT_INIT" >&2
  exit 1
fi
if [ ! -f "$SFT_DATA" ]; then
  echo "[sft-moe-v1-kd] ERROR: missing SFT_DATA $SFT_DATA" >&2
  exit 1
fi

echo "[sft-moe-v1-kd] start $(date '+%F %H:%M') pt_init=$PT_INIT steps=$MAX_STEPS lr=$LR wKL=$WKL bs=$BS gas=$GAS"
rm -rf "$OUT"

env OMP_NUM_THREADS="$OMP_THREADS" \
  PYTORCH_HIP_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=${GPU:-0,1} NPROC="$NPROC" DS="$DS" \
  BRANCH=moe STAGE=sft LOSS_TYPE=lrc_distill WARM_START=pt_ckpt PT_CKPT="$PT_INIT" \
  MODEL=/home/chenwh/models/LRC-1.5B-Base MODEL_TYPE=llama TEMPLATE=llama3_2 \
  TEACHER=/home/chenwh/models/Llama-3.2-3B-Instruct \
  DATASET="$SFT_DATA" TRUNC=delete DATA_CACHE=true DATA_PROC=12 DATALOADER_WORKERS=4 \
  MAX_LENGTH="$SEQ" BS="$BS" GRAD_ACCUM="$GAS" \
  MAX_STEPS="$MAX_STEPS" SAVE_STEPS="$SAVE_STEPS" \
  LR="$LR" CLIP="$CLIP" SEED=42 GATE=0 USE_AUX=0 \
  TWYD_T=1.0 TWYD_WKL="$WKL" TWYD_KL_TOPK=256 TWYD_RHO_KEEP=1.0 \
  TWYD_MOE_LB_COEF=0.02 TWYD_MOE_Z_COEF=0.001 \
  TWYD_MOE_ROUTER_LR_SCALE=1.0 TWYD_MOE_UNFREEZE_SHARED=0 \
  OUT="$OUT" \
  bash "$ROOT/scripts/train_core_lr_pilot.sh" > "$ROOT/output/sft_moe_v1_kd.log" 2>&1
RC=$?
echo "[sft-moe-v1-kd] SFT rc=$RC $(date '+%H:%M')"

CK=$(ls -d "$OUT"/v*/checkpoint-* 2>/dev/null | sort -V | tail -1)
echo "[sft-moe-v1-kd] ckpt=$CK"
if [ "$RC" -eq 0 ] && [ -n "$CK" ]; then
  find "$OUT" -maxdepth 3 -type d -name "global_step*" -exec rm -rf {} + 2>/dev/null || true
  env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PRESET=paper GPU=0 BATCH=16 \
    bash "$ROOT/scripts/run_lm_eval.sh" "$CK" "$EVAL_TAG" > "$ROOT/output/ev_${EVAL_TAG}.log" 2>&1
  echo "[sft-moe-v1-kd] eval rc=$? $(date '+%H:%M')"
fi
echo "[sft-moe-v1-kd] DONE $(date '+%F %H:%M')"
