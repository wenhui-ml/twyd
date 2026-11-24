#!/usr/bin/env bash
# Phase 2 Full Pipeline (MoE-v3 Pure Cosmopedia Textbooks + KD-Anchored SFT):
#
#   Lessons applied from v1 (+0.96 pt MMLU), v2 (-0.61 pt from Wiki dilution),
#   and SFT-MoE-KD (Avg8 0.6767, Beh7 0.7004, Pool 0.6028 all-time highs):
#
#   Stage 1: TWYD-MoE v3 PT (+200M tokens on 100% pure unseen Cosmopedia college
#     textbooks: openstax/stanford/khanacademy/auto_math_text/web_samples_v2,
#     ZERO Wikipedia! Packed with fresh FineWeb-Edu score>=4 at 65/35 ratio).
#     - Starts from MoE-v1-200M (checkpoint-1525, MMLU 0.5138)
#     - BS=2, GAS=16, NPROC=2, ds_zero2_offload_torch (16.7 GiB peak, zero GTT)
#     - LR=8e-5 (gentle continuation preserving v1 OpenStax gains), T=1.0 K=256 wKL=5.0 Rho-1=0.35
#     - Auto-evaluates offline -> paper_PTmoe_v3
#
#   Stage 2: KD-Anchored 0.5-epoch SFT on top of MoE-v3 checkpoint-1525:
#     - BS=1, GAS=32, NPROC=2, ds_zero2_offload_torch (18.95 GiB peak, zero GTT)
#     - LR=2e-5, T=1.0 K=256 wKL=2.0, sft_paper8.jsonl (1,585 steps)
#     - Auto-evaluates offline -> paper_SFTmoe_v3_kd
#
# Usage:
#   bash scripts/queue_twyd_moe_v3_full.sh
set -u
cd "$(dirname "$0")/.."
ROOT=$PWD

TOKENS_BUDGET=${TOKENS_BUDGET:-200000000}
CORPUS=${CORPUS:-$ROOT/data/pt_cosmopedia_v3_manifold_200M.jsonl}
SFT_DATA=${SFT_DATA:-$ROOT/data/sft_paper8.jsonl}
PT_INIT=${PT_INIT:-$ROOT/output/product/pt-moe-v1-200M/v0-20261119-161728/checkpoint-1525}
OUT_PT=${OUT_PT:-$ROOT/output/product/pt-moe-v3-200M}
OUT_SFT=${OUT_SFT:-$ROOT/output/product/sft-moe-v3-kd-halfep}
EVAL_PT_TAG=${EVAL_PT_TAG:-paper_PTmoe_v3}
EVAL_SFT_TAG=${EVAL_SFT_TAG:-paper_SFTmoe_v3_kd}

DS=${DS:-$ROOT/configs/ds_zero2_offload_torch.json}
OMP_THREADS=${OMP_THREADS:-8}
SEQ=${SEQ:-2048}
NPROC=${NPROC:-2}

PT_LR=${PT_LR:-8e-5}
PT_BS=${PT_BS:-2}
PT_GAS=${PT_GAS:-16}
SFT_LR=${SFT_LR:-2e-5}
SFT_WKL=${SFT_WKL:-2.0}
SFT_BS=${SFT_BS:-1}
SFT_GAS=${SFT_GAS:-32}
SFT_STEPS=${SFT_STEPS:-1585}

if [ ! -f "$CORPUS" ]; then
  echo "[moe-v3] ERROR: missing corpus $CORPUS" >&2
  exit 1
fi
if [ ! -d "$PT_INIT" ]; then
  echo "[moe-v3] ERROR: missing PT_INIT $PT_INIT" >&2
  exit 1
fi

PT_STEPS=$(( TOKENS_BUDGET / (SEQ * PT_BS * PT_GAS * NPROC) ))
PT_SAVE_STEPS=${PT_SAVE_STEPS:-$(( (PT_STEPS + 1) / 2 ))}

echo "[moe-v3] === STAGE 1: PT start $(date '+%F %H:%M') steps=$PT_STEPS lr=$PT_LR corpus=$CORPUS ==="
rm -rf "$OUT_PT"

env OMP_NUM_THREADS="$OMP_THREADS" \
  PYTORCH_HIP_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=${GPU:-0,1} NPROC="$NPROC" DS="$DS" \
  BRANCH=moe STAGE=pt WARM_START=pt_ckpt PT_CKPT="$PT_INIT" \
  MODEL=/home/chenwh/models/LRC-1.5B-Base MODEL_TYPE=llama TEMPLATE=llama3_2 \
  TEACHER=/home/chenwh/models/Llama-3.2-3B-Instruct \
  DATASET="$CORPUS" TRUNC=split DATA_CACHE=true DATA_PROC=16 \
  MAX_LENGTH="$SEQ" BS="$PT_BS" GRAD_ACCUM="$PT_GAS" \
  MAX_STEPS="$PT_STEPS" SAVE_STEPS="$PT_SAVE_STEPS" \
  LR="$PT_LR" CLIP=50.0 SEED=2026 GATE=0 USE_AUX=0 \
  TWYD_T=1.0 TWYD_WKL=5.0 TWYD_KL_TOPK=256 TWYD_RHO_KEEP=0.35 TWYD_RHO_MIN_WEIGHT=0.1 \
  TWYD_MOE_LB_COEF=0.02 TWYD_MOE_Z_COEF=0.001 \
  TWYD_MOE_ROUTER_LR_SCALE=1.0 TWYD_MOE_UNFREEZE_SHARED=0 \
  TWYD_WRITER_LR_SCALE=1.0 TWYD_WRITER_GRAD_SCALE=1.0 TWYD_WRITER_MAX_NORM=0.0 \
  OUT="$OUT_PT" \
  bash "$ROOT/scripts/train_core_lr_pilot.sh" > "$ROOT/output/moe_v3_pt.log" 2>&1
RC_PT=$?
echo "[moe-v3] PT rc=$RC_PT $(date '+%H:%M')"

CK_PT=$(ls -d "$OUT_PT"/v*/checkpoint-* 2>/dev/null | sort -V | tail -1)
echo "[moe-v3] PT ckpt=$CK_PT"
if [ "$RC_PT" -ne 0 ] || [ -z "$CK_PT" ]; then
  echo "[moe-v3] ABORT: Stage 1 PT failed." >&2
  exit 1
fi

find "$OUT_PT" -maxdepth 3 -type d -name "global_step*" -exec rm -rf {} + 2>/dev/null || true
env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PRESET=paper GPU=0 BATCH=16 \
  bash "$ROOT/scripts/run_lm_eval.sh" "$CK_PT" "$EVAL_PT_TAG" > "$ROOT/output/ev_${EVAL_PT_TAG}.log" 2>&1
echo "[moe-v3] PT eval rc=$? $(date '+%H:%M')"

echo "[moe-v3] === STAGE 2: KD-SFT start $(date '+%F %H:%M') steps=$SFT_STEPS lr=$SFT_LR wKL=$SFT_WKL ==="
rm -rf "$OUT_SFT"

env OMP_NUM_THREADS="$OMP_THREADS" \
  PYTORCH_HIP_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=${GPU:-0,1} NPROC="$NPROC" DS="$DS" \
  BRANCH=moe STAGE=sft LOSS_TYPE=lrc_distill WARM_START=pt_ckpt PT_CKPT="$CK_PT" \
  MODEL=/home/chenwh/models/LRC-1.5B-Base MODEL_TYPE=llama TEMPLATE=llama3_2 \
  TEACHER=/home/chenwh/models/Llama-3.2-3B-Instruct \
  DATASET="$SFT_DATA" TRUNC=delete DATA_CACHE=true DATA_PROC=12 DATALOADER_WORKERS=4 \
  MAX_LENGTH="$SEQ" BS="$SFT_BS" GRAD_ACCUM="$SFT_GAS" \
  MAX_STEPS="$SFT_STEPS" SAVE_STEPS="$SFT_STEPS" \
  LR="$SFT_LR" CLIP=50.0 SEED=42 GATE=0 USE_AUX=0 \
  TWYD_T=1.0 TWYD_WKL="$SFT_WKL" TWYD_KL_TOPK=256 TWYD_RHO_KEEP=1.0 \
  TWYD_MOE_LB_COEF=0.02 TWYD_MOE_Z_COEF=0.001 \
  TWYD_MOE_ROUTER_LR_SCALE=1.0 TWYD_MOE_UNFREEZE_SHARED=0 \
  OUT="$OUT_SFT" \
  bash "$ROOT/scripts/train_core_lr_pilot.sh" > "$ROOT/output/sft_moe_v3_kd.log" 2>&1
RC_SFT=$?
echo "[moe-v3] SFT rc=$RC_SFT $(date '+%H:%M')"

CK_SFT=$(ls -d "$OUT_SFT"/v*/checkpoint-* 2>/dev/null | sort -V | tail -1)
echo "[moe-v3] SFT ckpt=$CK_SFT"
if [ "$RC_SFT" -eq 0 ] && [ -n "$CK_SFT" ]; then
  find "$OUT_SFT" -maxdepth 3 -type d -name "global_step*" -exec rm -rf {} + 2>/dev/null || true
  env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PRESET=paper GPU=0 BATCH=16 \
    bash "$ROOT/scripts/run_lm_eval.sh" "$CK_SFT" "$EVAL_SFT_TAG" > "$ROOT/output/ev_${EVAL_SFT_TAG}.log" 2>&1
  echo "[moe-v3] SFT eval rc=$? $(date '+%H:%M')"
fi
echo "[moe-v3] ALL DONE $(date '+%F %H:%M')"
