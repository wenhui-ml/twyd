#!/usr/bin/env bash
# Phase 2 Plan A: Complete MoE-v3 Stage 1 (300M -> 400M pure Cosmopedia textbooks)
# from intact checkpoint-763 (MMLU 0.5165, Pool 0.6027), then automatically run
# Stage 2 KD-Anchored 0.5-epoch SFT -> paper_SFTmoe_v3_kd.
#
# Safeguards:
#   - Frequent checkpoints (SAVE_STEPS=254 in PT ~4.8h; SAVE_STEPS=529 in SFT ~8h)
#   - Continues at exact step-763 cosine LR (4.2e-5 -> 0) with fresh SEED=2027
#   - Uses proven zero-GTT Fused Loss + ds_zero2_offload_torch.json
#
# Usage:
#   bash scripts/queue_twyd_moe_v3_finish.sh
set -u
cd "$(dirname "$0")/.."
ROOT=$PWD

CORPUS=${CORPUS:-$ROOT/data/pt_cosmopedia_v3_manifold_200M.jsonl}
SFT_DATA=${SFT_DATA:-$ROOT/data/sft_paper8.jsonl}
PT_INIT=${PT_INIT:-$ROOT/output/product/pt-moe-v3-200M/v0-20261124-124935/checkpoint-763}
OUT_PT=${OUT_PT:-$ROOT/output/product/pt-moe-v3-400M}
OUT_SFT=${OUT_SFT:-$ROOT/output/product/sft-moe-v3-kd-halfep}
EVAL_PT_TAG=${EVAL_PT_TAG:-paper_PTmoe_v3}
EVAL_SFT_TAG=${EVAL_SFT_TAG:-paper_SFTmoe_v3_kd}

DS=${DS:-$ROOT/configs/ds_zero2_offload_torch.json}
OMP_THREADS=${OMP_THREADS:-8}
SEQ=${SEQ:-2048}
NPROC=${NPROC:-2}

PT_LR=${PT_LR:-4.2e-5}
PT_BS=${PT_BS:-2}
PT_GAS=${PT_GAS:-16}
PT_STEPS=${PT_STEPS:-762}
PT_SAVE_STEPS=${PT_SAVE_STEPS:-254}

SFT_LR=${SFT_LR:-2e-5}
SFT_WKL=${SFT_WKL:-2.0}
SFT_BS=${SFT_BS:-1}
SFT_GAS=${SFT_GAS:-32}
SFT_STEPS=${SFT_STEPS:-1585}
SFT_SAVE_STEPS=${SFT_SAVE_STEPS:-529}

if [ ! -f "$CORPUS" ]; then
  echo "[moe-v3-finish] ERROR: missing corpus $CORPUS" >&2
  exit 1
fi
if [ ! -d "$PT_INIT" ]; then
  echo "[moe-v3-finish] ERROR: missing PT_INIT $PT_INIT" >&2
  exit 1
fi

echo "[moe-v3-finish] === STAGE 1B: PT 300M->400M start $(date '+%F %H:%M') steps=$PT_STEPS save=$PT_SAVE_STEPS lr=$PT_LR ==="
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
  LR="$PT_LR" CLIP=50.0 SEED=2027 GATE=0 USE_AUX=0 \
  TWYD_T=1.0 TWYD_WKL=5.0 TWYD_KL_TOPK=256 TWYD_RHO_KEEP=0.35 TWYD_RHO_MIN_WEIGHT=0.1 \
  TWYD_MOE_LB_COEF=0.02 TWYD_MOE_Z_COEF=0.001 \
  TWYD_MOE_ROUTER_LR_SCALE=1.0 TWYD_MOE_UNFREEZE_SHARED=0 \
  TWYD_WRITER_LR_SCALE=1.0 TWYD_WRITER_GRAD_SCALE=1.0 TWYD_WRITER_MAX_NORM=0.0 \
  OUT="$OUT_PT" \
  bash "$ROOT/scripts/train_core_lr_pilot.sh" > "$ROOT/output/moe_v3_400m_pt.log" 2>&1
RC_PT=$?
echo "[moe-v3-finish] PT rc=$RC_PT $(date '+%H:%M')"

CK_PT=$(ls -d "$OUT_PT"/v*/checkpoint-* 2>/dev/null | sort -V | tail -1)
echo "[moe-v3-finish] PT ckpt=$CK_PT"
if [ "$RC_PT" -ne 0 ] || [ -z "$CK_PT" ]; then
  echo "[moe-v3-finish] ABORT: Stage 1B PT failed." >&2
  exit 1
fi

find "$OUT_PT" -maxdepth 3 -type d -name "global_step*" -exec rm -rf {} + 2>/dev/null || true
env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PRESET=paper GPU=0 BATCH=16 \
  bash "$ROOT/scripts/run_lm_eval.sh" "$CK_PT" "$EVAL_PT_TAG" > "$ROOT/output/ev_${EVAL_PT_TAG}.log" 2>&1
echo "[moe-v3-finish] PT eval rc=$? $(date '+%H:%M')"

echo "[moe-v3-finish] === STAGE 2: KD-SFT start $(date '+%F %H:%M') steps=$SFT_STEPS save=$SFT_SAVE_STEPS lr=$SFT_LR wKL=$SFT_WKL ==="
rm -rf "$OUT_SFT"

env OMP_NUM_THREADS="$OMP_THREADS" \
  PYTORCH_HIP_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=${GPU:-0,1} NPROC="$NPROC" DS="$DS" \
  BRANCH=moe STAGE=sft LOSS_TYPE=lrc_distill WARM_START=pt_ckpt PT_CKPT="$CK_PT" \
  MODEL=/home/chenwh/models/LRC-1.5B-Base MODEL_TYPE=llama TEMPLATE=llama3_2 \
  TEACHER=/home/chenwh/models/Llama-3.2-3B-Instruct \
  DATASET="$SFT_DATA" TRUNC=delete DATA_CACHE=true DATA_PROC=12 DATALOADER_WORKERS=4 \
  MAX_LENGTH="$SEQ" BS="$SFT_BS" GRAD_ACCUM="$SFT_GAS" \
  MAX_STEPS="$SFT_STEPS" SAVE_STEPS="$SFT_SAVE_STEPS" \
  LR="$SFT_LR" CLIP=50.0 SEED=42 GATE=0 USE_AUX=0 \
  TWYD_T=1.0 TWYD_WKL="$SFT_WKL" TWYD_KL_TOPK=256 TWYD_RHO_KEEP=1.0 \
  TWYD_MOE_LB_COEF=0.02 TWYD_MOE_Z_COEF=0.001 \
  TWYD_MOE_ROUTER_LR_SCALE=1.0 TWYD_MOE_UNFREEZE_SHARED=0 \
  OUT="$OUT_SFT" \
  bash "$ROOT/scripts/train_core_lr_pilot.sh" > "$ROOT/output/sft_moe_v3_kd.log" 2>&1
RC_SFT=$?
echo "[moe-v3-finish] SFT rc=$RC_SFT $(date '+%H:%M')"

CK_SFT=$(ls -d "$OUT_SFT"/v*/checkpoint-* 2>/dev/null | sort -V | tail -1)
echo "[moe-v3-finish] SFT ckpt=$CK_SFT"
if [ "$RC_SFT" -eq 0 ] && [ -n "$CK_SFT" ]; then
  find "$OUT_SFT" -maxdepth 3 -type d -name "global_step*" -exec rm -rf {} + 2>/dev/null || true
  env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PRESET=paper GPU=0 BATCH=16 \
    bash "$ROOT/scripts/run_lm_eval.sh" "$CK_SFT" "$EVAL_SFT_TAG" > "$ROOT/output/ev_${EVAL_SFT_TAG}.log" 2>&1
  echo "[moe-v3-finish] SFT eval rc=$? $(date '+%H:%M')"
fi
echo "[moe-v3-finish] ALL DONE $(date '+%F %H:%M')"
