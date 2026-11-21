#!/usr/bin/env bash
# Phase 2 Plan B: TWYD-MoE v2 (+200M-token Expanded Multi-Discipline Knowledge Manifold Continuation).
#
#   Corpus (data/pt_textbook_manifold_v2_200M.jsonl, 808.9 MB, 107,595 records):
#     - Built from 333,322 knowledge docs (1.35 GB): unseen 896 MB tail of
#       pt_knowledge.jsonl (Wikipedia + Open-Web-Math) merged & re-clustered with
#       450 MB Cosmopedia college textbooks (openstax + stanford + auto_math_text)
#       into 62,569 contrastive neighbor-packed windows (70% factual share, +56%
#       more knowledge windows than v1) + fresh unseen FineWeb-Edu score>=4 (offset 320 MB+).
#
#   Recipe:
#     - Continuation from best MoE-v1 checkpoint (default: selected winner of v1 / soup)
#     - NPROC=2, DS=zero2, SEQ=2048, BS=1, GAS=32 (1,525 steps = 200M tokens)
#     - LR=1e-4 (router_lr_scale=1.0, since routers & experts are already specialized),
#       CLIP=50.0, cosine decay, warmup 3%
#     - T=1.0, K=256, wKL=5.0, Rho-1=0.35 (min_w=0.1)
#
#   Usage:
#     bash scripts/queue_twyd_moe_v2_arm.sh
set -u
cd "$(dirname "$0")/.."
ROOT=$PWD

TOKENS_BUDGET=${TOKENS_BUDGET:-200000000}
CORPUS=${CORPUS:-$ROOT/data/pt_textbook_manifold_v2_200M.jsonl}
PT_INIT=${PT_INIT:-$ROOT/output/product/pt-moe-v1-200M/v0-20261119-161728/checkpoint-1525}
OUT=${OUT:-$ROOT/output/product/pt-moe-v2-200M}
EVAL_TAG=${EVAL_TAG:-paper_PTmoe_v2}

LR=${LR:-1e-4}
CLIP=${CLIP:-50.0}
DS=${DS:-$ROOT/configs/ds_zero2_offload_torch.json}
SEQ=${SEQ:-2048}
BS=${BS:-2}
GAS=${GAS:-16}
NPROC=${NPROC:-2}
OMP_THREADS=${OMP_THREADS:-8}

if [ ! -f "$CORPUS" ]; then
  echo "[twyd-moe-v2] ERROR: missing corpus $CORPUS" >&2
  exit 1
fi
if [ ! -d "$PT_INIT" ]; then
  echo "[twyd-moe-v2] ERROR: missing PT_INIT $PT_INIT" >&2
  exit 1
fi

STEPS=$(( TOKENS_BUDGET / (SEQ * BS * GAS * NPROC) ))
SAVE_STEPS=${SAVE_STEPS:-$(( (STEPS + 1) / 2 ))}

echo "[twyd-moe-v2] start $(date '+%F %H:%M') steps=$STEPS save_steps=$SAVE_STEPS lr=$LR"
echo "[twyd-moe-v2] init=$PT_INIT corpus=$CORPUS out=$OUT"
rm -rf "$OUT"

env OMP_NUM_THREADS="$OMP_THREADS" \
  PYTORCH_HIP_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=${GPU:-0,1} NPROC="$NPROC" DS="$DS" \
  BRANCH=moe STAGE=pt WARM_START=pt_ckpt PT_CKPT="$PT_INIT" \
  MODEL=/home/chenwh/models/LRC-1.5B-Base MODEL_TYPE=llama TEMPLATE=llama3_2 \
  TEACHER=/home/chenwh/models/Llama-3.2-3B-Instruct \
  DATASET="$CORPUS" TRUNC=split DATA_CACHE=true DATA_PROC=16 \
  MAX_LENGTH="$SEQ" BS="$BS" GRAD_ACCUM="$GAS" \
  MAX_STEPS="$STEPS" SAVE_STEPS="$SAVE_STEPS" \
  LR="$LR" CLIP="$CLIP" SEED=137 GATE=0 USE_AUX=0 \
  TWYD_T=1.0 TWYD_WKL=5.0 TWYD_KL_TOPK=256 TWYD_RHO_KEEP=0.35 TWYD_RHO_MIN_WEIGHT=0.1 \
  TWYD_MOE_LB_COEF=0.02 TWYD_MOE_Z_COEF=0.001 \
  TWYD_MOE_ROUTER_LR_SCALE=1.0 TWYD_MOE_UNFREEZE_SHARED=0 \
  TWYD_WRITER_LR_SCALE=1.0 TWYD_WRITER_GRAD_SCALE=1.0 TWYD_WRITER_MAX_NORM=0.0 \
  OUT="$OUT" \
  bash "$ROOT/scripts/train_core_lr_pilot.sh" > "$ROOT/output/moe_v2_pt.log" 2>&1
RC=$?
echo "[twyd-moe-v2] PT rc=$RC $(date '+%H:%M')"

CK=$(ls -d "$OUT"/v*/checkpoint-* 2>/dev/null | sort -V | tail -1)
echo "[twyd-moe-v2] ckpt=$CK"

if [ "$RC" -eq 0 ] && [ -n "$CK" ]; then
  find "$OUT" -maxdepth 3 -type d -name "global_step*" -exec rm -rf {} + 2>/dev/null || true
  env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PRESET=paper GPU=0 BATCH=16 \
    bash "$ROOT/scripts/run_lm_eval.sh" "$CK" "$EVAL_TAG" > "$ROOT/output/ev_${EVAL_TAG}.log" 2>&1
  echo "[twyd-moe-v2] eval rc=$? $(date '+%H:%M')"
fi
echo "[twyd-moe-v2] DONE $(date '+%F %H:%M')"
