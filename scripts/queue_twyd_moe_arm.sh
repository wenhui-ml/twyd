#!/usr/bin/env bash
# Phase 2 TWYD-MoE v1 (200M-token Textbook Manifold Pilot):
#
#   Architecture (strictly preserves hidden_size=1536 & KV-cache footprint):
#     - Shared-Dense Mother (frozen 1.6B): sftDenseV2_1-halfep/checkpoint-1585
#       (Avg8 0.6728, Beh 0.7060, Pool 0.5978, MMLU 0.5042)
#     - Routed Residual Knowledge Experts on 16 mid-to-late layers (8-23):
#       E=8 experts, TopK=2, d_e=1536 (+906.2M routed params; 2.51B total,
#       1.83B active = +14% active FLOPs).
#     - Bit-exact step-0 identity (w_down=0, orthogonal w_gate/w_up gain=0.8,
#       orthogonal router), GAS-normalized Switch LB (0.02) + Z-loss (0.001).
#
#   Recipe (200M tokens on Cosmopedia + Wiki + Math neighbor-packed corpus):
#     - NPROC=2, DS=zero2 (measured peak ~21.7 GiB / 24 GiB per 7900 XTX)
#     - SEQ=2048, BS=1, GAS=32 (131,072 tok/step -> 1,525 steps = 200M tokens)
#     - LR=2e-4 (router 2x = 4e-4), CLIP=50.0, cosine decay, warmup 3%
#     - T=1.0, K=256, wKL=5.0, Rho-1=0.35 (min_w=0.1)
#
#   Usage:
#     bash scripts/queue_twyd_moe_arm.sh
set -u
cd "$(dirname "$0")/.."
ROOT=$PWD

TOKENS_BUDGET=${TOKENS_BUDGET:-200000000}
CORPUS=${CORPUS:-$ROOT/data/pt_textbook_manifold_200M.jsonl}
PT_INIT=${PT_INIT:-$ROOT/output/product/sftDenseV2_1-halfep/v0-20261119-121153/checkpoint-1585}
OUT=${OUT:-$ROOT/output/product/pt-moe-v1-200M}
EVAL_TAG=${EVAL_TAG:-paper_PTmoe_v1}

MOE_LAYERS=${TWYD_MOE_LAYERS:-8-23}
MOE_EXPERTS=${TWYD_MOE_EXPERTS:-8}
MOE_TOPK=${TWYD_MOE_TOPK:-2}
MOE_INTER=${TWYD_MOE_INTER:-1536}
MOE_LB=${TWYD_MOE_LB_COEF:-0.02}
MOE_Z=${TWYD_MOE_Z_COEF:-0.001}
MOE_ROUTER_SCALE=${TWYD_MOE_ROUTER_LR_SCALE:-2.0}
LR=${LR:-2e-4}
CLIP=${CLIP:-50.0}
DS=${DS:-zero2}
SEQ=${SEQ:-2048}
BS=${BS:-1}
GAS=${GAS:-32}
NPROC=${NPROC:-2}

if [ ! -f "$CORPUS" ]; then
  echo "[twyd-moe-v1] ERROR: missing corpus $CORPUS" >&2
  exit 1
fi
if [ ! -d "$PT_INIT" ]; then
  echo "[twyd-moe-v1] ERROR: missing PT_INIT $PT_INIT" >&2
  exit 1
fi

STEPS=$(( TOKENS_BUDGET / (SEQ * BS * GAS * NPROC) ))
SAVE_STEPS=${SAVE_STEPS:-$(( (STEPS + 1) / 2 ))}

echo "[twyd-moe-v1] start $(date '+%F %H:%M') steps=$STEPS save_steps=$SAVE_STEPS layers=$MOE_LAYERS E=$MOE_EXPERTS topK=$MOE_TOPK d_e=$MOE_INTER"
echo "[twyd-moe-v1] mother=$PT_INIT corpus=$CORPUS out=$OUT"
rm -rf "$OUT"

env PYTORCH_HIP_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=${GPU:-0,1} NPROC="$NPROC" DS="$DS" \
  BRANCH=moe STAGE=pt WARM_START=pt_ckpt PT_CKPT="$PT_INIT" \
  MODEL=/home/chenwh/models/LRC-1.5B-Base MODEL_TYPE=llama TEMPLATE=llama3_2 \
  TEACHER=/home/chenwh/models/Llama-3.2-3B-Instruct \
  DATASET="$CORPUS" TRUNC=split DATA_CACHE=true DATA_PROC=16 \
  MAX_LENGTH="$SEQ" BS="$BS" GRAD_ACCUM="$GAS" \
  MAX_STEPS="$STEPS" SAVE_STEPS="$SAVE_STEPS" \
  LR="$LR" CLIP="$CLIP" SEED=42 GATE=0 USE_AUX=0 \
  TWYD_T=1.0 TWYD_WKL=5.0 TWYD_KL_TOPK=256 TWYD_RHO_KEEP=0.35 TWYD_RHO_MIN_WEIGHT=0.1 \
  TWYD_MOE_LAYERS="$MOE_LAYERS" TWYD_MOE_EXPERTS="$MOE_EXPERTS" \
  TWYD_MOE_TOPK="$MOE_TOPK" TWYD_MOE_INTER="$MOE_INTER" \
  TWYD_MOE_LB_COEF="$MOE_LB" TWYD_MOE_Z_COEF="$MOE_Z" \
  TWYD_MOE_ROUTER_LR_SCALE="$MOE_ROUTER_SCALE" \
  TWYD_WRITER_LR_SCALE=1.0 TWYD_WRITER_GRAD_SCALE=1.0 TWYD_WRITER_MAX_NORM=0.0 \
  OUT="$OUT" \
  bash "$ROOT/scripts/train_core_lr_pilot.sh" > "$ROOT/output/moe_v1_pt.log" 2>&1
RC=$?
echo "[twyd-moe-v1] PT rc=$RC $(date '+%H:%M')"

CK=$(ls -d "$OUT"/v*/checkpoint-* 2>/dev/null | sort -V | tail -1)
echo "[twyd-moe-v1] ckpt=$CK"

if [ "$RC" -eq 0 ] && [ -n "$CK" ]; then
  # Drop ~18G DeepSpeed ZeRO-2 optimizer shards once final HF checkpoint is safely written.
  find "$OUT" -maxdepth 3 -type d -name "global_step*" -exec rm -rf {} + 2>/dev/null || true
  PRESET=paper GPU=0 BATCH=16 bash "$ROOT/scripts/run_lm_eval.sh" "$CK" "$EVAL_TAG" > "$ROOT/output/ev_${EVAL_TAG}.log" 2>&1
  echo "[twyd-moe-v1] eval rc=$? $(date '+%H:%M')"
fi
echo "[twyd-moe-v1] DONE $(date '+%F %H:%M')"
