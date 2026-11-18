#!/usr/bin/env bash
# Dense-TWYD 2.1 Validation Arm (Fixing the two measured bottlenecks from v2):
#
#   Fixes over v2 (pt-dense-v2-200M):
#     1. Unlock the 60x CLIP=1.0 brake -> CLIP=50.0 (Dense 503M raw grad_norm
#        is ~50-90; keep WRITER_MAX_NORM=2.0 + WRITER_GRAD_SCALE=0.35 +
#        WRITER_LR_SCALE=0.4 so Attention/down_proj stay balanced without
#        starving gate/up).
#     2. Fix T=3.0 K=64 (which only covered 2.56% of teacher mass and gave klT2~0.2
#        vs ce~3.0) -> T=1.0, K=256, TWYD_WKL=5.0 (covers 94.0% of teacher mass
#        with ~1:1 KL:CE ratio).
#     3. Upgrade PT corpus from prose Wikipedia -> Cosmopedia college textbooks
#        (openstax + stanford + auto_math_text) + wiki + open-web-math with
#        neighbor-window packing (data/pt_textbook_manifold_200M.jsonl).
#   Warm-start: continue 200M tokens from pt-dense-v2-200M checkpoint-1525
#   (already at Avg8 0.6675, Beh 0.6988, MMLU 0.5061; total PT = 400M < 500M).
#
#   bash scripts/queue_dense_v2_1_arm.sh
set -u
cd "$(dirname "$0")/.."
ROOT=$PWD
PLANS_SPEC=${PLANS_SPEC:-0-3,10-15}
TOKENS_BUDGET=${TOKENS_BUDGET:-200000000}
CORPUS=${CORPUS:-$ROOT/data/pt_textbook_manifold_200M.jsonl}
PT_INIT=${PT_INIT:-$ROOT/output/product/pt-dense-v2-200M/0-3_10-15/v0-20261117-165744/checkpoint-1525}
OUT="$ROOT/output/product/pt-dense-v2_1-200M"

if [ ! -f "$CORPUS" ]; then
  echo "[dense-v2.1] ERROR: missing corpus $CORPUS" >&2
  exit 1
fi
if [ ! -d "$PT_INIT" ]; then
  echo "[dense-v2.1] ERROR: missing PT_INIT $PT_INIT" >&2
  exit 1
fi

echo "[dense-v2.1] start $(date '+%F %H:%M') plans=$PLANS_SPEC tokens=$TOKENS_BUDGET init=$PT_INIT"
rm -rf "$OUT"

env PYTORCH_HIP_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  GPU=0,1 NPROC=2 SEQ=2048 SAVE_STEPS=500 BS=1 GAS=32 GATE=0 USE_AUX=0 STAGE=pt FORCE=1 \
  DATA="$CORPUS" PLANS="$PLANS_SPEC" SEED=42 CLIP=50.0 LR=1e-4 \
  BRANCH=dense K=0 LORA_MODE=dense FACTORIZED=0 FREEZE_COL=false \
  WARM_START=pt_ckpt PT_CKPT="$PT_INIT" TOKENS="$TOKENS_BUDGET" \
  TWYD_T=1.0 TWYD_WKL=5.0 TWYD_KL_TOPK=256 TWYD_RHO_KEEP=0.35 TWYD_RHO_MIN_WEIGHT=0.1 \
  TWYD_UNFREEZE_ATTN=1 TWYD_UNFREEZE_NORM=1 \
  TWYD_WRITER_LR_SCALE=0.4 TWYD_WRITER_GRAD_SCALE=0.35 TWYD_WRITER_MAX_NORM=2.0 \
  OUTROOT="$OUT" \
  bash scripts/run_v15_w1.sh > "$ROOT/output/dense_v2_1_pt.log" 2>&1
echo "[dense-v2.1] PT rc=$? $(date '+%H:%M')"

CK=$(ls -d "$OUT"/*/v*/checkpoint-* 2>/dev/null | sort -V | tail -1)
echo "[dense-v2.1] ckpt=$CK"
if [ -n "$CK" ]; then
  PRESET=paper GPU=0 BATCH=16 bash scripts/run_lm_eval.sh "$CK" paper_PTdense_v2_1 > "$ROOT/output/ev_PTdense_v2_1.log" 2>&1
  echo "[dense-v2.1] eval rc=$? $(date '+%H:%M')"
fi
echo "[dense-v2.1] DONE $(date '+%F %H:%M')"
