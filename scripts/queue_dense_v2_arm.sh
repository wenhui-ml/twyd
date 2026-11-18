#!/usr/bin/env bash
# Dense-TWYD 2.0 Validation Arm (Manifold Alignment + Full-Bus Co-adaptation):
#
#   1. Full-Rank Dense-LRC (BRANCH=dense, K=0, WARM_START=base) across the
#      hunchback entry + knowledge-peak manifold layers (0-3, 10-15 = 10 layers,
#      503.35M trainable params, 0 GiB frozen basis, ~13.6 GiB peak VRAM).
#   2. Full-Bus Writer-Reader Co-adaptation: unfreeze Attention (q/k/v/o) +
#      RMSNorm alongside MLP (gate/up/down), with Writer/Attention LR scaled to
#      0.3x (3e-5) and pre-clip gradient scale 0.25x so down_proj and q/k/v
#      never starve gate_proj/up_proj under global max_grad_norm.
#   3. High-SNR Manifold Loss: Top-K(64) + Tail-Bucket KL at sharp T=3.0 +
#      Rho-1 Excess-Loss Token Selection (keep=0.35, min_weight=0.1).
#   4. In-Context Contrastive Neighbor-Packed Corpus: data/pt_manifold_200M.jsonl
#      (200M tokens = 1525 steps at seq=2048, bs=1, gas=32, 2 GPUs).
#
#   bash scripts/queue_dense_v2_arm.sh
set -u
cd "$(dirname "$0")/.."
ROOT=$PWD
PLANS_SPEC=${PLANS_SPEC:-0-3,10-15}
TOKENS_BUDGET=${TOKENS_BUDGET:-200000000}
CORPUS=${CORPUS:-$ROOT/data/pt_manifold_200M.jsonl}
OUT="$ROOT/output/product/pt-dense-v2-200M"

if [ ! -f "$CORPUS" ]; then
  echo "[dense-v2] building $CORPUS ..."
  /home/chenwh/miniconda3/envs/twyd/bin/python3.14 scripts/build_manifold_pt_corpus.py \
    --knowledge "$ROOT/data/pt_knowledge.jsonl" \
    --web "$ROOT/data/fw_score4_1B.jsonl" \
    --out "$CORPUS" --target-mb 800 || exit 1
fi

echo "[dense-v2] start $(date '+%F %H:%M') plans=$PLANS_SPEC tokens=$TOKENS_BUDGET"
rm -rf "$OUT"

env PYTORCH_HIP_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  GPU=0,1 NPROC=2 SEQ=2048 SAVE_STEPS=500 BS=1 GAS=32 GATE=0 USE_AUX=0 STAGE=pt FORCE=1 \
  DATA="$CORPUS" PLANS="$PLANS_SPEC" SEED=42 \
  BRANCH=dense K=0 LORA_MODE=dense FACTORIZED=0 FREEZE_COL=false \
  WARM_START=base TOKENS="$TOKENS_BUDGET" \
  TWYD_T=3.0 TWYD_WKL=8.0 TWYD_KL_TOPK=64 TWYD_RHO_KEEP=0.35 TWYD_RHO_MIN_WEIGHT=0.1 \
  TWYD_UNFREEZE_ATTN=1 TWYD_UNFREEZE_NORM=1 \
  TWYD_WRITER_LR_SCALE=0.3 TWYD_WRITER_GRAD_SCALE=0.25 TWYD_WRITER_MAX_NORM=5.0 \
  OUTROOT="$OUT" \
  bash scripts/run_v15_w1.sh > "$ROOT/output/dense_v2_pt.log" 2>&1
echo "[dense-v2] PT rc=$? $(date '+%H:%M')"

CK=$(ls -d "$OUT"/*/v*/checkpoint-* 2>/dev/null | sort -V | tail -1)
echo "[dense-v2] ckpt=$CK"
if [ -n "$CK" ]; then
  PRESET=paper GPU=0 BATCH=16 bash scripts/run_lm_eval.sh "$CK" paper_PTdense_v2 > "$ROOT/output/ev_PTdense_v2.log" 2>&1
  echo "[dense-v2] eval rc=$? $(date '+%H:%M')"
fi
echo "[dense-v2] DONE $(date '+%F %H:%M')"
