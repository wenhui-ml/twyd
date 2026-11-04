#!/usr/bin/env bash
# SFT with the paper's 8-dataset general mixture (Paper 1, App. A.6 / Table 18):
#   UltraChat, OpenHermes, SlimOrca, MetaMath, WizardLM, Platypus, CodeAlpaca, MathInstruct
# ~0.62B tokens in the paper; we start at TOKENS=2e8 (~12 h here) to check the direction.
#
# Starts from the best PT arm (6-layer: layers 0-2 + 11-13, 5.11M coords) with the
# SAME parameterization (CORE-LR k=64, col frozen), so the only change is the stage.
#
#   PREFLIGHT=1 bash scripts/run_sft_mixture.sh     # 30 steps, checks data+schema
#
# GATE=0: check_init.py only accepts warm-start slice/ridge/base, and for SFT-from-PT the
# right check is "the PT weights actually loaded" -- which _apply_pt_ckpt_ now enforces
# (it raises on 0 tensors) -- plus the merge/logit parity we verified on 2026-10-31.
set -uo pipefail
ROOT=${TWYD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ROOT"
PT=${PT_CKPT:-$(ls -d "$ROOT"/output/ablation/band-6L | tail -1)}
OUT=${OUT:-$ROOT/output/product/sft-mixture-6L}
TOKENS=${TOKENS:-200000000}
LR=${LR:-1e-5}
PRE=${PREFLIGHT:-0}

MIX="swift/ultrachat_200k#60000 swift/OpenHermes-2.5#50000 swift/SlimOrca#50000 \
AI-ModelScope/WizardLM_evol_instruct_V2_196k#40000 AI-ModelScope/Open-Platypus#30000 \
AI-ModelScope/CodeAlpaca-20k#15000 AI-ModelScope/MathInstruct#40000"
if [ "$PRE" = "1" ]; then TOKENS=4000000; OUT=$ROOT/output/product/preflight-sft-mix; fi
mkdir -p "$OUT"
echo "=== SFT stage: preflight=$PRE tokens=$TOKENS lr=$LR pt_ckpt=$PT"
echo "=== out=$OUT"

env GPU=0,1 NPROC=2 MAX_LENGTH=${MAX_LENGTH:-1024} GRAD_ACCUM=${GRAD_ACCUM:-64} SAVE_STEPS=1000 \
    STAGE=sft WARM_START=pt_ckpt PT_CKPT="$PT" TRUNC=delete GATE=0 \
    BRANCH=${BRANCH_ARM:-core} K=${K:-64} LORA_MODE=${LORA_MODE_ARM:-core} FACTORIZED=${FACTORIZED_ARM:-1} FREEZE_COL=${FREEZE_COL_ARM:-true} \
    PLANS=${PLANS_ARM:-0-2} LORA_MODULES=gate_proj,up_proj SEED=42 \
    LR="$LR" TOKENS="$TOKENS" DATASET="${SFT_DATA:-$MIX}" OUTROOT="$OUT" \
    bash scripts/train_core_lr_pilot.sh 2>&1 | tee "$OUT/run.log" | tail -40
