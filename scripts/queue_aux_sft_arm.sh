#!/usr/bin/env bash
# Arm A of the "better training at fixed rank" line - SFT-stage version.
#
# Why not PT-stage: with WARM_START=slice the student's hidden states are garbage, so
# the paper's Eq.7 alignment term (lambda=0.2) fights the KL instead of regularising it.
# Measured 2026-11-13 on the 500M mixed corpus: loss 346 vs 200 without aux, grad_norm
# 38 vs 10, KL pushed up by ~145. The launcher's own guard said the same thing
# ("aux loss is meaningless on an ungated student") - it was right.
#
# A PT checkpoint *is* a good student (CE 2.44, better than the paper's released
# student at 3.05), and WARM_START=pt_ckpt is explicitly allowed by the guard. So the
# alignment term is tested where it is a gentle regulariser: the 78M-token SFT stage,
# from the exact same ckpt that produced the +0.89 pt headline arm. 6.6 h instead of 44 h.
#
#   bash scripts/queue_aux_sft_arm.sh
set -u
cd "$(dirname "$0")/.."
ROOT=$PWD

while pgrep -f 'swift/cli/|lm_eval' >/dev/null 2>&1; do sleep 120; done
echo "[auxsft] GPU free, start $(date '+%F %H:%M')"

CK="$ROOT/output/product/pt-mixed-500s/0-2_11-13/v0-20261109-135200/checkpoint-3814"
if [ ! -d "$CK" ]; then
  echo "[auxsft] missing PT ckpt $CK"; exit 1
fi

env MODEL="$CK" MODEL_TYPE=llama TEMPLATE=llama3_2 LR=1e-4 LOSS_TYPE=lrc_distill \
  WARM_START_ARM=pt_ckpt PT_CKPT="$CK" \
  BRANCH_ARM=dense LORA_MODE_ARM=dense FACTORIZED_ARM=0 \
  USE_AUX=1 TWYD_AUX_NORM=1 DATA_PROC=12 DATALOADER_WORKERS=4 \
  MAX_LENGTH=2048 GRAD_ACCUM=32 SAVE_STEPS=1000 MAX_STEPS=3170 PLANS_ARM=0-2,11-13,25-27 \
  SFT_DATA="$ROOT/data/sft_paper8.jsonl" OUT="$ROOT/output/product/sftDIST-9L" \
  bash scripts/run_sft_mixture.sh > "$ROOT/output/auxsft_sft.log" 2>&1
echo "[auxsft] SFT rc=$? $(date '+%H:%M')"

CK2=$(ls -d output/product/sftDIST-9L/v*/checkpoint-* 2>/dev/null | sort -V | tail -1)
echo "[auxsft] ckpt=$CK2"
[ -n "$CK2" ] && PRESET=paper GPU=0 BATCH=16 bash scripts/run_lm_eval.sh "$CK2" paper_DIST9 > "$ROOT/output/ev_DIST9.log" 2>&1
echo "[auxsft] DONE $(date '+%F %H:%M')"
