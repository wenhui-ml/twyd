#!/usr/bin/env bash
# Arm A of the "better training at fixed rank" line.
#
# Finding (2026-11-13): `run_v15_w1.sh` passed a literal `USE_AUX=0`, so the paper's
# Eq.7 hidden-state alignment term (1/L sum_l ||h^S_l P - h^T_l||^2, lambda=0.2, 4
# matched layers, one shared least-squares P, zero deployable parameters) was OFF in
# *every* PT run of this study - including the +0.89 pt headline arm.
#
# This arm changes exactly that one thing. Corpus (ordered pt_mixed), budget (500M PT
# + 78M SFT), layer plan, rank k=64, LR, KD temperature and the SFT recipe are all
# identical to output/product/pt-mixed-500s (arm `mix9`). Kept in the repo rather than
# /tmp: the 2026-11-13 reboot wiped /tmp and took the previous queue with it.
#
#   bash scripts/queue_aux_arm.sh          # waits for the GPU, then runs PT -> eval -> SFT -> eval
set -u
cd "$(dirname "$0")/.."
ROOT=$PWD

while pgrep -f 'swift/cli/|lm_eval' >/dev/null 2>&1; do sleep 120; done
echo "[aux] GPU free, start $(date '+%F %H:%M')"

OUT="$ROOT/output/product/pt-mixed-aux-500s"
env PYTORCH_HIP_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  GPU=0,1 NPROC=2 SEQ=2048 SAVE_STEPS=800 BS=1 GAS=32 GATE=0 USE_AUX=1 STAGE=pt \
  DATA="$ROOT/data/pt_mixed.jsonl" PLANS=0-2,11-13 \
  LORA_MODULES=gate_proj,up_proj SEED=42 BRANCH=core K=64 LORA_MODE=core FACTORIZED=1 \
  FREEZE_COL=true PERP_BASIS=teacher TOKENS=500000000 WARM_START=slice OUTROOT="$OUT" \
  bash scripts/run_v15_w1.sh > "$ROOT/output/aux_pt.log" 2>&1
RC=$?
echo "[aux] PT rc=$RC $(date '+%H:%M')"
if [ "$RC" -ne 0 ]; then echo "[aux] PT failed, stopping chain (see output/aux_pt.log)"; exit 1; fi

CK=$(ls -d "$OUT"/*/v*/checkpoint-* 2>/dev/null | sort -V | tail -1)
echo "[aux] ckpt=$CK"
[ -n "$CK" ] && PRESET=paper GPU=0 BATCH=16 bash scripts/run_lm_eval.sh "$CK" paper_PTaux > "$ROOT/output/ev_PTaux.log" 2>&1
echo "[aux] PT eval done $(date '+%H:%M')"

if [ -n "$CK" ]; then
  # SFT recipe is byte-identical to the one used for `mix9` (USE_AUX stays 0 there so the
  # comparison isolates the PT-stage alignment term).
  env MODEL="$CK" MODEL_TYPE=llama TEMPLATE=llama3_2 LR=1e-4 LOSS_TYPE= WARM_START_ARM=pt_ckpt \
    PT_CKPT="$CK" BRANCH_ARM=dense LORA_MODE_ARM=dense FACTORIZED_ARM=0 USE_AUX=0 \
    DATA_PROC=12 DATALOADER_WORKERS=4 MAX_LENGTH=2048 GRAD_ACCUM=32 SAVE_STEPS=1000 \
    MAX_STEPS=3170 PLANS_ARM=0-2,11-13,25-27 SFT_DATA="$ROOT/data/sft_paper8.jsonl" \
    OUT="$ROOT/output/product/sftAUX-9L" bash scripts/run_sft_mixture.sh > "$ROOT/output/aux_sft.log" 2>&1
  echo "[aux] SFT done $(date '+%H:%M')"
  CK2=$(ls -d output/product/sftAUX-9L/v*/checkpoint-* 2>/dev/null | sort -V | tail -1)
  [ -n "$CK2" ] && PRESET=paper GPU=0 BATCH=16 bash scripts/run_lm_eval.sh "$CK2" paper_AUX9 > "$ROOT/output/ev_AUX9.log" 2>&1
  echo "[aux] SFT eval done $(date '+%H:%M')"
fi
echo "[aux] DONE $(date '+%F %H:%M')"
