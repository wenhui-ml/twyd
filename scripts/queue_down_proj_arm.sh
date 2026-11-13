#!/usr/bin/env bash
# Arm B of the "better training at fixed rank" line: train the module set the paper
# actually opens. App. B opens gate_proj / up_proj / down_proj; every arm in this study
# opened only gate/up, i.e. down_proj - a third of the MLP - has never been trained.
# Nothing about the deployed model changes: still low-rank, still the same parameter
# count, still Train-What-You-Deploy. Only which of the deployable factors receive
# gradient changes.
#
# Guarded by a 2-step smoke first: down_proj goes through the side="left" factor path of
# CoreLinear, which none of the 500M-token arms have exercised, and a 36 h run that dies
# at step 1 has already happened twice in this study.
#
#   bash scripts/queue_down_proj_arm.sh
set -u
cd "$(dirname "$0")/.."
ROOT=$PWD
MODS=gate_proj,up_proj,down_proj

while pgrep -f 'swift/cli/|lm_eval' >/dev/null 2>&1; do sleep 120; done
echo "[dp] GPU free, start $(date '+%F %H:%M')"

# ---- 1. smoke -----------------------------------------------------------------
rm -rf output/smoke_dp
env PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
  GPU=0,1 NPROC=2 SEQ=2048 BS=1 GAS=32 GATE=0 USE_AUX=0 STAGE=pt \
  DATA="$ROOT/data/pt_smoke.jsonl" PLANS=0-2 LORA_MODULES="$MODS" SEED=42 \
  BRANCH=core K=64 LORA_MODE=core FACTORIZED=1 FREEZE_COL=true PERP_BASIS=teacher \
  TOKENS=262144 WARM_START=slice OUTROOT="$ROOT/output/smoke_dp" \
  bash scripts/run_v15_w1.sh > "$ROOT/output/dp_smoke.log" 2>&1
if ! grep -q 'rc=0' "$ROOT/output/dp_smoke.log"; then
  echo "[dp] SMOKE FAILED - not launching the 36 h arm. See output/dp_smoke.log"; exit 1
fi
if ! grep -q 'model_parameter_info' output/smoke_dp/0-2.log 2>/dev/null; then
  echo "[dp] SMOKE produced no parameter line - aborting"; exit 1
fi
grep -a 'model_parameter_info' output/smoke_dp/0-2.log | head -1
echo "[dp] smoke ok $(date '+%H:%M')"
rm -rf output/smoke_dp

# ---- 2. PT 500M on the mixed corpus, gate+up+down ------------------------------
OUT="$ROOT/output/product/pt-mixed-dp-500s"
env PYTORCH_HIP_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  GPU=0,1 NPROC=2 SEQ=2048 SAVE_STEPS=800 BS=1 GAS=32 GATE=0 USE_AUX=0 STAGE=pt \
  DATA="$ROOT/data/pt_mixed.jsonl" PLANS=0-2,11-13 LORA_MODULES="$MODS" SEED=42 \
  BRANCH=core K=64 LORA_MODE=core FACTORIZED=1 FREEZE_COL=true PERP_BASIS=teacher \
  TOKENS=500000000 WARM_START=slice OUTROOT="$OUT" \
  bash scripts/run_v15_w1.sh > "$ROOT/output/dp_pt.log" 2>&1
echo "[dp] PT rc=$? $(date '+%H:%M')"
CK=$(ls -d "$OUT"/*/v*/checkpoint-* 2>/dev/null | sort -V | tail -1)
echo "[dp] ckpt=$CK"
[ -n "$CK" ] && PRESET=paper GPU=0 BATCH=16 bash scripts/run_lm_eval.sh "$CK" paper_PTdp > "$ROOT/output/ev_PTdp.log" 2>&1

# ---- 3. SFT 78M, same module set, plain CE (matches the mix9 SFT recipe) --------
if [ -n "$CK" ]; then
  env MODEL="$CK" MODEL_TYPE=llama TEMPLATE=llama3_2 LR=1e-4 LOSS_TYPE= \
    WARM_START_ARM=pt_ckpt PT_CKPT="$CK" BRANCH_ARM=dense LORA_MODE_ARM=dense \
    FACTORIZED_ARM=0 USE_AUX=0 LORA_MODULES_ARM="$MODS" \
    DATA_PROC=12 DATALOADER_WORKERS=4 MAX_LENGTH=2048 GRAD_ACCUM=32 SAVE_STEPS=1000 \
    MAX_STEPS=3170 PLANS_ARM=0-2,11-13,25-27 SFT_DATA="$ROOT/data/sft_paper8.jsonl" \
    OUT="$ROOT/output/product/sftDP-9L" bash scripts/run_sft_mixture.sh > "$ROOT/output/dp_sft.log" 2>&1
  echo "[dp] SFT rc=$? $(date '+%H:%M')"
  CK2=$(ls -d output/product/sftDP-9L/v*/checkpoint-* 2>/dev/null | sort -V | tail -1)
  [ -n "$CK2" ] && PRESET=paper GPU=0 BATCH=16 bash scripts/run_lm_eval.sh "$CK2" paper_DP9 > "$ROOT/output/ev_DP9.log" 2>&1
fi
echo "[dp] DONE $(date '+%F %H:%M')"
