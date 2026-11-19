#!/usr/bin/env bash
# Short 0.5-epoch pure-CE SFT wrap-up on top of Dense-TWYD 2.1 (checkpoint-1525).
#
# Why 0.5 epoch (1585 steps) + LR=5e-5 (H2 recipe from docs/HANDOFF.md §7.2):
#   - Captures the bulk of SFT behaviour/format conversion (~3.3 h on 2 GPUs)
#   - Cosine LR decays cleanly to 0 at step 1585 (no mid-schedule truncation)
#   - Minimizes MMLU forgetting compared to 1.0 epoch LR=1e-4 (which cost -0.18 to -0.37 pt)
#
#   bash scripts/queue_dense_v2_1_sft.sh
set -u
cd "$(dirname "$0")/.."
ROOT=$PWD
CK="$ROOT/output/product/pt-dense-v2_1-200M/0-3_10-15/v0-20261118-101426/checkpoint-1525"
OUT="$ROOT/output/product/sftDenseV2_1-halfep"

if [ ! -d "$CK" ]; then
  echo "[sft-v2.1] ERROR: missing PT ckpt $CK" >&2
  exit 1
fi

echo "[sft-v2.1] start $(date '+%F %H:%M') pt_ckpt=$CK"
rm -rf "$OUT"

env PYTORCH_HIP_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  MODEL="$CK" MODEL_TYPE=llama TEMPLATE=llama3_2 LR=5e-5 LOSS_TYPE= \
  WARM_START_ARM=pt_ckpt PT_CKPT="$CK" BRANCH_ARM=dense LORA_MODE_ARM=dense \
  K=64 FACTORIZED_ARM=0 USE_AUX=0 LORA_MODULES_ARM=gate_proj,up_proj \
  DATA_PROC=12 DATALOADER_WORKERS=4 MAX_LENGTH=2048 GRAD_ACCUM=32 SAVE_STEPS=1585 \
  MAX_STEPS=1585 PLANS_ARM=0-2,11-13,25-27 SFT_DATA="$ROOT/data/sft_paper8.jsonl" \
  OUT="$OUT" bash scripts/run_sft_mixture.sh > "$ROOT/output/sft_v2_1.log" 2>&1
echo "[sft-v2.1] SFT rc=$? $(date '+%H:%M')"

CK2=$(ls -d "$OUT"/v*/checkpoint-* 2>/dev/null | sort -V | tail -1)
echo "[sft-v2.1] ckpt=$CK2"
if [ -n "$CK2" ]; then
  PRESET=paper GPU=0 BATCH=16 bash scripts/run_lm_eval.sh "$CK2" paper_SFTdense_v2_1 > "$ROOT/output/ev_SFTdense_v2_1.log" 2>&1
  echo "[sft-v2.1] eval rc=$? $(date '+%H:%M')"
fi
echo "[sft-v2.1] DONE $(date '+%F %H:%M')"
