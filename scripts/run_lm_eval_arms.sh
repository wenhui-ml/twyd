#!/usr/bin/env bash
# Run the fixed lm-eval suite on base / arm A / arm B and print the paired table.
#
#   bash scripts/run_lm_eval_arms.sh <A_ckpt> <B_ckpt> <tag_prefix>
#
# e.g. after phase2 finishes:
#   bash scripts/run_lm_eval_arms.sh \
#     output/formal-b/slice-full-cont/all/v0-*/checkpoint-2000 \
#     output/formal-b/slice-sel9-cont/*/v0-*/checkpoint-2000 m525
#
# Runs on one GPU, sequentially (~10 min per model, ~30 min total). Do not start
# while a PT run owns both cards: it will slow that run down and pollute its s/step.
set -uo pipefail
ROOT=${TWYD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ROOT"

A_CK=${1:?usage: run_lm_eval_arms.sh <A_ckpt> <B_ckpt> <tag_prefix>}
B_CK=${2:?usage: run_lm_eval_arms.sh <A_ckpt> <B_ckpt> <tag_prefix>}
PREFIX=${3:?usage: run_lm_eval_arms.sh <A_ckpt> <B_ckpt> <tag_prefix>}
BASE=${BASE:-/home/chenwh/models/LRC-1.5B-Base}
GPU=${GPU:-0}

for ck in "$A_CK" "$B_CK"; do
  [ -d "$ck" ] || { echo "missing checkpoint: $ck" >&2; exit 1; }
done

run_one () {   # <ckpt> <tag>
  echo "=== $(date '+%F %H:%M')  $2  <- $1"
  GPU=$GPU bash scripts/run_lm_eval.sh "$1" "$2" || echo "    !! $2 failed" >&2
}

run_one "$BASE" "${PREFIX}_base"
run_one "$A_CK" "${PREFIX}_A"
run_one "$B_CK" "${PREFIX}_B"

OUT=output/eval_lm_eval_${PREFIX}.txt
python scripts/eval_mcnemar.py --metric acc_norm \
  "base=output/lm_eval/${PREFIX}_base" \
  "A=output/lm_eval/${PREFIX}_A" \
  "B=output/lm_eval/${PREFIX}_B" | tee "$OUT"
echo "=== wrote $OUT  ($(date '+%F %H:%M'))"
