#!/usr/bin/env bash
# Spend the post-PT idle GPU on the evaluation that CE alone cannot give us.
#
# Waits for phase2_matched.sh's paired CE (output/eval_matched_525M.json) and for
# every trainer to exit, then runs, in this order:
#   1. out-of-distribution CE on chat text (the holdout is fineweb-edu, i.e. in-domain)
#   2. the fixed 6-task lm-eval suite on base / A@525M / B@525M
#   3. paired McNemar + recovery fraction (scripts/eval_mcnemar.py)
#
#   setsid nohup bash scripts/after_pt_lm_eval.sh > /tmp/after_pt.log 2>&1 < /dev/null &
set -uo pipefail
ROOT=${TWYD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ROOT"

BASE=/home/chenwh/models/LRC-1.5B-Base
A_CK=$(ls -d "$ROOT"/output/formal-b/slice-full-cont/*/v*/checkpoint-2000 2>/dev/null | tail -1)
CE_JSON=$ROOT/output/eval_matched_525M.json
OOD_DATA=$ROOT/data/ultrachat_pt_35M.jsonl
export CUDA_VISIBLE_DEVICES=0

echo "=== after-pt eval start $(date '+%F %H:%M')"
[ -n "$A_CK" ] || { echo "!! no A checkpoint"; exit 1; }
echo "    A = $A_CK"

# 1) wait for the paired CE table and an idle GPU
while [ ! -f "$CE_JSON" ] || pgrep -f 'swift/cli/pt.py' >/dev/null; do
  sleep 120
done
B_CK=$(ls -d "$ROOT"/output/formal-b/slice-sel9-cont/*/v*/checkpoint-2000 2>/dev/null | tail -1)
[ -n "$B_CK" ] || { echo "!! no B checkpoint"; exit 1; }
echo "--- paired CE ready, GPU free ($(date '+%H:%M'))"
echo "    B = $B_CK"
cat "$ROOT/output/eval_matched_525M.txt" 2>/dev/null

# 2) out-of-distribution CE: ultrachat is a different distribution from the
#    fineweb-edu training corpus, so a win here is not just in-domain compression.
echo "=== OOD CE (ultrachat) $(date '+%H:%M')"
python scripts/eval_ce.py --data "$OOD_DATA" \
  --models "$BASE" "$A_CK" "$B_CK" --seq-len 512 --max-docs 600 --device cuda \
  --json-out output/eval_ood_525M.json > /tmp/eval_ood.log 2>&1
grep -E 'CE=' /tmp/eval_ood.log
python scripts/eval_pairwise.py output/eval_ood_525M.json > output/eval_ood_525M.txt 2>&1
cat output/eval_ood_525M.txt

# 3) downstream suite + paired statistics
echo "=== lm-eval suite $(date '+%H:%M')"
bash scripts/run_lm_eval_arms.sh "$A_CK" "$B_CK" m525

echo "=== after-pt eval done $(date '+%F %H:%M')"
