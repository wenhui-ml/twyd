#!/usr/bin/env bash
# Phase 4: paper-aligned downstream numbers + the SFT deliverable.
#
#   1. wait for the MMLU pre-cache to finish (no GPU needed for that)
#   2. score base / A@524M / B@524M on the paper's suite (Avg8: logiqa blocked)
#   3. SFT both arms for 100M tokens each from their 524M PT checkpoints
#   4. score the two SFT students on the same suite
#
#   setsid nohup bash scripts/phase4_paper9_sft.sh > /tmp/phase4.log 2>&1 < /dev/null &
set -uo pipefail
ROOT=${TWYD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ROOT"

BASE=/home/chenwh/models/LRC-1.5B-Base
SFT_DATA=$ROOT/data/sft_ultrachat_200M.jsonl
SFT_TOKENS=${SFT_TOKENS:-100000000}
SFT_LR=${SFT_LR:-2e-5}

A_PT=$(ls -d "$ROOT"/output/formal-b/slice-full-cont/*/v*/checkpoint-2000 2>/dev/null | tail -1)
B_PT=$(ls -d "$ROOT"/output/formal-b/slice-sel9-cont/*/v*/checkpoint-2000 2>/dev/null | tail -1)
[ -n "$A_PT" ] && [ -n "$B_PT" ] || { echo "!! missing PT checkpoints"; exit 1; }

echo "=== phase4 start $(date '+%F %H:%M')"
echo "    A_PT=$A_PT"
echo "    B_PT=$B_PT"

# 1) MMLU is 57 subject configs; without them the mmlu task dies mid-run.
echo "--- waiting for mmlu pre-cache"
for _ in $(seq 1 240); do
  grep -q '^done$' /tmp/precache_mmlu.log 2>/dev/null && break
  sleep 30
done
echo "    mmlu cache: $(grep -c 'ok$' /tmp/precache_mmlu.log 2>/dev/null)/57 subjects  ($(date '+%H:%M'))"

# 2) paper-aligned suite on the PT arms
for spec in "base:$BASE" "A:$A_PT" "B:$B_PT"; do
  name=${spec%%:*}; ck=${spec#*:}
  echo "=== paper-suite PT $name  $(date '+%H:%M')"
  PRESET=paper GPU=0 bash scripts/run_lm_eval.sh "$ck" "paper_pt_$name" \
    || echo "    !! paper_pt_$name failed" >&2
done
python scripts/eval_mcnemar.py --metric acc_norm \
  "base=output/lm_eval/paper_pt_base" \
  "A=output/lm_eval/paper_pt_A" \
  "B=output/lm_eval/paper_pt_B" | tee output/eval_paper_pt_avg8.txt

# 3) SFT: same arm definitions as PT (A = all core, B = 9 layers x gate/up),
#    identical data / tokens / LR / schedule, so the comparison stays matched.
run_sft () {   # <name> <plans> <modules> <pt_ckpt>
  local name=$1 plans=$2 mods=$3 ck=$4
  echo "=== SFT $name  plans='$plans' mods='$mods'  $(date '+%H:%M')"
  WARM_START=pt_ckpt PT_CKPT="$ck" STAGE=sft TRUNC=delete LR="$SFT_LR" \
  TOKENS="$SFT_TOKENS" SEQ=2048 GPU=0,1 NPROC=2 PLANS="$plans" LORA_MODULES="$mods" \
  DATA="$SFT_DATA" GATE=0 OUTROOT="$ROOT/output/formal-b/$name" \
  bash scripts/run_v15_w1.sh || { echo "!! SFT $name failed" >&2; return 1; }
}
run_sft sft-full all                  ""                     "$A_PT" || exit 1
run_sft sft-sel9 "0-2,11-13,25-27"    "gate_proj,up_proj"    "$B_PT" || exit 1

A_SFT=$(ls -d "$ROOT"/output/formal-b/sft-full/*/v*/checkpoint-* 2>/dev/null | sort -V | tail -1)
B_SFT=$(ls -d "$ROOT"/output/formal-b/sft-sel9/*/v*/checkpoint-* 2>/dev/null | sort -V | tail -1)
echo "    A_SFT=$A_SFT"
echo "    B_SFT=$B_SFT"

# 4) paper-aligned suite on the SFT students
for spec in "A:$A_SFT" "B:$B_SFT"; do
  name=${spec%%:*}; ck=${spec#*:}
  echo "=== paper-suite SFT $name  $(date '+%H:%M')"
  PRESET=paper GPU=0 bash scripts/run_lm_eval.sh "$ck" "paper_sft_$name" \
    || echo "    !! paper_sft_$name failed" >&2
done
python scripts/eval_mcnemar.py --metric acc_norm \
  "A=output/lm_eval/paper_sft_A" \
  "B=output/lm_eval/paper_sft_B" | tee output/eval_paper_sft_avg8.txt

echo "=== phase4 done $(date '+%F %H:%M')"
