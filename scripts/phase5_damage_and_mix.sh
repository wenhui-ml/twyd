#!/usr/bin/env bash
# Phase 5: (a) damage-vs-budget curve, (b) does SFT data composition heal the
# MMLU loss that 524M tokens of PT caused?
#
# Context: on the paper's suite the 524M PT arms sit 0.5-1.8 pt below base and the
# whole deficit is MMLU; 100M tokens of chat SFT did not bring it back. Two cheap
# follow-ups, in order of value per GPU-hour:
#
#   1. score the 262M checkpoints -> damage at 262M vs 524M for both arms
#   2. from B@524M (the better arm), SFT 200M tokens on:
#        mix  = 50% chat + 50% instruction-formatted knowledge (Alpaca-GPT4-style)
#        chat = 100% ultrachat  (length control for the 100M run we already have)
#
#   setsid nohup bash scripts/phase5_damage_and_mix.sh > /tmp/phase5.log 2>&1 < /dev/null &
set -uo pipefail
ROOT=${TWYD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ROOT"

PY=/home/chenwh/miniconda3/envs/twyd/bin/python
SFT_TOKENS=${SFT_TOKENS:-200000000}
SFT_LR=${SFT_LR:-2e-5}

A262=$(ls -d "$ROOT"/output/formal-b/slice-full/*/v*/checkpoint-2000 2>/dev/null | tail -1)
B262=$(ls -d "$ROOT"/output/formal-b/slice-sel9/*/v*/checkpoint-2000 2>/dev/null | tail -1)
B524=$(ls -d "$ROOT"/output/formal-b/slice-sel9-cont/*/v*/checkpoint-2000 2>/dev/null | tail -1)
[ -n "$A262" ] && [ -n "$B262" ] && [ -n "$B524" ] || { echo "!! missing checkpoints"; exit 1; }

echo "=== phase5 start $(date '+%F %H:%M')"
echo "    A262=$A262"
echo "    B262=$B262"
echo "    B524=$B524"

echo "=== damage curve: A/B at 262M  $(date '+%H:%M')"
for spec in "A262:$A262" "B262:$B262"; do
  name=${spec%%:*}; ck=${spec#*:}
  PRESET=paper GPU=0 bash scripts/run_lm_eval.sh "$ck" "paper_$name" \
    || echo "    !! paper_$name failed" >&2
done
$PY scripts/summarize_avg8.py \
  output/lm_eval/paper_pt_base \
  output/lm_eval/paper_A262 output/lm_eval/paper_pt_A \
  output/lm_eval/paper_B262 output/lm_eval/paper_pt_B \
  | tee output/eval_damage_curve.txt

run_sft () {   # <name> <data>
  local name=$1 data=$2
  [ -f "$data" ] || { echo "!! missing $data" >&2; return 1; }
  echo "=== SFT $name  data=$data  $(date '+%H:%M')"
  WARM_START=pt_ckpt PT_CKPT="$B524" STAGE=sft TRUNC=delete LR="$SFT_LR" \
  TOKENS="$SFT_TOKENS" SEQ=2048 GPU=0,1 NPROC=2 PLANS="0-2,11-13,25-27" \
  LORA_MODULES="gate_proj,up_proj" DATA="$data" GATE=0 \
  OUTROOT="$ROOT/output/formal-b/$name" bash scripts/run_v15_w1.sh \
    || { echo "!! SFT $name failed" >&2; return 1; }
}

run_sft sft-sel9-mix200  "$ROOT/data/sft_mix_200M.jsonl"      || exit 1
run_sft sft-sel9-chat200 "$ROOT/data/sft_ultrachat_200M.jsonl" || exit 1

for name in sft-sel9-mix200 sft-sel9-chat200; do
  ck=$(ls -d "$ROOT"/output/formal-b/$name/*/v*/checkpoint-* 2>/dev/null | sort -V | tail -1)
  echo "=== paper-suite $name  ck=$ck  $(date '+%H:%M')"
  [ -n "$ck" ] && PRESET=paper GPU=0 bash scripts/run_lm_eval.sh "$ck" "paper_$name" \
    || echo "    !! eval $name failed" >&2
done

$PY scripts/summarize_avg8.py \
  output/lm_eval/paper_pt_base output/lm_eval/paper_pt_B \
  output/lm_eval/paper_sft_B output/lm_eval/paper_sft-sel9-chat200 \
  output/lm_eval/paper_sft-sel9-mix200 | tee output/eval_sft_mix.txt

echo "=== phase5 done $(date '+%F %H:%M')"
