#!/usr/bin/env bash
# One-shot "milestone check" that does not waste GPU time:
#   wait for checkpoint-<N> -> stop training -> held-out CE -> resume from that checkpoint.
#
# The pause only happens once the checkpoint exists, so nothing is lost (the trainer would
# otherwise be killed between checkpoints and up to 1,000 steps of work thrown away).
#
#   STEPS=2000 setsid nohup bash scripts/pause_eval_resume.sh > /tmp/milestone.log 2>&1 < /dev/null &
set -uo pipefail

ROOT=${TWYD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
ARM=${ARM:-$ROOT/output/formal-b/slice-full}
STEPS=${STEPS:-2000}
TOTAL_TOKENS=${TOTAL_TOKENS:-1050000000}
SEQ=${SEQ:-2048}
TOK_PER_STEP=$(( SEQ * 32 * 2 ))          # seq * gas * ranks

VDIR=$(ls -d "$ARM"/*/v0-*/ 2>/dev/null | tail -1)
[ -z "$VDIR" ] && { echo "FATAL: no run dir under $ARM"; exit 2; }
CKPT="$VDIR/checkpoint-$STEPS"
echo "=== waiting for $CKPT  ($(date '+%F %H:%M'))"
while [ ! -d "$CKPT" ]; do sleep 60; done
sleep 10                                  # let the writer finish flushing
echo "=== checkpoint present, stopping training  ($(date '+%H:%M'))"
pkill -f run_v15_w1.sh; sleep 3
pkill -f 'swift/cli/pt.py'; sleep 10
pgrep -c -f 'swift/cli/pt.py' || true

echo "=== held-out CE for step $STEPS"
cd "$ROOT"
CUDA_VISIBLE_DEVICES=0 timeout 1800 python scripts/eval_ce.py \
  --data data/fineweb_edu_holdout.jsonl \
  --models /home/chenwh/models/LRC-1.5B-Base "$CKPT" \
  --seq-len 512 --max-docs 600 \
  --json-out "output/eval_$(basename "$ARM")_step$STEPS.json" > /tmp/eval_step.log 2>&1
grep -E 'CE=|dCE' /tmp/eval_step.log || tail -5 /tmp/eval_step.log

REMAIN=$(( TOTAL_TOKENS - STEPS * TOK_PER_STEP ))
echo "=== resuming from checkpoint-$STEPS, remaining tokens=$REMAIN  ($(date '+%H:%M'))"
GPU=0,1 NPROC=2 PLANS=all TOKENS="$REMAIN" SEQ="$SEQ" SAVE_STEPS=1000 \
  WARM_START=pt_ckpt PT_CKPT="$CKPT" DATA="$ROOT/data/fw_score4_1B.jsonl" GATE=0 \
  OUTROOT="$ROOT/output/formal-b/slice-full-cont" \
  setsid nohup bash "$ROOT/scripts/run_v15_w1.sh" > /tmp/pt_cont.log 2>&1 < /dev/null &
sleep 15
echo "=== resumed; monitor with: python scripts/watch_training.py output/formal-b/slice-full-cont"
