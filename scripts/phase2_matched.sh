#!/usr/bin/env bash
# Phase 2, option 1: stop both arms at 525M tokens (4,000 steps) with *identical* LR paths.
#
# Arm A already ran as two cosine segments (0-262M over 8,010 steps, then 262M-> over 6,010).
# Arm B must copy that shape exactly, otherwise the comparison is polluted by a different
# LR trajectory:
#     seg1: 2,000 steps of a cosine over 8,010   -> stop when checkpoint-2000 lands
#     seg2: 2,000 steps of a cosine over 6,010   -> stop when checkpoint-2000 lands
# Both arms then sit at 525M tokens / 4,000 steps with the same schedule shape.
#
#   setsid nohup bash scripts/phase2_matched.sh > /tmp/phase2.log 2>&1 < /dev/null &
set -uo pipefail

ROOT=${TWYD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ROOT"
DATA=$ROOT/data/fw_score4_1B.jsonl
BASE=/home/chenwh/models/LRC-1.5B-Base
A_ARM=$ROOT/output/formal-b/slice-full-cont
A_STEP=2000
SEL_PLANS='0-2,11-13,25-27'
SEL_MODS='gate_proj,up_proj'
B_ARM=$ROOT/output/formal-b/slice-sel9
B_CONT=$ROOT/output/formal-b/slice-sel9-cont
TOK_FULL=1050000000            # max_steps = 8,010
TOK_CONT=787856000             # max_steps = 6,010

wait_ckpt () {   # <outside dir> <step> -> prints checkpoint path
  local vdir ck
  while :; do
    vdir=$(ls -d "$1"/*/v*/ 2>/dev/null | tail -1)
    if [ -n "$vdir" ]; then
      ck="$vdir/checkpoint-$2"
      [ -d "$ck" ] && { sleep 10; echo "$ck"; return 0; }
    fi
    sleep 60
  done
}

stop_train () {
  pkill -f run_v15_w1.sh; sleep 3
  pkill -f 'swift/cli/pt.py'; sleep 10
  echo "    stopped ($(pgrep -c -f 'swift/cli/pt.py' || true) trainers left)"
}

launch () {      # <log> <plan> <mods> <tokens> <out> [resume-from-ckpt]
  local log="$1" plan="$2" mods="$3" tok="$4" out="$5" mckpt="${6:-}"
  local enva=(GPU=0,1 NPROC=2 PLANS="$plan" LORA_MODULES="$mods" TOKENS="$tok"
              SEQ=2048 SAVE_STEPS=1000 WARM_START=slice DATA="$DATA" GATE=0 OUTROOT="$out")
  if [ -n "$mckpt" ]; then
    enva+=(WARM_START=pt_ckpt PT_CKPT="$mckpt")
  fi
  env "${enva[@]}" setsid nohup bash "$ROOT/scripts/run_v15_w1.sh" > "$log" 2>&1 < /dev/null &
  sleep 20
  echo "    launched: $out ($(date '+%H:%M'))"
}

echo "=== phase2 start $(date '+%F %H:%M')"
echo "--- A: waiting for checkpoint-$A_STEP under $A_ARM"
A_CK=$(wait_ckpt "$A_ARM" "$A_STEP")
echo "--- A checkpoint ready: $A_CK"
stop_train

echo "--- B seg1: 2,000 steps, cosine over 8,010  ($(date '+%H:%M'))"
launch /tmp/B_seg1.log "$SEL_PLANS" "$SEL_MODS" "$TOK_FULL" "$B_ARM" ""
B_CK1=$(wait_ckpt "$B_ARM" "$A_STEP")
echo "--- B seg1 checkpoint ready: $B_CK1"
stop_train

echo "--- B seg2: 2,000 steps, cosine over 6,010  ($(date '+%H:%M'))"
launch /tmp/B_seg2.log "$SEL_PLANS" "$SEL_MODS" "$TOK_CONT" "$B_CONT" "$B_CK1"
B_CK2=$(wait_ckpt "$B_CONT" "$A_STEP")
echo "--- B seg2 checkpoint ready: $B_CK2"
stop_train

echo "--- final paired evaluation (base / A@525M / B@525M)  ($(date '+%H:%M'))"
CUDA_VISIBLE_DEVICES=0 python scripts/eval_ce.py --data data/fineweb_edu_holdout.jsonl \
  --models "$BASE" "$A_CK" "$B_CK2" --seq-len 512 --max-docs 600 --device cuda \
  --json-out output/eval_matched_525M.json > /tmp/eval_matched.log 2>&1
grep -E 'CE=|dCE' /tmp/eval_matched.log
python scripts/eval_pairwise.py output/eval_matched_525M.json > output/eval_matched_525M.txt 2>&1
cat output/eval_matched_525M.txt
echo "=== phase2 done $(date '+%F %H:%M')"
