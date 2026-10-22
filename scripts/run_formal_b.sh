#!/usr/bin/env bash
# Formal matched comparison, option B: both arms at seq=2048 on 2x7900XTX.
#
# Feasibility was measured with scripts/test_2048.sh (3-step probes):
#   full 28L      2 cards DDP  20.96 GiB/card  25.52 s/step  (5,136 tok/s)
#   full 28L      1 card       22.22 GiB      24.44 s/step  (only 0.3 GiB headroom)
#   full 28L      2 cards ZeRO-2 21.39 GiB    64.83 s/step  (2.5x slower -> unused)
#   9L gate+up    2 cards DDP  18.80 GiB/card  24.67 s/step  (5,313 tok/s)
#
# 1.05B tokens x 2 arms, sequential (each arm needs both cards), 8,011 steps each.
#   ~56.7 h per arm -> ~4.7 days for PT, then SFT + Avg9.
set -uo pipefail

ROOT=${TWYD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
OUTROOT=${OUTROOT:-$ROOT/output/formal-b}
TOKENS=${TOKENS:-1050000000}     # total across both ranks
SEQ=${SEQ:-2048}
SAVE_STEPS=${SAVE_STEPS:-2000}
DATA_1B=${DATA_1B:-$ROOT/data/fw_score4_1B.jsonl}
mkdir -p "$OUTROOT"

run_arm () {   # name | plans | modules
  local name="$1" plans="$2" mods="$3" attempt=1 rc=0
  if [ -n "$(ls -d "$OUTROOT/$name"/*/v*/checkpoint-* 2>/dev/null)" ]; then
    echo "=== $name: SKIP (checkpoint exists)"; return 0
  fi
  while [ "$attempt" -le "${RETRIES:-3}" ]; do
    echo "=== PT arm $name attempt $attempt  $(date '+%F %H:%M')"
    local resume=0
    [ "$attempt" -gt 1 ] && resume=1
    GPU=0,1 NPROC=2 PLANS="$plans" LORA_MODULES="$mods" TOKENS="$TOKENS" SEQ="$SEQ" \
    SAVE_STEPS="$SAVE_STEPS" RESUME="$resume" DATA="$DATA_1B" GATE=0 OUTROOT="$OUTROOT/$name" \
    bash "$ROOT/scripts/run_v15_w1.sh" >> "$OUTROOT/$name.log" 2>&1
    rc=$?
    echo "    rc=$rc $(date '+%F %H:%M')"
    [ "$rc" -eq 0 ] && return 0
    attempt=$((attempt + 1))
    sleep 30
  done
  echo "!!! $name FAILED after ${RETRIES:-3} attempts -- aborting the pipeline" >&2
  return 1
}

echo "=== formal-B: tokens=$TOKENS (total) seq=$SEQ steps=$(( TOKENS / (SEQ * 32 * 2) ))/arm"
run_arm full-28L      all                 ""                 || exit 1
run_arm sel9-gateup   "0-2,11-13,25-27"   "gate_proj,up_proj" || exit 1

echo
echo "=== PT done $(date '+%F %H:%M'). Next (manual, needs the PT ckpt as warm start):"
echo "    WARM_START=pt_ckpt PT_CKPT=$OUTROOT/full-28L/.../checkpoint-8011 STAGE=sft \\"
echo "      DATASET=data/sft_ultrachat_200M.jsonl bash scripts/run_v15_w1.sh"
