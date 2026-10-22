#!/usr/bin/env bash
# V1.5 night queue: short matched arms that each close one specific caveat, then a
# single joint held-out evaluation so the morning starts with a table.
#
# Job spec (pipe-separated):
#   name | plans | tokens | seq | lora_modules | seed | clip | branch | k
#
# Usage:
#   GPU=0 WAIT_LOCK=$PWD/output/v15-w1-20M/0-2_11-13_25-27.lock bash scripts/run_v15_night.sh
#   DRY=1 bash scripts/run_v15_night.sh            # print the schedule only
set -uo pipefail

ROOT=${TWYD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
GPU=${GPU:-0}
WAIT_LOCK=${WAIT_LOCK:-}
NIGHT_OUT=${NIGHT_OUT:-$ROOT/output/v15-night}
BASE=/home/chenwh/models/LRC-1.5B-Base
HOLDOUT=${HOLDOUT:-$ROOT/data/fineweb_edu_holdout.jsonl}
mkdir -p "$NIGHT_OUT"

job_gpu0=(
  "mod-gateup|0-5|5000000|512|gate_proj,up_proj|42|1.0|dense|0"
  "mod-down|0-5|5000000|512|down_proj|42|1.0|dense|0"
  "knee-1L|0|5000000|512||42|1.0|dense|0"
  "knee-3L|0-2|5000000|512||42|1.0|dense|0"
  "seed13-0-5|0-5|5000000|512||13|1.0|dense|0"
  "seed13-all|all|5000000|512||13|1.0|dense|0"
  "seq2k-spread|0-2,11-13,25-27|5000000|2048||42|1.0|dense|0"
)
job_gpu1=(
  "seq2k-0-5|0-5|5000000|2048||42|1.0|dense|0"
  "clip10-all|all|5000000|512||42|10.0|dense|0"
  "clip10-0-5|0-5|5000000|512||42|10.0|dense|0"
  "corelr5M-k128|all|5000000|512||42|1.0|corelr|128"
  "corelr5M-k128-clip10|all|5000000|512||42|10.0|corelr|128"
  "seed7-0-5|0-5|5000000|512||7|1.0|dense|0"
)

if [ "$GPU" = "0" ]; then JOBS_LIST=("${job_gpu0[@]}"); else JOBS_LIST=("${job_gpu1[@]}"); fi

echo "=== V1.5 night queue on GPU $GPU ($(date '+%H:%M'))"
printf '%-24s %-16s %6s %5s %-22s %5s %5s %-7s\n' job layers tokens seq modules seed clip branch
for spec in "${JOBS_LIST[@]}"; do
  IFS='|' read -r name plans tok seq mods seed clip branch k <<< "$spec"
  printf '%-24s %-16s %7s %5s %-22s %5s %5s %-7s\n' "$name" "$plans" "$tok" "$seq" \
         "${mods:-all}" "$seed" "$clip" "$branch/$k"
done
[ "${DRY:-0}" = "1" ] && exit 0

# Wait for the currently running 20M arm to release its lock (bounded).
if [ -n "$WAIT_LOCK" ]; then
  waited=0
  while [ -d "$WAIT_LOCK" ] && [ "$waited" -lt 21600 ]; do
    sleep 30; waited=$((waited + 30))
    [ $((waited % 600)) -eq 0 ] && echo "  waiting on $WAIT_LOCK (${waited}s)"
  done
  echo "=== lock released after ${waited}s -> starting ($(date '+%H:%M'))"
fi

# Preflight: 2 steps of a throwaway arm. A bad spec (e.g. a non-integer TOKENS, a
# bad swift flag) otherwise kills every job in 0.2 s and silently wastes the night.
if [ "${SMOKE:-1}" = "1" ]; then
  rm -rf "$NIGHT_OUT/_smoke"
  GPU="$GPU" PLANS=all TOKENS=32768 OUTROOT="$NIGHT_OUT/_smoke" \
    bash "$ROOT/scripts/run_v15_w1.sh" > "$NIGHT_OUT/_smoke.log" 2>&1
  if [ -z "$(ls -d "$NIGHT_OUT"/_smoke/*/v*/checkpoint-* 2>/dev/null)" ]; then
    echo "FATAL: preflight failed -- nothing trained. See $NIGHT_OUT/_smoke.log"
    tail -5 "$NIGHT_OUT/_smoke.log"
    exit 3
  fi
  echo "=== preflight ok (2 steps trained)"
  rm -rf "$NIGHT_OUT/_smoke" "$NIGHT_OUT/_smoke.log"
fi

for spec in "${JOBS_LIST[@]}"; do
  IFS='|' read -r name plans tok seq mods seed clip branch k <<< "$spec"
  out="$NIGHT_OUT/$name"
  if [ -n "$(ls -d "$out"/v*/logging.jsonl 2>/dev/null)" ]; then
    echo "=== $name: SKIP (done)"; continue
  fi
  # CORE-LR needs the core tuner + the rank-k factorized complement (docs/TECHNICAL.md §2.4);
  # the dense arms keep the plain full-MLP path.
  case "$branch" in
    core|corelr) br=core; lmode=core; fact=1 ;;
    *)           br=dense; lmode=dense; fact=0 ;;
  esac
  echo "=== job $name  ($(date '+%H:%M'))"
  GPU="$GPU" PLANS="$plans" TOKENS="$tok" SEQ="$seq" OUTROOT="$out" \
  LORA_MODULES="$mods" SEED="$seed" CLIP="$clip" BRANCH="$br" K="$k" \
  LORA_MODE="$lmode" FACTORIZED="$fact" \
  bash "$ROOT/scripts/run_v15_w1.sh" >> "$NIGHT_OUT/$name.log" 2>&1
  echo "    rc=$? ($(date '+%H:%M'))"
  ckpt=$(ls -d "$out"/*/v*/checkpoint-* 2>/dev/null | tail -1)
  if [ -n "$ckpt" ]; then
    CUDA_VISIBLE_DEVICES="$GPU" python "$ROOT/scripts/eval_ce.py" --data "$HOLDOUT" \
      --models "$BASE" "$ckpt" --seq-len 512 --max-docs 600 --device cuda \
      --json-out "$NIGHT_OUT/$name.eval.json" >> "$NIGHT_OUT/$name.log" 2>&1
    python "$ROOT/scripts/eval_pairwise.py" "$NIGHT_OUT/$name.eval.json" 2>/dev/null \
      | tail -1 >> "$NIGHT_OUT/$name.log"
    echo "    eval: $(grep -o 'dCE=[^ ]*' "$NIGHT_OUT/$name.log" | tail -1)"
  fi
done

echo "=== joint evaluation over every night checkpoint  ($(date '+%H:%M'))"
ckpts=$(ls -d "$NIGHT_OUT"/*/*/v*/checkpoint-* 2>/dev/null)
if [ -n "$ckpts" ]; then
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$GPU" python "$ROOT/scripts/eval_ce.py" --data "$HOLDOUT" \
    --models $BASE $ckpts --seq-len 512 --max-docs 600 --device cuda \
    --json-out "$NIGHT_OUT/night_all.json" > "$NIGHT_OUT/night_all.log" 2>&1
  python "$ROOT/scripts/eval_pairwise.py" "$NIGHT_OUT/night_all.json" \
    > "$NIGHT_OUT/night_pairwise.txt" 2>&1
  cat "$NIGHT_OUT/night_pairwise.txt"
fi
echo "=== night queue done ($(date '+%H:%M'))"
