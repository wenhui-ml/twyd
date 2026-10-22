#!/usr/bin/env bash
# V1.5 wave 2: the seed arms were dropped on purpose (training here is bit-reproducible
# — two concurrent identical runs matched every logged step — so multi-seed buys little
# for the GPU-hours). That budget is redirected to the recipes that move the product
# claim: projection-selective plans and the gate+up variants at the full 20M budget.
#
# Job spec: name | plans | tokens | seq | lora_modules | seed | clip | branch | k
#
#   GPU=0 WAIT_LOCK=$PWD/output/v15-night/knee-3L/0-2.lock      bash scripts/run_v15_night2.sh
#   GPU=1 WAIT_LOCK=$PWD/output/v15-night/corelr5M-k128/all.lock bash scripts/run_v15_night2.sh
set -uo pipefail

ROOT=${TWYD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
GPU=${GPU:-0}
WAIT_LOCK=${WAIT_LOCK:-}
NIGHT_OUT=${NIGHT_OUT:-$ROOT/output/v15-night}
BASE=/home/chenwh/models/LRC-1.5B-Base
HOLDOUT=${HOLDOUT:-$ROOT/data/fineweb_edu_holdout.jsonl}
mkdir -p "$NIGHT_OUT"

job_gpu0=(
  "gateup-0-5-seq2k|0-5|5000000|2048|gate_proj,up_proj|42|1.0|dense|0"
  "gateup-1L|0|5000000|512|gate_proj,up_proj|42|1.0|dense|0"
  "gateup-0-2|0-2|5000000|512|gate_proj,up_proj|42|1.0|dense|0"
  "gateup-0-5-20M|0-5|20000000|512|gate_proj,up_proj|42|1.0|dense|0"
  "seq2k-spread|0-2,11-13,25-27|5000000|2048||42|1.0|dense|0"
)
job_gpu1=(
  "corelr5M-k128-clip10|all|5000000|512||42|10.0|corelr|128"
  "gateup-spread-5M|0-2,11-13,25-27|5000000|512|gate_proj,up_proj|42|1.0|dense|0"
  "gateup-spread-20M|0-2,11-13,25-27|20000000|512|gate_proj,up_proj|42|1.0|dense|0"
)

if [ "$GPU" = "0" ]; then JOBS_LIST=("${job_gpu0[@]}"); else JOBS_LIST=("${job_gpu1[@]}"); fi

echo "=== V1.5 wave 2 on GPU $GPU ($(date '+%H:%M'))"
printf '%-24s %-16s %8s %5s %-22s %5s %5s %-9s\n' job layers tokens seq modules seed clip branch
for spec in "${JOBS_LIST[@]}"; do
  IFS='|' read -r name plans tok seq mods seed clip branch k <<< "$spec"
  printf '%-24s %-16s %8s %5s %-22s %5s %5s %-9s\n' "$name" "$plans" "$tok" "$seq" \
         "${mods:-all}" "$seed" "$clip" "$branch/$k"
done
[ "${DRY:-0}" = "1" ] && exit 0

if [ -n "$WAIT_LOCK" ]; then
  waited=0
  while [ -d "$WAIT_LOCK" ] && [ "$waited" -lt 21600 ]; do
    sleep 30; waited=$((waited + 30))
    [ $((waited % 600)) -eq 0 ] && echo "  waiting on $WAIT_LOCK (${waited}s)"
  done
  echo "=== lock released after ${waited}s -> starting ($(date '+%H:%M'))"
fi

if [ "${SMOKE:-1}" = "1" ]; then
  rm -rf "$NIGHT_OUT/_smoke-gpu$GPU"
  GPU="$GPU" PLANS=all TOKENS=32768 OUTROOT="$NIGHT_OUT/_smoke-gpu$GPU" \
    bash "$ROOT/scripts/run_v15_w1.sh" > "$NIGHT_OUT/_smoke-gpu$GPU.log" 2>&1
  if [ -z "$(ls -d "$NIGHT_OUT"/_smoke-gpu$GPU/*/v*/checkpoint-* 2>/dev/null)" ]; then
    echo "FATAL: preflight failed -- see $NIGHT_OUT/_smoke-gpu$GPU.log"
    tail -5 "$NIGHT_OUT/_smoke-gpu$GPU.log"; exit 3
  fi
  echo "=== preflight ok (2 steps trained)"
  rm -rf "$NIGHT_OUT/_smoke-gpu$GPU" "$NIGHT_OUT/_smoke-gpu$GPU.log"
fi

for spec in "${JOBS_LIST[@]}"; do
  IFS='|' read -r name plans tok seq mods seed clip branch k <<< "$spec"
  out="$NIGHT_OUT/$name"
  if [ -n "$(ls -d "$out"/v*/logging.jsonl 2>/dev/null)" ]; then
    echo "=== $name: SKIP (done)"; continue
  fi
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
    echo "    eval: $(grep -o 'dCE=[^ ]*' "$NIGHT_OUT/$name.log" | tail -1)"
  fi
done

echo "=== wave 2 done ($(date '+%H:%M'))"
