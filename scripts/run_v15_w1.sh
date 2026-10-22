#!/usr/bin/env bash
# V1.5 W1: layer-selective full-MLP ablations at 5M tokens, matched across arms.
#
# Reads the plans produced by scripts/rank_layers.py and runs one sequential arm
# per plan on a single GPU (same base, data, token budget, LR, seed).  Prints a
# compact table at the end: held-out-free, but peak VRAM / step time / loss are
# exactly what the V1.5 gate needs.
#
# Usage:
#   PLANS="0-2 0-2,4,7,9,14,20 all" GPU=0 bash scripts/run_v15_w1.sh
#   DRY=1 PLANS="0-2 all" bash scripts/run_v15_w1.sh      # print, don't train
set -uo pipefail

ROOT=${TWYD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
GPU=${GPU:-0}
PLANS=${PLANS:-all}
TOKENS=${TOKENS:-5000000}
SEQ=${SEQ:-512}
BS=${BS:-1}
GAS=${GAS:-32}
LR=${LR:-1e-4}
CLIP=${CLIP:-1.0}          # max_grad_norm; 10 silences the "everything is clipped" regime
SEED=${SEED:-42}
NPROC=${NPROC:-1}          # >1 = single job across N cards (swift reads NPROC_PER_NODE)
DS=${DS:-}                 # e.g. zero2_offload -- needed for the full 28L arm at seq=2048
BRANCH=${BRANCH:-dense}    # dense | core (with K=k)
K=${K:-0}
LORA_MODE=${LORA_MODE:-dense}   # dense arm = dense; CORE-LR arm = core
FACTORIZED=${FACTORIZED:-0}     # CORE-LR(k) uses the rank-k factorized complement
REPEATS=${REPEATS:-1}      # >1 only makes sense for nondeterministic kernels; the harness is bit-reproducible
FORCE=${FORCE:-0}          # 1 = re-run even if the arm already has a log
TEACHER=${TEACHER:-/home/chenwh/models/Llama-3.2-3B-Instruct}
MODEL=${MODEL:-/home/chenwh/models/LRC-1.5B-Base}
DATA=${DATA:-$ROOT/data/fineweb_edu_score4_60M.jsonl}
OUTROOT=${OUTROOT:-$ROOT/output/v15-w1}

# Accept 5000000 / 5e6 / 5M / 1B — bash arithmetic cannot parse the last two, and a
# typo here kills the arm in 0.2 s (which is how a whole night queue was lost once).
normalize_tokens() {
  local t="$1"
  case "$t" in
    *[Mm]) echo $(( ${t%[Mm]} * 1000000 )) ;;
    *[Bb]) echo $(( ${t%[Bb]} * 1000000000 )) ;;
    *[eE]*) python3 -c "print(int(float('$t')))" ;;
    *) echo "$t" ;;
  esac
}
TOKENS=$(normalize_tokens "$TOKENS")
if ! [[ "$TOKENS" =~ ^[0-9]+$ ]] || [ "$TOKENS" -le 0 ]; then
  echo "FATAL: bad TOKENS='$TOKENS' (use 5000000, 5e6 or 5M)" >&2
  exit 2
fi
STEPS=$(( TOKENS / (SEQ * BS * GAS * NPROC) ))
if [ "$STEPS" -lt 1 ]; then
  echo "FATAL: TOKENS=$TOKENS at seq=$SEQ bs=$BS gas=$GAS nproc=$NPROC gives 0 steps" >&2
  exit 2
fi
mkdir -p "$OUTROOT"

# Fail loudly on a wrong/missing corpus. This exact mistake (a driver passing DATASET=
# while the harness reads DATA=) silently trained the formal arm on the 60M corpus.
if [ -e "$DATA" ]; then
  echo "dataset=$DATA  docs=$(wc -l < "$DATA")  bytes=$(stat -c%s "$DATA")"
elif [[ "$DATA" == *"#"* ]] || [[ "$DATA" != */* ]]; then
  echo "dataset=$DATA  (remote/sharded id)"
else
  echo "FATAL: dataset '$DATA' does not exist" >&2
  exit 2
fi

echo "V1.5 W1: tokens=$TOKENS seq=$SEQ bs=$BS gas=$GAS nproc=$NPROC -> steps=$STEPS per arm"
for plan in $PLANS; do
  tag=$(echo "$plan" | tr ',' '_')
  # "all"/"none"/"full" mean "no layer filter"; any other value is a real range.
  # (The config treats an EMPTY spec as "all layers" — passing the literal string
  # "all" would be parsed as a layer name and yield an empty trainable set.)
  case "$plan" in
    all|none|full) plan_arg="" ;;
    *)             plan_arg="$plan" ;;
  esac
  out="$OUTROOT/$tag"
  # One arm at a time per tag. The mkdir is the lock: a second orchestrator that
  # was given overlapping PLANS must not write into the same directory (it would
  # spawn a second v*/ version, doubling cost and confusing every summary).
  if ! mkdir "$out.lock" 2>/dev/null; then
    echo "=== arm $plan: SKIP (locked by another run_v15_w1.sh; rm -rf $out.lock if stale)"
    continue
  fi
  trap 'rmdir "$out.lock" 2>/dev/null' EXIT
  if [ -n "$(ls -d "$out"/v*/logging.jsonl 2>/dev/null)" ] && [ "$FORCE" != "1" ]; then
    echo "=== arm $plan: SKIP (already has $(ls -d "$out"/v*/ | wc -l) run(s); FORCE=1 to redo)"
    rmdir "$out.lock" 2>/dev/null
    continue
  fi
  for rep in $(seq 1 "$REPEATS"); do
    rep_out="$out"
    [ "$rep" -gt 1 ] && rep_out="$OUTROOT/$tag-r$rep"
    echo "=== arm $plan (rep $rep/$REPEATS) -> $rep_out"
    if [ "${DRY:-0}" = "1" ]; then
      echo "DRY: would run CUDA_VISIBLE_DEVICES=$GPU BRANCH=dense K=0 LORA_LAYERS='$plan_arg' ... MAX_STEPS=$STEPS"
      continue
    fi
    CUDA_VISIBLE_DEVICES="$GPU" \
    GATE="${GATE:-1}" GATE_SEQ_LEN=64 USE_AUX=0 STAGE=pt BRANCH="$BRANCH" K="$K" \
    LORA_MODE="$LORA_MODE" FACTORIZED="$FACTORIZED" LORA_LAYERS="$plan_arg" LORA_MODULES="${LORA_MODULES:-}" \
    NPROC="$NPROC" DS="$DS" \
    BS="$BS" MAX_LENGTH="$SEQ" GRAD_ACCUM="$GAS" TRUNC=split MAX_STEPS="$STEPS" LR="$LR" \
    CLIP="$CLIP" SEED="$SEED" \
    DATASET="$DATA" MODEL="$MODEL" TEACHER="$TEACHER" \
    MODEL_TYPE=llama TEMPLATE=llama3_2 WARM_START=base OUT="$rep_out" \
    bash "$ROOT/scripts/train_core_lr_pilot.sh" > "$rep_out.log" 2>&1
    echo "    rc=$? log=$rep_out.log"
  done
  rmdir "$out.lock" 2>/dev/null
done

echo
if [ "${DRY:-0}" = "1" ]; then echo "(dry run: no summary)"; exit 0; fi
echo "=== summary (loss / token_acc / peak GiB / s per step)"
for plan in $PLANS; do
  tag=$(echo "$plan" | tr ',' '_')
  f=$(ls "$OUTROOT/$tag"/v0-*/logging.jsonl 2>/dev/null | head -1)
  [ -z "$f" ] && { printf '%-28s (no log)\n' "$plan"; continue; }
  python - "$plan" "$f" <<'PY'
import json, sys
plan, path = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(path)]
steps = [r for r in rows if 'loss' in r]
if not steps:
    print(f"{plan:<28s} (no step logged)"); raise SystemExit
last = steps[-1]
print(f"{plan:<28s} loss={last['loss']:.2f} acc={last['token_acc']:.4f} "
      f"peak={last['memory(GiB)']:.2f}GiB step={float(last['train_speed(s/it)']):.2f}s")
PY
done
