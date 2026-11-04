#!/usr/bin/env bash
# Product gate: CORE-LR k=256 at the matched 262M-token budget (2,000 steps).
#
# ROADMAP's decisive gate: "k <= 256 within 0.5 Avg9 of full CORE, coordinates <= 25%
# of full CORE, real step-time / optimizer-memory win".
#
# Coverage note (2026-10-29): the first attempt used PLANS=all and died of HIP OOM at
# step 0 -- CORE-LR stores the complement basis U_perp of shape d_ff x (d_ff - H)
# (~54.5M bf16 elements per module); 28 layers x 3 projections is ~9.2 GB on top of
# the same activations that already used 19 GiB for the dense 9-layer arm.  So the arm
# runs on the SAME 9 layers x {gate, up} as the best arm B, which makes the comparison
# clean: identical coverage, identical init/data/steps, only the coordinate set differs
# (B: 226M dense coordinates; this arm: rank-256 factorised complement, ~68M).
#
#   trainable: complement in a rank-k factorisation only
#   frozen:    the plain-LRC col(T) path (freeze_col=true) -> the deployed matrix is
#              rebuilt from frozen slice + learned complement, then merged
#
# Two guards before committing ~11 h:
#   1. 30-step preflight: model_parameter_info must show a trainable count far below
#      1,057M, and grad_norm must be non-zero (the 2026-10-21 bug made it identically 0).
#   2. first logged loss must be ~185.9, i.e. the step-0 function is the slice init
#      (B is zero-initialised), not a fresh random adapter.
#
#   setsid nohup bash scripts/product_corelr_k256.sh > /tmp/corelr.log 2>&1 < /dev/null &
#   touch /tmp/STOP_CORELR      # abort before the next stage
set -uo pipefail
ROOT=${TWYD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ROOT"
PY=/home/chenwh/miniconda3/envs/twyd/bin/python
DATA=$ROOT/data/fw_score4_1B.jsonl
OUT=$ROOT/output/product
K=${K:-256}
PLANS_ARM=${PLANS_ARM:-0-2,11-13,25-27}     # same layers as arm B
MODULES_ARM=${MODULES_ARM:-gate_proj,up_proj}
TOK_FULL=1050000000          # cosine span, same as every other arm
STOP=/tmp/STOP_CORELR
mkdir -p "$OUT"
# ROCm: reduce fragmentation so a 21-22 GiB working set fits on a 24 GiB card
export PYTORCH_HIP_ALLOC_CONF=${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

arm_env () {   # <outroot> <tokens> <save_steps>
  echo "GPU=0,1 NPROC=2 SEQ=2048 SAVE_STEPS=$3 GATE=0 DATA=$DATA \
PLANS=$PLANS_ARM LORA_MODULES=$MODULES_ARM SEED=42 BRANCH=core K=$K LORA_MODE=core FACTORIZED=1 \
FREEZE_COL=true TOKENS=$2 WARM_START=slice OUTROOT=$1"
}

run_train () {   # <name> <outroot> <tokens> <save_steps> <log>
  echo "=== start $1  $(date '+%F %H:%M')"
  env $(arm_env "$2" "$3" "$4") bash "$ROOT/scripts/run_v15_w1.sh" > "$5" 2>&1
  echo "    rc=$? $(date '+%F %H:%M')"
}

wait_ckpt () {   # <outroot> <step>
  while :; do
    [ -e "$STOP" ] && return 1
    local vdir ck
    vdir=$(ls -d "$1"/*/v*/ 2>/dev/null | tail -1)
    if [ -n "$vdir" ]; then
      ck="$vdir/checkpoint-$2"
      [ -d "$ck" ] && { sleep 15; echo "$ck"; return 0; }   # let the save settle
    fi
    sleep 60
  done
}

# ---------- 1. preflight: 30 steps, verify the arm is real ----------
# Verified by hand on 2026-10-29 23:16:
#   model_parameter_info: 976.44M Params (30.67M Trainable [3.14%]), 3095.40M Buffers
#   grad_norm 46 (non-zero), first loss 185.95 (== the slice init), 22.38 GiB, 36.31 s/step
# so it is skipped by default; SKIP_PREFLIGHT=0 re-runs the guard.
if [ "${SKIP_PREFLIGHT:-1}" = "1" ]; then
  echo "=== preflight skipped (verified 2026-10-29 23:16: 30.67M trainable, grad_norm 46, loss 185.95, 22.38 GiB)"
else
run_train preflight "$OUT/preflight-corelr-k$K" 4000000 1000 /tmp/preflight_corelr.log
PLOG=$(ls -t "$OUT"/preflight-corelr-k$K/[0-9]*.log 2>/dev/null | head -1)
[ -n "$PLOG" ] || { echo "!!! preflight produced no arm log"; exit 1; }
$PY - "$PLOG" <<'EOF'
import re, sys
log = open(sys.argv[1], "r", errors="ignore").read()
m = re.search(r"\(([0-9.]+)M Trainable", log)
tr = float(m.group(1)) if m else -1.0
gn = [float(x) for x in re.findall(r"'grad_norm': '([0-9.eE+-]+)'", log)]
ls = [float(x) for x in re.findall(r"'loss': '([0-9.]+)'", log)]
sp = re.findall(r"'train_speed\(s/it\)': '([0-9.]+)'", log)
mem = re.findall(r"'memory\(GiB\)': '([0-9.]+)'", log)
steps = re.findall(r"'(\d+)/(\d+)'", log)
print(f"    trainable={tr}M  grad_norm last={gn[-1] if gn else None}  "
      f"loss first={ls[0] if ls else None}  step={sp[-1] if sp else '?'}s  "
      f"peak={mem[-1] if mem else '?'}GiB  steps={steps[-1] if steps else '?'}")
bad = []
if "OutOfMemoryError" in log:
    bad.append("OOM")
if steps and steps[-1][0] != steps[-1][1]:
    bad.append(f"only {steps[-1][0]}/{steps[-1][1]} steps ran")
if tr <= 0 or tr > 900:
    bad.append(f"trainable {tr}M is not a rank-k complement arm")
if not gn or gn[-1] == 0.0:
    bad.append("grad_norm is zero (the 2026-10-21 bug)")
if not ls or not (150 < ls[0] < 220):
    bad.append(f"first loss {ls[0] if ls else None} outside 150-220 (step-0 function moved)")
if bad:
    print("!!! preflight FAILED: " + "; ".join(bad)); sys.exit(1)
print("    preflight passed (a non-zero rc from the DDP teardown is expected)")
EOF
[ $? -ne 0 ] && exit 1
fi

# ---------- 2. the real arm: 2,000 steps = 262M tokens ----------
[ -e "$STOP" ] && { echo "=== stop flag present, not training"; exit 0; }
echo "=== full arm $(date '+%F %H:%M')"
setsid nohup env $(arm_env "$OUT/corelr-k$K" "$TOK_FULL" 1000) \
  bash "$ROOT/scripts/run_v15_w1.sh" > /tmp/corelr-k$K.log 2>&1 < /dev/null &
sleep 30
CK=$(wait_ckpt "$OUT/corelr-k$K" 2000) || { echo "=== aborted"; pkill -f 'swift/cli/pt.py'; exit 1; }
echo "    checkpoint-2000 -> $CK"
pkill -f 'swift/cli/pt.py'; sleep 10

# ---------- 3. score it against the matched arms ----------
A262=$(ls -d "$ROOT"/output/formal-b/slice-full/*/v*/checkpoint-2000 2>/dev/null | tail -1)
B262=$(ls -d "$ROOT"/output/formal-b/slice-sel9/*/v*/checkpoint-2000 2>/dev/null | tail -1)
BASE=/home/chenwh/models/LRC-1.5B-Base
echo "=== in-domain + OOD CE  $(date '+%F %H:%M')"
$PY scripts/eval_ce.py --data data/fineweb_edu_holdout.jsonl \
  --models "$BASE" "$A262" "$B262" "$CK" \
  --seq-len 512 --max-docs 400 --device cuda \
  --json-out output/eval_ce_corelr.json 2>&1 | tail -12
$PY scripts/eval_ce.py --data data/ultrachat_pt_35M.jsonl \
  --models "$BASE" "$A262" "$B262" "$CK" \
  --seq-len 512 --max-docs 400 --device cuda \
  --json-out output/eval_ce_corelr_ood.json 2>&1 | tail -12
echo "=== Avg8  $(date '+%F %H:%M')"
PRESET=paper GPU=0 bash scripts/run_lm_eval.sh "$CK" "paper2_corelr-k$K" 2>&1 | tail -3
$PY - <<EOF
import json,glob
T=['arc_easy','arc_challenge','commonsense_qa','piqa','winogrande','boolq','sciq','mmlu']
def row(tag):
    fs=sorted(glob.glob(f'output/lm_eval/{tag}/**/results_*.json',recursive=True))
    if not fs: return None
    d=json.load(open(fs[-1]))['results']; v=[d[t].get('acc_norm,none',d[t].get('acc,none')) for t in T if t in d]
    return sum(v)/len(v), d.get('mmlu',{}).get('acc,none',0), len(v)
for tag in ['paper_pt_base','paper_A262','paper_B262','paper2_corelr-k$K']:
    r=row(tag)
    print(f'{tag:24s} ' + (f'Avg8={r[0]:.4f} MMLU={r[1]:.3f} n={r[2]}' if r else 'no results'))
EOF
echo "=== done $(date '+%F %H:%M')"
