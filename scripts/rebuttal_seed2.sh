#!/usr/bin/env bash
# Rebuttal support: the two GPU experiments we can still run without a cluster.
#
# Reviewers (all five) ask for multi-seed. At 10B tokens that is impossible on this
# box (one 10B arm = ~51-57 h), so we get the three most useful reduced-budget facts
# for ~63-100 h of GPU at 524M tokens per arm (measured: dense 23.5 s/step -> 26 h,
# plain-LRC 33.4 s/step -> 37 h):
#
#   1. plain-LRC / BRANCH=plain @ 524M, seed 42  -> matched col(T) baseline (396M
#      trainable, gate-verified). Together with the existing slice-full arm (seed 42,
#      524M) this is the paper's contrast at matched extra budget, with the "it is
#      just more training" confound removed.
#   2. full-matrix / BRANCH=dense @ 524M, seed 43 -> sigma_run against slice-full@42.
#   3. plain-LRC @ 524M, seed 43                 -> a second seed of the contrast itself
#      (only if it still fits before the rebuttal text freezes, see the cutoff below).
#
# Both arms follow the same two-segment cosine shape as the original runs, so the
# only difference between seeds is the data order.
#
#   setsid nohup bash scripts/rebuttal_seed2.sh > /tmp/rebuttal_seed.log 2>&1 < /dev/null &
#   touch /tmp/STOP_SEED2        # abort before the next stage starts
set -uo pipefail
ROOT=${TWYD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ROOT"
DATA=$ROOT/data/fw_score4_1B.jsonl
OUT=$ROOT/output/rebuttal
TOK_FULL=1050000000
TOK_CONT=787856000

mkdir -p "$OUT"
echo "=== rebuttal seed runs start $(date '+%F %H:%M')"

# Do not fight the product experiment for the two cards. It trains with
# swift/cli/... and then evaluates with lm_eval on GPU 0, so wait for the whole
# driver to exit rather than just for the trainer process.
while pgrep -f 'swift/cli/' >/dev/null \
   || pgrep -f 'phase5_damage_and_mix' >/dev/null \
   || pgrep -f 'lm_eval' >/dev/null; do sleep 120; done
echo "--- GPUs free ($(date '+%H:%M'))"

preflight () {   # <plans> <mods> <branch> -> 0 if the arm can actually train
  local plans=$1 mods=$2 branch=$3
  local dir=$OUT/preflight-$branch
  rm -rf "$dir"; mkdir -p "$dir"
  # 30 steps rather than 2: the same run doubles as the measured cost probe for the
  # plain-LRC arm (0.40B trainable), which the paper reports only analytically. We
  # already have measured numbers for 1.06B and 226M trainable arms.
  env GPU=0,1 NPROC=2 SEQ=2048 SAVE_STEPS=1000 GATE=0 DATA="$DATA" \
      PLANS="$plans" LORA_MODULES="$mods" SEED=42 BRANCH="$branch" \
      TOKENS=4000000 WARM_START=slice OUTROOT="$dir" \
    bash "$ROOT/scripts/run_v15_w1.sh" > /tmp/preflight_${branch}.log 2>&1
  local alog=$dir/all.log
  if grep -qa 'model_parameter_info' "$alog" 2>/dev/null \
     && ! grep -qa 'Traceback' "$alog" 2>/dev/null; then
    grep -a 'model_parameter_info' "$alog" | tail -1
    # Hard gate: the plain branch must be col(T)-restricted (paper: 0.40B trainable),
    # never the 1.06B full matrix -- otherwise the 24 h run silently duplicates
    # slice-full and the "matched vs more-training" contrast does not exist.
    local tr
    tr=$(grep -a -oE '\([0-9.]+M Trainable' "$alog" | tail -1 | grep -oE '[0-9.]+')
    if [ "$branch" = plain ] && [ -n "$tr" ] && awk "BEGIN{exit !($tr>700)}"; then
      echo "    !! plain branch reported ${tr}M trainable = full matrix, not col(T) -> ABORT"
      return 1
    fi
    echo "    measured cost (${branch}):"
    grep -a -A2 'summary (loss' /tmp/preflight_${branch}.log | tail -2
    return 0
  fi
  echo "    preflight $branch FAILED (tail):"
  tail -5 /tmp/preflight_${branch}.log
  return 1
}


wait_ckpt () {   # <outside dir> <step>
  local vdir ck
  while :; do
    [ -e /tmp/STOP_SEED2 ] && return 1
    vdir=$(ls -d "$1"/*/v*/ 2>/dev/null | tail -1)
    if [ -n "$vdir" ]; then
      ck="$vdir/checkpoint-$2"
      [ -d "$ck" ] && { sleep 5; echo "$ck"; return 0; }
    fi
    sleep 60
  done
}

stop_train () {
  pkill -f 'swift/cli/pt.py'; sleep 10
  echo "    stopped ($(pgrep -c -f 'swift/cli/pt.py' || true) trainers left)"
}

one_segment () {   # <log> <k>=env-string
  local log=$1; shift
  env GPU=0,1 NPROC=2 SEQ=2048 SAVE_STEPS=1000 GATE=0 DATA="$DATA" "$@" \
    setsid nohup bash "$ROOT/scripts/run_v15_w1.sh" > "$log" 2>&1 < /dev/null &
  sleep 20
}

run_two_seg () {   # <name> <plans> <mods> <seed> <branch>
  local name=$1 plans=$2 mods=$3 seed=$4 branch=$5
  local arm=$OUT/$name tag
  tag=$(echo "$plans" | tr ',' '_')
  echo "=== $name (plans='$plans' mods='$mods' seed=$seed branch=$branch)  $(date '+%H:%M')"
  one_segment "/tmp/${name}_seg1.log" PLANS="$plans" LORA_MODULES="$mods" SEED="$seed" \
    BRANCH="$branch" TOKENS="$TOK_FULL" WARM_START=slice OUTROOT="$arm"
  local ck1
  ck1=$(wait_ckpt "$arm" 2000) || { echo "    aborted"; stop_train; return 1; }
  stop_train
  echo "    seg1 ckpt: $ck1"
  one_segment "/tmp/${name}_seg2.log" PLANS="$plans" LORA_MODULES="$mods" SEED="$seed" \
    BRANCH="$branch" TOKENS="$TOK_CONT" WARM_START=pt_ckpt PT_CKPT="$ck1" OUTROOT="$arm-cont"
  local ck2
  ck2=$(wait_ckpt "$arm-cont" 2000) || { echo "    aborted"; stop_train; return 1; }
  stop_train
  echo "    seg2 ckpt: $ck2"
}

# 1) the matched contrast the paper actually claims: plain-LRC vs the existing Dense arm.
#    The 30-step preflight was run by hand on 2026-10-27 and passed:
#      model_parameter_info: 945.77M Params (396.36M Trainable [41.9%])   <- col(T), not 1.06B
#      summary: loss=180.86 acc=0.4769 peak=22.15GiB step=33.24s
#    so skip it here (SKIP_PREFLIGHT=0 re-enables the gate).
if [ "${SKIP_PREFLIGHT:-1}" = "1" ] || preflight all "" plain; then
  run_two_seg plain42 all "" 42 plain
else
  echo "!!! plain-LRC unusable here -> falling back to Dense seed-43 only"
fi

# 2) seed 43 of the Dense arm -> sigma_run against the existing seed-42 arm
[ -e /tmp/STOP_SEED2 ] || run_two_seg dense43 all "" 43 dense

# 3) seed 43 of the plain-LRC arm -> a second seed of the contrast itself, so the
#    matched delta (full - plain) has an observed seed spread instead of one number.
#    Guard: the plain parameterization measures 33.4 s/step here (vs 23.5 dense), i.e.
#    ~37 h per arm, so only start it if it can still finish before the text freezes.
if [ -e /tmp/STOP_SEED2 ]; then
  :
elif [ "$(date +%Y%m%d%H)" -gt "${PLAIN43_CUTOFF:-2026102906}" ]; then
  echo "=== skip plain43: past the ${PLAIN43_CUTOFF:-2026102906} cutoff ($(date '+%F %H:%M'))"
else
  run_two_seg plain43 all "" 43 plain
fi

# 3) score whatever exists
echo "=== paper-suite scoring $(date '+%H:%M')"
PBASE=$(ls -d "$ROOT"/output/formal-b/slice-full-cont/*/v*/checkpoint-2000 | tail -1)   # Dense, seed 42
P42=$(ls -d "$OUT"/plain42-cont/*/v*/checkpoint-2000 2>/dev/null | tail -1)
D43=$(ls -d "$OUT"/dense43-cont/*/v*/checkpoint-2000 2>/dev/null | tail -1)
P43=$(ls -d "$OUT"/plain43-cont/*/v*/checkpoint-2000 2>/dev/null | tail -1)
for spec in "plain42:$P42" "dense43:$D43" "plain43:$P43"; do
  n=${spec%%:*}; ck=${spec#*:}
  [ -n "$ck" ] && PRESET=paper GPU=0 bash scripts/run_lm_eval.sh "$ck" "paper_$n" \
    || echo "    !! skip eval $n"
done
PY=${PY:-/home/chenwh/miniconda3/envs/twyd/bin/python}
"$PY" scripts/summarize_avg8.py \
  output/lm_eval/paper_pt_base \
  output/lm_eval/paper_pt_A output/lm_eval/paper_plain42 \
  output/lm_eval/paper_dense43 output/lm_eval/paper_plain43 2>/dev/null | tee output/eval_rebuttal_seeds.txt
echo "=== rebuttal seed runs done $(date '+%F %H:%M')"
