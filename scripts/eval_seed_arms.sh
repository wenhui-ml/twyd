#!/usr/bin/env bash
# Evaluate the three *matched* 2,000-step arms (+ base) and print the rebuttal table.
#
# Design (2026-10-27): every arm below is a 2,000-step run from the identical
# LRC-1.5B-Base slice init, same data, same cosine span (max_steps 8,010),
# same seq/gas/2 ranks, seed 42.  The ONLY difference is which coordinates may move:
#
#   plain  = col(T) only          396M trainable   <- the paper's plain-LRC parameterization
#   full   = the deployed matrix  1,057M trainable <- Dense-LRC
#   sel9   = 9 layers x gate/up   226M trainable   <- layer-selective variant
#
# NOTE: do NOT use the `-cont` arms for this comparison.  `twyd/tuner.py` only honours
# WARM_START=pt_ckpt when STAGE=sft, so every "-cont" run silently restarted from the
# slice init; RESULTS_m525.md's "4,000 steps / 524M tokens" description is wrong.
#
#   setsid nohup bash scripts/eval_seed_arms.sh > /tmp/eval_seed_arms.log 2>&1 < /dev/null &
set -uo pipefail
ROOT=${TWYD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ROOT"
PY=${PY:-/home/chenwh/miniconda3/envs/twyd/bin/python}
BASE=${BASE:-/home/chenwh/models/LRC-1.5B-Base}

FULL=$(ls -d "$ROOT"/output/formal-b/slice-full/all/v*/checkpoint-2000 2>/dev/null | tail -1)
SEL=$(ls -d "$ROOT"/output/formal-b/slice-sel9/*/v*/checkpoint-2000 2>/dev/null | tail -1)
PLAIN=$(ls -d "$ROOT"/output/rebuttal/plain42/all/v*/checkpoint-2000 2>/dev/null | tail -1)
for v in FULL SEL PLAIN; do
  [ -n "${!v}" ] || { echo "!!! $v checkpoint missing -- abort"; exit 1; }
done
echo "base=$BASE"
echo "plain=$PLAIN"
echo "full=$FULL"
echo "sel9=$SEL"

echo "=== in-domain CE / BPP ($(date '+%F %H:%M'))"
$PY scripts/eval_ce.py --data data/fineweb_edu_holdout.jsonl \
  --models "$BASE" "$PLAIN" "$FULL" "$SEL" \
  --seq-len 512 --max-docs 400 --device cuda \
  --json-out output/eval_ce_seed.json 2>&1 | tail -20

echo "=== out-of-domain CE, chat ($(date '+%F %H:%M'))"
$PY scripts/eval_ce.py --data data/ultrachat_pt_35M.jsonl \
  --models "$BASE" "$PLAIN" "$FULL" "$SEL" \
  --seq-len 512 --max-docs 400 --device cuda \
  --json-out output/eval_ce_seed_ood.json 2>&1 | tail -20

echo "=== bits-per-byte summary (bytes/token = 4.8549 on the fineweb holdout)"
$PY - <<'EOF' 2>&1 | tail -20
import json, math
BPT = 4.8549
for name, path in (("in-domain", "output/eval_ce_seed.json"),
                   ("chat OOD ", "output/eval_ce_seed_ood.json")):
    try:
        d = json.load(open(path))
    except Exception as e:
        print(name, "missing:", e); continue
    rows = d.get("models") or d.get("results") or d
    print(f"-- {name}")
    if isinstance(rows, dict):
        rows = [dict(model=k, **v) for k, v in rows.items()]
    for r in rows:
        ce = r.get("ce") or r.get("loss")
        tag = (r.get("model") or r.get("path") or "?").split("/")[-3:]
        tag = "/".join(tag)
        if ce is None:
            print("  ", tag, r); continue
        print(f"   {tag:55s} CE={ce:.4f}  BPP={ce/math.log(2)/BPT:.4f}")
EOF

echo "=== Avg9 via lm-eval paper preset ($(date '+%F %H:%M'))"
for spec in "plain:$PLAIN" "full:$FULL" "sel9:$SEL"; do
  n=${spec%%:*}; ck=${spec#*:}
  echo "--- $n"
  PRESET=paper GPU=0 bash scripts/run_lm_eval.sh "$ck" "paper2_$n" 2>&1 | tail -3
done

echo "=== Avg8/9 table"
$PY scripts/summarize_avg8.py \
  output/lm_eval/paper_pt_base output/lm_eval/paper2_plain \
  output/lm_eval/paper2_full output/lm_eval/paper2_sel9 2>/dev/null \
  | tee output/eval_seed_arms.txt
echo "=== done $(date '+%F %H:%M')"
