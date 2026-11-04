#!/usr/bin/env bash
# Score the layer-band ablations of the k=64 arm (no training).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/chenwh/miniconda3/envs/twyd/bin/python
BASE=/home/chenwh/models/LRC-1.5B-Base
echo "=== ablation eval start $(date '+%F %H:%M')"
for b in 0-2 11-13 25-27 all; do
  d="output/ablation/band-$b"
  echo "--- band $b  $(date '+%H:%M')"
  $PY scripts/eval_ce.py --data data/fineweb_edu_holdout.jsonl \
      --models "$BASE" "$d" --seq-len 512 --max-docs 400 --device cuda \
      --json-out "output/eval_ce_ablate_${b}.json" 2>&1 | grep -E 'dCE=|CE=' | tail -3
  $PY scripts/eval_ce.py --data data/ultrachat_pt_35M.jsonl \
      --models "$BASE" "$d" --seq-len 512 --max-docs 400 --device cuda \
      --json-out "output/eval_ce_ablate_${b}_ood.json" 2>&1 | grep -E 'dCE=|CE=' | tail -3
  PRESET=paper GPU=0 bash scripts/run_lm_eval.sh "$d" "paper_ablate-$b" 2>&1 | tail -2
done
echo "--- summary"
$PY - <<'EOF'
import json, glob
T=['arc_easy','arc_challenge','commonsense_qa','piqa','winogrande','boolq','sciq','mmlu']
print(f"{'band':10s} {'Avg8':>7s} {'MMLU':>6s}")
for b in ['0-2','11-13','25-27','all']:
    fs=sorted(glob.glob(f'output/lm_eval/paper_ablate-{b}/**/results_*.json',recursive=True))
    if not fs: print(f'{b:10s}   n/a'); continue
    d=json.load(open(fs[-1]))['results']
    v=[d[t].get('acc_norm,none',d[t].get('acc,none')) for t in T if t in d]
    print(f"{b:10s} {sum(v)/len(v):7.4f} {d['mmlu'].get('acc,none'):6.3f}")
EOF
echo "=== ablation eval done $(date '+%F %H:%M')"
