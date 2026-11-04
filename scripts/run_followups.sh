#!/usr/bin/env bash
# Follow-ups after A2 (no training):
#   A2b: merge the two GOOD bands (0-2, 11-13) -> the product candidate, 5.1M coords (0.48%)
#   A3 : the teacher's score on the same 8 tasks -> the ceiling
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/chenwh/miniconda3/envs/twyd/bin/python
BASE=/home/chenwh/models/LRC-1.5B-Base
ARM=$(ls -d output/product/corelr-k64/*/v*/checkpoint-2000 | tail -1)

while pgrep -f 'eval_ce|lm_eval|swift/cli' >/dev/null; do sleep 60; done
echo "=== A2b merge start $(date '+%F %H:%M')"
$PY scripts/ablate_layer_bands.py --base "$BASE" --arm "$ARM" --out output/ablation --bands "0-2,11-13"
D="output/ablation/band-0-2,11-13"
echo "=== A2b eval $(date '+%H:%M')"
$PY scripts/eval_ce.py --data data/fineweb_edu_holdout.jsonl --models "$BASE" "$D" \
   --seq-len 512 --max-docs 400 --device cuda --json-out output/eval_ce_ablate_6L.json 2>&1 | grep -E 'dCE=' | tail -2
$PY scripts/eval_ce.py --data data/ultrachat_pt_35M.jsonl --models "$BASE" "$D" \
   --seq-len 512 --max-docs 400 --device cuda --json-out output/eval_ce_ablate_6L_ood.json 2>&1 | grep -E 'dCE=' | tail -2
PRESET=paper GPU=0 bash scripts/run_lm_eval.sh "$D" paper_ablate-6L 2>&1 | tail -2

while pgrep -f 'eval_ce|lm_eval|swift/cli' >/dev/null; do sleep 60; done
echo "=== A3 teacher start $(date '+%F %H:%M')"
PRESET=paper GPU=0 bash scripts/run_lm_eval.sh /home/chenwh/models/Llama-3.2-3B-Instruct paper_teacher-3b 2>&1 | tail -2

echo "=== final tables $(date '+%H:%M')"
$PY - <<'EOF'
import json, glob
T=['arc_easy','arc_challenge','commonsense_qa','piqa','winogrande','boolq','sciq','mmlu']
rows=[('base','paper_pt_base'),('teacher 3B','paper_teacher-3b'),('band 0-2','paper_ablate-0-2'),
      ('band 11-13','paper_ablate-11-13'),('band 25-27','paper_ablate-25-27'),
      ('6-layer','paper_ablate-6L'),('all 9 (k64)','paper2_corelr-k64'),('LoRA r256','paper2_lora-r256')]
print(f"{'arm':14s} {'Avg8':>7s} {'MMLU':>6s}")
for name,tag in rows:
    fs=sorted(glob.glob(f'output/lm_eval/{tag}/**/results_*.json',recursive=True))
    if not fs: print(f'{name:14s}     n/a'); continue
    d=json.load(open(fs[-1]))['results']
    v=[d[t].get('acc_norm,none',d[t].get('acc,none')) for t in T if t in d]
    print(f"{name:14s} {sum(v)/len(v):7.4f} {d['mmlu'].get('acc,none'):6.3f}")
EOF
echo "=== followups done $(date '+%F %H:%M')"
