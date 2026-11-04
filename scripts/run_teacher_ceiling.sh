#!/usr/bin/env bash
# A3: the teacher's score on the SAME 8-task suite = the achievable ceiling.
# Answers "how much of the MMLU gap is width?" quantitatively.
# Waits for the GPU to be free first (the A2 ablation owns it).
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== A3 waiting for the GPU $(date '+%F %H:%M')"
while pgrep -f 'eval_ce|lm_eval|swift/cli' >/dev/null; do sleep 60; done
echo "=== A3 start $(date '+%F %H:%M')"
PRESET=paper GPU=0 bash scripts/run_lm_eval.sh /home/chenwh/models/Llama-3.2-3B-Instruct paper_teacher-3b 2>&1 | tail -3
/home/chenwh/miniconda3/envs/twyd/bin/python - <<'EOF'
import json, glob
T=['arc_easy','arc_challenge','commonsense_qa','piqa','winogrande','boolq','sciq','mmlu']
fs=sorted(glob.glob('output/lm_eval/paper_teacher-3b/**/results_*.json',recursive=True))
if not fs: print('teacher: no results'); raise SystemExit
d=json.load(open(fs[-1]))['results']
v=[d[t].get('acc_norm,none',d[t].get('acc,none')) for t in T if t in d]
print(f"TEACHER 3B   Avg8={sum(v)/len(v):.4f}  MMLU={d['mmlu'].get('acc,none'):.3f}")
EOF
echo "=== A3 done $(date '+%F %H:%M')"
