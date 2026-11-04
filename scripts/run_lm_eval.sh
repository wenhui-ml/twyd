#!/usr/bin/env bash
# Zero-shot lm-eval-harness runner for the matched-budget arms (PT or SFT).
#
# Why this exists: held-out CE only measures in-domain compression. Reviewers of a
# "train a cheap slice instead of the full model" claim will ask whether the CE win
# survives on task accuracy, so every arm gets the same harness suite.
#
# Host notes (verified 2026-10-24):
#   * huggingface.co resolves to IPv6 only here and the route is dead -> use the mirror.
#   * all six default tasks are now cached locally (hellaswag, arc_easy,
#     arc_challenge, piqa, winogrande, boolq), so the suite runs offline; lambada
#     and friends would need one mirror download each.
#
#   bash scripts/run_lm_eval.sh <ckpt> <tag>                 # full suite, GPU
#   LIMIT=20 DEVICE=cpu bash scripts/run_lm_eval.sh <ckpt> smoke   # cheap smoke test
#
# Env knobs: TASKS, BATCH, LIMIT, DEVICE, DTYPE, GPU
# Output:    output/lm_eval/<tag>.log  and  output/lm_eval/<tag>/ (results + samples)
set -uo pipefail

ROOT=${TWYD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
CKPT=${1:?usage: run_lm_eval.sh <ckpt> <tag>}
TAG=${2:?usage: run_lm_eval.sh <ckpt> <tag>}

TASKS=${TASKS:-hellaswag,arc_easy,arc_challenge,piqa,winogrande,boolq}
# The paper's Avg9 (TWYD Tables 6-8) is a *different* suite: ARC-E, ARC-C, LogiQA,
# CSQA, PIQA, WinoG, BoolQ, SciQ, MMLU (0-shot, MathQA excluded). Use PRESET=paper
# whenever a number has to be comparable to a table in the paper.
# LogiQA is excluded from the default preset: datasets>=4 dropped script-based
# loaders and lm-eval still ships logiqa.py, so it cannot load in this env. Set
# PAPER_TASKS=... (9 names) once an env with datasets<4 exists.
PAPER_TASKS=${PAPER_TASKS:-arc_easy,arc_challenge,commonsense_qa,piqa,winogrande,boolq,sciq,mmlu}
PAPER_TASKS_FULL=arc_easy,arc_challenge,logiqa,commonsense_qa,piqa,winogrande,boolq,sciq,mmlu
[ "${PRESET:-}" = paper ] && TASKS=$PAPER_TASKS
DEVICE=${DEVICE:-cuda}
GPU=${GPU:-0}
BATCH=${BATCH:-16}
LIMIT=${LIMIT:-}
DTYPE=${DTYPE:-}
OUT=$ROOT/output/lm_eval
mkdir -p "$OUT"

# Same convention as the other run_*.sh here: absolute env paths, overridable.
LM_EVAL=${LM_EVAL:-/home/chenwh/miniconda3/envs/twyd/bin/lm_eval}

# bf16 on CPU is slow/unsupported for most ops -> fall back to fp32 there.
if [ -z "$DTYPE" ]; then
  [ "$DEVICE" = cpu ] && DTYPE=float32 || DTYPE=bfloat16
fi

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=$GPU        # ROCm reuses the CUDA_* vars
export HIP_VISIBLE_DEVICES=$GPU

args=(--model hf
      --model_args "pretrained=$CKPT,dtype=$DTYPE,trust_remote_code=True"
      --tasks "$TASKS"
      --num_fewshot 0
      --batch_size "$BATCH"
      --device "$DEVICE"
      --log_samples
      --output_path "$OUT/$TAG")
[ -n "$LIMIT" ] && args+=(--limit "$LIMIT")

echo "[lm_eval] ckpt=$CKPT tag=$TAG tasks=$TASKS device=$DEVICE dtype=$DTYPE limit=${LIMIT:-none}"
"$LM_EVAL" "${args[@]}" 2>&1 | tee "$OUT/$TAG.log"
