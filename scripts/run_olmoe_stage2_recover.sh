#!/usr/bin/env bash
# Stage 2 Online KD + Survivor Router-KL Recovery for LRC-MoE v2 (64->32, 3.70B Native OlmoeForCausalLM).
#
# Dual-GPU asymmetric setup (verified 10.6G / 14.0G peak VRAM, <2.5G host RAM, ~1740 tok/s):
#   - GPU 1: Frozen Teacher OLMoE-1B-7B-0125-Instruct (64 experts, 6.92B)
#   - GPU 0: Trainable olmoe-lrc-v2-e32 (32 experts, 3.70B) + mergeable MoeExpertLoRA (r=64, 236M params)
#   - Budget: 600 steps x 32,768 tok/step = 19.7M tokens (~3.1h) on pure Cosmopedia + FineWeb-Edu (0% Wiki)
#   - Auto-merges at checkpoint-300 (~10M tok) & checkpoint-600 (~20M tok) -> 100% native OlmoeForCausalLM
#   - Auto-evaluates BOTH merged checkpoints in parallel on GPU 0 & GPU 1 (PRESET=paper, 26,186 items).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
PY=/home/chenwh/miniconda3/envs/twyd/bin/python3.14
TEACHER=${TEACHER:-$HOME/models/OLMoE-1B-7B-0125-Instruct}
STUDENT=${STUDENT:-$ROOT/output/moe_lrc/olmoe-lrc-v2-e32}
OUT=${OUT:-$ROOT/output/moe_lrc/olmoe-lrc-v2-e32-rec20M}
DATA=${DATA:-$ROOT/data/pt_cosmopedia_v3_manifold_200M.jsonl}
MAX_STEPS=${MAX_STEPS:-600}
SAVE_STEPS=${SAVE_STEPS:-300}

mkdir -p "$OUT"
echo "[stage2] start $(date '+%F %H:%M') student=$STUDENT out=$OUT steps=$MAX_STEPS"

"$PY" -u "$ROOT/scripts/train_moe_lrc_recover.py" \
  --teacher "$TEACHER" \
  --student "$STUDENT" \
  --out "$OUT" \
  --data "$DATA" \
  --teacher-device cuda:1 --student-device cuda:0 \
  --seq-len 1024 --batch-size 4 --grad-accum 8 \
  --max-steps "$MAX_STEPS" --warmup-steps 30 --save-steps "$SAVE_STEPS" --log-steps 10 \
  --lr 2e-4 --router-lr-scale 0.5 --lora-rank 64 --lora-alpha 64 \
  --kl-weight 2.0 --ce-weight 1.0 --router-kl-weight 0.2 --router-target-mode survivor \
  --aux-loss-weight 0.01 --kl-topk 256 --kl-temp 1.0 \
  --trainable-scope moe \
  > "$OUT/train.log" 2>&1

echo "[stage2] training finished rc=$? $(date '+%F %H:%M')"

CK1="$OUT/checkpoint-${SAVE_STEPS}"
CK2="$OUT/checkpoint-${MAX_STEPS}"

echo "[stage2] launching parallel PRESET=paper evals: GPU0=$CK1 GPU1=$CK2"
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 EXPERTS_IMPL=eager PRESET=paper GPU=0 BATCH=16 \
  bash "$ROOT/scripts/run_lm_eval.sh" "$CK1" "paper_olmoe_lrc_v2_e32_rec10M" \
  > "$ROOT/output/ev_olmoe_lrc_v2_e32_rec10M.log" 2>&1 &
PID0=$!

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 EXPERTS_IMPL=eager PRESET=paper GPU=1 BATCH=16 \
  bash "$ROOT/scripts/run_lm_eval.sh" "$CK2" "paper_olmoe_lrc_v2_e32_rec20M" \
  > "$ROOT/output/ev_olmoe_lrc_v2_e32_rec20M.log" 2>&1 &
PID1=$!

wait "$PID0"
RC0=$?
wait "$PID1"
RC1=$?
echo "[stage2] evals done rc0=$RC0 rc1=$RC1 $(date '+%F %H:%M')"
