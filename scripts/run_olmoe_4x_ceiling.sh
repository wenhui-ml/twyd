#!/usr/bin/env bash
# 4x Expert Compression (64 -> 16 experts, top-8 -> top-4, 6.92B -> 2.09B = 30.1% params)
# Ceiling-Pushing Pipeline:
#   Stage 2a: Full-Bus PT Online KD + Same-Width Hidden Alignment + Composed 64->16 Survivor Router-KL (~49.2M tok)
#   Stage 2b: Instruct Online KD-SFT Ceiling Boost on sft_paper8.jsonl (~11.5M tok, 64-expert Instruct Teacher)
#   Eval    : Parallel dual-GPU PRESET=paper (26,186 items) on Stage 2a & Stage 2b 100% native OlmoeForCausalLM.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
PY=/home/chenwh/miniconda3/envs/twyd/bin/python3.14

TEACHER=${TEACHER:-$HOME/models/OLMoE-1B-7B-0125-Instruct}
STUDENT=${STUDENT:-$ROOT/output/moe_lrc/olmoe-prog-e16-k4}
CMAP=${CMAP:-$STUDENT/lrc_cluster_map_64to16.json}
OUT_PT=${OUT_PT:-$ROOT/output/moe_lrc/olmoe-prog-e16-k4-pt50M}
OUT_SFT=${OUT_SFT:-$ROOT/output/moe_lrc/olmoe-prog-e16-k4-pt50M-kdsft}

PT_DATA=${PT_DATA:-$ROOT/data/pt_cosmopedia_v3_manifold_200M.jsonl}
SFT_DATA=${SFT_DATA:-$ROOT/data/sft_paper8.jsonl}

LORA_RANK=${LORA_RANK:-128}
LORA_ALPHA=${LORA_ALPHA:-128}
CPU_OFFLOAD_FLAG=${CPU_OFFLOAD_FLAG:---no-cpu-offload}
PT_STEPS=${PT_STEPS:-1500}
PT_SAVE=${PT_SAVE:-500}
SFT_STEPS=${SFT_STEPS:-350}

mkdir -p "$OUT_PT" "$OUT_SFT"

echo "[4x-ceiling] === Stage 2a: Full-Bus PT Recovery (${PT_STEPS} steps) $(date '+%F %H:%M') ==="
OMP_NUM_THREADS=12 "$PY" -u "$ROOT/scripts/train_moe_lrc_recover.py" \
  --teacher "$TEACHER" \
  --student "$STUDENT" \
  --cluster-map "$CMAP" \
  --out "$OUT_PT" \
  --mode pt \
  --data "$PT_DATA" \
  --teacher-device cuda:1 --student-device cuda:0 \
  --seq-len 1024 --batch-size 4 --grad-accum 8 \
  --max-steps "$PT_STEPS" --warmup-steps 50 --save-steps "$PT_SAVE" --log-steps 20 \
  --lr 2e-4 --attn-lr-scale 0.3 --router-lr-scale 0.5 \
  --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
  --kl-weight 2.0 --ce-weight 1.0 --hidden-align-weight 0.5 --hidden-align-layers 3,7,11,15 \
  --router-kl-weight 0.2 --router-target-mode survivor --aux-loss-weight 0.01 \
  --kl-topk 256 --kl-temp 1.0 --trainable-scope all_no_embed \
  $CPU_OFFLOAD_FLAG \
  > "$OUT_PT/train.log" 2>&1
echo "[4x-ceiling] Stage 2a finished rc=$? $(date '+%F %H:%M')"

CK_PT="$OUT_PT/checkpoint-${PT_STEPS}"

echo "[4x-ceiling] === Stage 2b: Instruct Online KD-SFT (${SFT_STEPS} steps) $(date '+%F %H:%M') ==="
OMP_NUM_THREADS=12 "$PY" -u "$ROOT/scripts/train_moe_lrc_recover.py" \
  --teacher "$TEACHER" \
  --student "$CK_PT" \
  --cluster-map "$CMAP" \
  --out "$OUT_SFT" \
  --mode sft \
  --data "$SFT_DATA" \
  --teacher-device cuda:1 --student-device cuda:0 \
  --seq-len 1024 --batch-size 4 --grad-accum 8 \
  --max-steps "$SFT_STEPS" --warmup-steps 20 --save-steps "$SFT_STEPS" --log-steps 10 \
  --lr 8e-5 --attn-lr-scale 0.3 --router-lr-scale 0.5 \
  --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
  --kl-weight 2.0 --ce-weight 1.0 --hidden-align-weight 0.3 --hidden-align-layers 3,7,11,15 \
  --router-kl-weight 0.1 --router-target-mode survivor --aux-loss-weight 0.01 \
  --kl-topk 256 --kl-temp 1.0 --trainable-scope all_no_embed \
  $CPU_OFFLOAD_FLAG \
  > "$OUT_SFT/train.log" 2>&1
echo "[4x-ceiling] Stage 2b finished rc=$? $(date '+%F %H:%M')"

CK_SFT="$OUT_SFT/checkpoint-${SFT_STEPS}"

echo "[4x-ceiling] === Parallel Dual-GPU PRESET=paper Evals $(date '+%F %H:%M') ==="
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 EXPERTS_IMPL=eager PRESET=paper GPU=0 BATCH=16 \
  bash "$ROOT/scripts/run_lm_eval.sh" "$CK_PT" "paper_olmoe_prog_e16_k4_pt50M" \
  > "$ROOT/output/ev_olmoe_prog_e16_k4_pt50M.log" 2>&1 &
PID0=$!

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 EXPERTS_IMPL=eager PRESET=paper GPU=1 BATCH=16 \
  bash "$ROOT/scripts/run_lm_eval.sh" "$CK_SFT" "paper_olmoe_prog_e16_k4_kdsft" \
  > "$ROOT/output/ev_olmoe_prog_e16_k4_kdsft.log" 2>&1 &
PID1=$!

wait "$PID0"; RC0=$?
wait "$PID1"; RC1=$?
echo "[4x-ceiling] ALL DONE rc0=$RC0 rc1=$RC1 $(date '+%F %H:%M')"
