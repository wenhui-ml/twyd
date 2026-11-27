#!/usr/bin/env bash
# Stage 1 Benchmark: OLMoE-1B-7B-0125-Instruct (64 experts, top-8)
#   1. Teacher baseline (64 experts, top-8)
#   2. Naive Frequency Pruning baseline (64 -> 32, slice router, no LRC ridge)
#   3. LRC-MoE Top-32 (OBS Gram + functional cluster + KL router + Coupled LRC Ridge)
#   4. LRC-MoE Top-16 / Top-4 active (4x expert compression, 2x active FLOPs reduction)
#
# All exported checkpoints are 100% native OlmoeForCausalLM (zero trust_remote_code).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
PY=/home/chenwh/miniconda3/envs/twyd/bin/python3.14
MODEL=${MODEL:-$HOME/models/OLMoE-1B-7B-0125-Instruct}
OUTROOT=${OUTROOT:-$ROOT/output/moe_lrc}
GPU_ID=${GPU_ID:-0}
BATCH=${BATCH:-16}
CALIB_SEQS=${CALIB_SEQS:-64}
CALIB_SEQ_LEN=${CALIB_SEQ_LEN:-512}

mkdir -p "$OUTROOT" "$ROOT/output/lm_eval"

eval_model() {
  local ckpt="$1"
  local tag="$2"
  if [ -f "$ROOT/output/lm_eval/paper_${tag}.log" ] && grep -q "|mmlu" "$ROOT/output/lm_eval/paper_${tag}.log"; then
    echo "[bench] SKIP eval $tag (already exists)"
    return 0
  fi
  echo "[bench] eval $tag ($ckpt) ..."
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PRESET=paper GPU="$GPU_ID" BATCH="$BATCH" \
    bash "$ROOT/scripts/run_lm_eval.sh" "$ckpt" "paper_${tag}" \
    > "$ROOT/output/ev_${tag}.log" 2>&1
  echo "[bench] eval $tag done (rc=$?)"
}

echo "[bench] === 1/4 Teacher baseline (64 experts, top-8) ==="
eval_model "$MODEL" "olmoe_teacher_e64"

echo "[bench] === 2/4 Baseline: Freq Pruning 64->32 (slice router, no LRC ridge) ==="
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" "$ROOT/scripts/compress_moe_lrc.py" \
  --model "$MODEL" \
  --out "$OUTROOT/olmoe-freq-e32" \
  --target-experts 32 \
  --method freq \
  --router-mode slice \
  --no-lrc-ridge \
  --calib-seqs "$CALIB_SEQS" --calib-seq-len "$CALIB_SEQ_LEN" \
  > "$OUTROOT/compress_freq_e32.log" 2>&1
eval_model "$OUTROOT/olmoe-freq-e32" "olmoe_freq_e32"

echo "[bench] === 3/4 LRC-MoE 64->32 (OBS Gram + KL-Router + Coupled LRC Ridge) ==="
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" "$ROOT/scripts/compress_moe_lrc.py" \
  --model "$MODEL" \
  --out "$OUTROOT/olmoe-lrc-e32" \
  --target-experts 32 \
  --method lrc_cluster \
  --router-mode kl_fit \
  --lrc-ridge \
  --calib-seqs "$CALIB_SEQS" --calib-seq-len "$CALIB_SEQ_LEN" \
  > "$OUTROOT/compress_lrc_e32.log" 2>&1
eval_model "$OUTROOT/olmoe-lrc-e32" "olmoe_lrc_e32"

echo "[bench] === 4/4 LRC-MoE 64->16 (top-8 -> top-4 active) ==="
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" "$ROOT/scripts/compress_moe_lrc.py" \
  --model "$MODEL" \
  --out "$OUTROOT/olmoe-lrc-e16-k4" \
  --target-experts 16 \
  --target-top-k 4 \
  --method lrc_cluster \
  --router-mode kl_fit \
  --lrc-ridge \
  --calib-seqs "$CALIB_SEQS" --calib-seq-len "$CALIB_SEQ_LEN" \
  > "$OUTROOT/compress_lrc_e16_k4.log" 2>&1
eval_model "$OUTROOT/olmoe-lrc-e16-k4" "olmoe_lrc_e16_k4"

echo "[bench] ALL STEP-0 BENCHMARKS COMPLETE!"
