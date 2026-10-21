#!/usr/bin/env bash
# CORE-LR pilot: TWYD-native parameter-efficient variant.
#
#   W = T Z_col^T + U_perp (A B^T)^T,  Z_col frozen, only A/B train.
#   All trainable coordinates live in col(T)^perp (the stranded capacity).
#   At init B=0 => exactly plain LRC; after merge => one dense d_ff x r weight,
#   zero inference cost.
#
# Usage:
#   # Full CORE (sanity / spectrum): K=0
#   K=0 MAX_STEPS=100 bash scripts/train_core_lr_pilot.sh
#   # CORE-LR with factorized low-rank forward
#   K=128 FACTORIZED=1 MAX_STEPS=3000 bash scripts/train_core_lr_pilot.sh
#   # Debug smoke only (no step-0 gate; results are NOT scientific)
#   GATE=0 K=128 MAX_STEPS=5 DATASET=/tmp/long_sft.jsonl bash scripts/train_core_lr_pilot.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
TEACHER=${TEACHER:-/home/chenwh/models/Llama-3.2-3B-Instruct}
MODEL=${MODEL:-/home/chenwh/models/Llama-3.2-3B-Instruct}
R=${R:-1536}
BRANCH=${BRANCH:-core}                    # core | dense | plain (plain = matched LRC baseline)
LORA_MODE=${LORA_MODE:-core}              # core = rank-k complement (CORE-LR); dense = Dense-LoRA
PERP_BASIS=${PERP_BASIS:-teacher}         # teacher = CORE-LR; col = P6 equal-parameter control
K=${K:-256}
# LoRA scaling = alpha / k.  CORE-LR's update is Z_perp = (A B^T) * scaling, so
# alpha=2k doubles the step size relative to full CORE (which updates Z_perp
# directly); that makes the k-sweep incomparable with the full-CORE arm.  Keep
# scaling at 1.0 (alpha = k) unless you are deliberately tuning it per arm.
ALPHA=${ALPHA:-$K}
LR=${LR:-1e-4}
ATTN_IMPL=${ATTN_IMPL:-sdpa}
MAX_LENGTH=${MAX_LENGTH:-1024}
MAX_STEPS=${MAX_STEPS:-2000}
GRAD_ACCUM=${GRAD_ACCUM:-16}
# Per-device micro-batch.  bs=1 with seq>=1024 leaves the GPU under-utilized
# (~1.4k tok/s measured on one 7900 XTX); raise it until VRAM is the limit.
BS=${BS:-1}
STAGE=${STAGE:-sft}                       # sft | pt (paper: 10B PT then ~0.6B SFT)
# ms-swift drops samples longer than max_length by default ('delete'), which
# silently shrank a 30k-document PT corpus to 3k short ones.  'split' keeps all
# content as multiple max_length chunks (no cross-document attention).
TRUNC=${TRUNC:-split}                     # delete | left | right | split
DATASET=${DATASET:-swift/ultrachat_200k#100000}
SWIFT=${SWIFT:-/home/chenwh/miniconda3/envs/twyd/bin/swift}
OUT=${OUT:-/home/chenwh/Papers/twyd/output/core-lr-r${R}-k${K}}
WARM_START=${WARM_START:-slice}          # slice | base | ridge | ridge_ckpt | pt_ckpt
PT_CKPT=${PT_CKPT:-}
RIDGE_CKPT=${RIDGE_CKPT:-}
CALIB_IDS=${CALIB_IDS:-}
RANK_MAP=${RANK_MAP:-}
FACTORIZED=${FACTORIZED:-0}
GATE=${GATE:-1}
GATE_SEQ_LEN=${GATE_SEQ_LEN:-64}
USE_AUX=${USE_AUX:-1}
# ms-swift cannot auto-detect model_type/template for the official LRC-*-Base
# dirs (they ship a bare config.json with no architecture hints), so both must
# be pinned there.  Harmless for regular HF checkpoints.
MODEL_TYPE=${MODEL_TYPE:-}   # e.g. llama / qwen2
TEMPLATE=${TEMPLATE:-}       # e.g. llama3_2 / qwen2_5

# Teacher SVD bases cost ~2-3 s per projection on CPU (2-5 min GPU-idle
# startup per model).  Cache them keyed by teacher-weight hash.
BASIS_CACHE_DIR=${BASIS_CACHE_DIR:-$HOME/.cache/twyd/basis}
export TWYD_BASIS_CACHE_DIR="$BASIS_CACHE_DIR"
mkdir -p "$BASIS_CACHE_DIR"

# The teacher is loaded on CPU (VRAM budget) and pinned to the compute device
# once.  auto = pin on GPU, fall back to a per-step copy on OOM; cpu = never pin
# (slow, saves ~6 GiB).  TWYD_PROFILE=1 times teacher/kl/ce inside the loss.
export TWYD_TEACHER_DEVICE=${TWYD_TEACHER_DEVICE:-auto}
if [ -n "${TWYD_PROFILE:-}" ]; then export TWYD_PROFILE; fi

export TWYD_TEACHER="$TEACHER"
export TWYD_R="$R"
export TWYD_STAGE="$STAGE"
export TWYD_WARM_START="$WARM_START"
export TWYD_T=40.0
export TWYD_LAM=0.2
export TWYD_USE_AUX="$USE_AUX"
export TWYD_TARGET_MODULES='gate_proj,up_proj,down_proj'
export TWYD_LORA_RANK="$K"
export TWYD_LORA_ALPHA="$ALPHA"
export TWYD_LORA_MODE="$LORA_MODE"
export TWYD_FREEZE_COL=${FREEZE_COL:-true}
export TWYD_RANK_MAP="$RANK_MAP"
export TWYD_FACTORIZED="$FACTORIZED"
export TWYD_PERP_BASIS="$PERP_BASIS"
export PYTORCH_ALLOC_CONF=expandable_segments:True
if [ -n "$PT_CKPT" ]; then export TWYD_PT_CKPT="$PT_CKPT"; fi
if [ -n "$RIDGE_CKPT" ]; then export TWYD_RIDGE_CKPT="$RIDGE_CKPT"; fi
if [ -n "$CALIB_IDS" ]; then export TWYD_CALIB_IDS="$CALIB_IDS"; fi

mkdir -p "$OUT"
echo "=============================="
echo "branch=$BRANCH lora_mode=$LORA_MODE perp_basis=$PERP_BASIS stage=$STAGE r=$R k=$K alpha=$ALPHA lr=$LR steps=$MAX_STEPS"
echo "bs=$BS grad_accum=$GRAD_ACCUM seq=$MAX_LENGTH trunc=$TRUNC tokens/step=$((BS * GRAD_ACCUM * MAX_LENGTH * ${NPROC_PER_NODE:-${WORLD_SIZE:-1}}))"
echo "teacher=$TEACHER model=$MODEL dataset=$DATASET"
echo "warm_start=$WARM_START rank_map=${RANK_MAP:-uniform} factorized=$FACTORIZED"
echo "model_type=${MODEL_TYPE:-auto} template=${TEMPLATE:-auto}"
echo "out=$OUT"
echo "=============================="

# Step-0 gate: do not train on a random warm start.  A debug smoke (GATE=0)
# only checks code mechanics and must not be used for paper/funding claims.
if [ "$GATE" = "1" ]; then
  GATE_EXTRA=()
  if [ "$FACTORIZED" = "1" ]; then GATE_EXTRA+=(--factorized); fi
  if [ "${FREEZE_COL:-true}" = "true" ]; then
    GATE_EXTRA+=(--freeze-col)
  else
    GATE_EXTRA+=(--no-freeze-col)
  fi
  /home/chenwh/miniconda3/envs/twyd/bin/python -u scripts/check_init.py \
    --teacher "$TEACHER" --model "$MODEL" --r "$R" --branch "$BRANCH" \
    --warm-start "$WARM_START" --lora-rank "$K" --lora-mode "$LORA_MODE" \
    --perp-basis "$PERP_BASIS" \
    --rank-map "$RANK_MAP" \
    --seq-len "$GATE_SEQ_LEN" "${GATE_EXTRA[@]}" \
    || { echo "[gate] FAILED: fix the LRC warm start before training"; exit 1; }
else
  echo "[gate] SKIPPED (GATE=0): debug smoke only; NOT a scientific run"
  # An aux term is meaningless when the student is a random/slice init, but a
  # trusted checkpoint (official LRC base / PT / ridge ckpt) is a real model, so
  # keep aux there.
  case "$WARM_START" in
    base|pt_ckpt|ridge_ckpt) : ;;
    *)
      if [ "${USE_AUX:-1}" = "1" ]; then
        echo "[gate] aux loss is meaningless on an ungated (random) student;"
        echo "       forcing USE_AUX=0 for this smoke (override with USE_AUX=1)."
        export TWYD_USE_AUX=0
      fi
      ;;
  esac
fi

MODEL_ARGS=()
if [ -n "$MODEL_TYPE" ]; then MODEL_ARGS+=(--model_type "$MODEL_TYPE"); fi
if [ -n "$TEMPLATE" ]; then MODEL_ARGS+=(--template "$TEMPLATE"); fi

"$SWIFT" "$STAGE" \
  --model "$MODEL" \
  "${MODEL_ARGS[@]}" \
  --dataset "$DATASET" \
  --tuner_type "${BRANCH}_lrc" \
  --loss_type lrc_distill \
  --attn_impl "$ATTN_IMPL" \
  --external_plugins /home/chenwh/Papers/twyd/twyd_plugin.py \
  --torch_dtype bfloat16 \
  --max_length "$MAX_LENGTH" \
  --truncation_strategy "$TRUNC" \
  --per_device_train_batch_size "$BS" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --learning_rate "$LR" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --max_steps "$MAX_STEPS" \
  --save_steps "$MAX_STEPS" \
  --logging_steps 10 \
  --optim adamw_torch \
  --gradient_checkpointing true \
  --output_dir "$OUT" \
  --dataloader_num_workers 0 \
  --remove_unused_columns false \
  2>&1 | tee "$OUT/training.log"

echo "=============================="
echo "done: $OUT"
echo "next: compare CE/acc + Avg9 against plain LRC and dense at matched tokens"
echo "=============================="
