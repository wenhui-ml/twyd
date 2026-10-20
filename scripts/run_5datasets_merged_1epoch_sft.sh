#!/usr/bin/env bash
# =============================================================================
# TWYD Dense-LRC-SFT: 小型通用 SFT 数据集合并 + 1个完整epoch (GPU1)
# 使用更小、更通用的 SFT 数据集（替代之前过大的 MATH-plus/OpenOrca）
# context-length=1024, ms-swift 4.x 使用 TWYD_* 环境变量传递 tuner 超参数
# =============================================================================
set -uo pipefail

export CUDA_VISIBLE_DEVICES=1
export TWYD_TEACHER=/home/chenwh/models/Llama-3.2-3B-Instruct
export TWYD_R=512
export TWYD_STAGE=sft
export TWYD_WARM_START=slice
export TWYD_PT_CKPT=""
export TWYD_T=40.0
export TWYD_LAM=0.2
export TWYD_USE_AUX=true
export TWYD_TARGET_MODULES="gate_proj,up_proj,down_proj"

export OUTPUT_DIR=/home/chenwh/Papers/twyd/output/dense-sft-merged-1epoch-small
mkdir -p "$OUTPUT_DIR"

# 小型通用 SFT 数据集 (空格分隔，全量，训练 1 个完整 epoch)
# swift/ultrachat_200k (~20万条) - 用户建议
# AI-ModelScope/alpaca-cleaned (~5万条)
# swift/self-cognition (小型)
# AI-ModelScope/alpaca-gpt4-data-zh (~4.8万条)
DATASET_ARGS=(
  "swift/ultrachat_200k"
  "AI-ModelScope/alpaca-cleaned"
  "swift/self-cognition"
  "AI-ModelScope/alpaca-gpt4-data-zh"
)

SWIFT=/home/chenwh/miniconda3/envs/twyd/bin/swift

LOG="/tmp/dense_sft_merged_1epoch_small.log"
OUT="$OUTPUT_DIR"

echo "============================================================"
echo "[START] 小型通用 SFT 数据集合并训练 (1 epoch)"
echo "数据集: ${DATASET_ARGS[*]}"
echo "输出目录: $OUT"
echo "日志: $LOG"
echo "============================================================"

"$SWIFT" sft \
  --model "/home/chenwh/models/Llama-3.2-1B-Instruct" \
  --dataset "${DATASET_ARGS[@]}" \
  --tuner_type dense_lrc \
  --loss_type lrc_distill \
  --external_plugins /home/chenwh/Papers/twyd/twyd_plugin.py \
  --torch_dtype bfloat16 \
  --max_length 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-5 \
  --warmup_ratio 0.03 \
  --num_train_epochs 1 \
  --save_strategy epoch \
  --logging_steps 10 \
  --output_dir "$OUT" \
  --dataloader_num_workers 0 \
  --remove_unused_columns false \
  > "$LOG" 2>&1

RC=$?
echo "[EXIT] rc=$RC  log=$LOG"

if [ -d "$OUT" ]; then
  echo "[CKPT] $(ls -1h "$OUT"/checkpoint-* 2>/dev/null | tr '\n' ' ')"
  echo "[LINK] $(readlink "$OUT"/last-checkpoint 2>/dev/null)"
fi

echo "============================================================"
echo "[DONE] 小型通用 SFT 数据集合并 1 epoch 训练完成 (rc=$RC)"
echo "============================================================"
exit $RC