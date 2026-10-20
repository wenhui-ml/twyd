#!/usr/bin/env bash
# =============================================================================
# TWYD Dense-LRC-SFT: 5 个常见 SFT 数据集验证 (GPU1, Llama-3.2-1B student)
# 顺序运行 5 个数据集，每个 200 steps 冒烟验证。
# ms-swift 4.x 不支持 --tuner_kwargs，改用 TWYD_* 环境变量。
# PT_CKPT 不存在时 warm_start="slice"，直接从 teacher slice 初始化。
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

export OUTPUT_DIR=/home/chenwh/Papers/twyd/output/dense-sft
mkdir -p "$OUTPUT_DIR"

# 5 个数据集 (dataset_id # sample_count)
DATASETS=(
  "AI-ModelScope/alpaca-gpt4-data-en#10000"
  "AI-ModelScope/alpaca-gpt4-data-zh#10000"
  "AI-ModelScope/alpaca-cleaned#10000"
  "TIGER-Lab/MATH-plus#10000"
  "AI-ModelScope/OpenOrca-Chinese#10000"
)

SWIFT=/home/chenwh/miniconda3/envs/twyd/bin/swift

run_one() {
  local ds="$1"          # "id#count"
  local id="${ds%%#*}"
  local cnt="${ds##*#}"
  local tag
  tag=$(echo "$id" | tr '/' '_' | tr ':' '_')
  local out="$OUTPUT_DIR/$tag"
  local log="/tmp/dense_sft_${tag}.log"

  echo "============================================================"
  echo "[START] dataset=$id  samples=$cnt  output=$out"
  echo "============================================================"

  "$SWIFT" sft \
    --model "/home/chenwh/models/Llama-3.2-1B-Instruct" \
    --dataset "$id#$cnt" \
    --tuner_type dense_lrc \
    --loss_type lrc_distill \
    --external_plugins /home/chenwh/Papers/twyd/twyd_plugin.py \
    --torch_dtype bfloat16 \
    --max_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.03 \
    --max_steps 200 \
    --save_steps 200 \
    --logging_steps 10 \
    --output_dir "$out" \
    --dataloader_num_workers 0 \
    --remove_unused_columns false \
    > "$log" 2>&1

  local rc=$?
  echo "[EXIT] dataset=$id  rc=$rc  log=$log"
  if [ -d "$out" ]; then
    echo "[CKPT] $(ls -1h "$out"/checkpoint-* 2>/dev/null | tr '\n' ' ')"
    echo "[LINK] $(readlink "$out"/last-checkpoint 2>/dev/null)"
  fi
  return $rc
}

for ds in "${DATASETS[@]}"; do
  run_one "$ds" || echo "[WARN] dataset $ds 失败，继续下一个"
done

echo "============================================================"
echo "[ALL DONE] 5 个数据集训练完成"
echo "============================================================"