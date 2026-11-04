#!/usr/bin/env bash
# Queue 2 (2026-11-01): knee + external baseline.
#
#   1. CORE-LR k=64  -> the third coordinate-budget point (15.34M -> ~7.7M).
#      Preflight ON (new k). If it still ties, the knee is below 1.45% of the
#      full matrix and the headline becomes "~1% suffices".
#   2. LoRA r=256 on the same 9 layers x gate/up -> 44.83M coordinates, i.e. 46%
#      MORE than our k=256 arm, so a win cannot be attributed to having more
#      parameters. Verified config 2026-11-01 18:37: 44.8266M trainable,
#      first loss 186.0 (= same start), grad_norm 54.3, 18.82 GiB, 29.2 s/step.
#
#   setsid nohup bash scripts/product_queue2.sh > /tmp/queue2.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== queue2 start $(date '+%F %H:%M')"

K=64 ARM_NAME=corelr-k64 SEED_ARM=42 SKIP_PREFLIGHT=0 bash scripts/product_arm.sh
rc1=$?
echo "=== k64 finished rc=$rc1 $(date '+%F %H:%M')"

K=256 ARM_NAME=lora-r256 SEED_ARM=42 SKIP_PREFLIGHT=1 bash scripts/product_lora.sh
rc2=$?
echo "=== lora-r256 finished rc=$rc2 $(date '+%F %H:%M')"

echo "=== queue2 done $(date '+%F %H:%M') rc=$rc1/$rc2"
