#!/usr/bin/env bash
# Sequential product queue (2026-10-31): the two runs the allocation paper needs.
#
#   1. k=128  -> the second *coordinate-budget* point for the Pareto figure
#                (k=256 is 30.67M; expect ~15.3M here). Preflight ON (new k).
#   2. k=256, seed 43 -> a second seed of the champion arm, so "ties the dense
#                9-layer arm" can become a claim rather than an observation.
#
# P6 (2026-10-31) showed the col(T) basis is not load-bearing at this budget, so the
# paper is an *allocation* paper: coordinate budget x depth. Nothing here depends on
# the reachability mechanism.
#
#   setsid nohup bash scripts/product_queue.sh > /tmp/queue.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== queue start $(date '+%F %H:%M')"

K=128 ARM_NAME=corelr-k128 SEED_ARM=42 SKIP_PREFLIGHT=0 bash scripts/product_arm.sh
rc1=$?
echo "=== k128 finished rc=$rc1 $(date '+%F %H:%M')"

K=256 ARM_NAME=corelr-k256-s43 SEED_ARM=43 SKIP_PREFLIGHT=1 bash scripts/product_arm.sh
rc2=$?
echo "=== k256-s43 finished rc=$rc2 $(date '+%F %H:%M')"

echo "=== queue done $(date '+%F %H:%M') rc=$rc1/$rc2"
