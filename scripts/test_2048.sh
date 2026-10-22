#!/usr/bin/env bash
# Feasibility + speed probe for the formal matched run at seq=2048 on 2x7900XTX (24 GB).
#
# Each case trains 2 steps (GATE off, so startup is what the real run pays) and records
# peak VRAM and s/step from the trainer's own log, plus whether it OOM'd.
#
#   bash scripts/test_2048.sh              # full matrix, one case at a time
#   ONLY=full-ddp bash scripts/test_2048.sh
set -uo pipefail

ROOT=${TWYD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
OUTROOT=${OUTROOT:-$ROOT/output/t2048}
SEQ=${SEQ:-2048}
TOKENS=${TOKENS:-131072}        # 2 steps at seq=2048, bs=1, gas=32
mkdir -p "$OUTROOT"

# name | plans | modules | gpu(s) | nproc | deepspeed
CASES=(
  "sel6-gateup-1gpu|0-5|gate_proj,up_proj|0|1|"
  "sel9-gateup-1gpu|0-2,11-13,25-27|gate_proj,up_proj|0|1|"
  "full-1gpu|all||0|1|"
  "full-2gpu-ddp|all||0,1|2|"
  "full-2gpu-zero2|all||0,1|2|zero2"
  "sel9-gateup-2gpu|0-2,11-13,25-27|gate_proj,up_proj|0,1|2|"
  "sel9-gateup-2gpu-zero2|0-2,11-13,25-27|gate_proj,up_proj|0,1|2|zero2"
)

printf '%-24s %-16s %-22s %-6s %-4s %-14s %8s %10s\n' case layers modules gpus np ds peakGiB s/step
for spec in "${CASES[@]}"; do
  IFS='|' read -r name plans mods gpus nproc ds <<< "$spec"
  [ -n "${ONLY:-}" ] && [ "$name" != "$ONLY" ] && continue
  out="$OUTROOT/$name"
  rm -rf "$out" "$out.log"
  log="$OUTROOT/$name.log"
  start=$(date +%s)
  CUDA_VISIBLE_DEVICES="$gpus" GPU="$gpus" PLANS="$plans" TOKENS="$TOKENS" SEQ="$SEQ" OUTROOT="$out" \
  LORA_MODULES="$mods" NPROC="$nproc" DS="$ds" \
  GATE=0 STAGE=pt BRANCH=dense K=0 LORA_MODE=dense FACTORIZED=0 \
  BS=1 GAS=32 TRUNC=split DATASET="$ROOT/data/fw_score4_1B.jsonl" \
  MODEL=/home/chenwh/models/LRC-1.5B-Base TEACHER=/home/chenwh/models/Llama-3.2-3B-Instruct \
  MODEL_TYPE=llama TEMPLATE=llama3_2 WARM_START=base \
  bash "$ROOT/scripts/run_v15_w1.sh" > "$log" 2>&1
  rc=$?
  wall=$(( $(date +%s) - start ))
  peak=$(python - "$out" <<'PY' 2>/dev/null
import glob, json, sys
fs = glob.glob(sys.argv[1] + "/*/v0-*/logging.jsonl")
if not fs:
    print("-"); raise SystemExit
rows = [json.loads(l) for l in open(fs[0])]
steps = [r for r in rows if "loss" in r]
print(f"{steps[-1]['memory(GiB)']:.2f}" if steps else "-")
PY
)
  spd=$(python - "$out" <<'PY' 2>/dev/null
import glob, json, sys
fs = glob.glob(sys.argv[1] + "/*/v0-*/logging.jsonl")
if not fs:
    print("-"); raise SystemExit
rows = [json.loads(l) for l in open(fs[0])]
steps = [r for r in rows if "loss" in r and "train_speed(s/it)" in r]
print(f"{float(steps[-1]['train_speed(s/it)']):.2f}" if steps else "-")
PY
)
  if [ "$peak" = "-" ]; then
    if grep -qi 'out of memory\|OutOfMemory' "$log" 2>/dev/null; then peak="OOM"
    else peak="FAIL"; fi
  fi
  printf '%-24s %-16s %-22s %-6s %-4s %-14s %8s %10s\n' \
         "$name" "$plans" "${mods:-all}" "$gpus" "$nproc" "${ds:-none}" "$peak" "$spd"
  echo "    rc=$rc wall=${wall}s log=$(basename "$log")"
done
