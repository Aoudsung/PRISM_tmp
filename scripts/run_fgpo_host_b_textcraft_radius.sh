#!/usr/bin/env bash
set -euo pipefail
umask 077
test "$(hostname)" = ubuntu

deploy=$(cd -- "$(dirname -- "$0")" && pwd)
root=$(cd -- "$deploy/../.." && pwd)
standard="$root/output-textcraft-v4-20261120"
comparison="$root/output-textcraft-v4-radius-20261121"
base_config="$deploy/fgpo-v4-host-b-textcraft-20261120.json"
radius_config="$deploy/fgpo-v4-host-b-textcraft-radius-20261121.json"
python="$HOME/env/prism-native-v1/bin/python"
relative_cell=comparison/textcraft/fgpo/seed-1
replay="$comparison/radius-replay"

cd -- "$deploy"
mkdir -p "$comparison/launch/tmp"
export PYTHONPATH="$deploy/src" PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PIP_NO_INDEX=1 RAY_USAGE_STATS_ENABLED=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export TMPDIR="$comparison/launch/tmp"

exec 8>>"$root/output-webshop-menu-fixed-20261119/launch/webshop-gpu-0.queue.lock"
flock 8

cell="$standard/$relative_cell"
jq -e '.status == "interrupted"' "$cell/training_summary.json" >/dev/null
snapshot=$(jq -er .snapshot "$cell/checkpoint/CURRENT")
snapshot_dir="$cell/checkpoint/snapshots/$snapshot"
jq -e '.parameters_synced and .has_optimizer and .has_reference' "$snapshot_dir/metadata.json" >/dev/null
retained="$replay/state-$snapshot"
if [[ ! "$snapshot_dir/state.pt" -ef "$retained.pt" ]]; then
    ln "$snapshot_dir/state.pt" "$retained.pt"
fi
printf '%s\n' "$snapshot_dir/state.pt" > "$retained.origin"

taskset -c 0-37,76-113 "$python" -u "$deploy/compare_fgpo_webshop_radius.py" \
    --benchmark textcraft --snapshot "$retained.pt" --reference "$replay/reference.pt" \
    --output "$comparison/radius-comparison.json" >> "$comparison/launch/radius-replay.log" 2>&1

jq '.credit_radius = 0.05' "$base_config" > "$radius_config"
printf '%s\n' "$$" > "$comparison/launch/textcraft-fgpo-seed-1.pid"
exec taskset -c 0-37,76-113 "$python" -u "$root/deploy/capped_study.py" \
    --config "$radius_config" --output-dir "$comparison" \
    --suite comparison --benchmarks textcraft --methods fgpo --seeds 1 --gpu 0 --model-gpu 0 \
    >> "$comparison/launch/textcraft-fgpo-seed-1.log" 2>&1
