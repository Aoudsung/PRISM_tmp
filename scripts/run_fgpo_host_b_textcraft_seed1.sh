#!/usr/bin/env bash
set -euo pipefail
umask 077
test "$(hostname)" = ubuntu

deploy=$(cd -- "$(dirname -- "$0")" && pwd)
root=$(cd -- "$deploy/../.." && pwd)
out="$root/output-textcraft-v4-20261120"
python="$HOME/env/prism-native-v1/bin/python"

cd -- "$deploy"
mkdir -p "$out/launch/tmp"
export PYTHONPATH="$deploy/src" PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PIP_NO_INDEX=1 RAY_USAGE_STATS_ENABLED=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export TMPDIR="$out/launch/tmp"

exec 8>>"$root/output-webshop-menu-fixed-20261119/launch/webshop-gpu-0.queue.lock"
flock -n 8
printf '%s\n' "$$" > "$out/launch/textcraft-fgpo-seed-1.pid"
exec taskset -c 0-37,76-113 "$python" -u "$root/deploy/capped_study.py" \
    --config "$deploy/fgpo-v4-host-b-textcraft-20261120.json" --output-dir "$out" \
    --suite comparison --benchmarks textcraft --methods fgpo --seeds 1 --gpu 0 --model-gpu 0 "$@"
