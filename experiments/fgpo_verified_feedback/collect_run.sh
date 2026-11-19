#!/usr/bin/env bash
set -euo pipefail
umask 077
test "$(hostname -s)" = mgt01
cd -- "$(dirname -- "$0")"
root=$PWD
test "$root" = /apps/users/adminuser/fgpo-verified-feedback-20261118-v5

formal=/apps/users/adminuser/fgpo-v4-formal-host-a-20261118T030647Z
pilot=/apps/users/adminuser/fgpo-feedback-pilot-20261118-a1
export PYTHONPATH="$root/v5src" PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export HF_HOME="$root/cache/hf" XDG_CACHE_HOME="$root/cache/xdg"
mkdir -p -- "$root/logs" "$root/tmp" "$HF_HOME" "$XDG_CACHE_HOME"

run_shard() {
    local shard=$1 gpu=$2 cpus=$3 status=0
    local log="$root/logs/shard-$shard"
    export CUDA_VISIBLE_DEVICES="$gpu" TMPDIR="$root/tmp/shard-$shard"
    mkdir -p -- "$TMPDIR"
    printf '%s\tshard=%s\tphysical_gpu=%s\tcpus=%s\tstart\n' \
        "$(date -u +%FT%TZ)" "$shard" "$gpu" "$cpus" >> "$log.log"
    taskset --cpu-list "$cpus" "$formal/runtime/venv/bin/python" -u "$root/collect.py" \
        --root "$root" --config "$pilot/webshop-config.json" --shard "$shard" >> "$log.log" 2>&1 || status=$?
    printf '%s\n' "$status" > "$log.exit"
    printf '%s\tshard=%s\tphysical_gpu=%s\texit=%s\n' "$(date -u +%FT%TZ)" "$shard" "$gpu" "$status" >> "$log.log"
    return "$status"
}

run_shard 0 3 0-8,72-80 &
left=$!
printf '%s\n' "$left" > "$root/logs/shard-0.pid"
run_shard 1 4 9-17,81-89 &
right=$!
printf '%s\n' "$right" > "$root/logs/shard-1.pid"
status=0
wait "$left" || status=$?
wait "$right" || status=$?
exit "$status"
