#!/usr/bin/env bash
set -euo pipefail
umask 077

lane=$1
test "$2" = --gpu
gpu=$3

script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
deploy=$(cd -- "$script_dir/.." && pwd)
root=$(cd -- "$deploy/../.." && pwd)
host=$(hostname -s)

case "$host:$lane:$gpu" in
    mgt01:alfworld:1)
        lock="$root/queue-1-7.lock"
        worker_python="$root/runtime/venv/bin/python"
        runner=(taskset -c 18-35,90-107 "$worker_python" -u -m fgpo.runner)
        gpu_headroom_mib=1536
        queue_log_dir="$root/study-alfworld-r8-1p5b-20261123/launch"
        JOBS=(
            "r8-1p5b|alfworld|1|$deploy/configs/fgpo-r8-1p5b-alfworld-host-a.json|$root/study-alfworld-r8-1p5b-20261123"
        )
        ;;
    mgt01:textcraft:3)
        lock="$root/logs/webshop-3-4-v4-formal.lock"
        worker_python="$root/runtime/venv/bin/python"
        runner=(taskset -c 0-17,72-89 "$worker_python" -u -m fgpo.runner)
        gpu_headroom_mib=1536
        queue_log_dir="$root/study-textcraft-r8-1p5b-20261123/launch"
        JOBS=(
            "r8-1p5b|textcraft|0|$deploy/configs/fgpo-r8-1p5b-textcraft-host-a.json|$root/study-textcraft-r8-1p5b-20261123"
        )
        ;;
    ubuntu:textcraft:0)
        lock="$root/output-webshop-menu-fixed-20261119/launch/webshop-gpu-0.queue.lock"
        worker_python="$HOME/env/prism-native-v1/bin/python"
        runner=(taskset -c 0-37,76-113 "$worker_python" -u "$root/deploy/capped_study.py")
        gpu_headroom_mib=1536
        queue_log_dir="$root/output-textcraft-r8-1p5b-20261123/launch"
        JOBS=(
            "r8-1p5b|textcraft|1|$deploy/configs/fgpo-r8-1p5b-textcraft-host-b.json|$root/output-textcraft-r8-1p5b-20261123"
        )
        ;;
    ubuntu:webshop:2)
        export JAVA_HOME="$HOME/workspace/prism-native-external/native-text-data-20261108/jdk-root/usr/lib/jvm/java-21-openjdk-amd64"
        export PATH="$JAVA_HOME/bin:$PATH"
        lock="$root/output-webshop-menu-fixed-20261119/launch/webshop-gpu-2.queue.lock"
        worker_python="$HOME/env/prism-native-v1/bin/python"
        runner=(taskset -c 38-75,114-151 "$worker_python" -u "$root/deploy/capped_study.py")
        gpu_headroom_mib=1536
        queue_log_dir="$root/output-webshop150-r8-1p5b-20261123/launch"
        JOBS=(
            "r8-1p5b|webshop|1|$deploy/configs/fgpo-r8-1p5b-webshop-host-b.json|$root/output-webshop150-r8-1p5b-20261123"
            "r8-7b|webshop|1|$deploy/configs/fgpo-r8-7b-webshop-host-b.json|$root/output-webshop150-r8-7b-20261123"
        )
        ;;
    *)
        printf 'Unsupported host/lane/gpu: %s:%s:%s\n' "$host" "$lane" "$gpu" >&2
        exit 2
        ;;
esac

export PYTHONPATH="$deploy/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PIP_NO_INDEX=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$gpu"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1

cd -- "$deploy"
mkdir -p "$queue_log_dir" "$(dirname -- "$lock")"
printf '%s\n' "$$" > "$queue_log_dir/queue-gpu-$gpu.pid"

printf '%s Waiting for lock %s on %s GPU %s...\n' "$(date -u +%FT%TZ)" "$lock" "$host" "$gpu"
exec 8>>"$lock"
flock 8
printf '%s Acquired lock %s on %s GPU %s\n' "$(date -u +%FT%TZ)" "$lock" "$host" "$gpu"

for spec in "${JOBS[@]}"; do
    IFS='|' read -r tag benchmark seed config output <<< "$spec"
    mkdir -p "$output/launch"
    cell="$output/comparison/$benchmark/fgpo/seed-$seed"
    if [[ -s "$cell/.finalization_complete.json" ]]; then
        printf '%s Skip already completed %s %s seed %s\n' "$(date -u +%FT%TZ)" "$tag" "$benchmark" "$seed"
        continue
    fi
    resume=()
    if [[ -s "$cell/checkpoint/CURRENT" || -s "$cell/cell_summary.json" ]]; then
        resume=(--resume)
    fi

    cap_mib=$("$worker_python" -c 'import json, math, sys; print(math.ceil(json.load(open(sys.argv[1]))["gpu_allocator_cap_gib"] * 1024))' "$config")
    required_mib=$((cap_mib + gpu_headroom_mib))
    while :; do
        free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
        [[ "$free_mib" =~ ^[0-9]+$ ]]
        if (( free_mib >= required_mib )); then
            break
        fi
        printf '%s GPU %s: %s MiB free, %s MiB needed for %s\n' "$(date -u +%FT%TZ)" "$gpu" "$free_mib" "$required_mib" "$tag"
        sleep 20
    done
    printf '%s GPU %s: %s MiB free, allocator capped at %s MiB\n' "$(date -u +%FT%TZ)" "$gpu" "$free_mib" "$cap_mib"

    printf '%s Starting %s %s seed %s on %s GPU %s (%s)\n' "$(date -u +%FT%TZ)" "$tag" "$benchmark" "$seed" "$host" "$gpu" "${resume[*]:-fresh}"
    (
        exec 8>&-
        exec "${runner[@]}" --config "$config" --output-dir "$output" --suite comparison \
            --benchmarks "$benchmark" --methods fgpo --seeds "$seed" --gpu "$gpu" --model-gpu "$gpu" \
            "${resume[@]}"
    ) >> "$output/launch/seed-$seed.log" 2>&1 &
    worker=$!
    printf '%s\n' "$worker" > "$output/launch/seed-$seed.pid"
    trap 'kill -TERM "$worker" 2>/dev/null || true; wait "$worker" || true; exit 143' TERM INT
    wait "$worker"
    trap - TERM INT
    test -s "$cell/.finalization_complete.json"
    printf '%s Completed %s %s seed %s on %s GPU %s\n' "$(date -u +%FT%TZ)" "$tag" "$benchmark" "$seed" "$host" "$gpu"
done
