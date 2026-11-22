#!/usr/bin/env bash
set -euo pipefail
umask 077

benchmark=$1
test "$2" = --gpu
gpu=$3
shift 3
test "$#" -ge 1

script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
deploy=$(cd -- "$script_dir/.." && pwd)
root=$(cd -- "$deploy/../.." && pwd)
host=$(hostname -s)

case "$host:$benchmark:$gpu" in
    mgt01:alfworld:1)
        config="$deploy/configs/fgpo-r5-1p5b-alfworld-host-a.json"
        output="$root/study-alfworld-r5-1p5b-20261122"
        lock="$root/queue-1-7.lock"
        worker_python="$root/runtime/venv/bin/python"
        runner=(taskset -c 18-35,90-107 "$worker_python" -u -m fgpo.runner)
        gpu_headroom_mib=1536
        ;;
    mgt01:textcraft:3)
        config="$deploy/configs/fgpo-r5-1p5b-textcraft-host-a.json"
        output="$root/study-textcraft-r5-1p5b-20261122"
        lock="$root/logs/webshop-3-4-v4-formal.lock"
        worker_python="$root/runtime/venv/bin/python"
        runner=(taskset -c 0-17,72-89 "$worker_python" -u -m fgpo.runner)
        gpu_headroom_mib=1536
        ;;
    ubuntu:textcraft:0)
        config="$deploy/configs/fgpo-r5-1p5b-textcraft-host-b.json"
        output="$root/output-textcraft-r5-1p5b-20261122"
        lock="$root/output-webshop-menu-fixed-20261119/launch/webshop-gpu-0.queue.lock"
        worker_python="$HOME/env/prism-native-v1/bin/python"
        runner=(taskset -c 0-37,76-113 "$worker_python" -u "$root/deploy/capped_study.py")
        gpu_headroom_mib=1536
        ;;
    ubuntu:webshop:2)
        config="$deploy/configs/fgpo-r5-1p5b-webshop-host-b.json"
        output="$root/output-webshop150-r5-1p5b-20261122"
        export JAVA_HOME="$HOME/workspace/prism-native-external/native-text-data-20261108/jdk-root/usr/lib/jvm/java-21-openjdk-amd64"
        export PATH="$JAVA_HOME/bin:$PATH"
        lock="$root/output-webshop-menu-fixed-20261119/launch/webshop-gpu-2.queue.lock"
        worker_python="$HOME/env/prism-native-v1/bin/python"
        runner=(taskset -c 38-75,114-151 "$worker_python" -u "$root/deploy/capped_study.py")
        gpu_headroom_mib=1536
        ;;
    *)
        printf 'Unsupported host/benchmark/gpu: %s:%s:%s\n' "$host" "$benchmark" "$gpu" >&2
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
mkdir -p "$output/launch"
printf '%s\n' "$$" > "$output/launch/queue-gpu-$gpu.pid"

exec 8>>"$lock"
flock 8

for seed in "$@"; do
    cell="$output/comparison/$benchmark/fgpo/seed-$seed"
    [[ ! -s "$cell/.finalization_complete.json" ]] || continue
    resume=()
    if [[ -s "$cell/checkpoint/CURRENT" || -s "$cell/cell_summary.json" ]]; then
        resume=(--resume)
    fi

    cap_mib=$("$worker_python" -c 'import json, math, sys; print(math.ceil(json.load(open(sys.argv[1]))["gpu_allocator_cap_gib"] * 1024))' "$config")
    required_mib=$((cap_mib + gpu_headroom_mib))
    while :; do
        free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
        [[ "$free_mib" =~ ^[0-9]+$ ]]
        if (( free_mib >= required_mib )); then
            break
        fi
        printf 'GPU %s: %s MiB free, %s MiB needed to start\n' "$gpu" "$free_mib" "$required_mib"
        sleep 30
    done
    printf 'GPU %s: %s MiB free, allocator capped at %s MiB\n' "$gpu" "$free_mib" "$cap_mib"

    printf '%s Starting r5 %s seed %s on %s GPU %s (%s)\n' "$(date -u +%FT%TZ)" "$benchmark" "$seed" "$host" "$gpu" "${resume[*]:-fresh}"
    "${runner[@]}" --config "$config" --output-dir "$output" --suite comparison \
        --benchmarks "$benchmark" --methods fgpo --seeds "$seed" --gpu "$gpu" --model-gpu "$gpu" \
        "${resume[@]}" >> "$output/launch/seed-$seed.log" 2>&1 &
    worker=$!
    printf '%s\n' "$worker" > "$output/launch/seed-$seed.pid"
    trap 'kill -TERM "$worker" 2>/dev/null || true; wait "$worker" || true; exit 143' TERM INT
    wait "$worker"
    trap - TERM INT
    test -s "$cell/.finalization_complete.json"
done
