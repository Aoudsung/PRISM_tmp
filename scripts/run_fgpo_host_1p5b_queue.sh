#!/usr/bin/env bash
set -euo pipefail
umask 077

benchmark=$1
test "$2" = --gpu
gpu=$3
shift 3
deploy=$(cd -- "$(dirname -- "$0")" && pwd)
root=$(cd -- "$deploy/../.." && pwd)
adapted_output=
case "$(hostname -s):$benchmark:$gpu" in
    mgt01:alfworld:7)
        variant=joint
        output="$root/study-alfworld-1p5b-joint-20261121"
        lock="$root/queue-1-7.lock"
        worker_python="$root/runtime/venv/bin/python"
        runner=("$worker_python" -u -m fgpo.runner)
        gpu_headroom_mib=1536
        ;;
    mgt01:textcraft:3)
        variant=joint
        output="$root/study-textcraft-1p5b-joint-20261121"
        predecessor="$root/study-textcraft-v4-joint-20261120/comparison/textcraft/fgpo/seed-0/.finalization_complete.json"
        lock="$root/logs/webshop-3-4-v4-formal.lock"
        runner=(taskset -c 0-17,72-89 "$root/runtime/venv/bin/python" -u -m fgpo.runner)
        ;;
    ubuntu:webshop:0|ubuntu:webshop:2)
        variant=joint
        output="$root/output-webshop150-1p5b-joint-20261121"
        export JAVA_HOME="$HOME/workspace/prism-native-external/native-text-data-20261108/jdk-root/usr/lib/jvm/java-21-openjdk-amd64"
        export PATH="$JAVA_HOME/bin:$PATH"
        if [[ "$gpu" == 0 ]]; then
            predecessor="$root/output-textcraft-1p5b-joint-20261121/comparison/textcraft/fgpo/seed-1/.finalization_complete.json"
        elif [[ "${1:-}" == --adapted-pair ]]; then
            test "$#" -eq 1
            set -- 1
            adapted_output="$root/output-webshop150-1p5b-joint-adapted-20261121"
            predecessor="$output/comparison/$benchmark/fgpo/seed-1/.finalization_complete.json"
            gpu_headroom_mib=1536
        fi
        ;;
    ubuntu:textcraft:0)
        if [[ "${1:-}" == --joint-format ]]; then
            test "$#" -eq 1
            set -- 1
            variant=joint-format
            output="$root/output-textcraft-1p5b-joint-format-20261121"
            gpu_headroom_mib=1536
        else
            variant=joint
            output="$root/output-textcraft-1p5b-joint-20261121"
            predecessor="$root/output-textcraft-v4-radius-20261121/comparison/textcraft/fgpo/seed-1/.finalization_complete.json"
        fi
        ;;
    *) exit 2 ;;
esac
if [[ "$(hostname -s)" == ubuntu ]]; then
    case "$gpu" in
        0) cpus=0-37,76-113 ;;
        2) cpus=38-75,114-151 ;;
    esac
    lock="$root/output-webshop-menu-fixed-20261119/launch/webshop-gpu-$gpu.queue.lock"
    worker_python="$HOME/env/prism-native-v1/bin/python"
    runner=(taskset -c "$cpus" "$worker_python" -u "$root/deploy/capped_study.py")
fi

export PYTHONPATH="$deploy/src" PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PIP_NO_INDEX=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
cd -- "$deploy"
mkdir -p "$output/launch"
printf '%s\n' "$$" > "$output/launch/queue-gpu-$gpu.pid"
if [[ -n "$adapted_output" ]]; then
    mkdir -p "$adapted_output/launch"
    printf '%s\n' "$$" > "$adapted_output/launch/queue-gpu-$gpu.pid"
fi

if [[ -n "${predecessor:-}" ]]; then
    printf 'Waiting for evaluation finalization at %s\n' "$predecessor"
    until [[ -s "$predecessor" ]]; do sleep 60; done
fi

exec 8>>"$lock"
flock 8
if [[ -z "${gpu_headroom_mib:-}" ]]; then
    printf 'Waiting for GPU %s to have no active compute process\n' "$gpu"
fi
for seed in "$@"; do
    cell_output=$output
    cell_variant=$variant
    if [[ -n "$adapted_output" && "$seed" == 1 ]]; then
        cell_output=$adapted_output
        cell_variant="$variant-adapted"
    fi
    config="$deploy/fgpo-1p5b-$benchmark-$cell_variant.json"
    cell="$cell_output/comparison/$benchmark/fgpo/seed-$seed"
    [[ ! -s "$cell/.finalization_complete.json" ]] || continue
    resume=()
    if [[ -s "$cell/checkpoint/CURRENT" || -s "$cell/cell_summary.json" ]]; then resume=(--resume); fi

    if [[ -n "${gpu_headroom_mib:-}" ]]; then
        cap_mib=$("$worker_python" -c 'import json, math, sys; print(math.ceil(json.load(open(sys.argv[1]))["gpu_allocator_cap_gib"] * 1024))' "$config")
        required_mib=$((cap_mib + gpu_headroom_mib))
        while :; do
            free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
            [[ "$free_mib" =~ ^[0-9]+$ ]]
            if (( free_mib >= required_mib )); then break; fi
            printf 'GPU %s: %s MiB free, %s MiB needed to start\n' "$gpu" "$free_mib" "$required_mib"
            sleep 60
        done
        printf 'GPU %s: %s MiB free, allocator capped at %s MiB\n' "$gpu" "$free_mib" "$cap_mib"
    else
        active=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)
        while [[ -n "$active" ]]; do
            sleep 60
            active=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)
        done
    fi
    printf '%s Starting %s %s seed %s on GPU %s\n' "$(date -u +%FT%TZ)" "$benchmark" "$cell_variant" "$seed" "$gpu"
    "${runner[@]}" --config "$config" --output-dir "$cell_output" --suite comparison \
        --benchmarks "$benchmark" --methods fgpo --seeds "$seed" --gpu "$gpu" --model-gpu "$gpu" \
        "${resume[@]}" >> "$cell_output/launch/seed-$seed.log" 2>&1 &
    worker=$!
    printf '%s\n' "$worker" > "$cell_output/launch/seed-$seed.pid"
    trap 'kill -TERM "$worker"; wait "$worker" || true; exit 143' TERM INT
    wait "$worker"
    trap - TERM INT
    test -s "$cell/.finalization_complete.json"
done
