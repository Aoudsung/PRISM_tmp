#!/usr/bin/env bash
set -euo pipefail
umask 077

gpu=${1:?Specify physical GPU 0 or 2}
case "$gpu" in
    0) cpus=0-37,76-113 ;;
    2) cpus=38-75,114-151 ;;
    *) exit 2 ;;
esac

deploy=$(cd -- "$(dirname -- "$0")" && pwd)
formal=$(cd -- "$deploy/../.." && pwd)
out="$formal/output-webshop-menu-fixed-20261119"
launch="$out/launch"
python="$HOME/env/prism-native-v1/bin/python"
native="$HOME/workspace/prism-native-external/native-text-data-20261108"
export JAVA_HOME="$native/jdk-root/usr/lib/jvm/java-21-openjdk-amd64"
export PATH="$JAVA_HOME/bin:$PATH" PYTHONPATH="$deploy/src" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PIP_NO_INDEX=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1

cd -- "$deploy"
mkdir -p "$launch"
exec 8>"$launch/webshop-gpu-$gpu.queue.lock"
flock -n 8

pid_file() {
    local method=$1 seed=$2
    if [[ "$method" == fgpo && ( "$seed" == 0 || "$seed" == 1 ) ]]; then
        printf '%s/webshop-seed-%s.pid\n' "$launch" "$seed"
    else
        printf '%s/webshop-%s-seed-%s.pid\n' "$launch" "$method" "$seed"
    fi
}

await_running_cell() {
    local cell=$1 receipt=$2 running
    if [[ ! -s "$receipt" ]]; then
        return
    fi
    read -r running < "$receipt"
    if [[ "$running" =~ ^[0-9]+$ && -r "/proc/$running/cmdline" ]] &&
            grep -zFxq -- "$out" "/proc/$running/cmdline"; then
        printf 'waiting for PID %s (%s)\n' "$running" "$cell"
        tail --pid="$running" --sleep-interval=20 -f /dev/null 8>&-
    fi
}

while IFS=$'\t' read -r lane suite method seed; do
    [[ "$lane" == "$gpu" ]] || continue
    cell="$out/$suite/webshop/$method/seed-$seed"
    receipt=$(pid_file "$method" "$seed")
    await_running_cell "$cell" "$receipt"
    if [[ -s "$cell/.finalization_complete.json" ]]; then
        printf 'complete %s seed %s\n' "$method" "$seed"
        continue
    fi
    if [[ "$suite" == ablation ]]; then
        reference="$out/comparison/webshop/fgpo/seed-$seed"
        await_running_cell "$reference" "$(pid_file fgpo "$seed")"
        test -s "$reference/.finalization_complete.json"
    fi

    active=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)
    if grep -Eq '^[[:space:]]*[0-9]+' <<<"$active"; then
        printf 'GPU %s is occupied; stopped before %s seed %s\n' "$gpu" "$method" "$seed" >&2
        exit 1
    fi
    resume=()
    if [[ -s "$cell/checkpoint/CURRENT" ]]; then
        resume=(--resume)
    fi
    printf 'start %s seed %s on GPU %s\n' "$method" "$seed" "$gpu"
    taskset -c "$cpus" "$python" -u "$formal/deploy/capped_study.py" \
        --config "$deploy/host-b-webshop-study.json" --output-dir "$out" \
        --suite "$suite" --benchmarks webshop --methods "$method" --seeds "$seed" \
        --gpu "$gpu" --model-gpu "$gpu" "${resume[@]}" \
        >> "$launch/webshop-$method-seed-$seed.log" 2>&1 </dev/null 8>&- &
    child=$!
    printf '%s\n' "$child" > "$receipt"
    wait "$child"
    test -s "$cell/.finalization_complete.json"
    printf 'complete %s seed %s\n' "$method" "$seed"
done < "$deploy/webshop-queue.tsv"
