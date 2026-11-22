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
original_deploy="$formal/deploy/webshop-study-20261119"
original_out="$formal/output-webshop-menu-fixed-20261119"
out="$formal/output-webshop150-paired-20261119"
launch="$out/launch"
config="$deploy/host-b-webshop150.json"
python="$HOME/env/prism-native-v1/bin/python"
native="$HOME/workspace/prism-native-external/native-text-data-20261108"
launcher="$formal/deploy/capped_study.py"
export JAVA_HOME="$native/jdk-root/usr/lib/jvm/java-21-openjdk-amd64"
export PATH="$JAVA_HOME/bin:$PATH" PYTHONPATH="$deploy/src" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PIP_NO_INDEX=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1

cd -- "$deploy"
mkdir -p "$launch"
# Share the predecessor's physical-GPU queue lock through the handoff.
exec 8>>"$original_out/launch/webshop-gpu-$gpu.queue.lock"
flock -n 8

archive_incomplete_start() (
    local root=$1 suite=$2 method=$3 seed=$4 marker attempt attempts
    exec 9>>"${cell%/*}/.seed-$seed.lock"
    if ! flock -n 9; then
        printf 'Cell has an active owner: %s\n' "$cell" >&2
        exit 1
    fi
    if [[ ! -d "$cell" || -L "$cell" ]]; then
        printf 'Unexpected cell directory: %s\n' "$cell" >&2
        exit 1
    fi
    for marker in checkpoint/CURRENT checkpoint/FINAL final_checkpoint continuation.json \
        training.jsonl training_summary.json evaluation-test.jsonl test cell_summary.json .finalization_complete.json; do
        if [[ -e "$cell/$marker" || -L "$cell/$marker" ]]; then
            printf 'Preserving %s because it has %s\n' "$cell" "$marker" >&2
            exit 1
        fi
    done
    if [[ -e "$cell/costs.jsonl" ]]; then
        # A partial initial checkpoint can have initialization and serialization costs.
        if grep -Eqv '"kind": "initialization"|"role": "checkpoint.serialize_parameters"' "$cell/costs.jsonl"; then
            printf 'Preserving %s because its journal goes beyond initialization\n' "$cell" >&2
            exit 1
        elif [[ "$?" != 1 ]]; then
            exit 1
        fi
    fi
    attempts="$root/attempts/$suite/webshop/$method/seed-$seed"
    mkdir -p "$attempts"
    attempt=$(mktemp -d "$attempts/start.XXXXXXXX")
    if [[ "$(stat -c %d -- "$cell")" != "$(stat -c %d -- "$attempt")" ]]; then
        printf 'Attempt archive must share the filesystem of %s\n' "$cell" >&2
        exit 1
    fi
    mv -T -- "$cell" "$attempt"
    printf 'Retained interrupted startup at %s\n' "$attempt"
)

start_cell() {
    local root=$1 source=$2 config_path=$3 suite=$4 method=$5 seed=$6
    local receipt="$root/launch/webshop-$method-seed-$seed.pid" existing active
    local -a argv resume=()
    cell="$root/$suite/webshop/$method/seed-$seed"
    running=
    child=
    if [[ "$root/$suite/$method" == "$original_out/comparison/fgpo" && "$seed" != 2 ]]; then
        receipt="$root/launch/webshop-seed-$seed.pid"
    fi
    if [[ -s "$receipt" ]]; then
        read -r existing < "$receipt"
        if [[ "$existing" =~ ^[0-9]+$ && -r "/proc/$existing/cmdline" ]] &&
                mapfile -d '' -t argv < "/proc/$existing/cmdline" &&
                [[ " ${argv[*]} " == *" $launcher --config "* &&
                   " ${argv[*]} " == *" --output-dir $root --suite $suite --benchmarks webshop --methods $method --seeds $seed --gpu $gpu "* ]]; then
            running=$existing
            printf 'adopt PID %s (%s)\n' "$running" "$cell"
            return
        fi
    fi
    if [[ -s "$cell/.finalization_complete.json" ]]; then
        return
    fi

    active=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)
    while grep -Eq '^[[:space:]]*[0-9]+' <<<"$active"; do
        printf '%s waiting for GPU %s before starting %s\n' "$(date -u +%FT%TZ)" "$gpu" "$cell" >&2
        sleep 60
        active=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)
    done
    if [[ -s "$cell/checkpoint/CURRENT" || -s "$cell/cell_summary.json" ]]; then
        resume=(--resume)
    elif [[ -e "$cell" || -L "$cell" ]]; then
        archive_incomplete_start "$root" "$suite" "$method" "$seed"
    fi
    printf 'start %s on GPU %s\n' "$cell" "$gpu"
    (
        cd -- "$source"
        export PYTHONPATH="$source/src"
        exec taskset -c "$cpus" "$python" -u "$launcher" \
            --config "$config_path" --output-dir "$root" \
            --suite "$suite" --benchmarks webshop --methods "$method" --seeds "$seed" \
            --gpu "$gpu" --model-gpu "$gpu" "${resume[@]}"
    ) >> "$root/launch/webshop-$method-seed-$seed.log" 2>&1 </dev/null 8>&- &
    child=$!
    running=$child
    printf '%s\n' "$child" > "$receipt"
}

wait_cell() {
    if [[ -n "$child" ]]; then
        wait "$child" || return "$?"
    elif [[ -n "$running" ]]; then
        tail --pid="$running" --sleep-interval=5 -f /dev/null 8>&- || return "$?"
    fi
    if [[ ! -s "$cell/.finalization_complete.json" ]]; then
        printf 'Missing terminal receipt for %s\n' "$cell" >&2
        return 1
    fi
    printf 'complete %s\n' "$cell"
}

report_pairs() {
    CUDA_VISIBLE_DEVICES= "$python" -u "$deploy/scripts/report_fgpo_webshop_paired.py" \
        --study-root "$out" --v4-60-root "$original_out" >> "$launch/report-gpu-$gpu.log" 2>&1 8>&-
}

original_config="$original_deploy/host-b-webshop-study.json"
while IFS=$'\t' read -r lane phase method seed; do
    [[ "$lane" == "$gpu" ]] || continue
    archive="$out/continuation/seed-$seed"
    original_cell="$original_out/comparison/webshop/fgpo/seed-$seed"
    case "$phase" in
        original60)
            start_cell "$original_out" "$original_deploy" "$original_config" comparison "$method" "$seed"
            CUDA_VISIBLE_DEVICES= "$python" -u -m fgpo.webshop_continuation pin \
                --source-cell "$cell" --archive "$archive" \
                >> "$launch/pin-seed-$seed.log" 2>&1 </dev/null 8>&- &
            pin=$!
            if ! wait_cell; then
                kill "$pin" 2>/dev/null || true
                wait "$pin" || true
                exit 1
            fi
            wait "$pin"
            ;;
        continuation150)
            target_cell="$out/comparison/webshop/$method/seed-$seed"
            if [[ ! -s "$target_cell/checkpoint/CURRENT" && ! -s "$target_cell/cell_summary.json" ]]; then
                CUDA_VISIBLE_DEVICES= "$python" -u -m fgpo.webshop_continuation prepare \
                    --archive "$archive" --target-cell "$target_cell" --config "$config" \
                    >> "$launch/prepare-seed-$seed.log" 2>&1 </dev/null 8>&-
            fi
            start_cell "$out" "$deploy" "$config" comparison "$method" "$seed"
            wait_cell
            report_pairs
            ;;
        fresh150)
            start_cell "$out" "$deploy" "$config" comparison "$method" "$seed"
            wait_cell
            report_pairs
            ;;
        ablation60)
            test -s "$original_cell/.finalization_complete.json"
            start_cell "$original_out" "$original_deploy" "$original_config" ablation "$method" "$seed"
            wait_cell
            ;;
        *) printf 'Unknown queue phase %s\n' "$phase" >&2; exit 2 ;;
    esac
done < "$deploy/webshop150-queue.tsv"
