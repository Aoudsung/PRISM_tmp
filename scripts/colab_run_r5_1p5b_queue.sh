#!/usr/bin/env bash
set -euo pipefail

ROOT="${FGPO_R5_ROOT:-/content/fgpo-r5}"
OUT="${FGPO_R5_OUTPUT:-/content/outputs/fgpo-r5-1p5b-colab-20261122}"
PY="${FGPO_R5_PYTHON:-/content/venvs/fgpo-main/bin/python}"

export JAVA_HOME="${JAVA_HOME:-/content/tools/jre21}"
export PATH="$JAVA_HOME/bin:$PATH"
export PYTHONPATH="$ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PIP_NO_INDEX=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=1

cd "$ROOT"
mkdir -p "$OUT/launch"
exec 9>>"$OUT/launch/main.lock"
flock -n 9
printf '%s\n' "$$" > "$OUT/launch/queue.pid"

rm -f "$OUT/launch/main.exit_code"
write_exit() {
    local code=$?
    printf '%s\n' "$code" > "$OUT/launch/main.exit_code.tmp"
    mv -f "$OUT/launch/main.exit_code.tmp" "$OUT/launch/main.exit_code"
}
trap write_exit EXIT

JOBS=(
    "alfworld:0:$ROOT/configs/fgpo-r5-1p5b-alfworld-colab.json"
)

worker=
handle_signal() {
    if [[ -n "${worker:-}" ]]; then
        kill -TERM "$worker" 2>/dev/null || true
        wait "$worker" || true
    fi
    exit 143
}

for spec in "${JOBS[@]}"; do
    IFS=: read -r benchmark seed config <<< "$spec"
    cell="$OUT/comparison/$benchmark/fgpo/seed-$seed"
    if [[ -s "$cell/.finalization_complete.json" ]]; then
        printf '%s skip completed %s seed %s\n' "$(date -u +%FT%TZ)" "$benchmark" "$seed"
        continue
    fi
    resume=()
    if [[ -s "$cell/checkpoint/CURRENT" || -s "$cell/cell_summary.json" ]]; then
        resume=(--resume)
    fi
    printf '%s start %s seed %s (%s)\n' "$(date -u +%FT%TZ)" "$benchmark" "$seed" "${resume[*]:-fresh}"
    "$PY" -u -m fgpo.runner \
        --config "$config" \
        --output-dir "$OUT" \
        --suite comparison \
        --benchmarks "$benchmark" \
        --methods fgpo \
        --seeds "$seed" \
        --gpu 0 \
        --model-gpu 0 \
        "${resume[@]}" </dev/null >> "$OUT/launch/$benchmark-seed-$seed.log" 2>&1 &
    worker=$!
    printf '%s\n' "$worker" > "$OUT/launch/$benchmark-seed-$seed.pid"
    trap handle_signal TERM INT
    wait "$worker"
    trap - TERM INT
    worker=
    test -s "$cell/.finalization_complete.json"
    printf '%s complete %s seed %s\n' "$(date -u +%FT%TZ)" "$benchmark" "$seed"
done
