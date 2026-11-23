#!/usr/bin/env bash
set -euo pipefail

ROOT="${FGPO_R8_ROOT:-/content/fgpo-r8}"
OUT="${FGPO_R8_OUTPUT:-/content/outputs/fgpo-r8-1p5b-colab-20261123}"
R7_OUT="${FGPO_R7_OUTPUT:-/content/outputs/fgpo-r7-1p5b-colab-20261123}"
WAIT_LOCK="${FGPO_WAIT_LOCK:-$R7_OUT/launch/main.lock}"
PY="${FGPO_R8_PYTHON:-/content/venvs/fgpo-main/bin/python}"

export JAVA_HOME="${JAVA_HOME:-/content/tools/jre21}"
export PATH="$JAVA_HOME/bin:$PATH"
export PYTHONPATH="$ROOT/src"
export FGPO_GCS_ACCESS_TOKEN_FILE="${FGPO_GCS_ACCESS_TOKEN_FILE:-/run/fgpo-gcs/access-token}"
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

if [[ -n "$WAIT_LOCK" && -e "$WAIT_LOCK" ]]; then
    printf '%s waiting for prior r7 queue lock %s to release...\n' "$(date -u +%FT%TZ)" "$WAIT_LOCK"
    flock "$WAIT_LOCK" true
    printf '%s prior r7 queue lock %s released; checking r7 finalization...\n' "$(date -u +%FT%TZ)" "$WAIT_LOCK"
    test -s "$R7_OUT/comparison/alfworld/fgpo/seed-0/.finalization_complete.json"
fi

JOBS=(
    "alfworld:0:$ROOT/configs/fgpo-r8-1p5b-alfworld-colab-gcs.json"
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
    resume=()
    backend=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1])).get("checkpoint_backend", "local"))' "$config")
    if [[ "$backend" == "gcs" ]]; then
        gcs_dto=$("$PY" "$ROOT/scripts/fgpo_gcs_checkpoint.py" status \
            --config "$config" --cell-dir "$cell" \
            --suite comparison --method fgpo --benchmark "$benchmark" --seed "$seed")
        gcs_state=$("$PY" -c 'import json, sys
d = json.loads(sys.argv[1])
st = d.get("status", "absent")
if st == "complete" and d.get("terminal_export_completed"):
    print("complete")
elif st != "absent":
    print("resume")
else:
    print("fresh")' "$gcs_dto")
        if [[ "$gcs_state" == "complete" ]]; then
            printf '%s skip gcs-completed %s seed %s\n' "$(date -u +%FT%TZ)" "$benchmark" "$seed"
            continue
        elif [[ "$gcs_state" == "resume" ]]; then
            resume=(--resume)
        fi
    else
        if [[ -s "$cell/.finalization_complete.json" ]]; then
            printf '%s skip completed %s seed %s\n' "$(date -u +%FT%TZ)" "$benchmark" "$seed"
            continue
        fi
        if [[ -s "$cell/checkpoint/CURRENT" || -s "$cell/cell_summary.json" ]]; then
            resume=(--resume)
        fi
    fi
    printf '%s start r8 %s seed %s (%s)\n' "$(date -u +%FT%TZ)" "$benchmark" "$seed" "${resume[*]:-fresh}"
    (
        exec 9>&-
        exec "$PY" -u -m fgpo.runner \
            --config "$config" \
            --output-dir "$OUT" \
            --suite comparison \
            --benchmarks "$benchmark" \
            --methods fgpo \
            --seeds "$seed" \
            --gpu 0 \
            --model-gpu 0 \
            "${resume[@]}" </dev/null
    ) >> "$OUT/launch/$benchmark-seed-$seed.log" 2>&1 &
    worker=$!
    printf '%s\n' "$worker" > "$OUT/launch/$benchmark-seed-$seed.pid"
    trap handle_signal TERM INT
    wait "$worker"
    trap - TERM INT
    worker=
    test -s "$cell/.finalization_complete.json"
    printf '%s complete %s seed %s\n' "$(date -u +%FT%TZ)" "$benchmark" "$seed"
done
