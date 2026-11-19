#!/usr/bin/env bash
set -euo pipefail
umask 077
test "$(hostname)" = mgt01
root=/apps/users/adminuser/fgpo-v4-formal-host-a-20261118T030647Z
deploy=$(cd -- "$(dirname -- "$0")" && pwd)
cd -- "$root"
lane=$1
config=$2
study=$3
python="$root/runtime/venv/bin/python"
case "$lane" in
    3-4)
        gpu=(3 4)
        lock="$root/logs/webshop-3-4-v4-formal.lock"
        peer=1-7
        methods=(fgpo fgpo_obs_only fgpo_no_support_weight)
        ;;
    1-7)
        gpu=(1 7)
        lock="$root/queue-1-7.lock"
        peer=3-4
        methods=(graphgpo g2po fgpo_no_transfer)
        ;;
    *)
        exit 2
        ;;
esac
test -f "$config"
test -f "$deploy/src/fgpo/runner.py"
artifacts="$deploy/lane-$lane"
mkdir -p -- "$artifacts/tmp" "$study"
export PYTHONPATH="$deploy/src" PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PIP_NO_INDEX=1 RAY_USAGE_STATS_ENABLED=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu[0]},${gpu[1]}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export TMPDIR="$artifacts/tmp" HF_HOME="$root/runtime/hf"

exec 8>>"$lock"
flock -n 8
exec 9>>"$deploy/disk-admission.lock"
lease="$deploy/lane-$lane.disk-lease"
peer_lease="$deploy/lane-$peer.disk-lease"
child=
stop() {
    if [[ -n "$child" ]]; then
        kill -TERM "$child" 2>/dev/null || true
        wait "$child" || true
    fi
    exit 143
}
trap stop TERM INT

reserve_checkpoint_space() {
    local pending=$1 status account announced=false
    local -a cells
    while true; do
        flock 9
        cells=("$pending")
        if ! flock -n "$peer_lease" true; then
            cells+=("$(cat -- "$peer_lease")")
        fi
        if account=$("$python" "$deploy/launch-disk-space.py" "$study" "${cells[@]}"); then
            exec 7>>"$lease"
            flock -n 7
            printf '%s\n' "$pending" >"$lease"
            flock -u 9
            printf '%s admitted %s\n' "$(date -u +%FT%TZ)" "$account"
            return
        else
            status=$?
        fi
        flock -u 9
        if ((status != 75)); then
            return "$status"
        fi
        if [[ "$announced" == false ]]; then
            printf '%s waiting for checkpoint space %s\n' "$(date -u +%FT%TZ)" "$account"
            announced=true
        fi
        sleep 30
    done
}

for method in "${methods[@]}"; do
    case "$method" in
        fgpo_obs_only|fgpo_no_support_weight|fgpo_no_transfer) suite=ablation ;;
        *) suite=comparison ;;
    esac
    for seed in 0 1 2; do
        cell="$study/$suite/textcraft/$method/seed-$seed"
        name="$suite.textcraft.$method.seed-$seed"
        resume=()
        if [[ -f "$cell/cell_summary.json" ]]; then
            terminal=$("$python" -c 'import json,sys; s=json.load(open(sys.argv[1])); print(s["status"] == "complete" and s["finalization_stage"] == "finalized")' "$cell/cell_summary.json")
            if [[ "$terminal" == True ]]; then
                printf '%s already finalized %s\n' "$(date -u +%FT%TZ)" "$name"
                continue
            fi
        fi
        if [[ -e "$cell" ]]; then
            if [[ -f "$cell/checkpoint/CURRENT" || -f "$cell/final_checkpoint/CURRENT" ]]; then
                resume=(--resume)
            else
                if [[ -s "$cell/training.jsonl" || -e "$cell/training_summary.json" || -e "$cell/cell_summary.json" ]] ||
                    grep -Eq '"role": "checkpoint.publish"|"env_steps": [1-9]' "$cell/costs.jsonl" 2>/dev/null; then
                    printf '%s recorded work has no checkpoint pointer %s\n' "$(date -u +%FT%TZ)" "$cell" >&2
                    exit 1
                fi
                attempt="$artifacts/failed-startups/$name.$(date -u +%Y%m%dT%H%M%SZ).$$"
                mkdir -p -- "$(dirname -- "$attempt")"
                mv -T -- "$cell" "$attempt"
                printf '%s archived incomplete startup %s\n' "$(date -u +%FT%TZ)" "$attempt"
            fi
        fi
        reserve_checkpoint_space "$cell"
        command=("$python" -u -m fgpo.runner --config "$config" --output-dir "$study"
                 --suite "$suite" --benchmarks textcraft --methods "$method" --seeds "$seed"
                 --gpu "${gpu[@]}" --model-gpu "${gpu[@]}" "${resume[@]}")
        printf '%s command' "$(date -u +%FT%TZ)"
        printf ' %q' "${command[@]}"
        printf '\n'
        "${command[@]}" </dev/null >>"$artifacts/$name.log" 2>&1 9>&- &
        child=$!
        printf '%s started %s pid=%s\n' "$(date -u +%FT%TZ)" "$name" "$child"
        status=0
        wait "$child" || status=$?
        printf '%s exited %s pid=%s status=%s\n' "$(date -u +%FT%TZ)" "$name" "$child" "$status"
        child=
        flock -u 7
        exec 7>&-
        if ((status != 0)); then
            exit "$status"
        fi
    done
done
printf '%s lane=%s completed\n' "$(date -u +%FT%TZ)" "$lane"
