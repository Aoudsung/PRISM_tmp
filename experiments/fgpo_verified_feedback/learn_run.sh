#!/usr/bin/env bash
set -euo pipefail
umask 077
test "$(hostname)" = mgt01
cd -- "$(dirname -- "$0")"
root=/apps/users/adminuser/fgpo-verified-feedback-20261118-v5
test "$PWD" = "$root"
archive=/apps/users/adminuser/fgpo-feedback-pilot-20261118-a1
formal=/apps/users/adminuser/fgpo-v4-formal-host-a-20261118T030647Z
adapter=/apps/users/adminuser/fgpo-v4-webshop-adapter-20261118.in2tA1Hp

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3,4
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$root/updates/tmp" HF_HOME="$root/updates/hf" XDG_CACHE_HOME="$root/updates/cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
mkdir -p -- "$TMPDIR" "$XDG_CACHE_HOME"

for arm in "$@"; do
    arm_env=()
    case "$arm" in
        anchor_v4|archive_null|archive_success|archive_success_history) source_dir="$adapter/src" ;;
        anchor_v5|fresh_null_history|fresh_success_history)
            source_dir="$root/v5src"
            arm_env=(CUBLAS_WORKSPACE_CONFIG=:4096:8)
            ;;
        *) printf 'Unknown learning arm %s\n' "$arm" >&2; exit 2 ;;
    esac
    printf '%s\t%s\tSTART\n' "$(date -u +%FT%TZ)" "$arm" >> "$root/updates/exit.tsv"
    code=0
    env "${arm_env[@]}" PYTHONPATH="$source_dir:$archive:$root/experiments/fgpo_verified_feedback" \
        "$formal/runtime/venv/bin/python" -u "$root/learn.py" \
        --root "$root" --archive "$archive" --config "$adapter/fgpo-v4-webshop-host-a-l40-3-4.json" \
        --arm "$arm" > "$root/updates/$arm.log" 2>&1 || code=$?
    printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$arm" "$code" >> "$root/updates/exit.tsv"
    if (( code != 0 )); then
        exit "$code"
    fi
done
