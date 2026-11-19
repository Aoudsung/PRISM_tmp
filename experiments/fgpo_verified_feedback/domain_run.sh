#!/usr/bin/env bash
set -euo pipefail
umask 077
test "$(hostname -s)" = mgt01
cd -- "$(dirname -- "$0")"
domain=$PWD
root=${domain%/domain}
test "$root" = /apps/users/adminuser/fgpo-verified-feedback-20261118-v5

formal=/apps/users/adminuser/fgpo-v4-formal-host-a-20261118T030647Z
alf=/apps/users/adminuser/fgpo-v4-valid-seen-20261118
v4=/apps/users/adminuser/fgpo-v4-webshop-adapter-20261118.in2tA1Hp/src
v5="$root/v5src"
reference="$root/experiments/fgpo_verified_feedback"
app_config="$formal/study/comparison/appworld/g2po/seed-2/config.json"
alf_config="$alf/host-a.json"
python="$formal/runtime/venv/bin/python"

export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

wait_for_shard() {
    local shard=$1 status='' pid='' state=''
    local receipt="$root/logs/shard-$shard"
    until [[ -s "$receipt.exit" ]]; do sleep 20; done
    read -r status < "$receipt.exit"
    if [[ "$status" != 0 ]]; then
        printf 'Collection shard %s exited %s\n' "$shard" "$status"
        return 1
    fi
    if [[ ! -s "$receipt.pid" ]]; then
        printf 'Collection shard %s has no sampler process receipt\n' "$shard"
        return 1
    fi
    read -r pid < "$receipt.pid"
    while state=$(ps -p "$pid" -o stat=) && [[ "$state" != Z* ]]; do sleep 1; done
    printf '%s\tcollection_shard=%s\tpid=%s\thandoff\n' "$(date -u +%FT%TZ)" "$shard" "$pid"
}

run_phase() {
    local phase=$1 config=$2 source=$3 status=0
    shift 3
    export PYTHONPATH="$source:$reference" TMPDIR="$domain/tmp/$phase"
    export HF_HOME="$TMPDIR/hf" XDG_CACHE_HOME="$TMPDIR/xdg"
    export APPWORLD_ROOT="$domain/appworld-data" APPWORLD_CACHE="$TMPDIR/appworld-cache"
    mkdir -p -- "$HF_HOME" "$XDG_CACHE_HOME"
    local -a command=(taskset -c "$cpus" "$python" -u "$domain/domain_checks.py"
        "$phase" --root "$domain" --config "$config" "$@")
    {
        printf '%s\tphase=%s\tCUDA_VISIBLE_DEVICES=%s\tPYTHONPATH=%s\tstart\n' \
            "$(date -u +%FT%TZ)" "$phase" "$CUDA_VISIBLE_DEVICES" "$PYTHONPATH"
        printf '%q ' "${command[@]}"
        printf '\n'
        "${command[@]}" || status=$?
        printf '%s\tphase=%s\texit=%s\n' "$(date -u +%FT%TZ)" "$phase" "$status"
    } >> "$domain/$phase.log" 2>&1
    printf '%s\n' "$status" > "$domain/$phase.exit"
    return "$status"
}

case "${1:?Expected appworld or alfworld}" in
    appworld)
        wait_for_shard 0
        cpus=0-8,72-80
        export CUDA_VISIBLE_DEVICES=3
        run_phase appworld-v4 "$app_config" "$v4"
        run_phase appworld-v5 "$app_config" "$v5"
        ;;
    alfworld)
        wait_for_shard 1
        cpus=9-17,81-89
        export CUDA_VISIBLE_DEVICES=4
        run_phase alf-actor "$alf_config" "$v4" --actor "$alf/artifacts/seed-0/actor-cache/model"
        run_phase alf-credit "$alf_config" "$v4"
        ;;
    *) exit 2 ;;
esac
