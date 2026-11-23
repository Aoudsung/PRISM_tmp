#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${FGPO_KEEPALIVE_DIR:-/tmp/fgpo-colab-keepalive}"
SESSION="${COLAB_SESSION:-colab}"
REMOTE_TOKEN_FILE="${REMOTE_TOKEN_FILE:-/run/fgpo-gcs/access-token}"
COMMON_SH="${COMMON_SH:-$SCRIPT_DIR/colab_common.sh}"
LOCK_DIR="$BASE_DIR/.colab_gcs_keepalive.lock"

mkdir -p "$BASE_DIR"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "$(date -u +%FT%TZ) another colab_gcs_keepalive holds $LOCK_DIR" >&2
    exit 1
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT INT TERM

printf '%s\n' "$$" > "$BASE_DIR/colab_gcs_keepalive.pid"
# shellcheck source=/dev/null
source "$COMMON_SH"

run_with_hard_timeout() {
    local timeout_sec="$1"
    shift
    python3 -c '
import os, signal, subprocess, sys
t = float(sys.argv[1])
p = subprocess.Popen(sys.argv[2:], start_new_session=True)
try:
    sys.exit(p.wait(timeout=t))
except subprocess.TimeoutExpired:
    sys.stderr.write(f"hard-timeout ({t:.0f}s) killing pgid={p.pid}: {sys.argv[2]}\n")
    sys.stderr.flush()
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    p.wait()
    sys.exit(124)
' "$timeout_sec" "$@"
}

push_token_once() {
    run_with_hard_timeout 90 bash -c '
        set -euo pipefail
        source "$1"
        colab_push_access_token ssh "$2" "$3" </dev/null
    ' _ "$COMMON_SH" "$SESSION" "$REMOTE_TOKEN_FILE"
}

while true; do
    if ! run_with_hard_timeout 60 "$COLAB_CLI" sessions </dev/null >/dev/null 2>&1; then
        echo "$(date -u +%FT%TZ) colab sessions refresh failed for session=$SESSION" >&2
    fi

    token_ok=0
    for attempt in 1 2 3; do
        if push_token_once; then
            echo "$(date -u +%FT%TZ) token push ok for session=$SESSION (attempt=$attempt)"
            token_ok=1
            break
        fi
        echo "$(date -u +%FT%TZ) token push attempt=$attempt failed for session=$SESSION; sleeping 5s" >&2
        sleep 5
    done
    if [ "$token_ok" -eq 0 ]; then
        echo "$(date -u +%FT%TZ) keepalive/token push failed after 3 attempts for session=$SESSION" >&2
    fi

    if run_with_hard_timeout 540 "$COLAB_CLI" exec -s "$SESSION" --timeout 510 <<'PYEOF'
import time
for _ in range(8):
    time.sleep(60)
    print(".", end="", flush=True)
print("ok", flush=True)
PYEOF
    then
        echo "$(date -u +%FT%TZ) kernel keepalive cycle ok for session=$SESSION"
    else
        echo "$(date -u +%FT%TZ) kernel keepalive exec exited early/timed out for session=$SESSION; sleeping 15s" >&2
        sleep 15
    fi
done
