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
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

printf '%s\n' "$$" > "$BASE_DIR/colab_gcs_keepalive.pid"
# shellcheck source=/dev/null
source "$COMMON_SH"

while true; do
    if ! "$COLAB_CLI" sessions >/dev/null 2>&1; then
        echo "$(date -u +%FT%TZ) colab sessions refresh failed for session=$SESSION" >&2
    fi

    if ( set -euo pipefail; colab_push_access_token ssh "$SESSION" "$REMOTE_TOKEN_FILE" </dev/null ); then
        echo "$(date -u +%FT%TZ) token push ok for session=$SESSION"
    else
        echo "$(date -u +%FT%TZ) keepalive/token push failed for session=$SESSION" >&2
    fi

    if "$COLAB_CLI" exec -s "$SESSION" --timeout 600 <<'PYEOF'
import time
for _ in range(9):
    time.sleep(60)
    print(".", end="", flush=True)
print("ok", flush=True)
PYEOF
    then
        echo "$(date -u +%FT%TZ) kernel keepalive cycle ok for session=$SESSION"
    else
        echo "$(date -u +%FT%TZ) kernel keepalive exec exited early for session=$SESSION; sleeping 15s" >&2
        sleep 15
    fi
done
