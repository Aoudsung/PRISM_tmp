#!/usr/bin/env bash
# Unified day-to-day Colab operations for F-GPO r3:
#   sync-code [session] [--force]
#   status [--cell SUITE/BM/METHOD/seed-N] [--session SESSION] [--ssh]
#   pull <SUITE/BM/METHOD/seed-N> [dest_dir] [--with-final-model]
#   stop <session> [--cell SUITE/BM/METHOD/seed-N] [--backup-checkpoint]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/colab_common.sh"

CMD="${1:-}"
[ -n "$CMD" ] || { echo "Usage: $0 {sync-code|status|pull|stop} [options]" >&2; exit 2; }
shift

case "$CMD" in
  sync-code)
    SESSION=""
    FORCE="0"
    while [ $# -gt 0 ]; do
      case "$1" in
        --force) FORCE="1"; shift ;;
        --bucket) FGPO_GCS_BUCKET="$2"; shift 2 ;;
        --prefix) FGPO_GCS_PREFIX="$2"; shift 2 ;;
        -*) echo "Unknown flag: $1" >&2; exit 2 ;;
        *) SESSION="$1"; shift ;;
      esac
    done
    SRC_TAR="/tmp/fgpo_r3_src.tar.gz"
    pack_fgpo_source "$SRC_TAR"
    gcs_py upload "$SRC_TAR" "${FGPO_GCS_PREFIX}/code/fgpo_r3_src.tar.gz" "application/gzip"
    if [ -n "$SESSION" ]; then
      if [ "$FORCE" != "1" ] && colab_ssh_run "$SESSION" "pgrep -f fgpo.runner >/dev/null 2>&1"; then
        echo "[sync-code] Refused: fgpo.runner is active in '$SESSION'. Stop training cleanly first or pass --force." >&2
        exit 1
      fi
      cat "$SRC_TAR" | colab_ssh_run "$SESSION" \
        "rm -rf /content/fgpo-r3/src/fgpo && mkdir -p /content/fgpo-r3 && tar --warning=no-unknown-keyword -xzf - -C /content/fgpo-r3"
      echo "[sync-code] Replaced source in '$SESSION:/content/fgpo-r3'"
    fi
    ;;

  status)
    CELL_ID="comparison/alfworld/fgpo/seed-0"
    SESSION="fgpo-alfworld-s0"
    USE_SSH="0"
    while [ $# -gt 0 ]; do
      case "$1" in
        --cell) CELL_ID="$2"; shift 2 ;;
        -s|--session) SESSION="$2"; shift 2 ;;
        --ssh) USE_SSH="1"; shift ;;
        --bucket) FGPO_GCS_BUCKET="$2"; shift 2 ;;
        --prefix) FGPO_GCS_PREFIX="$2"; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
      esac
    done
    echo "=== Colab Sessions & Usage ==="
    "$COLAB_CLI" sessions || true
    "$COLAB_CLI" usage || true

    if [ "$USE_SSH" = "1" ]; then
      colab_ssh_run "$SESSION" "
        echo '=== GPU & System ==='
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
        free -g
        df -BG /content
        ps -ef | grep -E 'fgpo\.runner|fgpo_gcs_monitor\.py' | grep -v grep || true
      "
      exit 0
    fi

    GCS_CELL_PREFIX="${FGPO_GCS_PREFIX}/cells/${CELL_ID}"
    TMP_DIR="/tmp/fgpo_status_$$"
    mkdir -p "$TMP_DIR"
    trap 'rm -rf "$TMP_DIR"' EXIT

    echo "=== GCS Status (${CELL_ID}) ==="
    gcs_py stat "${GCS_CELL_PREFIX}/live_status.json" || echo "(no live_status.json)"
    if gcs_py download "${GCS_CELL_PREFIX}/live_status.json" "$TMP_DIR/live_status.json" 2>/dev/null; then
      cat "$TMP_DIR/live_status.json"
    fi
    if gcs_py download "${GCS_CELL_PREFIX}/training.jsonl" "$TMP_DIR/training.jsonl" 2>/dev/null; then
      echo "=== Latest training.jsonl ==="
      tail -n 3 "$TMP_DIR/training.jsonl"
    fi
    if gcs_py download "${GCS_CELL_PREFIX}/runner.log" "$TMP_DIR/runner.log" 2>/dev/null; then
      echo "=== Runner log tail ==="
      tail -n 10 "$TMP_DIR/runner.log"
    fi
    ;;

  pull)
    [ $# -ge 1 ] || { echo "Usage: $0 pull <SUITE/BM/METHOD/seed-N> [dest_dir] [--with-final-model]" >&2; exit 2; }
    CELL_ID="$1"
    shift
    DEST_ROOT="$REPO_ROOT/outputs/colab_synced"
    WITH_MODEL="0"
    while [ $# -gt 0 ]; do
      case "$1" in
        --with-final-model) WITH_MODEL="1"; shift ;;
        --bucket) FGPO_GCS_BUCKET="$2"; shift 2 ;;
        --prefix) FGPO_GCS_PREFIX="$2"; shift 2 ;;
        -*) echo "Unknown flag: $1" >&2; exit 2 ;;
        *) DEST_ROOT="$1"; shift ;;
      esac
    done
    GCS_CELL_PREFIX="${FGPO_GCS_PREFIX}/cells/${CELL_ID}"
    DEST_DIR="${DEST_ROOT}/${CELL_ID}"
    mkdir -p "$DEST_DIR"
    PULLED=0
    while IFS=$'\t' read -r obj size updated; do
      rel="${obj#${GCS_CELL_PREFIX}/}"
      [ -n "$rel" ] || continue
      if [ "$WITH_MODEL" != "1" ] && [[ "$rel" == final_checkpoint/* ]]; then
        continue
      fi
      if [[ "$rel" == checkpoint/* ]]; then
        continue
      fi
      gcs_py download "$obj" "${DEST_DIR}/${rel}"
      echo "[pull] ${rel} (${size} bytes)"
      PULLED=$((PULLED + 1))
    done < <(gcs_py list "${GCS_CELL_PREFIX}/")
    if [ "$PULLED" -eq 0 ]; then
      echo "[pull] Warning: no objects found under gs://${FGPO_GCS_BUCKET}/${GCS_CELL_PREFIX}/" >&2
      exit 1
    fi
    echo "[pull] Synced ${PULLED} files into ${DEST_DIR}"
    ;;

  stop)
    [ $# -ge 1 ] || { echo "Usage: $0 stop <session> [--cell SUITE/BM/METHOD/seed-N] [--backup-checkpoint]" >&2; exit 2; }
    SESSION="$1"
    shift
    CELL_ID=""
    BACKUP_CKPT="0"
    while [ $# -gt 0 ]; do
      case "$1" in
        --cell) CELL_ID="$2"; shift 2 ;;
        --backup-checkpoint) BACKUP_CKPT="1"; shift ;;
        --bucket) FGPO_GCS_BUCKET="$2"; shift 2 ;;
        --prefix) FGPO_GCS_PREFIX="$2"; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
      esac
    done
    echo "[stop] Sending SIGTERM to fgpo.runner on '$SESSION'..."
    colab_ssh_run "$SESSION" "
      pkill -TERM -f 'fgpo\.runner' 2>/dev/null || true
      for i in \$(seq 1 60); do
        pgrep -f 'fgpo\.runner' >/dev/null 2>&1 || break
        sleep 2
      done
    " || true
    if [ "$BACKUP_CKPT" = "1" ]; then
      [ -n "$CELL_ID" ] || { echo "--backup-checkpoint requires --cell" >&2; exit 2; }
      colab_ssh_run "$SESSION" "BUCKET='$FGPO_GCS_BUCKET' CELL_DIR='/content/outputs/fgpo-r3/${CELL_ID}' GCS_PREFIX='${FGPO_GCS_PREFIX}/cells/${CELL_ID}' bash -s" << 'BKP_EOF'
set -euo pipefail
/content/venvs/fgpo-main/bin/python - "$BUCKET" "$CELL_DIR" "$GCS_PREFIX" << 'PY'
import json, os, sys, requests
bucket, cell_dir, prefix = sys.argv[1:4]
cur_path = os.path.join(cell_dir, "checkpoint", "CURRENT")
if not os.path.isfile(cur_path):
    raise SystemExit("No checkpoint/CURRENT found to back up.")
snap_id = json.load(open(cur_path))["snapshot"]
snap_dir = os.path.join(cell_dir, "checkpoint", "snapshots", snap_id)
creds = json.load(open("/root/.gcs_adc.json"))
token = requests.post("https://oauth2.googleapis.com/token", data={
    "client_id": creds["client_id"], "client_secret": creds["client_secret"],
    "refresh_token": creds["refresh_token"], "grant_type": "refresh_token",
}, timeout=20).json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}
def up(local_p, rel_p):
    with open(local_p, "rb") as f:
        r = requests.post(
            f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o",
            params={"uploadType": "media", "name": f"{prefix}/{rel_p}"},
            headers={**headers, "Content-Type": "application/octet-stream"},
            data=f, timeout=1800,
        )
    r.raise_for_status()
for rel in ("config.json", "official_tasks.json", "costs.jsonl", "training_summary.json"):
    p = os.path.join(cell_dir, rel)
    if os.path.isfile(p):
        up(p, rel)
for root, _, files in os.walk(snap_dir):
    for fn in sorted(files):
        full = os.path.join(root, fn)
        up(full, os.path.relpath(full, cell_dir).replace(os.sep, "/"))
up(cur_path, "checkpoint/CURRENT")
print(f"[stop] Atomically backed up snapshot {snap_id} to gs://{bucket}/{prefix}/")
PY
BKP_EOF
    fi
    "$COLAB_CLI" stop -s "$SESSION"
    "$COLAB_CLI" sessions || true
    ;;

  *)
    echo "Unknown command: $CMD (expected sync-code|status|pull|stop)" >&2
    exit 2
    ;;
esac
