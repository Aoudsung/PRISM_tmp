#!/usr/bin/env bash
# Unified day-to-day Colab operations for F-GPO r3 (published-v4 manifest & incremental GCS layout):
#   sync-code [session] [--force] [--bucket B] [--prefix P] [--transport ssh|exec]
#   refresh-token -s SESSION [--remote-token-file PATH] [--transport ssh|exec]
#   status -s SESSION [--cell SUITE/BM/METHOD/seed-N --config REMOTE_CFG] [--output-dir REMOTE_OUT] [--remote-token-file PATH] [--ssh] [--transport ssh|exec]
#   pull -s SESSION <SUITE/BM/METHOD/seed-N> [dest_dir] --config REMOTE_CFG [--output-dir REMOTE_OUT] [--remote-token-file PATH] [--with-final-model] [--transport ssh|exec]
#   aggregate -s SESSION --config REMOTE_CFG [--output-dir REMOTE_OUT] [--suite comparison|ablation] [--method fgpo] [--benchmark alfworld] [--seeds 0,1,2] [--records-format json|jsonl] [--dest-dir LOCAL_DEST] [--transport ssh|exec]
#   stop <session> --cell SUITE/BM/METHOD/seed-N [--cell ...] --config REMOTE_CFG --dest-dir LOCAL_DEST [--output-dir REMOTE_OUT] [--remote-token-file PATH] [--wait-seconds N] [--with-final-model] [--transport ssh|exec]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/colab_common.sh"

DEFAULT_REMOTE_OUTPUT="/content/outputs/fgpo-r3-published-v4-alf"
DEFAULT_REMOTE_TOKEN_FILE="/run/fgpo-gcs/access-token"
REMOTE_PY="/content/venvs/fgpo-main/bin/python"
REMOTE_GCS_CLI="/content/fgpo-r3/scripts/fgpo_gcs_checkpoint.py"

parse_cell_id() {
  local cell_id="$1"
  IFS=/ read -r CELL_SUITE CELL_BENCHMARK CELL_METHOD CELL_SEED_DIR <<< "$cell_id"
  if [ -z "${CELL_SUITE:-}" ] || [ -z "${CELL_BENCHMARK:-}" ] || [ -z "${CELL_METHOD:-}" ] || [ -z "${CELL_SEED_DIR:-}" ]; then
    echo "Invalid cell ID '$cell_id' (expected SUITE/BENCHMARK/METHOD/seed-N)" >&2
    exit 2
  fi
  case "$CELL_SUITE" in comparison|ablation) ;; *) echo "Invalid suite '$CELL_SUITE' in '$cell_id'" >&2; exit 2 ;; esac
  case "$CELL_BENCHMARK" in alfworld|appworld) ;; *) echo "Invalid benchmark '$CELL_BENCHMARK' in '$cell_id'" >&2; exit 2 ;; esac
  case "$CELL_METHOD" in fgpo|fgpo_obs_only|fgpo_no_transfer|fgpo_no_support_weight|gigpo|hgpo|graphgpo) ;; *) echo "Invalid method '$CELL_METHOD' in '$cell_id'" >&2; exit 2 ;; esac
  case "$CELL_SEED_DIR" in
    seed-0|seed-1|seed-2) CELL_SEED="${CELL_SEED_DIR#seed-}" ;;
    *) echo "Invalid seed directory '$CELL_SEED_DIR' in cell ID '$cell_id'" >&2; exit 2 ;;
  esac
}

validate_safe_path() {
  local label="$1"
  local value="$2"
  case "$value" in
    /*) ;;
    *) echo "$label must be an absolute path: $value" >&2; exit 2 ;;
  esac
  case "$value" in
    *[!a-zA-Z0-9_./-]*) echo "Unsupported characters in $label: $value" >&2; exit 2 ;;
  esac
}

REMOTE_PROC_SCANNER_B64="$("$COLAB_PY" -c 'import base64, sys; print(base64.b64encode(sys.stdin.read().encode("utf-8")).decode("ascii"))' <<'SCANNER_B64_EOF'
import os, pathlib, signal, sys

action = sys.argv[1]
kinds = set(sys.argv[2].split(",")) if len(sys.argv) > 2 and sys.argv[2] else set()

def ancestor_pids():
    anc = set()
    cur = os.getpid()
    while cur > 1 and cur not in anc:
        anc.add(cur)
        try:
            st = pathlib.Path(f"/proc/{cur}/stat").read_text()
            cur = int(st.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    anc.add(1)
    return anc

def classify_argv(argv):
    if not argv:
        return None
    exe = os.path.basename(argv[0])
    if exe == ".fgpo-colab-supervisor.sh":
        return "supervisor"
    if exe.startswith("python"):
        i = 1
        while i < len(argv):
            tok = argv[i]
            if tok == "-c":
                return None
            if tok == "-m":
                if i + 1 < len(argv) and argv[i + 1] == "fgpo.runner":
                    return "runner"
                return None
            if tok in ("-W", "-X"):
                i += 2
                continue
            if tok.startswith("-"):
                i += 1
                continue
            if os.path.basename(tok) == "fgpo_gcs_checkpoint.py":
                return "checkpoint"
            return None
    if exe in ("bash", "sh"):
        i = 1
        while i < len(argv):
            tok = argv[i]
            if tok in ("-c", "-s"):
                return None
            if tok == "-o":
                i += 2
                continue
            if tok == "--":
                i += 1
                if i < len(argv) and os.path.basename(argv[i]) == ".fgpo-colab-supervisor.sh":
                    return "supervisor"
                return None
            if tok.startswith("-"):
                i += 1
                continue
            if os.path.basename(tok) == ".fgpo-colab-supervisor.sh":
                return "supervisor"
            return None
    return None

anc = ancestor_pids()
matches = []
for entry in sorted(pathlib.Path("/proc").iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 10**9):
    if not entry.name.isdigit():
        continue
    pid = int(entry.name)
    if pid in anc:
        continue
    try:
        st = (entry / "stat").read_text()
        fields = st.rsplit(")", 1)[1].split()
        if fields[0] in ("Z", "X"):
            continue
        ppid = int(fields[1])
        raw = (entry / "cmdline").read_bytes()
    except OSError:
        continue
    argv = [x.decode("utf-8", "replace") for x in raw.rstrip(b"\x00").split(b"\x00") if x]
    kind = classify_argv(argv)
    if kind in kinds:
        matches.append((pid, ppid, kind, " ".join(argv)))

if action == "any":
    raise SystemExit(0 if matches else 1)
elif action == "list":
    for pid, ppid, kind, cmd in matches:
        print(f"{pid}\t{ppid}\t{kind}\t{cmd}")
elif action == "term":
    for pid, _, _, _ in matches:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
SCANNER_B64_EOF
)"

CMD="${1:-}"
[ -n "$CMD" ] || { echo "Usage: $0 {sync-code|refresh-token|status|pull|aggregate|stop} [options]" >&2; exit 2; }
shift

case "$CMD" in
  sync-code)
    SESSION=""
    FORCE="0"
    TRANSPORT="ssh"
    while [ $# -gt 0 ]; do
      case "$1" in
        --force) FORCE="1"; shift ;;
        --bucket) FGPO_GCS_BUCKET="$2"; shift 2 ;;
        --prefix) FGPO_GCS_PREFIX="$2"; shift 2 ;;
        --transport) TRANSPORT="$2"; shift 2 ;;
        -s|--session) SESSION="$2"; shift 2 ;;
        -*) echo "Unknown flag: $1" >&2; exit 2 ;;
        *) SESSION="$1"; shift ;;
      esac
    done
    colab_validate_transport "$TRANSPORT"
    SRC_TAR="/tmp/fgpo_r3_src.tar.gz"
    pack_fgpo_source "$SRC_TAR"
    gcs_py upload "$SRC_TAR" "${FGPO_GCS_PREFIX}/code/fgpo_r3_src.tar.gz" "application/gzip"
    if [ -n "$SESSION" ]; then
      if [ "$FORCE" != "1" ] && colab_remote_run "$TRANSPORT" "$SESSION" "SCANNER_B64='$REMOTE_PROC_SCANNER_B64' python3 -c 'import base64, os, sys; sys.argv = [\"proc_scanner\", *sys.argv[1:]]; exec(compile(base64.b64decode(os.environ[\"SCANNER_B64\"]), \"proc_scanner\", \"exec\"))' any 'runner,supervisor'"; then
        echo "[sync-code] Refused: fgpo.runner or supervisor is active in '$SESSION'. Stop training cleanly first or pass --force." >&2
        exit 1
      fi
      cat "$SRC_TAR" | colab_remote_run "$TRANSPORT" "$SESSION" \
        "rm -rf /content/fgpo-r3/src/fgpo && mkdir -p /content/fgpo-r3 && tar --warning=no-unknown-keyword -xzf - -C /content/fgpo-r3"
      echo "[sync-code] Replaced source in '$SESSION:/content/fgpo-r3' via $TRANSPORT"
    fi
    ;;

  refresh-token)
    SESSION=""
    REMOTE_TOKEN_FILE="$DEFAULT_REMOTE_TOKEN_FILE"
    TRANSPORT="ssh"
    while [ $# -gt 0 ]; do
      case "$1" in
        -s|--session) SESSION="$2"; shift 2 ;;
        --remote-token-file) REMOTE_TOKEN_FILE="$2"; shift 2 ;;
        --transport) TRANSPORT="$2"; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
      esac
    done
    [ -n "$SESSION" ] || { echo "[refresh-token] --session SESSION is required." >&2; exit 2; }
    colab_validate_transport "$TRANSPORT"
    validate_safe_path "--remote-token-file" "$REMOTE_TOKEN_FILE"
    colab_push_access_token "$TRANSPORT" "$SESSION" "$REMOTE_TOKEN_FILE"
    ;;

  status)
    CELL_ID=""
    CONFIG_PATH=""
    SESSION=""
    REMOTE_OUTPUT="$DEFAULT_REMOTE_OUTPUT"
    REMOTE_TOKEN_FILE="$DEFAULT_REMOTE_TOKEN_FILE"
    USE_PROBE="0"
    TRANSPORT="ssh"
    while [ $# -gt 0 ]; do
      case "$1" in
        --cell) CELL_ID="$2"; shift 2 ;;
        --config) CONFIG_PATH="$2"; shift 2 ;;
        -s|--session) SESSION="$2"; shift 2 ;;
        --output-dir) REMOTE_OUTPUT="$2"; shift 2 ;;
        --remote-token-file) REMOTE_TOKEN_FILE="$2"; shift 2 ;;
        --ssh|--probe) USE_PROBE="1"; shift ;;
        --transport) TRANSPORT="$2"; shift 2 ;;
        --bucket|--prefix)
          echo "[status] Unsupported override '$1': study GCS location is resolved strictly from --config." >&2
          exit 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
      esac
    done
    [ -n "$SESSION" ] || { echo "[status] --session SESSION is required to query the designated runtime." >&2; exit 2; }
    colab_validate_transport "$TRANSPORT"
    validate_safe_path "--output-dir" "$REMOTE_OUTPUT"
    validate_safe_path "--remote-token-file" "$REMOTE_TOKEN_FILE"

    echo "=== Colab Sessions & Usage ==="
    "$COLAB_CLI" sessions || true
    "$COLAB_CLI" usage || true

    if [ "$USE_PROBE" = "1" ]; then
      colab_remote_run "$TRANSPORT" "$SESSION" "REMOTE_OUTPUT='$REMOTE_OUTPUT' SCANNER_B64='$REMOTE_PROC_SCANNER_B64' bash -s" << 'PROBE_EOF'
        echo '=== GPU & System ==='
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
        free -g
        df -BG /content
        python3 -c 'import base64, os, sys; sys.argv = ["proc_scanner", *sys.argv[1:]]; exec(compile(base64.b64decode(os.environ["SCANNER_B64"]), "proc_scanner", "exec"))' list 'runner,checkpoint,supervisor' || true
        for f in "$REMOTE_OUTPUT/.fgpo-colab-supervisor.pid" "$REMOTE_OUTPUT/.fgpo-colab-supervisor.ready" "$REMOTE_OUTPUT/.fgpo-stop-requested"; do
          [ -e "$f" ] && echo "MARKER $f: $(head -n 1 "$f" 2>/dev/null || echo present)"
        done || true
PROBE_EOF
      if [ -z "$CELL_ID" ] && [ -z "$CONFIG_PATH" ]; then
        exit 0
      fi
    fi

    [ -n "$CELL_ID" ] || { echo "[status] --cell SUITE/BM/METHOD/seed-N is required." >&2; exit 2; }
    [ -n "$CONFIG_PATH" ] || { echo "[status] --config REMOTE_CONFIG_PATH is required." >&2; exit 2; }
    parse_cell_id "$CELL_ID"
    validate_safe_path "--config" "$CONFIG_PATH"
    validate_safe_path "--output-dir" "$REMOTE_OUTPUT"
    validate_safe_path "--remote-token-file" "$REMOTE_TOKEN_FILE"

    echo "=== Remote GCS Manifest Status (${CELL_ID} via ${SESSION}, transport=${TRANSPORT}) ==="
    colab_remote_run "$TRANSPORT" "$SESSION" \
      "PY='$REMOTE_PY' CLI='$REMOTE_GCS_CLI' CONFIG_PATH='$CONFIG_PATH' CELL_DIR='${REMOTE_OUTPUT}/${CELL_ID}' CELL_SUITE='$CELL_SUITE' CELL_BENCHMARK='$CELL_BENCHMARK' CELL_METHOD='$CELL_METHOD' CELL_SEED='$CELL_SEED' REMOTE_TOKEN_FILE='$REMOTE_TOKEN_FILE' bash -s" << 'STATUS_EOF'
set -euo pipefail
export PYTHONPATH=/content/fgpo-r3/src
export FGPO_GCS_ACCESS_TOKEN_FILE="$REMOTE_TOKEN_FILE"
[ -f "$CONFIG_PATH" ] || { echo "Missing remote config: $CONFIG_PATH" >&2; exit 1; }
TMP_STAGE="$(mktemp -d /tmp/fgpo_status_XXXXXX)"
trap 'rm -rf "$TMP_STAGE"' EXIT
STATUS_JSON="$("$PY" "$CLI" status \
  --config "$CONFIG_PATH" --cell-dir "$CELL_DIR" \
  --suite "$CELL_SUITE" --method "$CELL_METHOD" \
  --benchmark "$CELL_BENCHMARK" --seed "$CELL_SEED")"
printf '%s\n' "$STATUS_JSON"
REMOTE_STATUS="$("$PY" -c 'import json,sys; print(json.loads(sys.argv[1])["status"])' "$STATUS_JSON")"
if [ "$REMOTE_STATUS" != "absent" ]; then
  "$PY" "$CLI" pull \
    --config "$CONFIG_PATH" --cell-dir "$CELL_DIR" \
    --suite "$CELL_SUITE" --method "$CELL_METHOD" \
    --benchmark "$CELL_BENCHMARK" --seed "$CELL_SEED" \
    --dest-dir "$TMP_STAGE" >/dev/null
  if [ -f "$TMP_STAGE/training.jsonl" ]; then
    echo "=== Latest training.jsonl ==="
    tail -n 3 "$TMP_STAGE/training.jsonl"
  fi
  if [ -f "$TMP_STAGE/costs.jsonl" ]; then
    echo "=== Latest costs.jsonl ==="
    tail -n 5 "$TMP_STAGE/costs.jsonl"
  fi
fi
STATUS_EOF
    ;;

  pull)
    [ $# -ge 1 ] || { echo "Usage: $0 pull -s SESSION <SUITE/BM/METHOD/seed-N> [dest_dir] --config REMOTE_CFG [--output-dir REMOTE_OUT] [--remote-token-file PATH] [--with-final-model] [--transport ssh|exec]" >&2; exit 2; }
    SESSION=""
    CELL_ID=""
    DEST_ROOT="$REPO_ROOT/outputs/colab_synced"
    DEST_SET="0"
    CONFIG_PATH=""
    REMOTE_OUTPUT="$DEFAULT_REMOTE_OUTPUT"
    REMOTE_TOKEN_FILE="$DEFAULT_REMOTE_TOKEN_FILE"
    WITH_MODEL="0"
    TRANSPORT="ssh"
    while [ $# -gt 0 ]; do
      case "$1" in
        -s|--session) SESSION="$2"; shift 2 ;;
        --cell) CELL_ID="$2"; shift 2 ;;
        --dest-dir) DEST_ROOT="$2"; DEST_SET="1"; shift 2 ;;
        --with-final-model) WITH_MODEL="1"; shift ;;
        --config) CONFIG_PATH="$2"; shift 2 ;;
        --output-dir) REMOTE_OUTPUT="$2"; shift 2 ;;
        --remote-token-file) REMOTE_TOKEN_FILE="$2"; shift 2 ;;
        --transport) TRANSPORT="$2"; shift 2 ;;
        --bucket|--prefix)
          echo "[pull] Unsupported override '$1': study GCS location is resolved strictly from --config." >&2
          exit 2 ;;
        -*) echo "Unknown flag: $1" >&2; exit 2 ;;
        *)
          if [ -z "$CELL_ID" ]; then
            CELL_ID="$1"
          elif [ "$DEST_SET" = "0" ]; then
            DEST_ROOT="$1"; DEST_SET="1"
          else
            echo "Unexpected positional argument: $1" >&2; exit 2
          fi
          shift ;;
      esac
    done
    [ -n "$SESSION" ] || { echo "[pull] --session SESSION is required to reconstruct via the designated runtime." >&2; exit 2; }
    [ -n "$CELL_ID" ] || { echo "[pull] Cell ID SUITE/BM/METHOD/seed-N is required." >&2; exit 2; }
    [ -n "$CONFIG_PATH" ] || { echo "[pull] --config REMOTE_CONFIG_PATH is required." >&2; exit 2; }
    colab_validate_transport "$TRANSPORT"
    if [ "$TRANSPORT" = "exec" ] && [ "$WITH_MODEL" = "1" ]; then
      echo "[pull] --with-final-model exceeds Jupyter exec streaming limits; use --transport ssh." >&2
      exit 2
    fi
    parse_cell_id "$CELL_ID"
    validate_safe_path "--config" "$CONFIG_PATH"
    validate_safe_path "--output-dir" "$REMOTE_OUTPUT"
    validate_safe_path "--remote-token-file" "$REMOTE_TOKEN_FILE"

    DEST_DIR="${DEST_ROOT}/${CELL_ID}"
    mkdir -p "$DEST_ROOT"
    colab_remote_run "$TRANSPORT" "$SESSION" \
      "PY='$REMOTE_PY' CLI='$REMOTE_GCS_CLI' CONFIG_PATH='$CONFIG_PATH' REMOTE_OUTPUT='$REMOTE_OUTPUT' CELL_ID='$CELL_ID' CELL_DIR='${REMOTE_OUTPUT}/${CELL_ID}' CELL_SUITE='$CELL_SUITE' CELL_BENCHMARK='$CELL_BENCHMARK' CELL_METHOD='$CELL_METHOD' CELL_SEED='$CELL_SEED' REMOTE_TOKEN_FILE='$REMOTE_TOKEN_FILE' WITH_MODEL='$WITH_MODEL' bash -s" << 'PULL_EOF' | tar -xzf - -C "$DEST_ROOT"
set -euo pipefail
export PYTHONPATH=/content/fgpo-r3/src
export FGPO_GCS_ACCESS_TOKEN_FILE="$REMOTE_TOKEN_FILE"
[ -f "$CONFIG_PATH" ] || { echo "Missing remote config: $CONFIG_PATH" >&2; exit 1; }
STAGE_ROOT="$(mktemp -d /tmp/fgpo_pull_XXXXXX)"
trap 'rm -rf "$STAGE_ROOT"' EXIT
CELL_STAGE="$STAGE_ROOT/$CELL_ID"
SUITE_STAGE="$STAGE_ROOT/$CELL_SUITE"
mkdir -p "$CELL_STAGE" "$SUITE_STAGE"
EXTRA=()
if [ "$WITH_MODEL" = "1" ]; then
  EXTRA+=(--with-final-model)
fi
"$PY" "$CLI" pull \
  --config "$CONFIG_PATH" --cell-dir "$CELL_DIR" \
  --suite "$CELL_SUITE" --method "$CELL_METHOD" \
  --benchmark "$CELL_BENCHMARK" --seed "$CELL_SEED" \
  --dest-dir "$CELL_STAGE" ${EXTRA[@]+"${EXTRA[@]}"} > "$CELL_STAGE/.pull_receipt.json"
if [ -f "$CELL_DIR/reporting_summary.json" ]; then
  cp -f "$CELL_DIR/reporting_summary.json" "$CELL_STAGE/reporting_summary.json"
fi
for report_name in study_reporting_summary.json study_reporting_table1.md; do
  if [ -f "$REMOTE_OUTPUT/$CELL_SUITE/$report_name" ]; then
    cp -f "$REMOTE_OUTPUT/$CELL_SUITE/$report_name" "$SUITE_STAGE/$report_name"
  fi
done
tar -czf - -C "$STAGE_ROOT" .
PULL_EOF
    if [ ! -f "$DEST_DIR/.pull_receipt.json" ]; then
      echo "[pull] Missing pull receipt after transfer into $DEST_DIR" >&2
      exit 1
    fi
    cat "$DEST_DIR/.pull_receipt.json"
    printf '\n'
    ;;

  aggregate)
    SESSION=""
    CONFIG_PATH=""
    REMOTE_OUTPUT="$DEFAULT_REMOTE_OUTPUT"
    AGG_SUITE="comparison"
    AGG_METHOD="fgpo"
    AGG_BENCHMARK="alfworld"
    AGG_SEEDS="0,1,2"
    RECORDS_FORMAT="json"
    DEST_ROOT=""
    TRANSPORT="ssh"
    while [ $# -gt 0 ]; do
      case "$1" in
        -s|--session) SESSION="$2"; shift 2 ;;
        --config) CONFIG_PATH="$2"; shift 2 ;;
        --output-dir) REMOTE_OUTPUT="$2"; shift 2 ;;
        --suite) AGG_SUITE="$2"; shift 2 ;;
        --method) AGG_METHOD="$2"; shift 2 ;;
        --benchmark) AGG_BENCHMARK="$2"; shift 2 ;;
        --seeds) AGG_SEEDS="$2"; shift 2 ;;
        --records-format) RECORDS_FORMAT="$2"; shift 2 ;;
        --dest-dir) DEST_ROOT="$2"; shift 2 ;;
        --transport) TRANSPORT="$2"; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
      esac
    done
    [ -n "$SESSION" ] || { echo "[aggregate] --session SESSION is required." >&2; exit 2; }
    [ -n "$CONFIG_PATH" ] || { echo "[aggregate] --config REMOTE_CONFIG_PATH is required." >&2; exit 2; }
    case "$AGG_SUITE" in comparison|ablation) ;; *) echo "Invalid --suite: $AGG_SUITE" >&2; exit 2 ;; esac
    case "$AGG_BENCHMARK" in alfworld|appworld) ;; *) echo "Invalid --benchmark: $AGG_BENCHMARK" >&2; exit 2 ;; esac
    case "$AGG_METHOD" in fgpo|fgpo_obs_only|fgpo_no_transfer|fgpo_no_support_weight|gigpo|hgpo|graphgpo) ;; *) echo "Invalid --method: $AGG_METHOD" >&2; exit 2 ;; esac
    case "$RECORDS_FORMAT" in json|jsonl) ;; *) echo "Invalid --records-format: $RECORDS_FORMAT" >&2; exit 2 ;; esac
    IFS=, read -r -a _AGG_SEED_ARR <<< "$AGG_SEEDS"
    [ "${#_AGG_SEED_ARR[@]}" -ge 1 ] || { echo "[aggregate] At least one seed is required." >&2; exit 2; }
    for _sd in "${_AGG_SEED_ARR[@]}"; do
      case "$_sd" in 0|1|2) ;; *) echo "Invalid seed: $_sd" >&2; exit 2 ;; esac
    done
    colab_validate_transport "$TRANSPORT"
    validate_safe_path "--config" "$CONFIG_PATH"
    validate_safe_path "--output-dir" "$REMOTE_OUTPUT"

    colab_remote_run "$TRANSPORT" "$SESSION" \
      "PY='$REMOTE_PY' CONFIG_PATH='$CONFIG_PATH' REMOTE_OUTPUT='$REMOTE_OUTPUT' AGG_SUITE='$AGG_SUITE' AGG_METHOD='$AGG_METHOD' AGG_BENCHMARK='$AGG_BENCHMARK' AGG_SEEDS='$AGG_SEEDS' RECORDS_FORMAT='$RECORDS_FORMAT' bash -s" << 'AGG_EOF'
set -euo pipefail
export PYTHONPATH=/content/fgpo-r3/src
[ -f "$CONFIG_PATH" ] || { echo "Missing remote config: $CONFIG_PATH" >&2; exit 1; }
IFS=, read -r -a seed_list <<< "$AGG_SEEDS"
"$PY" -m fgpo.runner \
  --config "$CONFIG_PATH" \
  --output-dir "$REMOTE_OUTPUT" \
  --suite "$AGG_SUITE" \
  --methods "$AGG_METHOD" \
  --benchmarks "$AGG_BENCHMARK" \
  --seeds "${seed_list[@]}" \
  --aggregate-only \
  --evaluation-records-format "$RECORDS_FORMAT"
AGG_EOF
    if [ -n "$DEST_ROOT" ]; then
      mkdir -p "$DEST_ROOT"
      colab_remote_run "$TRANSPORT" "$SESSION" \
        "REMOTE_OUTPUT='$REMOTE_OUTPUT' AGG_SUITE='$AGG_SUITE' AGG_METHOD='$AGG_METHOD' AGG_BENCHMARK='$AGG_BENCHMARK' AGG_SEEDS='$AGG_SEEDS' bash -s" << 'AGG_PULL_EOF' | tar -xzf - -C "$DEST_ROOT"
set -euo pipefail
STAGE_ROOT="$(mktemp -d /tmp/fgpo_agg_pull_XXXXXX)"
trap 'rm -rf "$STAGE_ROOT"' EXIT
mkdir -p "$STAGE_ROOT/$AGG_SUITE"
for report_name in study_reporting_summary.json study_reporting_table1.md; do
  if [ -f "$REMOTE_OUTPUT/$AGG_SUITE/$report_name" ]; then
    cp -f "$REMOTE_OUTPUT/$AGG_SUITE/$report_name" "$STAGE_ROOT/$AGG_SUITE/$report_name"
  fi
done
IFS=, read -r -a seed_list <<< "$AGG_SEEDS"
for sd in "${seed_list[@]}"; do
  rel="$AGG_SUITE/$AGG_BENCHMARK/$AGG_METHOD/seed-$sd"
  if [ -f "$REMOTE_OUTPUT/$rel/reporting_summary.json" ]; then
    mkdir -p "$STAGE_ROOT/$rel"
    cp -f "$REMOTE_OUTPUT/$rel/reporting_summary.json" "$STAGE_ROOT/$rel/reporting_summary.json"
  fi
done
tar -czf - -C "$STAGE_ROOT" .
AGG_PULL_EOF
      echo "[aggregate] Saved derived Table 1 reporting artifacts under '$DEST_ROOT/$AGG_SUITE/'"
    fi
    ;;

  stop)
    [ $# -ge 1 ] || { echo "Usage: $0 stop <session> --cell SUITE/BM/METHOD/seed-N [--cell ...] --config REMOTE_CFG --dest-dir LOCAL_DEST [--output-dir REMOTE_OUT] [--remote-token-file PATH] [--wait-seconds N] [--with-final-model] [--transport ssh|exec]" >&2; exit 2; }
    SESSION=""
    case "${1:-}" in
      -*) ;;
      *) SESSION="$1"; shift ;;
    esac
    CELL_IDS=()
    REMOTE_OUTPUT="$DEFAULT_REMOTE_OUTPUT"
    REMOTE_CONFIG=""
    DEST_ROOT=""
    REMOTE_TOKEN_FILE="$DEFAULT_REMOTE_TOKEN_FILE"
    WAIT_SECONDS="0"
    WITH_MODEL="0"
    TRANSPORT="ssh"
    while [ $# -gt 0 ]; do
      case "$1" in
        -s|--session) SESSION="$2"; shift 2 ;;
        --cell) CELL_IDS+=("$2"); shift 2 ;;
        --cells) IFS=, read -r -a _MORE <<< "$2"; CELL_IDS+=("${_MORE[@]}"); shift 2 ;;
        --output-dir) REMOTE_OUTPUT="$2"; shift 2 ;;
        --config) REMOTE_CONFIG="$2"; shift 2 ;;
        --dest-dir) DEST_ROOT="$2"; shift 2 ;;
        --remote-token-file) REMOTE_TOKEN_FILE="$2"; shift 2 ;;
        --wait-seconds) WAIT_SECONDS="$2"; shift 2 ;;
        --with-final-model) WITH_MODEL="1"; shift ;;
        --transport) TRANSPORT="$2"; shift 2 ;;
        --bucket|--prefix|--backup-checkpoint)
          echo "[stop] Unsupported legacy option '$1': use --cell, --config, and --dest-dir for verified off-instance retention." >&2
          exit 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
      esac
    done
    [ -n "$SESSION" ] || { echo "[stop] SESSION is required (pass positional <session> or -s SESSION)." >&2; exit 2; }
    [ "${#CELL_IDS[@]}" -ge 1 ] || { echo "[stop] At least one --cell SUITE/BM/METHOD/seed-N is required." >&2; exit 2; }
    [ -n "$REMOTE_CONFIG" ] || { echo "[stop] --config REMOTE_CONFIG_PATH is required." >&2; exit 2; }
    [ -n "$DEST_ROOT" ] || { echo "[stop] --dest-dir LOCAL_DEST is required to retain recovery state off-instance before VM shutdown." >&2; exit 2; }
    case "$WAIT_SECONDS" in
      ''|*[!0-9]*) echo "--wait-seconds must be a nonnegative integer (0 waits until complete exit)" >&2; exit 2 ;;
    esac
    colab_validate_transport "$TRANSPORT"
    if [ "$TRANSPORT" = "exec" ] && [ "$WITH_MODEL" = "1" ]; then
      echo "[stop] --with-final-model exceeds Jupyter exec streaming limits; use --transport ssh." >&2
      exit 2
    fi
    validate_safe_path "--config" "$REMOTE_CONFIG"
    validate_safe_path "--output-dir" "$REMOTE_OUTPUT"
    validate_safe_path "--remote-token-file" "$REMOTE_TOKEN_FILE"
    for cid in "${CELL_IDS[@]}"; do
      parse_cell_id "$cid"
    done
    CELL_CSV="$(IFS=,; echo "${CELL_IDS[*]}")"

    STOP_EXEC_TIMEOUT="${COLAB_EXEC_TIMEOUT:-1800}"
    if [ "$WAIT_SECONDS" -gt 0 ] && [ "$((WAIT_SECONDS + 300))" -gt "$STOP_EXEC_TIMEOUT" ]; then
      STOP_EXEC_TIMEOUT="$((WAIT_SECONDS + 300))"
    fi

    echo "[stop] Requesting graceful shutdown on '$SESSION' via $TRANSPORT and verifying cell persistence..."
    BACKEND="$(COLAB_EXEC_TIMEOUT="$STOP_EXEC_TIMEOUT" colab_remote_run "$TRANSPORT" "$SESSION" \
      "PY='$REMOTE_PY' CLI='$REMOTE_GCS_CLI' WAIT_SECONDS='$WAIT_SECONDS' CELL_CSV='$CELL_CSV' REMOTE_OUTPUT='$REMOTE_OUTPUT' REMOTE_CONFIG='$REMOTE_CONFIG' REMOTE_TOKEN_FILE='$REMOTE_TOKEN_FILE' SCANNER_B64='$REMOTE_PROC_SCANNER_B64' bash -s" << 'STOP_VERIFY_EOF'
set -euo pipefail
export PYTHONPATH=/content/fgpo-r3/src
export FGPO_GCS_ACCESS_TOKEN_FILE="$REMOTE_TOKEN_FILE"

[ -f "$REMOTE_CONFIG" ] || { echo "[stop] Missing remote config: $REMOTE_CONFIG" >&2; exit 1; }
BACKEND=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("checkpoint_backend","local"))' "$REMOTE_CONFIG")

if [ -d "$REMOTE_OUTPUT" ]; then
  touch "${REMOTE_OUTPUT}/.fgpo-stop-requested"
fi

"$PY" -c 'import base64, os, sys; sys.argv = ["proc_scanner", *sys.argv[1:]]; exec(compile(base64.b64decode(os.environ["SCANNER_B64"]), "proc_scanner", "exec"))' term runner

started=$(date +%s)
while "$PY" -c 'import base64, os, sys; sys.argv = ["proc_scanner", *sys.argv[1:]]; exec(compile(base64.b64decode(os.environ["SCANNER_B64"]), "proc_scanner", "exec"))' any 'runner,checkpoint,supervisor'; do
  if [ "$WAIT_SECONDS" -gt 0 ]; then
    elapsed=$(( $(date +%s) - started ))
    if [ "$elapsed" -ge "$WAIT_SECONDS" ]; then
      echo "[stop] Refused VM destruction: fgpo.runner, supervisor, or checkpoint sync still running after ${elapsed}s" >&2
      exit 1
    fi
  fi
  sleep 5
done

IFS=, read -r -a EXPECTED_CELLS <<< "$CELL_CSV"
if [ -d "$REMOTE_OUTPUT" ]; then
  while IFS= read -r discovered_dir; do
    rel="${discovered_dir#${REMOTE_OUTPUT}/}"
    found=0
    for cid in "${EXPECTED_CELLS[@]}"; do
      if [ "$cid" = "$rel" ]; then found=1; break; fi
    done
    if [ "$found" = "0" ]; then
      echo "[stop] Refused VM destruction: unreferenced cell '${rel}' exists under ${REMOTE_OUTPUT}; list all affected cells via --cell." >&2
      exit 1
    fi
  done < <(find "$REMOTE_OUTPUT" -mindepth 4 -maxdepth 4 -type d -name 'seed-*' 2>/dev/null)
fi

for cid in "${EXPECTED_CELLS[@]}"; do
  IFS=/ read -r s b m sd <<< "$cid"
  seed="${sd#seed-}"
  CELL_DIR="${REMOTE_OUTPUT}/${cid}"
  [ -d "$CELL_DIR" ] || { echo "[stop] Missing cell directory on runtime: ${CELL_DIR}" >&2; exit 1; }
  [ -f "${CELL_DIR}/costs.jsonl" ] && [ -f "${CELL_DIR}/config.json" ] || {
    echo "[stop] Cell ${cid} lacks config.json or costs.jsonl on runtime" >&2; exit 1;
  }
  CUR_PTR="${CELL_DIR}/checkpoint/CURRENT"
  FIN_PTR="${CELL_DIR}/final_checkpoint/CURRENT"
  if [ ! -f "$CUR_PTR" ] && [ ! -f "$FIN_PTR" ]; then
    echo "[stop] Refused VM destruction: no durable checkpoint/CURRENT or final_checkpoint/CURRENT in ${CELL_DIR}" >&2
    exit 1
  fi
  if [ "$BACKEND" = "gcs" ]; then
    [ -f "$REMOTE_TOKEN_FILE" ] || {
      echo "[stop] Refused VM destruction: missing GCS token file ${REMOTE_TOKEN_FILE}" >&2
      exit 1
    }
    "$PY" "$CLI" sync-logs \
      --config "$REMOTE_CONFIG" --cell-dir "$CELL_DIR" \
      --suite "$s" --method "$m" --benchmark "$b" --seed "$seed" >/dev/null
    STATUS_JSON=$("$PY" "$CLI" status \
      --config "$REMOTE_CONFIG" --cell-dir "$CELL_DIR" \
      --suite "$s" --method "$m" --benchmark "$b" --seed "$seed")
    "$PY" - "$STATUS_JSON" "$CUR_PTR" "$FIN_PTR" << 'CHK_PY'
import json, pathlib, sys
status = json.loads(sys.argv[1])
cur_ptr, fin_ptr = pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
if status.get("status") not in ("resume", "complete"):
    raise SystemExit(f"Remote GCS checkpoint status is {status.get('status')!r}")
local_ptr = fin_ptr if (fin_ptr.is_file() and status["status"] == "complete") else cur_ptr if cur_ptr.is_file() else fin_ptr
local_snap = json.loads(local_ptr.read_text()).get("snapshot")
if local_snap != status.get("snapshot"):
    raise SystemExit(f"Local snapshot {local_snap!r} differs from remote GCS snapshot {status.get('snapshot')!r}")
CHK_PY
  else
    "$PY" - "$CELL_DIR" << 'LOC_PY'
import json, pathlib, sys
cell = pathlib.Path(sys.argv[1])
for ptr_name in ("final_checkpoint", "checkpoint"):
    ptr = cell / ptr_name / "CURRENT"
    if ptr.is_file():
        snap = json.loads(ptr.read_text()).get("snapshot")
        snap_dir = cell / ptr_name / "snapshots" / snap
        if not (snap_dir / "metadata.json").is_file() or not (snap_dir / "state.pt").is_file() or not (snap_dir / "model").is_dir():
            raise SystemExit(f"Incomplete local snapshot directory {snap_dir}")
        break
else:
    raise SystemExit("No valid local CURRENT pointer")
LOC_PY
  fi
done
printf '%s\n' "$BACKEND"
STOP_VERIFY_EOF
)"

    for cid in "${CELL_IDS[@]}"; do
      parse_cell_id "$cid"
      CELL_DEST="${DEST_ROOT}/${cid}"
      mkdir -p "$DEST_ROOT"
      if [ "$BACKEND" = "gcs" ]; then
        echo "[stop] Pulling retained GCS artifacts/journals and derived reporting outputs for ${cid} into ${CELL_DEST}..."
        colab_remote_run "$TRANSPORT" "$SESSION" \
          "PY='$REMOTE_PY' CLI='$REMOTE_GCS_CLI' CONFIG_PATH='$REMOTE_CONFIG' REMOTE_OUTPUT='$REMOTE_OUTPUT' CELL_ID='$cid' CELL_DIR='${REMOTE_OUTPUT}/${cid}' CELL_SUITE='$CELL_SUITE' CELL_BENCHMARK='$CELL_BENCHMARK' CELL_METHOD='$CELL_METHOD' CELL_SEED='$CELL_SEED' REMOTE_TOKEN_FILE='$REMOTE_TOKEN_FILE' WITH_MODEL='$WITH_MODEL' bash -s" << 'STOP_PULL_EOF' | tar -xzf - -C "$DEST_ROOT"
set -euo pipefail
export PYTHONPATH=/content/fgpo-r3/src
export FGPO_GCS_ACCESS_TOKEN_FILE="$REMOTE_TOKEN_FILE"
STAGE_ROOT="$(mktemp -d /tmp/fgpo_stop_pull_XXXXXX)"
trap 'rm -rf "$STAGE_ROOT"' EXIT
CELL_STAGE="$STAGE_ROOT/$CELL_ID"
SUITE_STAGE="$STAGE_ROOT/$CELL_SUITE"
mkdir -p "$CELL_STAGE" "$SUITE_STAGE"
EXTRA=()
if [ "$WITH_MODEL" = "1" ]; then
  EXTRA+=(--with-final-model)
fi
"$PY" "$CLI" pull \
  --config "$CONFIG_PATH" --cell-dir "$CELL_DIR" \
  --suite "$CELL_SUITE" --method "$CELL_METHOD" \
  --benchmark "$CELL_BENCHMARK" --seed "$CELL_SEED" \
  --dest-dir "$CELL_STAGE" ${EXTRA[@]+"${EXTRA[@]}"} > "$CELL_STAGE/.pull_receipt.json"
if [ -f "$CELL_DIR/reporting_summary.json" ]; then
  cp -f "$CELL_DIR/reporting_summary.json" "$CELL_STAGE/reporting_summary.json"
fi
for report_name in study_reporting_summary.json study_reporting_table1.md; do
  if [ -f "$REMOTE_OUTPUT/$CELL_SUITE/$report_name" ]; then
    cp -f "$REMOTE_OUTPUT/$CELL_SUITE/$report_name" "$SUITE_STAGE/$report_name"
  fi
done
tar -czf - -C "$STAGE_ROOT" .
STOP_PULL_EOF
        [ -f "$CELL_DEST/.pull_receipt.json" ] && [ -f "$CELL_DEST/costs.jsonl" ] && [ -f "$CELL_DEST/config.json" ] || {
          echo "[stop] Refused VM destruction: failed off-instance GCS retention in ${CELL_DEST}" >&2
          exit 1
        }
      else
        echo "[stop] Transferring full local cell state and derived reporting outputs for ${cid} off-instance into ${CELL_DEST}..."
        colab_remote_run "$TRANSPORT" "$SESSION" \
          "REMOTE_OUTPUT='$REMOTE_OUTPUT' CELL_ID='$cid' CELL_SUITE='$CELL_SUITE' bash -s" << 'STOP_LOCAL_PULL_EOF' | tar -xzf - -C "$DEST_ROOT"
set -euo pipefail
STAGE_ROOT="$(mktemp -d /tmp/fgpo_stop_local_XXXXXX)"
trap 'rm -rf "$STAGE_ROOT"' EXIT
mkdir -p "$STAGE_ROOT/$CELL_ID" "$STAGE_ROOT/$CELL_SUITE"
tar --exclude='environments' -cf - -C "$REMOTE_OUTPUT/$CELL_ID" . | tar -xf - -C "$STAGE_ROOT/$CELL_ID"
for report_name in study_reporting_summary.json study_reporting_table1.md; do
  if [ -f "$REMOTE_OUTPUT/$CELL_SUITE/$report_name" ]; then
    cp -f "$REMOTE_OUTPUT/$CELL_SUITE/$report_name" "$STAGE_ROOT/$CELL_SUITE/$report_name"
  fi
done
tar -czf - -C "$STAGE_ROOT" .
STOP_LOCAL_PULL_EOF
        [ -f "$CELL_DEST/costs.jsonl" ] && [ -f "$CELL_DEST/config.json" ] && { [ -f "$CELL_DEST/checkpoint/CURRENT" ] || [ -f "$CELL_DEST/final_checkpoint/CURRENT" ]; } || {
          echo "[stop] Refused VM destruction: failed off-instance local cell transfer in ${CELL_DEST}" >&2
          exit 1
        }
      fi
    done

    echo "[stop] Off-instance retention verified in '${DEST_ROOT}'. Stopping VM '${SESSION}'..."
    "$COLAB_CLI" stop -s "$SESSION"
    "$COLAB_CLI" sessions || true
    ;;

  *)
    echo "Unknown command: $CMD (expected sync-code|refresh-token|status|pull|aggregate|stop)" >&2
    exit 2
    ;;
esac
