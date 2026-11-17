#!/usr/bin/env bash
# Launch or adopt the declared FGPO seed sequence on an already provisioned Colab instance
# using a detached remote supervisor so no foreground SSH tunnel stays open.
# The original operator owns instance lifecycle, source deployment and token refresh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/colab_common.sh"

SESSION=""
BENCHMARK="alfworld"
SUITE="comparison"
METHOD="fgpo"
CONFIG="/content/fgpo-r3/configs/fgpo-r3-colab-a100.json"
OUTPUT="/content/outputs/fgpo-r3-published-v4-alf"
SEEDS=(0 1 2)
RESUME=0
ADOPT_PID=""
REMOTE_TOKEN_FILE="/run/fgpo-gcs/access-token"
TRANSPORT="ssh"
while [ $# -gt 0 ]; do
  case "$1" in
    -s|--session) SESSION="$2"; shift 2 ;;
    -b|--benchmark) BENCHMARK="$2"; shift 2 ;;
    --suite) SUITE="$2"; shift 2 ;;
    --method) METHOD="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --output-dir) OUTPUT="$2"; shift 2 ;;
    --seed) SEEDS=("$2"); shift 2 ;;
    --seeds) IFS=, read -r -a SEEDS <<< "$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --adopt-pid) ADOPT_PID="$2"; shift 2 ;;
    --remote-token-file) REMOTE_TOKEN_FILE="$2"; shift 2 ;;
    --transport) TRANSPORT="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --session EXISTING_SESSION [--benchmark alfworld] [--suite comparison|ablation] [--method fgpo] [--config PATH] [--output-dir PATH] [--seeds 0,1,2 | --seed 0] [--resume] [--adopt-pid PID] [--remote-token-file /run/fgpo-gcs/access-token] [--transport ssh|exec]"
      echo "Detaches the remote seed/log-sync supervisor under setsid (optionally adopting an existing orphan runner PID for the first seed) so SSH disconnects cleanly after launch."
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$SESSION" ] || { echo "An existing --session is required" >&2; exit 2; }
[ "$BENCHMARK" = alfworld ] || { echo "Colab is assigned ALFWorld only" >&2; exit 2; }
case "$SUITE" in comparison|ablation) ;; *) echo "Invalid suite: $SUITE" >&2; exit 2 ;; esac
case "$METHOD" in fgpo|fgpo_obs_only|fgpo_no_transfer|fgpo_no_support_weight|gigpo|hgpo|graphgpo) ;; *) echo "Invalid method: $METHOD" >&2; exit 2 ;; esac
[ "${#SEEDS[@]}" -ge 1 ] || { echo "At least one seed is required" >&2; exit 2; }
for seed in "${SEEDS[@]}"; do
  case "$seed" in 0|1|2) ;; *) echo "Invalid seed" >&2; exit 2 ;; esac
done
if [ -n "$ADOPT_PID" ]; then
  case "$ADOPT_PID" in
    ''|*[!0-9]*|0) echo "Invalid --adopt-pid '$ADOPT_PID' (expected positive integer PID)" >&2; exit 2 ;;
  esac
fi
colab_validate_transport "$TRANSPORT"
# Pass only constrained shell values to the existing instance. No credentials
# are copied; the pathname names the operator's externally refreshed token.
for path_arg in "$REMOTE_TOKEN_FILE" "$CONFIG" "$OUTPUT"; do
  case "$path_arg" in /*) ;; *) echo "Path must be absolute: $path_arg" >&2; exit 2 ;; esac
  case "$path_arg" in *[!a-zA-Z0-9_./-]*) echo "Unsupported path characters: $path_arg" >&2; exit 2 ;; esac
done
SEED_CSV=$(IFS=,; echo "${SEEDS[*]}")

COLAB_EXEC_TIMEOUT="${COLAB_EXEC_TIMEOUT:-600}" colab_remote_run "$TRANSPORT" "$SESSION" \
  "FGPO_GCS_ACCESS_TOKEN_FILE='$REMOTE_TOKEN_FILE' FGPO_SUITE='$SUITE' FGPO_METHOD='$METHOD' FGPO_CONFIG='$CONFIG' FGPO_OUTPUT='$OUTPUT' FGPO_SEEDS='$SEED_CSV' FGPO_RESUME='$RESUME' FGPO_ADOPT_PID='$ADOPT_PID' FGPO_LAUNCH_READY_TIMEOUT='${FGPO_LAUNCH_READY_TIMEOUT:-480}' bash -s" <<'OUTER_REMOTE_EOF'
set -euo pipefail
PY=/content/venvs/fgpo-main/bin/python
[ -x "$PY" ] || { echo "Missing $PY" >&2; exit 1; }
[ -d /content/fgpo-r3/src/fgpo ] || { echo "Missing /content/fgpo-r3/src/fgpo" >&2; exit 1; }
[ -f "$FGPO_CONFIG" ] || { echo "Missing remote config: $FGPO_CONFIG" >&2; exit 1; }
mkdir -p /content/logs "$FGPO_OUTPUT"

SUP_BODY_FILE="$(mktemp /tmp/fgpo_sup_body_XXXXXX)"
trap 'rm -f "$SUP_BODY_FILE"' EXIT

cat > "$SUP_BODY_FILE" <<'INNER_SUP_BODY_EOF'
SUPERVISOR_LOCK_FILE="$OUTPUT/.fgpo-colab-supervisor.lock"
LEGACY_LOCK_FILE="$OUTPUT/.fgpo-colab-sequence.lock"
PID_FILE="$OUTPUT/.fgpo-colab-supervisor.pid"
READY_FILE="$OUTPUT/.fgpo-colab-supervisor.ready"
STOP_FLAG="$OUTPUT/.fgpo-stop-requested"
CURRENT_RUNNER=""
CURRENT_RUNNER_IS_ADOPTED=0

pid_tmp="${PID_FILE}.tmp.$$"
printf '%s\n' "$$" > "$pid_tmp"
mv -f "$pid_tmp" "$PID_FILE"

run_unlocked() {
  (exec 8>&- 9>&-; exec "$@")
}

publish_ready() {
  local ready_seed="$1" ready_mode="$2" ready_runner="$3"
  local ready_tmp="${READY_FILE}.tmp.$$"
  run_unlocked "$PY" -c '
import json, sys
print(json.dumps({
    "pid": int(sys.argv[1]),
    "seed": int(sys.argv[2]),
    "mode": sys.argv[3],
    "runner_pid": int(sys.argv[4]),
}))
' "$$" "$ready_seed" "$ready_mode" "$ready_runner" > "$ready_tmp"
  mv -f "$ready_tmp" "$READY_FILE"
}

run_aggregate_reports() {
  local agg_csv
  agg_csv="$(run_unlocked "$PY" - "$OUTPUT/$FGPO_SUITE/alfworld/$FGPO_METHOD" "$FGPO_SUITE" "$FGPO_METHOD" alfworld <<'INNER_AGG_SEEDS_PY_EOF'
import json, sys
from pathlib import Path

base, suite, method, benchmark = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
ready_seeds = []
for s in (0, 1, 2):
    cdir = base / f"seed-{s}"
    sp = cdir / "cell_summary.json"
    mp = cdir / ".finalization_complete.json"
    if not sp.is_file() or not mp.is_file():
        continue
    summary = json.loads(sp.read_text())
    marker = json.loads(mp.read_text())
    ev = summary.get("evaluation") or {}
    if (summary.get("status") == "complete"
            and summary.get("finalization_stage") == "finalized"
            and summary.get("suite") == suite
            and summary.get("method") == method
            and summary.get("benchmark") == benchmark
            and summary.get("seed") == s
            and ev
            and not any(not item.get("dataset_complete") for item in ev.values())
            and marker.get("terminal_export_completed")):
        ready_seeds.append(str(s))
print(",".join(ready_seeds))
INNER_AGG_SEEDS_PY_EOF
)"
  if [ -n "$agg_csv" ]; then
    local agg_seeds=()
    IFS=, read -r -a agg_seeds <<< "$agg_csv"
    echo "[supervisor] Generating report-only Table 1 outputs for finalized seed(s): ${agg_seeds[*]}"
    (exec 8>&- 9>&-; exec "$PY" -m fgpo.runner \
      --config "$CONFIG" \
      --output-dir "$OUTPUT" \
      --suite "$FGPO_SUITE" \
      --methods "$FGPO_METHOD" \
      --benchmarks alfworld \
      --seeds "${agg_seeds[@]}" \
      --aggregate-only)
  fi
}

pid_is_non_zombie() {
  run_unlocked "$PY" -c '
import pathlib, sys
try:
    st = pathlib.Path(f"/proc/{int(sys.argv[1])}/stat").read_text()
    state = st.rsplit(")", 1)[1].split()[0]
    raise SystemExit(0 if state not in ("Z", "X") else 1)
except OSError:
    raise SystemExit(1)
' "$1"
}

adopt_pid_matches() {
  run_unlocked "$PY" -c '
import base64, pathlib, sys
pid, exp_start, exp_cmd_b64 = int(sys.argv[1]), sys.argv[2], sys.argv[3]
try:
    st = pathlib.Path(f"/proc/{pid}/stat").read_text()
    fields = st.rsplit(")", 1)[1].split()
    if fields[0] in ("Z", "X") or fields[19] != exp_start:
        raise SystemExit(1)
    raw = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes()
    if base64.b64encode(raw).decode("ascii") != exp_cmd_b64:
        raise SystemExit(1)
except OSError:
    raise SystemExit(1)
' "$ADOPT_PID" "$ADOPT_STARTTIME" "$ADOPT_CMDLINE_B64"
}

verify_cell_completed() {
  local check_cell="$1" check_seed="$2"
  local gcs_dto="{}"
  if [ "$BACKEND" = gcs ]; then
    gcs_dto="$(run_unlocked "$PY" scripts/fgpo_gcs_checkpoint.py status \
      --config "$CONFIG" --cell-dir "$check_cell" \
      --suite "$FGPO_SUITE" --method "$FGPO_METHOD" \
      --benchmark alfworld --seed "$check_seed")"
  fi
  run_unlocked "$PY" - "$check_cell/cell_summary.json" "$check_cell/.finalization_complete.json" \
    "$BACKEND" "$gcs_dto" "$FGPO_SUITE" "$FGPO_METHOD" alfworld "$check_seed" <<'INNER_VERIFY_PY_EOF'
import json, sys
from pathlib import Path

summary_path, marker_path = Path(sys.argv[1]), Path(sys.argv[2])
backend, gcs_raw, suite, method, benchmark, seed_str = sys.argv[3:9]
seed = int(seed_str)

summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
marker = json.loads(marker_path.read_text()) if marker_path.is_file() else {}
evaluation = summary.get("evaluation") or {}

if (summary.get("status") != "complete"
        or summary.get("finalization_stage") != "finalized"
        or summary.get("suite") != suite
        or summary.get("method") != method
        or summary.get("benchmark") != benchmark
        or summary.get("seed") != seed
        or not evaluation
        or any(not item.get("dataset_complete") for item in evaluation.values())
        or not marker.get("terminal_export_completed")):
    raise SystemExit(
        f"Cell seed-{seed} stopped before durable finalized completion "
        f"(summary_status={summary.get('status')!r}, stage={summary.get('finalization_stage')!r}, "
        f"terminal_marker={marker.get('terminal_export_completed')!r}); next seed not dispatched"
    )

if backend == "gcs":
    dto = json.loads(gcs_raw)
    if (dto.get("status") != "complete"
            or not dto.get("terminal_export_completed")
            or not dto.get("final_generation")):
        raise SystemExit(
            f"GCS DTO for seed-{seed} is not finalized/complete ({dto!r}); next seed not dispatched"
        )
INNER_VERIFY_PY_EOF
}

cleanup() {
  if [ -f "$PID_FILE" ] && [ "$(cat "$PID_FILE" 2>/dev/null || true)" = "$$" ]; then
    rm -f "$PID_FILE" "$READY_FILE"
  fi
}

on_signal() {
  touch "$STOP_FLAG"
  if [ -n "$CURRENT_RUNNER" ]; then
    if [ "$CURRENT_RUNNER_IS_ADOPTED" = "1" ]; then
      if adopt_pid_matches; then
        kill -TERM "$CURRENT_RUNNER" 2>/dev/null || true
        while adopt_pid_matches; do
          sleep 2
        done
      fi
    elif pid_is_non_zombie "$CURRENT_RUNNER"; then
      kill -TERM "$CURRENT_RUNNER" 2>/dev/null || true
      wait "$CURRENT_RUNNER" 2>/dev/null || true
    fi
  fi
  exit 130
}

trap cleanup EXIT
trap on_signal INT TERM

BACKEND=$(run_unlocked "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("checkpoint_backend","local"))' "$CONFIG")
MODEL_DIR=$(run_unlocked "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_path"])' "$CONFIG")
mkdir -p /content/logs "$OUTPUT"

IFS=, read -r -a seeds <<< "$FGPO_SEEDS"
seed_index=0
for seed in "${seeds[@]}"; do
  [ ! -f "$STOP_FLAG" ] || { echo "[supervisor] Stop requested before seed-$seed; exiting cleanly."; exit 0; }
  cell="$OUTPUT/$FGPO_SUITE/alfworld/$FGPO_METHOD/seed-$seed"

  if [ "$seed_index" -eq 0 ] && [ -n "$ADOPT_PID" ]; then
    echo "[supervisor] Verifying readiness to adopt existing fgpo.runner PID=$ADOPT_PID for seed-$seed ($cell)"
    adopt_pid_matches || { echo "Adopted PID=$ADOPT_PID vanished or mismatched before readiness." >&2; exit 1; }
    if [ "$BACKEND" = gcs ]; then
      run_unlocked "$PY" scripts/fgpo_gcs_checkpoint.py status --config "$CONFIG" --cell-dir "$cell" --suite "$FGPO_SUITE" --method "$FGPO_METHOD" --benchmark alfworld --seed "$seed" >/dev/null
    fi
    publish_ready "$seed" "adopted" "$ADOPT_PID"
    echo "[supervisor] Adopting existing fgpo.runner PID=$ADOPT_PID for seed-$seed ($cell)"
    CURRENT_RUNNER="$ADOPT_PID"
    CURRENT_RUNNER_IS_ADOPTED=1
    last_sync=$(date +%s)
    while adopt_pid_matches; do
      sleep 5
      adopt_pid_matches || break
      now=$(date +%s)
      if [ ! -f "$STOP_FLAG" ] && [ "$BACKEND" = gcs ] && [ $((now - last_sync)) -ge 45 ] && { [ -f "$cell/checkpoint/CURRENT" ] || [ -f "$cell/final_checkpoint/CURRENT" ]; }; then
        if ! run_unlocked "$PY" scripts/fgpo_gcs_checkpoint.py sync-logs --config "$CONFIG" --cell-dir "$cell" --suite "$FGPO_SUITE" --method "$FGPO_METHOD" --benchmark alfworld --seed "$seed"; then
          echo "GCS physical-log publication failed; sequence stopped. Adopted runner PID=$ADOPT_PID remains operator-owned." >&2
          exit 1
        fi
        last_sync="$now"
      fi
    done
    CURRENT_RUNNER=""
    CURRENT_RUNNER_IS_ADOPTED=0
    if [ -f "$STOP_FLAG" ]; then
      echo "[supervisor] Stop requested; adopted seed-$seed (PID=$ADOPT_PID) exited."
      exit 0
    fi
    verify_cell_completed "$cell" "$seed"
    run_aggregate_reports
    if [ "$LEGACY_LOCK_HELD" != "1" ]; then
      exec 9>"$LEGACY_LOCK_FILE"
      flock -n 9 || {
        echo "Legacy lock $LEGACY_LOCK_FILE remains held after adopted PID=$ADOPT_PID exited; stopping sequence." >&2
        exit 1
      }
      LEGACY_LOCK_HELD=1
    fi
    seed_index=$((seed_index + 1))
    continue
  fi

  resume_args=()
  receipt_file="/content/logs/fgpo_${FGPO_SUITE}_${FGPO_METHOD}_alfworld_s${seed}_cold_asset.json"
  rm -f "$receipt_file"
  startup_receipt_env=""
  if [ "$BACKEND" = gcs ]; then
    status_json=$(run_unlocked "$PY" scripts/fgpo_gcs_checkpoint.py status --config "$CONFIG" --cell-dir "$cell" --suite "$FGPO_SUITE" --method "$FGPO_METHOD" --benchmark alfworld --seed "$seed")
    status=$(printf '%s' "$status_json" | run_unlocked "$PY" -c 'import json,sys; print(json.load(sys.stdin)["status"])')
    if [ "$status" != absent ]; then
      [ "$FGPO_RESUME" = 1 ] || { echo "Existing GCS seed-$seed requires --resume" >&2; exit 1; }
      resume_args=(--resume)
    else
      # Every fresh seed restores the ORIGINAL Qwen asset, including all weight
      # shards. Previous seeds' trained actor caches are never initialization.
      run_unlocked "$PY" scripts/fgpo_gcs_checkpoint.py restore-asset \
        --uri gs://r032-pact-rebuild-f1be230a/fgpo-r3/assets/qwen25_7b_instruct.tar.zst \
        --destination /content/models > "$receipt_file"
      startup_receipt_env="$receipt_file"
    fi
  else
    if [ -e "$cell" ]; then
      [ "$FGPO_RESUME" = 1 ] || { echo "Existing local seed-$seed requires --resume" >&2; exit 1; }
      resume_args=(--resume)
    fi
    if ! compgen -G "$MODEL_DIR/*.safetensors" > /dev/null && [ -f "${FGPO_GCS_ACCESS_TOKEN_FILE:-}" ]; then
      run_unlocked "$PY" scripts/fgpo_gcs_checkpoint.py restore-asset \
        --uri gs://r032-pact-rebuild-f1be230a/fgpo-r3/assets/qwen25_7b_instruct.tar.zst \
        --destination /content/models > "$receipt_file"
      startup_receipt_env="$receipt_file"
    fi
  fi

  [ ! -f "$STOP_FLAG" ] || { echo "[supervisor] Stop requested before launching seed-$seed runner; exiting cleanly."; exit 0; }

  # Close supervisor lock FDs 8/9 in the child subshell before exec'ing fgpo.runner
  # so runners and their worker pools never inherit supervisor flock descriptors.
  (exec 8>&- 9>&-; exec env FGPO_STARTUP_RECEIPT_FILE="$startup_receipt_env" "$PY" -u -m fgpo.runner --config "$CONFIG" --output-dir "$OUTPUT" \
    --suite "$FGPO_SUITE" --methods "$FGPO_METHOD" --benchmarks alfworld --seeds "$seed" --gpu 0 \
    ${resume_args[@]+"${resume_args[@]}"}) >> "/content/logs/fgpo_${FGPO_SUITE}_${FGPO_METHOD}_alfworld_s${seed}.log" 2>&1 &
  runner=$!
  CURRENT_RUNNER="$runner"
  CURRENT_RUNNER_IS_ADOPTED=0
  if [ "$seed_index" -eq 0 ]; then
    sleep 1
    if ! pid_is_non_zombie "$runner"; then
      runner_rc=0
      wait "$runner" || runner_rc=$?
      CURRENT_RUNNER=""
      echo "[supervisor] Initial seed-$seed runner exited before readiness (rc=$runner_rc)." >&2
      exit "${runner_rc:-1}"
    fi
    publish_ready "$seed" "spawned" "$runner"
  fi
  last_sync=$(date +%s)
  # Check /proc/<pid>/stat state != Z so an un-reaped exited child falls through
  # immediately to `wait` instead of stalling in `kill -0`.
  while pid_is_non_zombie "$runner"; do
    sleep 5
    pid_is_non_zombie "$runner" || break
    now=$(date +%s)
    if [ ! -f "$STOP_FLAG" ] && [ "$BACKEND" = gcs ] && [ $((now - last_sync)) -ge 45 ] && { [ -f "$cell/checkpoint/CURRENT" ] || [ -f "$cell/final_checkpoint/CURRENT" ]; }; then
      if ! run_unlocked "$PY" scripts/fgpo_gcs_checkpoint.py sync-logs --config "$CONFIG" --cell-dir "$cell" --suite "$FGPO_SUITE" --method "$FGPO_METHOD" --benchmark alfworld --seed "$seed"; then
        echo "GCS physical-log publication failed; sequence stopped. Training PID=$runner remains operator-owned." >&2
        exit 1
      fi
      last_sync="$now"
    fi
  done
  runner_rc=0
  wait "$runner" || runner_rc=$?
  CURRENT_RUNNER=""
  if [ -f "$STOP_FLAG" ]; then
    echo "[supervisor] Stop requested; seed-$seed exited cleanly with status=$runner_rc."
    exit 0
  fi
  [ "$runner_rc" = 0 ] || exit "$runner_rc"
  verify_cell_completed "$cell" "$seed"
  run_aggregate_reports
  seed_index=$((seed_index + 1))
done
INNER_SUP_BODY_EOF

"$PY" - "$SUP_BODY_FILE" <<'INNER_LAUNCH_PY_EOF'
import base64
import fcntl
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

sup_body_path = Path(sys.argv[1])
sup_body_text = sup_body_path.read_text(encoding="utf-8")

token_file = os.environ["FGPO_GCS_ACCESS_TOKEN_FILE"]
suite = os.environ["FGPO_SUITE"]
method = os.environ["FGPO_METHOD"]
config = os.environ["FGPO_CONFIG"]
output = Path(os.environ["FGPO_OUTPUT"])
seeds_csv = os.environ["FGPO_SEEDS"]
resume = os.environ["FGPO_RESUME"]
adopt_pid_raw = os.environ.get("FGPO_ADOPT_PID", "").strip()
py_bin = "/content/venvs/fgpo-main/bin/python"

seeds = [s for s in seeds_csv.split(",") if s]
if not seeds:
    raise SystemExit("No seeds specified")
first_seed = seeds[0]

def norm_path(value: str | None) -> str:
    if not value:
        return ""
    return os.path.normpath(value)

def parse_argv_flag(argv: list[str], flag: str) -> str | None:
    prefix = flag + "="
    for idx, tok in enumerate(argv):
        if tok == flag and idx + 1 < len(argv):
            return argv[idx + 1]
        if tok.startswith(prefix):
            return tok[len(prefix):]
    return None

def has_runner_module(argv: list[str]) -> bool:
    for idx in range(len(argv) - 1):
        if argv[idx] == "-m" and argv[idx + 1] == "fgpo.runner":
            return True
    return False

def scan_matching_runners(target_output: Path) -> list[tuple[int, str]]:
    norm_out = norm_path(str(target_output))
    matches = []
    for entry in sorted(Path("/proc").iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 10**9):
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            st = (entry / "stat").read_text()
            if st.rsplit(")", 1)[1].split()[0] in ("Z", "X"):
                continue
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        argv = [x.decode("utf-8", "replace") for x in raw.rstrip(b"\x00").split(b"\x00") if x]
        if has_runner_module(argv) and norm_path(parse_argv_flag(argv, "--output-dir")) == norm_out:
            matches.append((pid, " ".join(argv)))
    return matches

def scan_lock_holders(lock_path: Path) -> list[tuple[int, str, str]]:
    if not lock_path.exists():
        return []
    lst = os.stat(lock_path)
    target = (lst.st_dev, lst.st_ino)
    self_pid = os.getpid()
    holders = []
    for entry in sorted(Path("/proc").iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 10**9):
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == self_pid:
            continue
        fd_dir = entry / "fd"
        try:
            fds = list(fd_dir.iterdir())
        except OSError:
            continue
        matched_fds = []
        for fd_link in fds:
            try:
                fst = os.stat(fd_link)
                if (fst.st_dev, fst.st_ino) == target:
                    fi_txt = (entry / "fdinfo" / fd_link.name).read_text()
                    locks = [ln.strip() for ln in fi_txt.splitlines() if "lock:" in ln.lower()]
                    matched_fds.append(f"fd/{fd_link.name}({';'.join(locks) or 'open'})")
            except OSError:
                continue
        if matched_fds:
            try:
                cmd = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
            except OSError:
                cmd = "<exited>"
            holders.append((pid, ",".join(matched_fds), cmd))
    return holders

def cmdline_looks_like_launcher(pid: int) -> str | None:
    try:
        cmd = (Path(f"/proc/{pid}/cmdline")).read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
    except OSError:
        return None
    for marker in ("fgpo-colab-supervisor", "colab_launch_fgpo", "fgpo_launch", "FGPO_SEEDS="):
        if marker in cmd:
            return cmd
    return None

# Single-writer critical section: hold .fgpo-colab-supervisor.lock continuously
# from preflight validation through script creation and detached setsid handoff.
sup_lock_path = output / ".fgpo-colab-supervisor.lock"
sup_raw = os.open(sup_lock_path, os.O_CREAT | os.O_RDWR, 0o644)
try:
    fcntl.flock(sup_raw, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    os.close(sup_raw)
    raise SystemExit(f"Existing Colab sequence supervisor already holds {sup_lock_path}")

leg_lock_path = output / ".fgpo-colab-sequence.lock"
leg_raw = os.open(leg_lock_path, os.O_CREAT | os.O_RDWR, 0o644)
try:
    fcntl.flock(leg_raw, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    os.close(leg_raw)
    leg_raw = None

stop_flag = output / ".fgpo-stop-requested"
runners = scan_matching_runners(output)
adopt_starttime = ""
adopt_cmdline_b64 = ""

if not adopt_pid_raw:
    if leg_raw is None:
        holders = scan_lock_holders(leg_lock_path)
        raise SystemExit(
            f"Legacy sequence lock {leg_lock_path} is still held (holders={holders}); "
            "wait for it to exit or pass explicit --adopt-pid <PID>."
        )
    if runners:
        raise SystemExit(
            f"Existing fgpo.runner process(es) active for {output}: {runners}; "
            "refusing duplicate launch (use --adopt-pid <PID> if adopting an orphan runner)."
        )
    try:
        stop_flag.unlink()
    except FileNotFoundError:
        pass
else:
    adopt_pid = int(adopt_pid_raw)
    if stop_flag.exists():
        raise SystemExit(f"Stop flag {stop_flag} already exists; refusing --adopt-pid {adopt_pid}")
    proc_dir = Path(f"/proc/{adopt_pid}")
    try:
        stat_txt = (proc_dir / "stat").read_text()
        stat_fields = stat_txt.rsplit(")", 1)[1].split()
        state = stat_fields[0]
        ppid = int(stat_fields[1])
        adopt_starttime = stat_fields[19]
        cmd_raw = (proc_dir / "cmdline").read_bytes()
    except OSError as exc:
        raise SystemExit(f"Cannot inspect --adopt-pid {adopt_pid}: {exc}") from exc
    if state in ("Z", "X"):
        raise SystemExit(f"Adopted PID {adopt_pid} is not running (state={state})")
    argv = [x.decode("utf-8", "replace") for x in cmd_raw.rstrip(b"\x00").split(b"\x00") if x]
    if not has_runner_module(argv):
        raise SystemExit(f"PID {adopt_pid} is not executing -m fgpo.runner: {argv!r}")
    checks = {
        "--config": (norm_path(parse_argv_flag(argv, "--config")), norm_path(config)),
        "--output-dir": (norm_path(parse_argv_flag(argv, "--output-dir")), norm_path(str(output))),
        "--suite": (parse_argv_flag(argv, "--suite"), suite),
        "--methods": (parse_argv_flag(argv, "--methods"), method),
        "--benchmarks": (parse_argv_flag(argv, "--benchmarks"), "alfworld"),
        "--seeds": (parse_argv_flag(argv, "--seeds"), first_seed),
    }
    mismatches = {k: v for k, v in checks.items() if v[0] != v[1]}
    if mismatches:
        raise SystemExit(f"PID {adopt_pid} runner identity mismatch vs requested study/seed: {mismatches}")
    other_runners = [r for r in runners if r[0] != adopt_pid]
    if other_runners:
        raise SystemExit(f"Other fgpo.runner process(es) active for {output}: {other_runners}")
    if ppid != 1:
        parent_launcher = cmdline_looks_like_launcher(ppid)
        if parent_launcher is not None:
            raise SystemExit(
                f"PID {adopt_pid} still has active launcher/supervisor parent PID {ppid} ({parent_launcher}); refusing adoption."
            )
    if leg_raw is None:
        holders = scan_lock_holders(leg_lock_path)
        non_adopt = [h for h in holders if h[0] != adopt_pid]
        if non_adopt:
            raise SystemExit(
                f"Legacy sequence lock {leg_lock_path} is held by non-adopted process(es): {non_adopt}"
            )
    adopt_cmdline_b64 = base64.b64encode(cmd_raw).decode("ascii")

sup_script = output / ".fgpo-colab-supervisor.sh"
sup_pid_file = output / ".fgpo-colab-supervisor.pid"
sup_ready_file = output / ".fgpo-colab-supervisor.ready"
sup_log = Path(f"/content/logs/fgpo_{suite}_{method}_supervisor.log")
legacy_held_flag = "1" if leg_raw is not None else "0"
ready_timeout = max(5.0, float(os.environ.get("FGPO_LAUNCH_READY_TIMEOUT", "480")))

for stale_file in (sup_pid_file, sup_ready_file):
    try:
        stale_file.unlink()
    except FileNotFoundError:
        pass

header = "\n".join([
    "#!/usr/bin/env bash",
    "set -euo pipefail",
    "cd /content/fgpo-r3",
    "export PYTHONPATH=/content/fgpo-r3/src",
    "export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8",
    "export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1",
    f"PY={shlex.quote(py_bin)}",
    f"export FGPO_GCS_ACCESS_TOKEN_FILE={shlex.quote(token_file)}",
    f"FGPO_SUITE={shlex.quote(suite)}",
    f"FGPO_METHOD={shlex.quote(method)}",
    f"CONFIG={shlex.quote(config)}",
    f"OUTPUT={shlex.quote(str(output))}",
    f"FGPO_SEEDS={shlex.quote(seeds_csv)}",
    f"FGPO_RESUME={shlex.quote(resume)}",
    f"ADOPT_PID={shlex.quote(adopt_pid_raw)}",
    f"ADOPT_STARTTIME={shlex.quote(adopt_starttime)}",
    f"ADOPT_CMDLINE_B64={shlex.quote(adopt_cmdline_b64)}",
    f"LEGACY_LOCK_HELD={shlex.quote(legacy_held_flag)}",
    "",
])

tmp_script = output / f".fgpo-colab-supervisor.sh.tmp.{os.getpid()}"
tmp_script.write_text(header + sup_body_text, encoding="utf-8")
os.chmod(tmp_script, 0o700)
os.replace(tmp_script, sup_script)

# Move lock FDs above 20 first so dup2(8)/dup2(9) cannot collide with open FDs.
sup_high = fcntl.fcntl(sup_raw, fcntl.F_DUPFD, 20)
os.close(sup_raw)
leg_high = None
if leg_raw is not None:
    leg_high = fcntl.fcntl(leg_raw, fcntl.F_DUPFD, 20)
    os.close(leg_raw)

os.dup2(sup_high, 8)
os.close(sup_high)
pass_fds = [8]

if leg_high is not None:
    os.dup2(leg_high, 9)
    os.close(leg_high)
    pass_fds.append(9)

log_fh = open(sup_log, "ab")
sup_log_fd = fcntl.fcntl(log_fh.fileno(), fcntl.F_DUPFD, 20)
log_fh.close()
try:
    proc = subprocess.Popen(
        ["bash", str(sup_script)],
        cwd="/content/fgpo-r3",
        stdin=subprocess.DEVNULL,
        stdout=sup_log_fd,
        stderr=sup_log_fd,
        start_new_session=True,
        pass_fds=tuple(pass_fds),
    )
finally:
    os.close(sup_log_fd)
    for fd_num in pass_fds:
        os.close(fd_num)

import json as _json

ready_deadline = time.monotonic() + ready_timeout
ready_payload = None
while time.monotonic() < ready_deadline:
    rc = proc.poll()
    if rc is not None:
        tail = "\n".join(sup_log.read_text(errors="replace").splitlines()[-30:]) if sup_log.is_file() else ""
        raise SystemExit(f"Detached supervisor exited before readiness (rc={rc}); inspect {sup_log}:\n{tail}")
    if sup_ready_file.is_file():
        try:
            candidate = _json.loads(sup_ready_file.read_text())
        except Exception:
            candidate = None
        if isinstance(candidate, dict) and candidate.get("pid") == proc.pid:
            ready_payload = candidate
            break
    time.sleep(0.2)

if ready_payload is None or proc.poll() is not None:
    tail = "\n".join(sup_log.read_text(errors="replace").splitlines()[-30:]) if sup_log.is_file() else ""
    raise SystemExit(
        f"Detached supervisor (pid={proc.pid}, owner_pid_file={sup_pid_file}) did not publish readiness "
        f"{sup_ready_file} within {ready_timeout:.0f}s; inspect {sup_log}:\n{tail}"
    )

mode_tag = f"adopted_pid={adopt_pid_raw}" if adopt_pid_raw else "fresh"
print(f"[launch] Detached supervisor ready (pid={proc.pid}, {mode_tag}, first_seed={ready_payload.get('seed')}, runner_pid={ready_payload.get('runner_pid')}, log={sup_log}, seeds={seeds_csv})")
INNER_LAUNCH_PY_EOF
OUTER_REMOTE_EOF
