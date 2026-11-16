#!/usr/bin/env bash
# Launch one F-GPO r3 cell on a dedicated Colab High-RAM A100 (80 GB) with:
# - Strict hardware/RAM/disk pre-flight gating (stops VM immediately if unmet)
# - Internal-network streaming restore of pre-staged .tar.zst cold assets
# - Lightweight 60s JSONL/status GCS sync (NO periodic tar-czf of 100GB checkpoints)
# - Resilient keep-alive and clean auto-stop after final upload
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/colab_common.sh"

BENCHMARK="alfworld"
SEED="0"
SUITE="comparison"
METHOD="fgpo"
GPU_TYPE="G4"
SESSION=""
RESUME="0"
MIN_RAM_GIB="130"
MIN_DISK_GIB="225"
ALLOW_LOW_DISK="0"
NO_KEEPALIVE="0"

while [ $# -gt 0 ]; do
  case "$1" in
    -s|--session) SESSION="$2"; shift 2 ;;
    -b|--benchmark) BENCHMARK="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --suite) SUITE="$2"; shift 2 ;;
    -m|--method) METHOD="$2"; shift 2 ;;
    --gpu) GPU_TYPE="$2"; shift 2 ;;
    --resume) RESUME="1"; shift ;;
    --bucket) FGPO_GCS_BUCKET="$2"; shift 2 ;;
    --prefix) FGPO_GCS_PREFIX="$2"; shift 2 ;;
    --min-ram-gib) MIN_RAM_GIB="$2"; shift 2 ;;
    --min-disk-gib) MIN_DISK_GIB="$2"; shift 2 ;;
    --allow-low-disk) ALLOW_LOW_DISK="1"; shift ;;
    --no-keepalive) NO_KEEPALIVE="1"; shift ;;
    -h|--help)
      echo "Usage: $0 [-s SESSION] [--gpu G4|A100] [-b alfworld|appworld] [--seed 0|1|2] [--suite comparison|ablation] [-m fgpo] [--resume] [--bucket BUCKET] [--prefix PREFIX] [--min-ram-gib 130] [--min-disk-gib 225] [--allow-low-disk] [--no-keepalive]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

SESSION="${SESSION:-fgpo-${BENCHMARK}-s${SEED}}"
CELL_ID="${SUITE}/${BENCHMARK}/${METHOD}/seed-${SEED}"
GCS_CELL_PREFIX="${FGPO_GCS_PREFIX}/cells/${CELL_ID}"

SRC_TAR="/tmp/fgpo_r3_src.tar.gz"
pack_fgpo_source "$SRC_TAR"
gcs_py upload "$SRC_TAR" "${FGPO_GCS_PREFIX}/code/fgpo_r3_src.tar.gz" "application/gzip"

HIGH_MEM_FLAG=""
case "$GPU_TYPE" in
  G4|g4|L4|l4) HIGH_MEM_FLAG="" ;;
  *) HIGH_MEM_FLAG="--high-mem" ;;
esac

echo "[launch] Requesting Colab GPU session '$SESSION' (--gpu $GPU_TYPE $HIGH_MEM_FLAG)..."
"$COLAB_CLI" new -s "$SESSION" --gpu "$GPU_TYPE" $HIGH_MEM_FLAG
"$COLAB_CLI" upload -s "$SESSION" "$ADC_PATH" /root/.gcs_adc.json

echo "[launch] Running pre-flight VRAM/RAM/disk verification on '$SESSION'..."
if ! colab_ssh_run "$SESSION" \
  "MIN_RAM_GIB='$MIN_RAM_GIB' MIN_DISK_GIB='$MIN_DISK_GIB' ALLOW_LOW_DISK='$ALLOW_LOW_DISK' bash -s" << 'PRECHECK_EOF'
set -euo pipefail
VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
RAM_GIB=$(awk '/MemTotal:/ {printf "%d", $2 / 1024 / 1024}' /proc/meminfo)
DISK_GIB=$(df -BG --output=avail /content | tail -n 1 | tr -dc '0-9')
echo "[preflight] GPU='$GPU_NAME' VRAM_FREE=${VRAM_FREE}MiB RAM=${RAM_GIB}GiB DISK_FREE=${DISK_GIB}GiB"

if [ "${VRAM_FREE:-0}" -lt 75000 ]; then
  echo "[preflight] FAIL: Full-parameter F-GPO requires >= 75000 MiB free VRAM (80GB A100 or 96GB G4 PRO 6000)." >&2
  exit 1
fi
if [ "${RAM_GIB:-0}" -lt "$MIN_RAM_GIB" ]; then
  echo "[preflight] FAIL: OffloadedAdamW requires >= ${MIN_RAM_GIB} GiB system RAM (got ${RAM_GIB} GiB)." >&2
  exit 1
fi
if [ "$ALLOW_LOW_DISK" != "1" ] && [ "${DISK_GIB:-0}" -lt "$MIN_DISK_GIB" ]; then
  echo "[preflight] FAIL: /content disk too small (got ${DISK_GIB} GiB < ${MIN_DISK_GIB} GiB)." >&2
  exit 1
fi
if [ "${DISK_GIB:-0}" -lt 260 ]; then
  echo "[preflight] WARN: Total /content disk is ${DISK_GIB} GiB (~$((DISK_GIB - 25)) GiB free after assets); F-GPO double-snapshot save peak (~212+ GiB) is tight without Host B offload or non-atomic checkpoint cleanup." >&2
fi
PRECHECK_EOF
then
  echo "[launch] Pre-flight check failed. Stopping '$SESSION' immediately to avoid Compute Unit waste." >&2
  "$COLAB_CLI" stop -s "$SESSION" >/dev/null 2>&1 || true
  exit 1
fi

echo "[launch] Provisioning cold assets from GCS and starting '$CELL_ID' on '$SESSION'..."
colab_ssh_run "$SESSION" \
  "chmod 600 /root/.gcs_adc.json && BUCKET='$FGPO_GCS_BUCKET' PREFIX='$FGPO_GCS_PREFIX' SUITE='$SUITE' BENCHMARK='$BENCHMARK' METHOD='$METHOD' SEED='$SEED' CELL_ID='$CELL_ID' GCS_CELL_PREFIX='$GCS_CELL_PREFIX' RESUME='$RESUME' bash -s" << 'REMOTE_EOF'
set -euo pipefail
set -o pipefail

if ! command -v zstd >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq zstd >/dev/null
fi

mkdir -p /content/fgpo-r3 /content/venvs /content/models /content/data /content/outputs /content/logs

get_token() {
  python3 -c '
import json, urllib.request, urllib.parse
c = json.load(open("/root/.gcs_adc.json"))
data = urllib.parse.urlencode({
    "client_id": c["client_id"],
    "client_secret": c["client_secret"],
    "refresh_token": c["refresh_token"],
    "grant_type": "refresh_token",
}).encode()
req = urllib.request.Request("https://oauth2.googleapis.com/token", data=data)
print(json.loads(urllib.request.urlopen(req, timeout=20).read())["access_token"])
'
}

stream_gcs_zst() {
  local obj="$1"
  local dest="$2"
  local token enc
  token="$(get_token)"
  enc="$(python3 -c 'import urllib.parse, sys; print(urllib.parse.quote(sys.argv[1], safe=""))' "$obj")"
  echo "[setup] Streaming gs://${BUCKET}/${obj} -> ${dest}"
  curl -fsSL --retry 2 -H "Authorization: Bearer $token" \
    "https://storage.googleapis.com/storage/v1/b/${BUCKET}/o/${enc}?alt=media" \
    | zstd -d -c | tar -xf - -C "$dest"
}

stream_gcs_gz() {
  local obj="$1"
  local dest="$2"
  local token enc
  token="$(get_token)"
  enc="$(python3 -c 'import urllib.parse, sys; print(urllib.parse.quote(sys.argv[1], safe=""))' "$obj")"
  curl -fsSL --retry 2 -H "Authorization: Bearer $token" \
    "https://storage.googleapis.com/storage/v1/b/${BUCKET}/o/${enc}?alt=media" \
    | tar --warning=no-unknown-keyword -xzf - -C "$dest"
}

rm -rf /content/fgpo-r3/src/fgpo
stream_gcs_gz "${PREFIX}/code/fgpo_r3_src.tar.gz" /content/fgpo-r3

[ -x /content/venvs/fgpo-main/bin/python ] || stream_gcs_zst "${PREFIX}/assets/fgpo_venv_main.tar.zst" /content/venvs
[ -d /content/models/Qwen2.5-7B-Instruct ] || stream_gcs_zst "${PREFIX}/assets/qwen25_7b_instruct.tar.zst" /content/models

if [ "$BENCHMARK" = "alfworld" ]; then
  [ -x /content/venvs/alfworld/bin/python ] || stream_gcs_zst "${PREFIX}/assets/fgpo_venv_alfworld.tar.zst" /content/venvs
  [ -d /content/data/alfworld ] || stream_gcs_zst "${PREFIX}/assets/alfworld_data.tar.zst" /content/data
else
  [ -d /content/data/appworld-root ] || stream_gcs_zst "${PREFIX}/assets/appworld_data.tar.zst" /content/data
fi

CELL_DIR="/content/outputs/fgpo-r3/${CELL_ID}"
if [ "$RESUME" = "1" ]; then
  /content/venvs/fgpo-main/bin/python - "$BUCKET" "$GCS_CELL_PREFIX" "$CELL_DIR" << 'RESTORE_PY'
import json, os, sys, urllib.parse, requests

bucket, prefix, cell_dir = sys.argv[1:4]
creds = json.load(open("/root/.gcs_adc.json"))
token = requests.post("https://oauth2.googleapis.com/token", data={
    "client_id": creds["client_id"], "client_secret": creds["client_secret"],
    "refresh_token": creds["refresh_token"], "grant_type": "refresh_token",
}, timeout=20).json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

items = []
page = None
while True:
    params = {"prefix": prefix + "/"}
    if page:
        params["pageToken"] = page
    r = requests.get(f"https://storage.googleapis.com/storage/v1/b/{bucket}/o", headers=headers, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    items.extend(data.get("items", []))
    page = data.get("nextPageToken")
    if not page:
        break

if not items:
    raise SystemExit(f"No saved state found in GCS at gs://{bucket}/{prefix}/ for --resume")

os.makedirs(cell_dir, exist_ok=True)
deferred_current = None
for item in items:
    name = item["name"]
    rel = name[len(prefix) + 1:]
    if not rel or rel in ("DONE", "live_status.json", "runner.log"):
        continue
    if rel == "checkpoint/CURRENT":
        deferred_current = name
        continue
    target = os.path.join(cell_dir, rel)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    enc = urllib.parse.quote(name, safe="")
    with requests.get(f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{enc}?alt=media", headers=headers, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        tmp = target + ".part"
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, target)

if deferred_current is not None:
    target = os.path.join(cell_dir, "checkpoint/CURRENT")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    enc = urllib.parse.quote(deferred_current, safe="")
    resp = requests.get(f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{enc}?alt=media", headers=headers, timeout=30)
    resp.raise_for_status()
    tmp = target + ".part"
    with open(tmp, "wb") as f:
        f.write(resp.content)
    os.replace(tmp, target)
print(f"[restore] Restored {len(items)} objects into {cell_dir}")
RESTORE_PY
fi

cat << 'MONITOR_PY' > /root/fgpo_gcs_monitor.py
import fcntl, json, os, subprocess, sys, time, urllib.parse, requests

BUCKET = os.environ["FGPO_BUCKET"]
CELL_DIR = os.environ["FGPO_CELL_DIR"]
GCS_PREFIX = os.environ["FGPO_GCS_CELL_PREFIX"]
RUNNER_LOG = os.environ["FGPO_RUNNER_LOG"]
ADC_PATH = "/root/.gcs_adc.json"
DONE_FLAG = "/tmp/fgpo_monitor_done"
LOCK_PATH = "/tmp/fgpo_gcs_sync.lock"

def get_token():
    creds = json.load(open(ADC_PATH))
    r = requests.post("https://oauth2.googleapis.com/token", data={
        "client_id": creds["client_id"], "client_secret": creds["client_secret"],
        "refresh_token": creds["refresh_token"], "grant_type": "refresh_token",
    }, timeout=20)
    r.raise_for_status()
    return r.json()["access_token"]

def upload_file(token, local_path, obj_name, ctype="application/octet-stream"):
    with open(local_path, "rb") as f:
        r = requests.post(
            f"https://storage.googleapis.com/upload/storage/v1/b/{BUCKET}/o",
            params={"uploadType": "media", "name": obj_name},
            headers={"Authorization": f"Bearer {token}", "Content-Type": ctype},
            data=f,
            timeout=600,
        )
    r.raise_for_status()

def upload_bytes(token, data_bytes, obj_name, ctype="application/json"):
    r = requests.post(
        f"https://storage.googleapis.com/upload/storage/v1/b/{BUCKET}/o",
        params={"uploadType": "media", "name": obj_name},
        headers={"Authorization": f"Bearer {token}", "Content-Type": ctype},
        data=data_bytes,
        timeout=60,
    )
    r.raise_for_status()

def collect_status():
    pid_out = subprocess.run(["pgrep", "-f", "fgpo.runner"], capture_output=True, text=True).stdout.strip()
    gpu_out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout.strip()
    disk_out = subprocess.run(["df", "-BG", "--output=avail", "/content"], capture_output=True, text=True).stdout.splitlines()
    disk_avail = disk_out[-1].strip() if disk_out else ""
    mem_avail = ""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                mem_avail = f"{int(line.split()[1]) // (1024 * 1024)}G"
                break
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runner_pids": pid_out.splitlines() if pid_out else [],
        "gpu": gpu_out,
        "content_avail": disk_avail,
        "ram_avail": mem_avail,
        "cell_complete": os.path.isfile(os.path.join(CELL_DIR, "cell_summary.json")),
    }

def sync_small(token, uploaded_cache):
    candidates = [
        ("runner.log", RUNNER_LOG, "text/plain"),
        ("config.json", os.path.join(CELL_DIR, "config.json"), "application/json"),
        ("official_tasks.json", os.path.join(CELL_DIR, "official_tasks.json"), "application/json"),
        ("training.jsonl", os.path.join(CELL_DIR, "training.jsonl"), "application/x-ndjson"),
        ("costs.jsonl", os.path.join(CELL_DIR, "costs.jsonl"), "application/x-ndjson"),
        ("training_summary.json", os.path.join(CELL_DIR, "training_summary.json"), "application/json"),
        ("cell_summary.json", os.path.join(CELL_DIR, "cell_summary.json"), "application/json"),
    ]
    if os.path.isdir(CELL_DIR):
        for fn in os.listdir(CELL_DIR):
            if fn.startswith("evaluation-") and fn.endswith(".jsonl"):
                candidates.append((fn, os.path.join(CELL_DIR, fn), "application/x-ndjson"))
    for rel, path, ctype in candidates:
        if not os.path.isfile(path):
            continue
        st = os.stat(path)
        sig = (st.st_size, st.st_mtime_ns)
        if uploaded_cache.get(rel) == sig:
            continue
        upload_file(token, path, f"{GCS_PREFIX}/{rel}", ctype)
        uploaded_cache[rel] = sig

def sync_completed_artifacts(token, uploaded_cache):
    skip_dirs = {"checkpoint", "environments", "groups", "trajectories"}
    deferred_current = None
    for root, dirs, files in os.walk(CELL_DIR):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fn in sorted(files):
            if fn.startswith(".") or fn.endswith(".incoming") or fn.startswith("reference-round-"):
                continue
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, CELL_DIR).replace(os.sep, "/")
            if rel == "final_checkpoint/CURRENT":
                deferred_current = (rel, full)
                continue
            st = os.stat(full)
            sig = (st.st_size, st.st_mtime_ns)
            if uploaded_cache.get(rel) == sig:
                continue
            upload_file(token, full, f"{GCS_PREFIX}/{rel}")
            uploaded_cache[rel] = sig
    if deferred_current is not None:
        rel, full = deferred_current
        st = os.stat(full)
        sig = (st.st_size, st.st_mtime_ns)
        if uploaded_cache.get(rel) != sig:
            upload_file(token, full, f"{GCS_PREFIX}/{rel}")
            uploaded_cache[rel] = sig

def main():
    if os.path.exists(DONE_FLAG):
        os.remove(DONE_FLAG)
    uploaded_cache = {}
    saw_runner = False
    while True:
        try:
            with open(LOCK_PATH, "w") as lf:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                token = get_token()
                status = collect_status()
                if status["runner_pids"]:
                    saw_runner = True
                sync_small(token, uploaded_cache)
                upload_bytes(token, json.dumps(status, indent=2).encode(), f"{GCS_PREFIX}/live_status.json")
                if saw_runner and not status["runner_pids"]:
                    if status["cell_complete"]:
                        sync_completed_artifacts(token, uploaded_cache)
                        upload_bytes(token, b"complete\n", f"{GCS_PREFIX}/DONE", "text/plain")
                        with open(DONE_FLAG, "w") as df:
                            df.write("COMPLETE\n")
                    else:
                        with open(DONE_FLAG, "w") as df:
                            df.write("STOPPED_INCOMPLETE\n")
                    break
        except Exception as exc:
            print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] monitor error: {exc}", file=sys.stderr, flush=True)
        time.sleep(60)

if __name__ == "__main__":
    main()
MONITOR_PY

RUNNER_LOG="/content/logs/fgpo_${BENCHMARK}_s${SEED}.log"
CELL_DIR="/content/outputs/fgpo-r3/${CELL_ID}"
RESUME_ARG=""
if [ "$RESUME" = "1" ]; then RESUME_ARG="--resume"; fi

pkill -f "/root/fgpo_gcs_monitor.py" 2>/dev/null || true
FGPO_BUCKET="$BUCKET" FGPO_CELL_DIR="$CELL_DIR" FGPO_GCS_CELL_PREFIX="$GCS_CELL_PREFIX" FGPO_RUNNER_LOG="$RUNNER_LOG" \
  nohup /content/venvs/fgpo-main/bin/python -u /root/fgpo_gcs_monitor.py \
  > /root/fgpo_gcs_monitor.log 2>&1 < /dev/null &

cd /content/fgpo-r3
PYTHONPATH=/content/fgpo-r3/src \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
nohup /content/venvs/fgpo-main/bin/python -u -m fgpo.runner \
  --config /content/fgpo-r3/configs/fgpo-r3-colab-a100.json \
  --output-dir /content/outputs/fgpo-r3 \
  --suite "$SUITE" \
  --methods "$METHOD" \
  --benchmarks "$BENCHMARK" \
  --seeds "$SEED" \
  --gpu 0 \
  $RESUME_ARG \
  >> "$RUNNER_LOG" 2>&1 < /dev/null &

sleep 8
if ! pgrep -f "fgpo.runner" >/dev/null; then
  echo "[launch] ERROR: fgpo.runner exited immediately. Log tail:" >&2
  tail -n 30 "$RUNNER_LOG" >&2 || true
  exit 1
fi
ps -ef | grep -E "fgpo\.runner|fgpo_gcs_monitor\.py" | grep -v grep
REMOTE_EOF

if [ "$NO_KEEPALIVE" = "0" ]; then
  pkill -f "fgpo_keepalive_loop_${SESSION}" 2>/dev/null || true
  KEEPALIVE_PY="/tmp/fgpo_keepalive_${SESSION}.py"
  cat << 'KA_PY' > "$KEEPALIVE_PY"
import os, subprocess
done_file = "/tmp/fgpo_monitor_done"
if os.path.isfile(done_file):
    print("KEEPALIVE_DONE:" + open(done_file).read().strip(), flush=True)
else:
    pids = subprocess.run(["pgrep", "-f", "fgpo.runner"], capture_output=True, text=True).stdout.strip()
    print("KEEPALIVE_ACTIVE:" + (pids.replace("\n", ",") if pids else "monitor_flushing"), flush=True)
KA_PY

  KA_LOG="/tmp/fgpo_keepalive_${SESSION}.log"
  CAFFEINATE_BIN=""
  if command -v caffeinate >/dev/null 2>&1; then CAFFEINATE_BIN="caffeinate -i"; fi
  nohup $CAFFEINATE_BIN bash -c "
    # fgpo_keepalive_loop_${SESSION}
    export PATH=\"$HOME/.local/bin:\$PATH\"
    FAIL_STREAK=0
    while true; do
      if ! \"$COLAB_CLI\" sessions 2>/dev/null | grep -qE \"(^|\\[|[[:space:]])${SESSION}(\\]|[[:space:]]|$)\"; then
        FAIL_STREAK=\$((FAIL_STREAK + 1))
        echo \"[\$(date -u +%Y-%m-%dT%H:%M:%SZ)] session probe miss (\${FAIL_STREAK}/5)\" >> \"$KA_LOG\"
        if [ \"\$FAIL_STREAK\" -ge 5 ]; then
          echo \"[keepalive] Session '$SESSION' absent for 5 consecutive probes; exiting.\" >> \"$KA_LOG\"
          break
        fi
        sleep 30
        continue
      fi
      OUT=\$($COLAB_CLI exec -s \"$SESSION\" -f \"$KEEPALIVE_PY\" 2>&1 || true)
      echo \"[\$(date -u +%Y-%m-%dT%H:%M:%SZ)] \$OUT\" >> \"$KA_LOG\"
      if echo \"\$OUT\" | grep -q 'KEEPALIVE_DONE:'; then
        echo \"[keepalive] Monitor finished final sync; stopping '$SESSION'...\" >> \"$KA_LOG\"
        \"$COLAB_CLI\" stop -s \"$SESSION\" >> \"$KA_LOG\" 2>&1 || true
        break
      elif echo \"\$OUT\" | grep -q 'KEEPALIVE_ACTIVE:'; then
        FAIL_STREAK=0
      else
        FAIL_STREAK=\$((FAIL_STREAK + 1))
        if [ \"\$FAIL_STREAK\" -ge 10 ]; then
          echo \"[keepalive] Too many consecutive exec failures; exiting without stopping VM.\" >> \"$KA_LOG\"
          break
        fi
      fi
      sleep 45
    done
  " > /dev/null 2>&1 < /dev/null &
  echo "[launch] Started resilient keep-alive watcher (log: $KA_LOG)"
fi
