#!/usr/bin/env bash
# Shared environment, GCS REST helper, and source packaging utilities for F-GPO Colab operations.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$HOME/google-cloud-sdk/bin:$PATH"

COLAB_CLI="${COLAB_CLI:-$HOME/.local/bin/colab}"
if [ -z "${COLAB_PY:-}" ] && [ -r "$COLAB_CLI" ]; then
  _colab_shebang="$(head -n 1 "$COLAB_CLI" 2>/dev/null || true)"
  case "$_colab_shebang" in
    '#!'*)
      _colab_py_cand="${_colab_shebang#\#!}"
      _colab_py_cand="${_colab_py_cand#"${_colab_py_cand%%[![:space:]]*}"}"
      _colab_py_cand="${_colab_py_cand%%[[:space:]]*}"
      case "$_colab_py_cand" in
        */env) ;;
        *) [ -x "$_colab_py_cand" ] && COLAB_PY="$_colab_py_cand" ;;
      esac ;;
  esac
fi
COLAB_PY="${COLAB_PY:-$HOME/.local/share/uv/tools/google-colab-cli/bin/python}"
ADC_PATH="${GOOGLE_APPLICATION_CREDENTIALS:-$HOME/.config/gcloud/application_default_credentials.json}"
FGPO_GCS_BUCKET="${FGPO_GCS_BUCKET:-r032-pact-rebuild-f1be230a}"
FGPO_GCS_PREFIX="${FGPO_GCS_PREFIX:-fgpo-r3}"

if [ -z "${HTTPS_PROXY:-}" ] && [ -z "${https_proxy:-}" ]; then
  if command -v python3 >/dev/null 2>&1 && python3 -c "import socket; s=socket.create_connection(('127.0.0.1',7897),0.2); s.close()" 2>/dev/null; then
    export HTTPS_PROXY="http://127.0.0.1:7897"
    export HTTP_PROXY="http://127.0.0.1:7897"
  fi
fi

colab_validate_transport() {
  local transport="${1:-ssh}"
  case "$transport" in
    ssh|exec) ;;
    *)
      echo "[colab] Unsupported --transport '$transport' (expected 'ssh' or 'exec'; silent fallback is disabled)." >&2
      exit 2 ;;
  esac
}

colab_ssh_run() {
  local session="$1"
  shift
  local err_file rc
  err_file="$(mktemp /tmp/fgpo_ssh_err_XXXXXX)"
  if ssh -o ProxyCommand="$COLAB_CLI ssh --proxy-mode -s $session" \
      -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o LogLevel=ERROR \
      root@colab-runtime "$@" 2>"$err_file"; then
    cat "$err_file" >&2
    rm -f "$err_file"
    return 0
  else
    rc=$?
    cat "$err_file" >&2
    if grep -qE 'Already-active SSH session|HTTP 429' "$err_file" 2>/dev/null; then
      echo "[colab] SSH connection slot for '$session' is already held (HTTP 429). Refusing automatic 429 retry or silent fallback; wait for the SSH owner or pass explicit --transport exec." >&2
    fi
    rm -f "$err_file"
    return "$rc"
  fi
}

colab_exec_run() {
  local session="$1"
  local remote_cmd="${2:-bash -s}"
  local exec_timeout="${COLAB_EXEC_TIMEOUT:-300}"
  "$COLAB_PY" -c '
import base64
import binascii
import os
import re
import secrets
import sys
import threading

session, remote_cmd, exec_timeout_raw = sys.argv[1:4]
exec_timeout = float(exec_timeout_raw)
stdin_b64 = base64.b64encode(sys.stdin.buffer.read()).decode("ascii")
nonce = secrets.token_hex(12)

import jupyter_kernel_client
from jupyter_kernel_client.client import output_hook as jkc_output_hook
from colab_cli.auth import AuthProvider
from colab_cli.common import state
from colab_cli.utils import is_terminal_error

# Match the installed colab launcher ADC selection before proxy refresh.
if (not os.path.exists(os.path.expanduser("~/.config/colab-cli/token.json"))
        and os.path.exists(os.path.expanduser("~/.config/gcloud/application_default_credentials.json"))):
    state.auth_provider = AuthProvider.ADC

s = state.get_session(session, ignore_missing_session=True)
if s is None:
    raise SystemExit(f"[colab-exec] Session {session!r} not found.")

if hasattr(jupyter_kernel_client, "ColabKernelClient"):
    client_cls = jupyter_kernel_client.ColabKernelClient
    token_kw = "proxy_token"
elif hasattr(jupyter_kernel_client, "KernelClient"):
    client_cls = jupyter_kernel_client.KernelClient
    token_kw = "token"
else:
    client_cls = jupyter_kernel_client.JupyterKernelClient
    token_kw = "token"

client_kwargs = {
    "subprotocol": jupyter_kernel_client.JupyterSubprotocol.DEFAULT,
    "extra_params": {"colab-runtime-proxy-token": s.token},
    "timeout": exec_timeout,
}
if s.session_id:
    client_kwargs["session"] = s.session_id

# Connect directly in a single attempt (bypassing ColabRuntime 3-retry kernel_client
# and colab_cli history/colab.log logging).
client = client_cls(
    server_url=s.url,
    **{token_kw: s.token},
    kernel_id=s.kernel_id,
    client_kwargs=client_kwargs,
    headers={
        "X-Colab-Client-Agent": "colab-cli",
        "X-Colab-Runtime-Proxy-Token": s.token,
    },
)
client._own_kernel = False

remote_code = (
    "def _fgpo_exec_entry(_nonce, _cmd):\n"
    "    import base64, getpass, os, subprocess, sys\n"
    "    os.makedirs(\"/content\", exist_ok=True)\n"
    "    os.chdir(\"/content\")\n"
    "    _b64_in = None\n"
    "    _raw_in = b\"\"\n"
    "    _prev_tblimit = getattr(sys, \"tracebacklimit\", 1000)\n"
    "    sys.tracebacklimit = 0\n"
    "    try:\n"
    "        _b64_in = getpass.getpass(\"\")\n"
    "        _raw_in = base64.b64decode(_b64_in.encode(\"ascii\"), validate=True) if _b64_in else b\"\"\n"
    "    except Exception as _exc:\n"
    "        sys.stdout.write(f\"__FGPO_EXEC_{_nonce}_OUT_START____FGPO_EXEC_{_nonce}_OUT_END__\\n\")\n"
    "        _err_b64 = base64.b64encode(f\"exec_stdin_error:{type(_exc).__name__}\\n\".encode(\"utf-8\")).decode(\"ascii\")\n"
    "        sys.stdout.write(f\"__FGPO_EXEC_{_nonce}_ERR_START__{_err_b64}__FGPO_EXEC_{_nonce}_ERR_END__\\n\")\n"
    "        sys.stdout.write(f\"__FGPO_EXEC_{_nonce}_RC__:125\\n\")\n"
    "        sys.stdout.flush()\n"
    "        return\n"
    "    finally:\n"
    "        _b64_in = None\n"
    "        sys.tracebacklimit = _prev_tblimit\n"
    "    try:\n"
    "        _proc = subprocess.run(_cmd, shell=True, input=_raw_in, capture_output=True)\n"
    "        _out_b64 = base64.b64encode(_proc.stdout).decode(\"ascii\")\n"
    "        _err_b64 = base64.b64encode(_proc.stderr).decode(\"ascii\")\n"
    "        _rc = int(_proc.returncode)\n"
    "    finally:\n"
    "        _raw_in = b\"\"\n"
    "    sys.stdout.write(f\"__FGPO_EXEC_{_nonce}_OUT_START__{_out_b64}__FGPO_EXEC_{_nonce}_OUT_END__\\n\")\n"
    "    sys.stdout.write(f\"__FGPO_EXEC_{_nonce}_ERR_START__{_err_b64}__FGPO_EXEC_{_nonce}_ERR_END__\\n\")\n"
    "    sys.stdout.write(f\"__FGPO_EXEC_{_nonce}_RC__:{_rc}\\n\")\n"
    "    sys.stdout.flush()\n"
    f"_fgpo_exec_entry({nonce!r}, {remote_cmd!r})\n"
    "del _fgpo_exec_entry\n"
)

outputs = []
stdin_sent = False
timed_out = False
timer = None
ws = None

try:
    try:
        client.start(timeout=exec_timeout)
    except Exception as exc:
        if is_terminal_error(exc):
            state.prune_session(session)
        raise

    ws = client._manager.client
    if not ws.connection_ready.is_set():
        raise TimeoutError(f"Jupyter websocket connection for session {session!r} timed out")

    changed = False
    if not s.kernel_id and client.id:
        s.kernel_id = client.id
        changed = True
    sess_id = getattr(getattr(ws, "session", None), "session", None)
    if not s.session_id and sess_id:
        s.session_id = sess_id
        changed = True
    if changed:
        state.store.add(s)

    def _on_timeout():
        global timed_out
        timed_out = True
        try:
            ws.stop_channels()
        except Exception:
            pass

    timer = threading.Timer(exec_timeout, _on_timeout)
    timer.daemon = True
    timer.start()

    def _out_hook(msg):
        jkc_output_hook(outputs, msg)

    def _stdin_hook(_req):
        global stdin_sent, stdin_b64
        if stdin_sent:
            raise RuntimeError("Unexpected duplicate stdin request from remote kernel")
        stdin_sent = True
        payload = stdin_b64
        stdin_b64 = ""
        ws.input(payload)

    try:
        reply = client.execute_interactive(
            remote_code,
            silent=False,
            store_history=False,
            allow_stdin=True,
            stop_on_error=True,
            timeout=exec_timeout,
            output_hook=_out_hook,
            stdin_hook=_stdin_hook,
        )
    except Exception:
        if timed_out:
            raise SystemExit(f"[colab-exec] Remote execution timed out after {exec_timeout}s")
        raise
finally:
    stdin_b64 = ""
    if timer is not None:
        timer.cancel()
    client._own_kernel = False
    if ws is None:
        try:
            ws = client._manager.client if client._manager.has_kernel else None
        except Exception:
            ws = None
    if ws is not None:
        try:
            ws.stop_channels()
        except Exception:
            pass
        try:
            sock = getattr(ws, "kernel_socket", None)
            if sock is not None:
                sock.close()
        except Exception:
            pass

if timed_out:
    raise SystemExit(f"[colab-exec] Remote execution timed out after {exec_timeout}s")

reply_content = reply.get("content") if isinstance(reply, dict) else None
if not isinstance(reply_content, dict) or reply_content.get("status") != "ok":
    err_items = [o for o in outputs if o.get("output_type") == "error"]
    if err_items:
        ename = err_items[0].get("ename")
        evalue = err_items[0].get("evalue")
        sys.stderr.write(f"[colab-exec] Remote kernel error: {ename}: {evalue}\n")
    else:
        rstatus = None if reply_content is None else reply_content.get("status")
        sys.stderr.write(f"[colab-exec] Remote kernel reply status={rstatus!r}\n")
    raise SystemExit(1)

if any(o.get("output_type") == "error" for o in outputs):
    sys.stderr.write("[colab-exec] Unexpected kernel error output despite ok reply\n")
    raise SystemExit(1)

unexpected_stderr = "".join(
    o.get("text", "") for o in outputs
    if o.get("output_type") == "stream" and o.get("name") != "stdout"
)
if unexpected_stderr:
    sys.stderr.write(f"[colab-exec] Unexpected unframed kernel stream output: {unexpected_stderr[:200]}\n")
    raise SystemExit(1)

out_text = "".join(
    o.get("text", "") for o in outputs
    if o.get("output_type") == "stream" and o.get("name") == "stdout"
)
frame_re = re.compile(
    rf"__FGPO_EXEC_{nonce}_OUT_START__([A-Za-z0-9+/=]*)__FGPO_EXEC_{nonce}_OUT_END__\r?\n"
    rf"__FGPO_EXEC_{nonce}_ERR_START__([A-Za-z0-9+/=]*)__FGPO_EXEC_{nonce}_ERR_END__\r?\n"
    rf"__FGPO_EXEC_{nonce}_RC__:(-?\d+)\r?\n"
)
m = frame_re.fullmatch(out_text)
if m is None:
    sys.stderr.write("[colab-exec] Incomplete or ambiguous remote exec framing (missing/corrupted stdout/stderr/rc frame)\n")
    raise SystemExit(1)

try:
    stdout_bytes = base64.b64decode(m.group(1), validate=True)
    stderr_bytes = base64.b64decode(m.group(2), validate=True)
except (binascii.Error, ValueError) as exc:
    sys.stderr.write(f"[colab-exec] Invalid base64 in remote exec frame: {exc}\n")
    raise SystemExit(1)

if stdout_bytes:
    sys.stdout.buffer.write(stdout_bytes)
    sys.stdout.buffer.flush()
if stderr_bytes:
    sys.stderr.buffer.write(stderr_bytes)
    sys.stderr.buffer.flush()
rc = int(m.group(3))
if rc != 0:
    raise SystemExit(rc)
' "$session" "$remote_cmd" "$exec_timeout"
}

colab_remote_run() {
  local transport="$1"
  local session="$2"
  shift 2
  colab_validate_transport "$transport"
  if [ "$transport" = "ssh" ]; then
    colab_ssh_run "$session" "$@"
  else
    colab_exec_run "$session" "${1:-bash -s}"
  fi
}

colab_push_access_token() {
  local transport="$1"
  local session="$2"
  local remote_token_file="$3"
  colab_validate_transport "$transport"
  local remote_dir remote_base remote_cmd
  remote_dir="$(dirname "$remote_token_file")"
  remote_base="$(basename "$remote_token_file")"
  remote_cmd="set -eu; umask 077; mkdir -p '$remote_dir'; chmod 700 '$remote_dir'; tmp=\$(mktemp '${remote_dir}/${remote_base}.tmp.XXXXXX'); trap 'rm -f \"\$tmp\"' EXIT; cat > \"\$tmp\"; [ -s \"\$tmp\" ]; chmod 600 \"\$tmp\"; mv -f \"\$tmp\" '$remote_token_file'; trap - EXIT"
  "$COLAB_PY" -c '
import json, sys, requests
creds = json.load(open(sys.argv[1]))
resp = requests.post(
    "https://oauth2.googleapis.com/token",
    data={
        "client_id": creds["client_id"],
        "client_secret": creds["client_secret"],
        "refresh_token": creds["refresh_token"],
        "grant_type": "refresh_token",
    },
    timeout=20,
)
resp.raise_for_status()
tok = resp.json()["access_token"].strip()
if not tok or any(ch.isspace() for ch in tok):
    raise SystemExit("Invalid access_token received")
sys.stdout.write(tok)
del tok
' "$ADC_PATH" | colab_remote_run "$transport" "$session" "$remote_cmd"
  echo "[token] Atomically refreshed $remote_token_file on '$session' via $transport"
}

pack_fgpo_source() {
  local out_tar="$1"
  COPYFILE_DISABLE=1 tar \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='._*' \
    -czf "$out_tar" -C "$REPO_ROOT" src pyproject.toml configs scripts
}

gcs_py() {
  "$COLAB_PY" - "$ADC_PATH" "$FGPO_GCS_BUCKET" "$@" << 'PY'
import json, os, sys, time, urllib.parse, requests

adc_path, bucket, action = sys.argv[1:4]
args = sys.argv[4:]
creds = json.load(open(adc_path))
token_resp = requests.post(
    "https://oauth2.googleapis.com/token",
    data={
        "client_id": creds["client_id"],
        "client_secret": creds["client_secret"],
        "refresh_token": creds["refresh_token"],
        "grant_type": "refresh_token",
    },
    timeout=20,
)
token_resp.raise_for_status()
token = token_resp.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

if action == "upload":
    local_path, obj_name = args[0], args[1]
    ctype = args[2] if len(args) > 2 else "application/octet-stream"
    t0 = time.time()
    with open(local_path, "rb") as f:
        r = requests.post(
            f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o",
            params={"uploadType": "media", "name": obj_name},
            headers={**headers, "Content-Type": ctype},
            data=f,
            timeout=600,
        )
    r.raise_for_status()
    size_kb = os.path.getsize(local_path) / 1024.0
    print(f"[GCS] Uploaded gs://{bucket}/{obj_name} ({size_kb:.1f} KB) in {time.time()-t0:.2f}s")
elif action == "download":
    obj_name, local_path = args[0], args[1]
    enc = urllib.parse.quote(obj_name, safe="")
    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
    with requests.get(
        f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{enc}?alt=media",
        headers=headers,
        stream=True,
        timeout=600,
    ) as r:
        if r.status_code == 404:
            sys.exit(44)
        r.raise_for_status()
        tmp = local_path + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, local_path)
elif action == "stat":
    obj_name = args[0]
    enc = urllib.parse.quote(obj_name, safe="")
    r = requests.get(
        f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{enc}",
        headers=headers,
        timeout=20,
    )
    if r.status_code == 404:
        sys.exit(44)
    r.raise_for_status()
    meta = r.json()
    print(json.dumps({"name": meta.get("name"), "size": meta.get("size"), "updated": meta.get("updated")}))
elif action == "list":
    prefix = args[0]
    page_token = None
    while True:
        params = {"prefix": prefix}
        if page_token:
            params["pageToken"] = page_token
        r = requests.get(
            f"https://storage.googleapis.com/storage/v1/b/{bucket}/o",
            headers=headers,
            params=params,
            timeout=30,
        )
        r.raise_for_status()
        payload = r.json()
        for item in payload.get("items", []):
            print(f"{item['name']}\t{item.get('size', 0)}\t{item.get('updated', '')}")
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
else:
    raise SystemExit(f"Unknown gcs_py action: {action}")
PY
}
