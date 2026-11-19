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

fgpo_emit_history_archive_py() {
  cat << 'FGPO_HISTORY_ARCHIVE_PY_EOF'
import argparse
from contextlib import contextmanager
import fcntl
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import time
from urllib.parse import urlsplit
import uuid

ARCHIVE_FORMAT = "fgpo-history-bundle-v2"
SUPPORTED_MANIFEST_FORMATS = frozenset({"fgpo-history-bundle-v1", ARCHIVE_FORMAT})
RECLAIMABLE_FILE_ROOTS = frozenset({"costs.jsonl"})
RECLAIMABLE_DIR_ROOTS = frozenset({"environments", "groups", "trajectories"})
REFERENCE_ROUND_RE = re.compile(r"^reference-round-[0-9]+\.pt$")
SEED_DIR_RE = re.compile(r"^seed-([0-9]+)$")
TRACKED_COST_ROLES = frozenset({
    "rollout.group", "rollout.step_native", "rollout.decode",
    "reference.single_round_activation", "credit.peer_values",
    "ppo.backward", "ppo.adamw", "evaluation.rollout",
})


def fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json_durable(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.incoming.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    try:
        with pending.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, path)
        fsync_dir(path.parent)
    finally:
        pending.unlink(missing_ok=True)


def is_allowed_reclaimable_file_root(name: str) -> bool:
    return name in RECLAIMABLE_FILE_ROOTS or bool(REFERENCE_ROUND_RE.fullmatch(name))


def is_allowed_reclaimable_root(name: str) -> bool:
    return is_allowed_reclaimable_file_root(name) or name in RECLAIMABLE_DIR_ROOTS


def normalize_archive_member(raw_name: str, *, is_dir: bool) -> tuple[str, tuple[str, ...]]:
    trimmed = raw_name.rstrip("/")
    if not trimmed:
        raise ValueError(f"Empty archive member name: {raw_name!r}")
    posix = PurePosixPath(trimmed)
    if posix.is_absolute() or any(part in ("", ".", "..") for part in posix.parts) or str(posix) != trimmed:
        raise ValueError(f"Non-canonical or escaping archive member path: {raw_name!r}")
    parts = posix.parts
    root = parts[0]
    if is_allowed_reclaimable_file_root(root):
        if len(parts) != 1 or is_dir:
            raise ValueError(f"File-only reclaimable member must be a single regular file: {raw_name!r}")
    elif root in RECLAIMABLE_DIR_ROOTS:
        pass
    else:
        raise ValueError(f"Forbidden non-reclaimable archive member: {raw_name!r}")
    return str(posix), parts


@contextmanager
def sibling_seed_lock(cell: Path):
    if cell.is_symlink() or cell.parent.is_symlink():
        raise RuntimeError(f"Refusing lock acquisition on symlinked cell path: {cell}")
    lock_path = cell.parent / f".{cell.name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield lock_path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def ensure_fgpo_importable(cfg_path: Path | None = None) -> None:
    try:
        import fgpo  # noqa: F401
        return
    except ImportError:
        pass
    candidates = []
    if cfg_path is not None:
        candidates.append(cfg_path.resolve().parent.parent / "src")
    candidates.append(Path("/content/fgpo-r3/src"))
    for cand in candidates:
        if (cand / "fgpo").is_dir() and str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
            return


def summarize_costs_stream(text_stream) -> dict:
    role_counts, role_seconds, groups_by_round, recent = {}, {}, {}, []
    total_events = in_tok = out_tok = fwd_tok = env_steps = 0
    sess_elapsed = last_gid = last_role = None
    for raw_ln in text_stream:
        ln = raw_ln.strip()
        if not ln:
            continue
        try:
            ev = json.loads(ln)
        except Exception:
            continue
        total_events += 1
        role = ev.get("role", "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1
        role_seconds[role] = role_seconds.get(role, 0.0) + float(ev.get("seconds") or 0.0)
        in_tok += int(ev.get("input_tokens") or 0)
        out_tok += int(ev.get("output_tokens") or 0)
        fwd_tok += int(ev.get("forward_tokens") or 0)
        env_steps += int(ev.get("env_steps") or 0)
        if ev.get("session_elapsed_seconds") is not None:
            sess_elapsed = ev.get("session_elapsed_seconds")
        det = ev.get("details") or {}
        gid = det.get("group_id")
        if gid:
            last_gid = gid
        if role in TRACKED_COST_ROLES:
            last_role = role
        if role == "rollout.group" and isinstance(gid, str):
            parts = gid.split(":")
            if len(parts) >= 4:
                groups_by_round[parts[2]] = groups_by_round.get(parts[2], 0) + 1
        recent.append({
            "event_index": ev.get("event_index"),
            "kind": ev.get("kind"),
            "role": role,
            "seconds": ev.get("seconds"),
            "env_steps": ev.get("env_steps"),
            "input_tokens": ev.get("input_tokens"),
            "output_tokens": ev.get("output_tokens"),
            "group_id": gid,
            "outcome": det.get("outcome"),
            "elapsed": ev.get("session_elapsed_seconds"),
        })
        if len(recent) > 20:
            recent.pop(0)
    return {
        "total_events": total_events,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "forward_tokens": fwd_tok,
        "env_steps": env_steps,
        "session_elapsed_seconds": sess_elapsed,
        "role_counts": role_counts,
        "role_seconds": role_seconds,
        "last_group_id": last_gid,
        "last_phase_role": last_role,
        "groups_by_round": groups_by_round,
        "recent": recent,
    }


def list_present_reclaimable_members(cell: Path) -> list[str]:
    found = []
    costs_p = cell / "costs.jsonl"
    if costs_p.is_symlink():
        raise RuntimeError(f"Refusing symlinked reclaimable file: {costs_p}")
    if costs_p.exists():
        if not costs_p.is_file():
            raise RuntimeError(f"Expected regular file at {costs_p}")
        found.append("costs.jsonl")
    for dname in sorted(RECLAIMABLE_DIR_ROOTS):
        dp = cell / dname
        if dp.is_symlink():
            raise RuntimeError(f"Refusing symlinked reclaimable directory: {dp}")
        if dp.exists():
            if not dp.is_dir():
                raise RuntimeError(f"Expected directory at {dp}")
            found.append(dname)
    for entry in sorted(cell.iterdir(), key=lambda p: p.name):
        if REFERENCE_ROUND_RE.fullmatch(entry.name):
            if entry.is_symlink() or not entry.is_file():
                raise RuntimeError(f"Refusing non-regular reference artifact: {entry}")
            found.append(entry.name)
    return found


def tree_regular_bytes_and_count(cell: Path, roots: list[str]) -> tuple[int, int]:
    total_bytes = 0
    member_count = 0
    for name in roots:
        target = cell / name
        if target.is_symlink():
            raise RuntimeError(f"Symlink encountered inside reclaimable root: {target}")
        if target.is_file():
            total_bytes += target.stat().st_size
            member_count += 1
        elif target.is_dir():
            member_count += 1
            for dirpath, dirnames, filenames in os.walk(target, followlinks=False):
                dp = Path(dirpath)
                for dn in dirnames:
                    subd = dp / dn
                    if subd.is_symlink():
                        raise RuntimeError(f"Symlink directory encountered inside {target}: {subd}")
                    member_count += 1
                for fn in filenames:
                    subf = dp / fn
                    if subf.is_symlink() or not subf.is_file():
                        raise RuntimeError(f"Non-regular file encountered inside {target}: {subf}")
                    total_bytes += subf.stat().st_size
                    member_count += 1
    return total_bytes, member_count


def build_reclaimable_bundle_once(cell: Path, members: list[str], dest_bundle: Path) -> dict:
    """Single-pass PAX+zstd creation linked non-overwritingly via os.link."""
    if dest_bundle.exists():
        raise FileExistsError(f"Refusing to overwrite existing bundle {dest_bundle}")
    if "costs.jsonl" not in members or not (cell / "costs.jsonl").is_file():
        raise RuntimeError(f"Cannot build {dest_bundle}: raw costs.jsonl is missing in {cell}")
    for name in members:
        if not is_allowed_reclaimable_root(name):
            raise RuntimeError(f"Refusing to archive non-reclaimable member {name!r}")
    uncompressed_bytes, member_count = tree_regular_bytes_and_count(cell, members)
    tmp_bundle = cell / f".history_bundle.tar.zst.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    proc = None
    try:
        proc = subprocess.Popen(
            ["zstd", "-3", "-q", "-f", "-o", str(tmp_bundle)],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.stdin is not None
        with tarfile.open(fileobj=proc.stdin, mode="w|", format=tarfile.PAX_FORMAT) as tf:
            for name in members:
                tf.add(str(cell / name), arcname=name, recursive=True)
        proc.stdin.close()
        proc.stdin = None
        stderr_bytes = proc.stderr.read() if proc.stderr is not None else b""
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"zstd compression failed (rc={rc}): {stderr_bytes.decode('utf-8', 'replace').strip()}")
        fsync_file(tmp_bundle)
        bundle_bytes = tmp_bundle.stat().st_size
        if bundle_bytes <= 0:
            raise RuntimeError("Compressed bundle is empty")
        os.link(tmp_bundle, dest_bundle)
        tmp_bundle.unlink()
        fsync_dir(cell)
        return {
            "bundle_bytes": bundle_bytes,
            "member_roots": list(members),
            "member_count": member_count,
            "uncompressed_bytes": uncompressed_bytes,
        }
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        tmp_bundle.unlink(missing_ok=True)


def inspect_existing_bundle_for_manifest_recovery(bundle_path: Path, *, need_costs_summary: bool) -> dict:
    """Used ONLY when an existing bundle lacks its local manifest (crash recovery)."""
    if bundle_path.is_symlink() or not bundle_path.is_file():
        raise ValueError(f"Bundle must be a regular non-symlink file: {bundle_path}")
    bundle_bytes = bundle_path.stat().st_size
    if bundle_bytes <= 0:
        raise ValueError(f"Existing bundle is empty: {bundle_path}")
    roots = []
    seen_roots = set()
    member_count = 0
    uncompressed_bytes = 0
    has_costs = False
    costs_summary = None
    proc = subprocess.Popen(["zstd", "-dc", "--", str(bundle_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert proc.stdout is not None
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
            for member in tf:
                if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                    raise ValueError(f"Existing bundle contains non-regular/link entry: {member.name!r}")
                norm_path, parts = normalize_archive_member(member.name, is_dir=member.isdir())
                if parts[0] not in seen_roots:
                    seen_roots.add(parts[0])
                    roots.append(parts[0])
                member_count += 1
                if member.isfile():
                    uncompressed_bytes += int(member.size)
                    if norm_path == "costs.jsonl":
                        has_costs = True
                        if need_costs_summary:
                            fobj = tf.extractfile(member)
                            if fobj is not None:
                                costs_summary = summarize_costs_stream(
                                    io.TextIOWrapper(fobj, encoding="utf-8", errors="replace")
                                )
        while proc.stdout.read(1024 * 1024):
            pass
        stderr_bytes = proc.stderr.read() if proc.stderr is not None else b""
        rc = proc.wait()
        if rc != 0:
            raise ValueError(f"Existing bundle decompression failed (rc={rc}): {stderr_bytes.decode('utf-8', 'replace').strip()}")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    if not has_costs:
        raise ValueError(f"Existing bundle {bundle_path} lacks required costs.jsonl member")
    return {
        "bundle_bytes": bundle_bytes,
        "member_roots": roots,
        "member_count": member_count,
        "uncompressed_bytes": uncompressed_bytes,
        "costs_summary": costs_summary,
    }


def verify_finalized_cell_identity(cell: Path, summary: dict, marker: dict, *, backend: str,
                                   suite: str, benchmark: str, method: str, seed: int) -> str:
    if (summary.get("status") != "complete"
            or summary.get("finalization_stage") != "finalized"
            or summary.get("suite") != suite
            or summary.get("benchmark") != benchmark
            or summary.get("method") != method
            or summary.get("seed") != seed
            or not marker.get("terminal_export_completed")):
        raise ValueError(f"Cell {suite}/{benchmark}/{method}/seed-{seed} is not durably finalized")
    for req_name in ("config.json", "official_tasks.json", "training.jsonl", "training_summary.json", "cell_summary.json"):
        req_p = cell / req_name
        if req_p.is_symlink() or not req_p.is_file():
            raise ValueError(f"Missing authoritative in-place file {req_p}")
    eval_map = summary.get("evaluation") or {}
    if not eval_map:
        raise ValueError(f"Cell summary lacks official evaluation splits in {cell}")
    for split, ev_item in eval_map.items():
        if not ev_item.get("dataset_complete"):
            raise ValueError(f"Evaluation split {split!r} is incomplete in {cell}")
        for ev_rel in (f"evaluation-{split}.jsonl", f"{split}/evaluation_summary.json", f"{split}/evaluation_records.json"):
            ev_p = cell / ev_rel
            if ev_p.is_symlink() or not ev_p.is_file():
                raise ValueError(f"Missing authoritative evaluation artifact {ev_p}")
    fin_ptr_p = cell / "final_checkpoint" / "CURRENT"
    if fin_ptr_p.is_symlink() or not fin_ptr_p.is_file():
        raise ValueError(f"Missing authoritative final_checkpoint/CURRENT in {cell}")
    fin_ptr = json.loads(fin_ptr_p.read_text(encoding="utf-8"))
    final_snapshot = fin_ptr.get("snapshot")
    expected_snap = marker.get("snapshot") or (summary.get("completed_checkpoint") or {}).get("snapshot")
    if not final_snapshot or (expected_snap and final_snapshot != expected_snap):
        raise ValueError(f"Final snapshot mismatch in {cell}: {final_snapshot!r} vs {expected_snap!r}")
    if backend == "local":
        snap_dir = cell / "final_checkpoint" / "snapshots" / str(final_snapshot)
        if (not (snap_dir / "metadata.json").is_file()
                or not (snap_dir / "state.pt").is_file() or not (snap_dir / "model").is_dir()):
            raise ValueError(f"Incomplete local final_checkpoint snapshot {snap_dir}")
    return str(final_snapshot)


def parse_cell_rel(rel_str: str) -> tuple[str, str, str, int]:
    posix = PurePosixPath(rel_str.strip("/"))
    if len(posix.parts) != 4 or any(p in ("", ".", "..") for p in posix.parts):
        raise SystemExit(f"Invalid cell relative path {rel_str!r} (expected SUITE/BENCHMARK/METHOD/seed-N)")
    suite, benchmark, method, sd = posix.parts
    if suite not in ("comparison", "ablation") or benchmark not in ("alfworld", "appworld"):
        raise SystemExit(f"Invalid suite/benchmark in {rel_str!r}")
    m = SEED_DIR_RE.fullmatch(sd)
    if m is None:
        raise SystemExit(f"Invalid seed directory {sd!r} in {rel_str!r}")
    return suite, benchmark, method, int(m.group(1))


def cmd_pack(args: argparse.Namespace) -> int:
    out_root = Path(args.output_dir).resolve()
    cfg_path = Path(args.config).resolve()
    if not cfg_path.is_file():
        raise SystemExit(f"[history-archive] Missing config file: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    backend = cfg.get("checkpoint_backend", "local")
    if backend not in ("local", "gcs"):
        raise SystemExit(f"[history-archive] Unsupported checkpoint_backend: {backend!r}")
    ensure_fgpo_importable(cfg_path)

    explicit_cell = bool(args.cell)
    candidates: list[tuple[Path, str, str, str, int]] = []
    if explicit_cell:
        suite, benchmark, method, seed = parse_cell_rel(args.cell)
        raw_cell = out_root / suite / benchmark / method / f"seed-{seed}"
        if raw_cell.is_symlink() or not raw_cell.is_dir():
            raise SystemExit(f"[history-archive] Missing or symlinked cell directory: {raw_cell}")
        candidates.append((raw_cell, suite, benchmark, method, seed))
    elif out_root.is_dir():
        for entry in sorted(out_root.glob("*/*/*/seed-*")):
            if entry.is_symlink() or not entry.is_dir():
                continue
            rel_parts = entry.relative_to(out_root).parts
            if len(rel_parts) != 4:
                continue
            suite, benchmark, method, sd = rel_parts
            m = SEED_DIR_RE.fullmatch(sd)
            if suite in ("comparison", "ablation") and benchmark in ("alfworld", "appworld") and m is not None:
                candidates.append((entry, suite, benchmark, method, int(m.group(1))))

    packed_count = 0
    for cell, suite, benchmark, method, seed in candidates:
        rel = f"{suite}/{benchmark}/{method}/seed-{seed}"
        try:
            with sibling_seed_lock(cell):
                summary_path = cell / "cell_summary.json"
                marker_path = cell / ".finalization_complete.json"
                if not summary_path.is_file() or not marker_path.is_file():
                    if explicit_cell:
                        raise SystemExit(f"[history-archive] Refused: {rel} is not finalized yet")
                    print(f"[history-archive] Skip active/unfinalized cell: {rel}")
                    continue
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                if summary.get("status") != "complete" or summary.get("finalization_stage") != "finalized" or not marker.get("terminal_export_completed"):
                    if explicit_cell:
                        raise SystemExit(f"[history-archive] Refused: {rel} finalization incomplete")
                    print(f"[history-archive] Skip incomplete cell: {rel}")
                    continue
                final_snapshot = verify_finalized_cell_identity(
                    cell, summary, marker, backend=backend,
                    suite=suite, benchmark=benchmark, method=method, seed=seed,
                )

                bundle_path = cell / "history_bundle.tar.zst"
                manifest_path = cell / "history_bundle.manifest.json"
                if bundle_path.is_symlink() or manifest_path.is_symlink():
                    raise RuntimeError(f"Refusing symlinked bundle/manifest in {cell}")

                existing_man = None
                if manifest_path.is_file():
                    existing_man = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if existing_man.get("format") not in SUPPORTED_MANIFEST_FORMATS:
                        raise RuntimeError(f"Unsupported manifest format in {manifest_path}")

                # Fast complete check: existing bundle + complete manifest already committed.
                # Note (seq=885 #2): Reusing an existing bundle NEVER reclaims local files (such as
                # unpacked/restored costs.jsonl); reclamation happens ONLY in the same lock session
                # that newly created history_bundle.tar.zst!
                if bundle_path.exists():
                    if not bundle_path.is_file() or bundle_path.stat().st_size <= 0:
                        raise RuntimeError(f"Invalid existing bundle at {bundle_path}")
                    act_bytes = bundle_path.stat().st_size
                    if (existing_man is not None
                            and (cell / "costs_summary.json").is_file()
                            and existing_man.get("format") == ARCHIVE_FORMAT
                            and existing_man.get("snapshot") == final_snapshot
                            and int(existing_man.get("bundle_bytes", -1)) == act_bytes
                            and (backend == "local" or (
                                bool(existing_man.get("gcs_bundle_name"))
                                and bool(existing_man.get("gcs_bundle_generation"))
                                and bool(existing_man.get("gcs_manifest_name"))
                                and bool(existing_man.get("gcs_manifest_generation"))
                            ))):
                        print(f"[history-archive] Already packed: {rel} ({act_bytes/(1024**2):.2f} MiB)")
                        continue

                # GCS readiness check before building or publishing; preserve raw files if unavailable.
                gcs_client = None
                gcs_bucket = None
                gcs_obj_prefix = None
                if backend == "gcs":
                    from fgpo.checkpoint_gcs import GCSCheckpoint, GCSClient, StorageError
                    from fgpo.protocol import resolve_method_config
                    parts = urlsplit(cfg["checkpoint_gcs_uri"])
                    if parts.scheme != "gs" or not parts.netloc:
                        raise RuntimeError(f"Invalid checkpoint_gcs_uri for {rel}")
                    gcs_bucket = parts.netloc
                    gcs_obj_prefix = parts.path.lstrip("/").rstrip("/") + f"/archives/{suite}/{benchmark}/{method}/seed-{seed}"
                    try:
                        gcs_client = GCSClient(gcs_bucket)
                        _ = gcs_client._token()
                        mcfg = resolve_method_config(dict(cfg, suite=suite, benchmark=benchmark, method=method, seed=seed))
                        gcs_st = GCSCheckpoint(mcfg, cell / "checkpoint").status()
                    except StorageError as err:
                        if args.defer_on_missing_gcs_token:
                            print(f"[history-archive] Deferred {rel}: GCS unavailable ({err}); raw source files preserved intact")
                            continue
                        raise SystemExit(f"[history-archive] Aborted {rel} before any reclamation: GCS unavailable ({err})") from None
                    if (gcs_st.get("status") != "complete"
                            or not gcs_st.get("terminal_export_completed")
                            or gcs_st.get("snapshot") != final_snapshot
                            or not gcs_st.get("final_generation")):
                        raise RuntimeError(f"GCS FINAL status mismatch for {rel}: {gcs_st}")

                costs_path = cell / "costs.jsonl"
                costs_summary_path = cell / "costs_summary.json"
                created_new_bundle = False
                packed_inputs: list[str] = []

                if bundle_path.exists():
                    act_bytes = bundle_path.stat().st_size
                    if existing_man is not None and int(existing_man.get("bundle_bytes", -1)) == act_bytes and existing_man.get("member_roots"):
                        bundle_info = {
                            "bundle_bytes": act_bytes,
                            "member_roots": list(existing_man["member_roots"]),
                            "member_count": int(existing_man.get("member_count") or len(existing_man["member_roots"])),
                            "uncompressed_bytes": int(existing_man.get("uncompressed_member_bytes") or 0),
                            "costs_summary": None,
                        }
                        if not costs_summary_path.is_file() and not costs_path.is_file():
                            bundle_info = inspect_existing_bundle_for_manifest_recovery(bundle_path, need_costs_summary=True)
                    else:
                        bundle_info = inspect_existing_bundle_for_manifest_recovery(
                            bundle_path,
                            need_costs_summary=(not costs_summary_path.is_file() and not costs_path.is_file()),
                        )
                    if not costs_summary_path.is_file():
                        if costs_path.is_file():
                            with costs_path.open("r", encoding="utf-8", errors="replace") as fh:
                                write_json_durable(costs_summary_path, summarize_costs_stream(fh))
                        elif bundle_info.get("costs_summary") is not None:
                            write_json_durable(costs_summary_path, bundle_info["costs_summary"])
                else:
                    present_members = list_present_reclaimable_members(cell)
                    if "costs.jsonl" not in present_members:
                        raise RuntimeError(f"Refusing to pack {rel}: neither history_bundle.tar.zst nor raw costs.jsonl exists")
                    with costs_path.open("r", encoding="utf-8", errors="replace") as fh:
                        write_json_durable(costs_summary_path, summarize_costs_stream(fh))
                    bundle_info = build_reclaimable_bundle_once(cell, present_members, bundle_path)
                    created_new_bundle = True
                    packed_inputs = list(present_members)

                archive_id = (
                    existing_man["archive_id"]
                    if (existing_man is not None
                        and existing_man.get("snapshot") == final_snapshot
                        and int(existing_man.get("bundle_bytes", -1)) == bundle_info["bundle_bytes"]
                        and isinstance(existing_man.get("archive_id"), str)
                        and existing_man["archive_id"])
                    else f"{final_snapshot[:12]}-{uuid.uuid4().hex[:8]}"
                )

                reclaimed_bytes_val = (
                    bundle_info["uncompressed_bytes"]
                    if created_new_bundle
                    else int((existing_man or {}).get("reclaimed_local_bytes") or 0)
                )

                manifest_payload = {
                    "format": ARCHIVE_FORMAT,
                    "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "protocol": summary["protocol"],
                    "suite": suite,
                    "benchmark": benchmark,
                    "method": method,
                    "seed": seed,
                    "snapshot": final_snapshot,
                    "archive_id": archive_id,
                    "bundle_file": "history_bundle.tar.zst",
                    "bundle_bytes": bundle_info["bundle_bytes"],
                    "member_roots": bundle_info["member_roots"],
                    "member_count": bundle_info["member_count"],
                    "uncompressed_member_bytes": bundle_info["uncompressed_bytes"],
                    "reclaimed_local_bytes": reclaimed_bytes_val,
                    "net_saved_bytes": max(0, reclaimed_bytes_val - bundle_info["bundle_bytes"]),
                    "gcs_bucket": None,
                    "gcs_bundle_name": None,
                    "gcs_bundle_uri": None,
                    "gcs_bundle_generation": None,
                    "gcs_manifest_name": None,
                    "gcs_manifest_uri": None,
                    "gcs_manifest_generation": None,
                }

                # Publish immutable remote objects (ifGenerationMatch=0) and record exact GCS generations
                # BEFORE reclaiming any local raw source file; never delete other objects under seed prefix.
                if backend == "gcs":
                    assert gcs_client is not None and gcs_bucket is not None and gcs_obj_prefix is not None
                    remote_bundle_name = f"{gcs_obj_prefix}/history_bundle.{archive_id}.tar.zst"
                    remote_manifest_name = f"{gcs_obj_prefix}/history_bundle.{archive_id}.manifest.json"

                    if (existing_man is not None
                            and existing_man.get("gcs_bundle_name") == remote_bundle_name
                            and existing_man.get("gcs_bundle_generation")):
                        bundle_gen = str(existing_man["gcs_bundle_generation"])
                    else:
                        existing_b = gcs_client.metadata(remote_bundle_name, optional=True)
                        if existing_b is not None:
                            if int(existing_b["size"]) != bundle_info["bundle_bytes"]:
                                raise RuntimeError(f"Remote bundle size mismatch on {remote_bundle_name}")
                            bundle_desc = gcs_client.descriptor(existing_b)
                        else:
                            bundle_desc = gcs_client.upload_file(remote_bundle_name, bundle_path)
                        bundle_gen = str(bundle_desc["generation"])

                    manifest_payload["gcs_bucket"] = gcs_bucket
                    manifest_payload["gcs_bundle_name"] = remote_bundle_name
                    manifest_payload["gcs_bundle_uri"] = f"gs://{gcs_bucket}/{remote_bundle_name}"
                    manifest_payload["gcs_bundle_generation"] = bundle_gen
                    manifest_payload["gcs_manifest_name"] = remote_manifest_name
                    manifest_payload["gcs_manifest_uri"] = f"gs://{gcs_bucket}/{remote_manifest_name}"

                    if (existing_man is not None
                            and existing_man.get("gcs_manifest_name") == remote_manifest_name
                            and existing_man.get("gcs_manifest_generation")):
                        man_gen = str(existing_man["gcs_manifest_generation"])
                    else:
                        existing_m = gcs_client.metadata(remote_manifest_name, optional=True)
                        if existing_m is not None:
                            man_desc = gcs_client.descriptor(existing_m)
                        else:
                            man_desc = gcs_client.write_json(remote_manifest_name, manifest_payload, generation=0)
                        man_gen = str(man_desc["generation"])
                    manifest_payload["gcs_manifest_generation"] = man_gen

                write_json_durable(manifest_path, manifest_payload)

                # Reclaim ONLY the exact inputs newly archived & durably published in THIS lock session!
                actually_reclaimed = 0
                if created_new_bundle:
                    if not bundle_path.is_file() or bundle_path.stat().st_size != bundle_info["bundle_bytes"] or not manifest_path.is_file():
                        raise RuntimeError(f"Local bundle/manifest missing before reclamation in {rel}")
                    if backend == "gcs" and (not manifest_payload["gcs_bundle_generation"] or not manifest_payload["gcs_manifest_generation"]):
                        raise RuntimeError(f"Remote GCS generation missing before reclamation in {rel}")
                    for name in packed_inputs:
                        if not is_allowed_reclaimable_root(name):
                            raise RuntimeError(f"Refusing to reclaim non-allowlisted path: {name!r}")
                        target = cell / name
                        if target.is_symlink():
                            raise RuntimeError(f"Refusing to reclaim symlink: {target}")
                        if target.is_file():
                            actually_reclaimed += target.stat().st_size
                            target.unlink()
                        elif target.is_dir():
                            b_sz, _ = tree_regular_bytes_and_count(cell, [name])
                            actually_reclaimed += b_sz
                            shutil.rmtree(target)
                    fsync_dir(cell)
                    Path(f"/tmp/fgpo_mon_cache_{method}_{benchmark}_{seed}.json").unlink(missing_ok=True)

                packed_count += 1
                print(
                    f"[history-archive] Packed {rel} -> {bundle_info['bundle_bytes']/(1024**2):.2f} MiB "
                    f"(new_bundle={created_new_bundle}, reclaimed={actually_reclaimed/(1024**2):.2f} MiB, "
                    f"gcs_gen={manifest_payload['gcs_bundle_generation']})"
                )
        except BlockingIOError:
            if explicit_cell:
                raise SystemExit(f"[history-archive] Refused: {rel} (.{cell.name}.lock) is held by an active process") from None
            print(f"[history-archive] Skip active locked cell: {rel}")
            continue

    print(f"[history-archive] Pack complete (updated_cells={packed_count})")
    return 0


def cmd_unpack(args: argparse.Namespace) -> int:
    """Single-pass streaming unpack under .seed-N.lock: creates ONLY missing allowlisted raw members."""
    out_root = Path(args.output_dir).resolve()
    suite, benchmark, method, seed = parse_cell_rel(args.cell)
    cell = out_root / suite / benchmark / method / f"seed-{seed}"
    if cell.is_symlink() or not cell.is_dir():
        raise SystemExit(f"[history-archive] Missing or symlinked cell directory: {cell}")

    requested_roots = None
    if args.members:
        requested_roots = set()
        for item in args.members.split(","):
            tok = item.strip()
            if not tok:
                continue
            if not is_allowed_reclaimable_root(tok):
                raise SystemExit(f"[history-archive] Forbidden unpack member root: {tok!r}")
            requested_roots.add(tok)

    try:
        with sibling_seed_lock(cell):
            bundle_path = cell / "history_bundle.tar.zst"
            manifest_path = cell / "history_bundle.manifest.json"
            if bundle_path.is_symlink() or not bundle_path.is_file():
                raise SystemExit(f"[history-archive] Missing regular bundle: {bundle_path}")
            bundle_bytes = bundle_path.stat().st_size
            if bundle_bytes <= 0:
                raise SystemExit(f"[history-archive] Empty bundle: {bundle_path}")
            if manifest_path.exists():
                if manifest_path.is_symlink() or not manifest_path.is_file():
                    raise SystemExit(f"[history-archive] Invalid manifest: {manifest_path}")
                man = json.loads(manifest_path.read_text(encoding="utf-8"))
                if man.get("format") not in SUPPORTED_MANIFEST_FORMATS:
                    raise SystemExit(f"[history-archive] Unsupported manifest format: {man.get('format')!r}")
                if int(man.get("bundle_bytes", -1)) != bundle_bytes:
                    raise SystemExit(f"[history-archive] Bundle byte length mismatch vs manifest in {cell}")
                if (man.get("suite"), man.get("benchmark"), man.get("method"), man.get("seed")) != (suite, benchmark, method, seed):
                    raise SystemExit(f"[history-archive] Manifest cell identity mismatch in {cell}")

            cell_resolved = cell.resolve()
            created_files = 0
            preserved_existing = 0
            proc = subprocess.Popen(["zstd", "-dc", "--", str(bundle_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                assert proc.stdout is not None
                with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
                    for member in tf:
                        if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                            raise RuntimeError(f"Refusing non-regular archive entry: {member.name!r}")
                        norm_path, parts = normalize_archive_member(member.name, is_dir=member.isdir())
                        root = parts[0]
                        if requested_roots is not None and root not in requested_roots:
                            continue
                        cur = cell
                        for seg in parts[:-1]:
                            cur = cur / seg
                            if cur.is_symlink():
                                raise RuntimeError(f"Refusing symlink directory in cell: {cur}")
                        target = cell / norm_path
                        if target.is_symlink():
                            raise RuntimeError(f"Refusing symlink target in cell: {target}")
                        if member.isdir():
                            if not target.exists():
                                target.mkdir(parents=True, exist_ok=True)
                                resolved_t = target.resolve()
                                if resolved_t != cell_resolved and cell_resolved not in resolved_t.parents:
                                    raise RuntimeError(f"Directory escaped cell root: {norm_path!r}")
                            elif not target.is_dir():
                                raise RuntimeError(f"Cannot create directory over existing file: {target}")
                        else:
                            if target.exists():
                                preserved_existing += 1
                                continue
                            target.parent.mkdir(parents=True, exist_ok=True)
                            resolved_parent = target.parent.resolve()
                            if resolved_parent != cell_resolved and cell_resolved not in resolved_parent.parents:
                                raise RuntimeError(f"File parent escaped cell root: {norm_path!r}")
                            fobj = tf.extractfile(member)
                            if fobj is None:
                                raise RuntimeError(f"Cannot read archive file member: {norm_path!r}")
                            tmp_f = target.with_name(f".{target.name}.unpack.{os.getpid()}.{uuid.uuid4().hex[:8]}")
                            try:
                                written = 0
                                with tmp_f.open("wb") as out_fh:
                                    while True:
                                        chunk = fobj.read(1024 * 1024)
                                        if not chunk:
                                            break
                                        out_fh.write(chunk)
                                        written += len(chunk)
                                    out_fh.flush()
                                    os.fsync(out_fh.fileno())
                                if written != member.size:
                                    raise RuntimeError(f"Short read while extracting {norm_path!r}")
                                try:
                                    os.link(tmp_f, target)
                                    created_files += 1
                                except FileExistsError:
                                    preserved_existing += 1
                            finally:
                                tmp_f.unlink(missing_ok=True)
                while proc.stdout.read(1024 * 1024):
                    pass
                stderr_bytes = proc.stderr.read() if proc.stderr is not None else b""
                rc = proc.wait()
                if rc != 0:
                    raise RuntimeError(f"zstd decompression failed during unpack (rc={rc}): {stderr_bytes.decode('utf-8', 'replace').strip()}")
            finally:
                if proc is not None and proc.poll() is None:
                    proc.kill()
                    proc.wait()
            fsync_dir(cell)
            print(f"[history-archive] Unpack {args.cell}: created_files={created_files}, preserved_existing={preserved_existing}")
            return 0
    except BlockingIOError:
        raise SystemExit(f"[history-archive] Refused: {args.cell} (.{cell.name}.lock) is held by an active process") from None


def copy_live_reclaimables(cell_dir: Path, stage_dir: Path) -> list[str]:
    """Copy all live raw materials actually present on runtime so stop/pull never drops unpacked data."""
    copied = []
    for name in list_present_reclaimable_members(cell_dir):
        src = cell_dir / name
        dst = stage_dir / name
        if src.is_file():
            if not dst.exists() or dst.stat().st_size < src.stat().st_size:
                shutil.copy2(src, dst)
            copied.append(name)
        elif src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, symlinks=False)
            copied.append(name)
    return copied


def cmd_ensure_stage_bundle(args: argparse.Namespace) -> int:
    """Stage live raw materials AND/OR generation-pinned real history bundle for off-instance retention."""
    cell_dir = Path(args.cell_dir).resolve()
    stage_dir = Path(args.stage_dir).resolve()
    cfg_path = Path(args.config).resolve() if args.config else None
    stage_dir.mkdir(parents=True, exist_ok=True)

    for extra_name in ("reporting_summary.json", "costs_summary.json", ".finalization_complete.json"):
        src_extra = cell_dir / extra_name
        if src_extra.is_file() and not src_extra.is_symlink():
            shutil.copy2(src_extra, stage_dir / extra_name)

    live_roots = []
    if args.copy_live_reclaimables:
        live_roots = copy_live_reclaimables(cell_dir, stage_dir)

    src_bundle = cell_dir / "history_bundle.tar.zst"
    src_manifest = cell_dir / "history_bundle.manifest.json"
    dst_bundle = stage_dir / "history_bundle.tar.zst"
    dst_manifest = stage_dir / "history_bundle.manifest.json"

    if src_bundle.is_file() and not src_bundle.is_symlink():
        if not dst_bundle.is_file():
            shutil.copy2(src_bundle, dst_bundle)
    if src_manifest.is_file() and not src_manifest.is_symlink():
        if not dst_manifest.is_file():
            shutil.copy2(src_manifest, dst_manifest)

    expected_bundle_bytes = None
    if dst_manifest.is_file():
        man = json.loads(dst_manifest.read_text(encoding="utf-8"))
        expected_bundle_bytes = int(man.get("bundle_bytes", -1))
        if expected_bundle_bytes <= 0:
            raise SystemExit(f"[history-archive] Invalid bundle_bytes in {dst_manifest}")
        if not dst_bundle.is_file():
            gcs_bucket = man.get("gcs_bucket")
            gcs_name = man.get("gcs_bundle_name")
            gcs_gen = man.get("gcs_bundle_generation")
            gcs_uri = man.get("gcs_bundle_uri") or man.get("gcs_archive_uri")
            if (not gcs_bucket or not gcs_name) and gcs_uri:
                parts = urlsplit(gcs_uri)
                if parts.scheme == "gs" and parts.netloc:
                    gcs_bucket = parts.netloc
                    gcs_name = parts.path.lstrip("/")
            if not gcs_bucket or not gcs_name or not gcs_gen:
                raise SystemExit(
                    f"[history-archive] Dangling {src_manifest} without local history_bundle.tar.zst or immutable GCS generation"
                )
            ensure_fgpo_importable(cfg_path)
            from fgpo.checkpoint_gcs import GCSClient
            client = GCSClient(gcs_bucket)
            client.download_file(
                {"name": gcs_name, "generation": str(gcs_gen), "size": expected_bundle_bytes},
                dst_bundle,
            )
        if dst_bundle.stat().st_size != expected_bundle_bytes:
            raise SystemExit(
                f"[history-archive] Staged bundle size ({dst_bundle.stat().st_size}) != manifest ({expected_bundle_bytes})"
            )
    elif dst_bundle.is_file():
        expected_bundle_bytes = dst_bundle.stat().st_size
        if expected_bundle_bytes <= 0:
            raise SystemExit(f"[history-archive] Empty staged bundle {dst_bundle}")

    # Materialize costs.jsonl in stage_dir if needed (e.g. for stop verification when local costs.jsonl was reclaimed)
    if args.materialize_costs and not (stage_dir / "costs.jsonl").is_file() and dst_bundle.is_file():
        proc = subprocess.Popen(["zstd", "-dc", "--", str(dst_bundle)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            assert proc.stdout is not None
            with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
                for member in tf:
                    norm_path, _ = normalize_archive_member(member.name, is_dir=member.isdir())
                    if norm_path == "costs.jsonl" and member.isfile():
                        fobj = tf.extractfile(member)
                        assert fobj is not None
                        tmp_c = stage_dir / ".costs.jsonl.incoming"
                        with tmp_c.open("wb") as out_fh:
                            shutil.copyfileobj(fobj, out_fh, length=1024 * 1024)
                            out_fh.flush()
                            os.fsync(out_fh.fileno())
                        os.replace(tmp_c, stage_dir / "costs.jsonl")
                        break
            while proc.stdout.read(1024 * 1024):
                pass
            proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    if args.require_retention and not live_roots and not dst_bundle.is_file() and not (stage_dir / "costs.jsonl").is_file():
        raise SystemExit(f"[history-archive] Refused: neither live raw materials nor history_bundle.tar.zst staged in {stage_dir}")

    write_json_durable(stage_dir / ".stage_retention_receipt.json", {
        "live_reclaimable_roots": live_roots,
        "has_bundle": dst_bundle.is_file(),
        "expected_bundle_bytes": expected_bundle_bytes,
    })
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="fgpo_history_archive")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pack = sub.add_parser("pack")
    p_pack.add_argument("--output-dir", required=True)
    p_pack.add_argument("--config", required=True)
    p_pack.add_argument("--cell", default="")
    p_pack.add_argument("--defer-on-missing-gcs-token", action="store_true")

    p_unpack = sub.add_parser("unpack")
    p_unpack.add_argument("--output-dir", required=True)
    p_unpack.add_argument("--cell", required=True)
    p_unpack.add_argument("--members", default="")

    p_stage = sub.add_parser("ensure-stage-bundle")
    p_stage.add_argument("--cell-dir", required=True)
    p_stage.add_argument("--stage-dir", required=True)
    p_stage.add_argument("--config", default="")
    p_stage.add_argument("--copy-live-reclaimables", action="store_true")
    p_stage.add_argument("--materialize-costs", action="store_true")
    p_stage.add_argument("--require-retention", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "pack":
        return cmd_pack(args)
    if args.cmd == "unpack":
        return cmd_unpack(args)
    if args.cmd == "ensure-stage-bundle":
        return cmd_ensure_stage_bundle(args)
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
FGPO_HISTORY_ARCHIVE_PY_EOF
}

fgpo_local_history_archive() {
  local py_bin="${FGPO_ARCHIVE_PYTHON:-python3}"
  fgpo_emit_history_archive_py | "$py_bin" - "$@"
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
