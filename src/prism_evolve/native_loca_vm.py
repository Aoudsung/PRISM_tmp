"""Pure user-space QEMU VM full-state snapshot backend (savevm/loadvm) for LOCA-Bench.

All MCP server processes, child processes, login sessions, PDF search caches,
live SQLite connections, and service databases execute inside an isolated QEMU
virtual machine backed by a per-environment qcow2 image. Host-side private
groundtruth and evaluator code never enter the guest VM.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
from typing import Any, Mapping, Sequence


def _json_line(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def _pack_dir_b64(source: Path) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        if source.exists():
            for item in sorted(source.iterdir()):
                tar.add(str(item), arcname=item.name, recursive=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _unpack_dir_b64(data_b64: str, target: Path, *, clean: bool = True) -> None:
    target.mkdir(parents=True, exist_ok=True)
    if clean:
        for existing in list(target.iterdir()):
            if existing.is_dir() and not existing.is_symlink():
                shutil.rmtree(existing)
            else:
                existing.unlink(missing_ok=True)
    if not data_b64:
        return
    raw = base64.b64decode(data_b64.encode("ascii"))
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        tar.extractall(path=str(target), filter="fully_trusted")


class _QMPSocket:
    """Minimal synchronous QMP client over a Unix domain socket."""

    def __init__(self, sock_path: Path, timeout: float = 60.0) -> None:
        self.sock_path = sock_path
        self.timeout = timeout
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(timeout)
        self._sock.connect(str(sock_path))
        self._file = self._sock.makefile("r", encoding="utf-8")
        greeting = self._read_msg()
        if "QMP" not in greeting:
            raise RuntimeError(f"Unexpected QMP greeting: {greeting}")
        self.execute("qmp_capabilities")

    def _read_msg(self) -> dict[str, Any]:
        while True:
            line = self._file.readline()
            if not line:
                raise EOFError("QMP socket closed unexpectedly.")
            msg = json.loads(line)
            if "event" in msg:
                continue
            return msg

    def execute(self, command: str, arguments: Mapping[str, Any] | None = None) -> Any:
        packet: dict[str, Any] = {"execute": command}
        if arguments is not None:
            packet["arguments"] = dict(arguments)
        self._sock.sendall(_json_line(packet))
        reply = self._read_msg()
        if "error" in reply:
            raise RuntimeError(f"QMP command {command} failed: {reply['error']}")
        return reply.get("return")

    def hmp(self, command_line: str) -> str:
        out = self.execute("human-monitor-command", {"command-line": command_line})
        return str(out or "")

    def close(self) -> None:
        try:
            self._file.close()
        finally:
            self._sock.close()


class LOCAVMSession:
    """Manage one isolated QEMU VM instance and its live MCP tool processes."""

    @staticmethod
    def _stage_in_shm(src: Path) -> Path:
        shm_root = Path("/dev/shm")
        if not shm_root.is_dir():
            return src
        try:
            usage = shutil.disk_usage(shm_root)
            if usage.free < 16 * (1024 ** 3):
                return src
            cache_dir = shm_root / f"prism-loca-vm-cache-{os.getuid()}"
            cache_dir.mkdir(parents=True, exist_ok=True)
            dst = cache_dir / src.name
            src_stat = src.stat()
            if dst.is_file():
                dst_stat = dst.stat()
                if dst_stat.st_size == src_stat.st_size and int(dst_stat.st_mtime) == int(src_stat.st_mtime):
                    return dst
            tmp_dst = cache_dir / f".{src.name}.{os.getpid()}.tmp"
            shutil.copy2(src, tmp_dst)
            os.replace(tmp_dst, dst)
            return dst
        except OSError:
            return src

    def __init__(self, vm_config: Mapping[str, Any], work_dir: Path) -> None:
        self.vm_config = dict(vm_config)
        self.work_dir = work_dir.resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.qemu_bin = Path(self.vm_config["qemu_system_executable"]).expanduser().resolve()
        self.qemu_img = Path(self.vm_config["qemu_img_executable"]).expanduser().resolve()
        self.kernel = Path(self.vm_config["kernel_path"]).expanduser().resolve()
        self.initrd = Path(self.vm_config["initrd_path"]).expanduser().resolve()
        self.runtime_sqsh = Path(self.vm_config["runtime_sqsh_path"]).expanduser().resolve()
        self.base_qcow2 = Path(self.vm_config["base_qcow2_path"]).expanduser().resolve()
        self.boot_snapshot = str(self.vm_config.get("boot_snapshot_tag", "boot_ready"))
        self.memory_mb = int(self.vm_config.get("memory_mb", 2048))
        self.smp = int(self.vm_config.get("smp", 4))
        for path in (self.qemu_bin, self.qemu_img, self.kernel, self.initrd, self.runtime_sqsh, self.base_qcow2):
            if not path.is_file():
                raise FileNotFoundError(f"Missing LOCA QEMU VM artifact: {path}")
        self.kernel = self._stage_in_shm(self.kernel)
        self.initrd = self._stage_in_shm(self.initrd)
        self.runtime_sqsh = self._stage_in_shm(self.runtime_sqsh)
        self.base_qcow2 = self._stage_in_shm(self.base_qcow2)

        # Keep Unix socket paths and per-episode VM state image in /tmp.
        self._sock_dir = Path(tempfile.mkdtemp(prefix="loca-vm-"))
        self.qmp_sock_path = self._sock_dir / "qmp.sock"
        self.serial_sock_path = self._sock_dir / "serial.sock"
        self.mux_sock_path = self._sock_dir / "mux.sock"
        self.overlay_qcow2 = self._sock_dir / "instance.qcow2"

        self._process: subprocess.Popen[bytes] | None = None
        self._qmp: _QMPSocket | None = None
        self._serial_sock: socket.socket | None = None
        self._serial_file: Any = None
        self._serial_lock = threading.Lock()
        self._mux_server: socket.socket | None = None
        self._mux_thread: threading.Thread | None = None
        self._closed = False
        self._snapshot_seq = 0
        self._agent_workspace: Path | None = None
        self._artifact_dir: Path | None = None
        self._databases: dict[str, Path] = {}

    def _start_vm(self) -> None:
        self._stop_vm()
        if self.overlay_qcow2.exists():
            self.overlay_qcow2.unlink()
        shutil.copyfile(self.base_qcow2, self.overlay_qcow2)
        cmd = [
            str(self.qemu_bin),
            "-machine", "q35",
            "-accel", "tcg,thread=multi",
            "-cpu", "max",
            "-smp", str(self.smp),
            "-m", str(self.memory_mb),
            "-nographic",
            "-no-reboot",
            "-nic", "none",
            "-kernel", str(self.kernel),
            "-initrd", str(self.initrd),
            "-append", "console=ttyS0 quiet panic=-1 rdinit=/init",
            "-drive", f"file={self.runtime_sqsh},if=virtio,format=raw,readonly=on",
            "-drive", f"file={self.overlay_qcow2},if=virtio,cache=unsafe,format=qcow2",
            "-chardev", f"socket,id=locach,path={self.serial_sock_path},server=on,wait=off",
            "-device", "virtio-serial-pci",
            "-device", "virtserialport,chardev=locach,name=loca.0",
            "-qmp", f"unix:{self.qmp_sock_path},server=on,wait=off",
        ]
        if self.boot_snapshot:
            cmd.extend(["-loadvm", self.boot_snapshot])
        log_file = (self.work_dir / "qemu.log").open("ab")
        self._process = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=log_file, stderr=log_file,
        )
        deadline = time.monotonic() + 45.0
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(f"QEMU exited prematurely with code {self._process.returncode}")
            if self.qmp_sock_path.exists() and self.serial_sock_path.exists():
                break
            time.sleep(0.05)
        else:
            raise TimeoutError("Timed out waiting for QEMU QMP and virtio-serial sockets.")

        self._qmp = _QMPSocket(self.qmp_sock_path, timeout=60.0)
        self._serial_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._serial_sock.settimeout(120.0)
        self._serial_sock.connect(str(self.serial_sock_path))
        self._serial_file = self._serial_sock.makefile("r", encoding="utf-8")

        # Verify guest agent readiness via ping.
        pong = self._guest_rpc({"op": "ping"})
        if not pong.get("ok"):
            raise RuntimeError(f"Guest daemon ping failed: {pong}")
        self._ensure_mux_server()

    def _ensure_mux_server(self) -> None:
        if self._mux_server is not None:
            return
        if self.mux_sock_path.exists():
            self.mux_sock_path.unlink()
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(self.mux_sock_path))
        server.listen(32)
        self._mux_server = server

        def accept_loop() -> None:
            while not self._closed:
                try:
                    conn, _ = server.accept()
                except OSError:
                    break
                threading.Thread(target=self._handle_mux_client, args=(conn,), daemon=True).start()

        self._mux_thread = threading.Thread(target=accept_loop, daemon=True)
        self._mux_thread.start()

    def _handle_mux_client(self, conn: socket.socket) -> None:
        with conn:
            reader = conn.makefile("r", encoding="utf-8")
            header_line = reader.readline()
            if not header_line:
                return
            header = json.loads(header_line)
            service = str(header["service"])
            for line in reader:
                stripped = line.rstrip("\r\n")
                if not stripped:
                    continue
                try:
                    parsed = json.loads(stripped)
                except ValueError:
                    parsed = {}
                is_tool_call = isinstance(parsed, dict) and parsed.get("method") == "tools/call"
                if is_tool_call:
                    self.sync_artifacts_to_guest()
                reply = self._guest_rpc({"op": "mcp_line", "service": service, "line": stripped})
                if not reply.get("ok"):
                    err_msg = reply.get("error", "Guest MCP forwarding failed")
                    raise RuntimeError(f"Guest MCP service {service} error: {err_msg}")
                if is_tool_call:
                    self.sync_workspace_to_host()
                out_line = str(reply.get("line", ""))
                conn.sendall((out_line + "\n").encode("utf-8"))

    def _guest_rpc(self, request: Mapping[str, Any]) -> dict[str, Any]:
        with self._serial_lock:
            if self._serial_sock is None or self._serial_file is None:
                raise RuntimeError("LOCA VM serial channel is not connected.")
            req_id = uuid.uuid4().hex
            packet = dict(request, id=req_id)
            self._serial_sock.sendall(_json_line(packet))
            while True:
                line = self._serial_file.readline()
                if not line:
                    raise EOFError("Guest virtio-serial channel returned EOF.")
                try:
                    reply = json.loads(line)
                except ValueError:
                    continue
                if isinstance(reply, dict) and reply.get("id") == req_id:
                    return reply

    def prepare_episode(
        self,
        service_specs: Mapping[str, Mapping[str, Any]],
        *,
        agent_workspace: Path,
        artifact_dir: Path,
        databases: Mapping[str, Path],
        code_mounts: Sequence[tuple[Path, Path]],
    ) -> dict[str, Any]:
        self._agent_workspace = agent_workspace.resolve()
        self._artifact_dir = artifact_dir.resolve()
        self._databases = {name: path.resolve() for name, path in databases.items()}
        self._start_vm()

        mounts_payload = [
            {"target": str(target), "tar_b64": _pack_dir_b64(source), "ro": True}
            for source, target in code_mounts
        ]
        db_payload = {
            str(path): _pack_dir_b64(path)
            for path in self._databases.values()
        }
        init_reply = self._guest_rpc({
            "op": "init_episode",
            "agent_workspace": str(self._agent_workspace),
            "agent_workspace_b64": _pack_dir_b64(self._agent_workspace),
            "artifact_dir": str(self._artifact_dir),
            "artifact_dir_b64": _pack_dir_b64(self._artifact_dir),
            "databases_b64": db_payload,
            "code_mounts": mounts_payload,
            "services": dict(service_specs),
        })
        if not init_reply.get("ok"):
            raise RuntimeError(f"Failed to initialize episode inside LOCA VM: {init_reply}")

        wrapped: dict[str, Any] = {}
        for name in service_specs:
            wrapped[name] = {
                "command": sys.executable,
                "args": [
                    "-u", "-m", "prism_evolve.native_loca_vm",
                    "--mcp-stdio-proxy",
                    "--socket", str(self.mux_sock_path),
                    "--service", str(name),
                ],
                "env": {
                    "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                    "HOME": "/tmp",
                },
            }
        return {"mcpServers": wrapped}

    def sync_artifacts_to_guest(self) -> None:
        if self._artifact_dir is None or not self._artifact_dir.exists():
            return
        reply = self._guest_rpc({
            "op": "sync_to_guest",
            "target": str(self._artifact_dir),
            "tar_b64": _pack_dir_b64(self._artifact_dir),
            "clean": False,
        })
        if not reply.get("ok"):
            raise RuntimeError(f"Failed to sync artifacts to guest VM: {reply}")

    def sync_workspace_to_host(self) -> None:
        if self._agent_workspace is None:
            return
        targets = [str(self._agent_workspace), *(str(p) for p in self._databases.values())]
        reply = self._guest_rpc({"op": "sync_from_guest", "targets": targets})
        if not reply.get("ok"):
            raise RuntimeError(f"Failed to sync workspace from guest VM: {reply}")
        archives = reply.get("archives", {})
        _unpack_dir_b64(archives.get(str(self._agent_workspace), ""), self._agent_workspace, clean=True)
        for db_path in self._databases.values():
            if str(db_path) in archives:
                _unpack_dir_b64(archives[str(db_path)], db_path, clean=True)

    def save_vm_snapshot(self) -> str:
        if self._qmp is None:
            raise RuntimeError("LOCA VM is not running; cannot save snapshot.")
        self._snapshot_seq += 1
        tag = f"loca_snap_{self._snapshot_seq}"
        with self._serial_lock:
            out = self._qmp.hmp(f"savevm {tag}")
            if "Error" in out or "error" in out.lower():
                raise RuntimeError(f"QEMU savevm {tag} failed: {out.strip()}")
            snapshots = self._qmp.hmp("info snapshots")
            if tag not in snapshots:
                raise RuntimeError(f"QEMU snapshot {tag} not found in info snapshots: {snapshots}")
        return tag

    def restore_vm_snapshot(self, tag: str) -> None:
        if self._qmp is None:
            raise RuntimeError("LOCA VM is not running; cannot restore snapshot.")
        with self._serial_lock:
            out = self._qmp.hmp(f"loadvm {tag}")
            if "Error" in out or "error" in out.lower() or "does not exist" in out:
                raise RuntimeError(f"QEMU loadvm {tag} failed: {out.strip()}")
        pong = self._guest_rpc({"op": "ping"})
        if not pong.get("ok"):
            raise RuntimeError(f"Guest VM failed ping after loadvm {tag}: {pong}")
        self.sync_workspace_to_host()

    def _stop_vm(self) -> None:
        for obj in (self._serial_file, self._serial_sock, self._qmp):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
        self._serial_file = None
        self._serial_sock = None
        self._qmp = None
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5)
            self._process = None

    def close(self) -> None:
        self._closed = True
        if self._mux_server is not None:
            try:
                self._mux_server.close()
            except Exception:
                pass
            self._mux_server = None
        self._stop_vm()
        if self.overlay_qcow2.exists():
            self.overlay_qcow2.unlink(missing_ok=True)
        shutil.rmtree(self._sock_dir, ignore_errors=True)


def _run_mcp_stdio_proxy(sock_path: Path, service: str) -> None:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(str(sock_path))
    reader = sock.makefile("r", encoding="utf-8")
    sock.sendall(_json_line({"service": service}))
    for line in sys.stdin:
        stripped = line.rstrip("\r\n")
        if not stripped:
            continue
        sock.sendall((stripped + "\n").encode("utf-8"))
        reply = reader.readline()
        if not reply:
            break
        if reply.rstrip("\r\n"):
            sys.stdout.write(reply)
            sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcp-stdio-proxy", action="store_true", required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--service", type=str, required=True)
    args = parser.parse_args()
    _run_mcp_stdio_proxy(args.socket, args.service)
