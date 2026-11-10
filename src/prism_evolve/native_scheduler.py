"""Remote scheduling for immutable, model-resident native-v1 worker bundles.

Each worker loads its backbone once. The server coordinator claims complete
cohorts once; POSIX-local GPU UUID quotas authorize shared resident workers.
The scheduler never changes scientific results.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import socket
import tempfile
import time
from typing import Any, Mapping, Sequence


_CELL_STATUSES = {"pending", "running", "done", "failed", "blocked"}
_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
_NVML_INITIALIZED = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clock() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC)


def _encoded(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(_encoded(value))
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _journal(run_dir: Path, event: str, **fields: Any) -> None:
    with (run_dir / "scheduler_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": _now(), "event": event, **fields}, ensure_ascii=False,
                                allow_nan=False, sort_keys=True) + "\n")


@dataclass(frozen=True)
class GPUInventory:
    index: int
    uuid: str
    name: str
    total_mib: int
    used_mib: int
    utilization_percent: int
    compute_pids: tuple[int, ...]
    native_worker_pids: tuple[int, ...] = ()
    process_memory_mib: tuple[tuple[int, int | None], ...] = ()

    @property
    def free_mib(self) -> int:
        return self.total_mib - self.used_mib

    @property
    def external_pids(self) -> tuple[int, ...]:
        return tuple(pid for pid in self.compute_pids if pid not in self.native_worker_pids)


def _native_worker_pids(pids: Sequence[int]) -> tuple[int, ...]:
    """Inspect only the native-runner marker; never retain process arguments."""
    native = []
    for pid in pids:
        try:
            arguments = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"prism_evolve.native_runner" in arguments:
            native.append(pid)
    return tuple(sorted(native))


def _inventory_record(gpu: GPUInventory) -> dict[str, Any]:
    return {**asdict(gpu), "free_mib": gpu.free_mib, "external_pids": list(gpu.external_pids)}


def nvidia_inventory() -> list[GPUInventory]:
    """Read current driver counters through one persistent NVML session."""
    import pynvml
    global _NVML_INITIALIZED
    if not _NVML_INITIALIZED:
        pynvml.nvmlInit()
        _NVML_INITIALIZED = True
    selected = os.environ.get("PRISM_NATIVE_NVML_GPU_IDS", "")
    selected_ids = {int(value) for value in selected.split(",")} if selected else None
    result = []
    for index in range(pynvml.nvmlDeviceGetCount()):
        if selected_ids is not None and index not in selected_ids:
            continue
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        uuid = pynvml.nvmlDeviceGetUUID(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        pids = tuple(sorted({int(process.pid) for process in processes}))
        process_memory = tuple(sorted((int(process.pid),
                                      int(process.usedGpuMemory // 1048576)
                                      if isinstance(process.usedGpuMemory, int)
                                      and 0 <= process.usedGpuMemory <= memory.total else None)
                                     for process in processes))
        result.append(GPUInventory(
            index, uuid.decode() if isinstance(uuid, bytes) else uuid,
            name.decode() if isinstance(name, bytes) else name,
            int(memory.total // 1048576), int(memory.used // 1048576),
            int(utilization.gpu), pids, _native_worker_pids(pids), process_memory,
        ))
    if not result:
        raise RuntimeError("NVIDIA inventory returned no devices.")
    return sorted(result, key=lambda gpu: gpu.index)


def eligible_gpus(
    inventory: Sequence[GPUInventory], allowed_gpu_ids: Sequence[int] | None,
    *, occupied: set[int] | None = None, required_memory_mib: int = 0,
    safety_margin_mib: int = 8192, idle_memory_mib: int = 1024,
    share_occupied_gpus: bool = False,
) -> list[GPUInventory]:
    """Exclusive inventory filter; coordinated sharing uses native_gpu_quota.reserve."""
    allowed = set(allowed_gpu_ids) if allowed_gpu_ids is not None else None
    occupied = occupied or set()
    return [gpu for gpu in inventory
            if (allowed is None or gpu.index in allowed) and gpu.index not in occupied
            and not gpu.native_worker_pids
            and (share_occupied_gpus or (not gpu.compute_pids and gpu.used_mib <= idle_memory_mib))
            and gpu.free_mib >= required_memory_mib + safety_margin_mib]


def _safe_id(value: str, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"{name} must be a safe filename component.")
    return value


def _prepare_bundles(bundles: Sequence[Mapping[str, Any]], run_dir: Path) -> list[dict[str, Any]]:
    prepared, seen_cells, seen_bundles = [], set(), set()
    for raw in bundles:
        bundle = dict(raw)
        for field in ("backbone_id", "model_path", "manifest_path"):
            if not isinstance(bundle.get(field), str) or not bundle[field].strip():
                raise ValueError(f"Every bundle needs {field}.")
        manifest = Path(bundle["manifest_path"]).resolve(strict=True)
        manifest_id = json.loads(manifest.read_text())["manifest_id"]
        cells = bundle.get("cells")
        if not isinstance(cells, list) or not cells:
            raise ValueError("Each resident backbone bundle needs at least one cell.")
        for cell in cells:
            if not isinstance(cell, Mapping):
                raise ValueError("A cell must be an object.")
            cell_id = _safe_id(cell.get("cell_id"), "cell_id")
            if cell_id in seen_cells:
                raise ValueError(f"Repeated cell_id across bundles: {cell_id}")
            seen_cells.add(cell_id)
        if bundle.get("run_dir") is not None and Path(bundle["run_dir"]).resolve() != run_dir:
            raise ValueError("A bundle cannot write into another run directory.")
        bundle.update(run_dir=str(run_dir), manifest_path=str(manifest), manifest_id=manifest_id)
        for field in ("estimated_seconds", "required_memory_mib", "parameters_b"):
            value = bundle.get(field, 0)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{field} must be finite and nonnegative.")
        bundle_id = _safe_id(bundle["bundle_id"], "bundle_id")
        if bundle_id in seen_bundles:
            raise ValueError(f"Repeated bundle_id: {bundle_id}")
        seen_bundles.add(bundle_id)
        bundle["bundle_id"] = bundle_id
        path = run_dir / "bundles" / f"{bundle_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _encoded(bundle)
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        prepared.append({"payload": bundle, "path": path, "bundle_id": bundle_id})
    return prepared


def _cell_receipts(bundle: Mapping[str, Any], run_dir: Path) -> dict[str, dict[str, Any]]:
    receipts = {}
    for cell in bundle["cells"]:
        cell_id = cell["cell_id"]
        path = run_dir / "cells" / cell_id / "status.json"
        if not path.exists():
            receipts[cell_id] = {"status": "pending"}
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if (record.get("cell_id") != cell_id or record.get("manifest_id") != bundle["manifest_id"]
                or record.get("status") not in _CELL_STATUSES):
            raise ValueError(f"Cell receipt has another identity or invalid status: {path}")
        receipts[cell_id] = record
    return receipts


def _priority(prepared: Mapping[str, Any], jobs: Mapping[str, Any]) -> tuple[float, float, str]:
    bundle = prepared["payload"]
    historical = [job["wall_seconds"] / max(1, len(job["cells"])) for job in jobs.values()
                  if job.get("status") == "done" and job.get("backbone_id") == bundle["backbone_id"]
                  and job.get("wall_seconds", 0) > 0]
    estimate = (sum(historical) / len(historical) * len(bundle["cells"]) if historical
                else float(bundle.get("priority_weight", bundle.get("estimated_seconds", 0))))
    return (-estimate, -float(bundle.get("parameters_b", 0)), bundle["bundle_id"])


@contextmanager
def _scheduler_lock(run_dir: Path):
    with (run_dir / ".scheduler.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This run already has an active scheduler.") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_schedule(
    bundles: list[dict[str, Any]], run_dir: Path, python_executable: str,
    allowed_gpu_ids: list[int] | None = None, max_workers: int = 3, poll_seconds: float = 10,
    *, safety_margin_mib: int = 8192, idle_memory_mib: int = 1024,
    cpu_threads: int = 4, monitor_seconds: float = 30,
    max_idle_wait_seconds: float | None = None,
    expected_hostname: str = "ubuntu", shutdown_grace_seconds: float = 30,
    share_occupied_gpus: bool = False, memory_fraction: float = 0.85,
    coordination: Mapping[str, Any] | None = None, max_workers_per_gpu: int = 1,
    worker_memory_mib: int | None = None, allocator_memory_mib: int | None = None,
) -> dict[str, Any]:
    """Claim each complete cohort once; resource quotas are keyed by GPU UUID."""
    if coordination is None or coordination.get("enabled") is not True:
        raise ValueError("New schedules require the explicit server-side unique-claim coordinator.")
    if not allowed_gpu_ids or len(set(allowed_gpu_ids)) != len(allowed_gpu_ids):
        raise ValueError("A coordinated node needs distinct explicit physical GPU IDs.")
    if max_idle_wait_seconds is not None:
        raise ValueError("Global pending work cannot be silently expired by a local idle-wait bound.")
    from .native_dispatch import run_coordinated_schedule
    return run_coordinated_schedule(
        bundles, Path(run_dir), python_executable, allowed_gpu_ids=allowed_gpu_ids,
        max_workers=max_workers, poll_seconds=poll_seconds, cpu_threads=cpu_threads,
        expected_hostname=expected_hostname, memory_fraction=memory_fraction,
        share_occupied_gpus=share_occupied_gpus, safety_margin_mib=safety_margin_mib,
        max_workers_per_gpu=max_workers_per_gpu, worker_memory_mib=worker_memory_mib,
        allocator_memory_mib=allocator_memory_mib, shutdown_grace_seconds=shutdown_grace_seconds,
    )
