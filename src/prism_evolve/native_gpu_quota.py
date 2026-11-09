"""POSIX-local GPU UUID reservations shared by all native schedulers on a host."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import re
import socket
from typing import Any
import uuid

from .native_coordination import atomic_json, immutable_json, now


ROOT = Path("/tmp/prism-native-gpu-quotas-v1")


def process_identity(pid: int) -> str | None:
    path = Path('/proc') / str(pid) / 'stat'
    return path.read_text().rsplit(') ', 1)[-1].split()[19] if path.exists() else None


@contextmanager
def quota_lock(gpu_uuid: str):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", gpu_uuid):
        raise ValueError("Unsafe GPU UUID.")
    ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (ROOT / f"{gpu_uuid}.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        path = ROOT / f"{gpu_uuid}.json"
        state = json.loads(path.read_text()) if path.exists() else {"gpu_uuid": gpu_uuid, "reservations": {}}
        yield path, state


def reserve(gpu: Any, *, max_workers: int, worker_memory_mib: int | None,
            allocator_memory_mib: int | None, safety_margin_mib: int,
            required_memory_mib: int, memory_fraction: float, share_occupied_gpus: bool) -> dict[str, Any] | None:
    with quota_lock(gpu.uuid) as (path, state):
        reservations = state["reservations"]
        if any(process_identity(row['owner_pid']) != row['owner_start'] for row in reservations.values()):
            raise RuntimeError("A GPU quota owner disappeared; the reservation is retained for explicit diagnosis.")
        known = {row["worker_pid"] for row in reservations.values() if row.get("worker_pid") is not None}
        unknown = set(gpu.native_worker_pids) - known
        if unknown:
            raise RuntimeError(f"GPU {gpu.uuid} has native workers outside the UUID quota ledger: {sorted(unknown)}")
        if len(reservations) >= max_workers:
            return None
        mode = "shared" if worker_memory_mib is not None else "exclusive"
        if any(row["mode"] != mode or row["max_workers"] != max_workers for row in reservations.values()):
            raise ValueError("Concurrent schedulers disagree on the GPU sharing contract.")
        if any(row['safety_margin_mib'] != safety_margin_mib for row in reservations.values()):
            raise ValueError('Concurrent GPU reservations disagree on the safety margin.')
        if not share_occupied_gpus and gpu.compute_pids:
            return None
        if mode == "exclusive":
            if max_workers != 1 or reservations:
                return None
            if gpu.free_mib < required_memory_mib + safety_margin_mib:
                return None
            quota = {"mode": mode, "reservation_mib": math.floor(gpu.total_mib * memory_fraction),
                     "allocator_fraction": memory_fraction, "allocator_mib": None}
        else:
            if any(row['quota']['reservation_mib'] != worker_memory_mib
                   or row['quota']['allocator_mib'] != allocator_memory_mib for row in reservations.values()):
                raise ValueError('Concurrent shared workers disagree on their fixed memory quotas.')
            if not 0 < allocator_memory_mib < worker_memory_mib:
                raise ValueError("A shared reservation needs a positive allocator cap and CUDA overhead allowance.")
            if max_workers * worker_memory_mib + safety_margin_mib > gpu.total_mib:
                raise ValueError("The planned shared worker caps and safety margin exceed this physical GPU.")
            process_memory = dict(gpu.process_memory_mib)
            if any(process_memory.get(pid) is None for pid in known if pid in gpu.compute_pids):
                raise RuntimeError("NVML cannot account for an existing native worker's memory.")
            external = max(0, gpu.used_mib - sum(value for pid, value in process_memory.items()
                                               if pid in known and value is not None))
            reserved = sum(row["quota"]["reservation_mib"] for row in reservations.values())
            if reserved + worker_memory_mib + external + safety_margin_mib > gpu.total_mib:
                return None
            quota = {"mode": mode, "reservation_mib": worker_memory_mib,
                     "allocator_mib": allocator_memory_mib,
                     "allocator_fraction": allocator_memory_mib / gpu.total_mib,
                     "external_used_mib_at_admission": external}
        reservation_id = uuid.uuid4().hex
        lease = {"reservation_id": reservation_id, "gpu": gpu.index, "gpu_uuid": gpu.uuid,
                 "host": socket.gethostname(), "owner_pid": os.getpid(), "worker_pid": None,
                 "owner_start": process_identity(os.getpid()),
                 "max_workers": max_workers, "mode": mode, "quota": quota, "created_at": now(),
                 "inventory_at_admission": {"used_mib": gpu.used_mib, "total_mib": gpu.total_mib,
                                             "utilization_percent": gpu.utilization_percent,
                                             "process_memory_mib": list(gpu.process_memory_mib)},
                 "safety_margin_mib": safety_margin_mib}
        reservations[reservation_id] = lease
        atomic_json(path, state)
        return lease


def bind(lease: dict[str, Any], execution_id: str, manifest_id: str) -> None:
    with quota_lock(lease["gpu_uuid"]) as (path, state):
        row = state["reservations"][lease["reservation_id"]]
        if row["owner_pid"] != os.getpid() or "execution_id" in row:
            raise ValueError("A GPU reservation cannot be rebound.")
        row.update(execution_id=execution_id, manifest_id=manifest_id)
        atomic_json(path, state)


def worker_start(manifest: dict[str, Any], manifest_id: str, gpu_uuid: str) -> None:
    grant = manifest["execution"]
    session = grant["session_id"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", session):
        raise ValueError("Unsafe coordination session identity.")
    if grant["host"] != socket.gethostname() or grant["gpu_uuid"] != gpu_uuid:
        raise ValueError("The worker is on a different physical host/GPU from its grant.")
    authorization = json.loads((Path(manifest["run_dir"]) / "authorization.json").read_text())
    if (authorization["execution_id"] != grant["execution_id"]
            or authorization["manifest_id"] != manifest_id
            or authorization["bundle_id"] != grant["bundle_id"]):
        raise ValueError("The worker manifest has no matching server-side execution authorization.")
    if not math.isclose(float(os.environ["PRISM_NATIVE_GPU_MEMORY_FRACTION"]),
                        grant["quota"]["allocator_fraction"], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Worker allocator limit differs from its recorded GPU quota.")
    with quota_lock(gpu_uuid) as (path, state):
        row = state["reservations"][grant["reservation_id"]]
        if (row.get("execution_id") != grant["execution_id"] or row.get("manifest_id") != manifest_id
                or row["worker_pid"] is not None):
            raise ValueError("Duplicate or mismatching worker startup claim.")
        immutable_json(ROOT / "started" / session / f"{grant['job_key']}.json",
                       {"execution_id": grant["execution_id"], "manifest_id": manifest_id,
                        "worker_pid": os.getpid(), "gpu_uuid": gpu_uuid, "started_at": now()})
        row.update(worker_pid=os.getpid(), worker_start=process_identity(os.getpid()), started_at=now())
        atomic_json(path, state)


def release(lease: dict[str, Any]) -> None:
    with quota_lock(lease["gpu_uuid"]) as (path, state):
        row = state["reservations"][lease["reservation_id"]]
        if row["owner_pid"] != os.getpid():
            raise ValueError("A scheduler cannot release another scheduler's GPU reservation.")
        del state["reservations"][lease["reservation_id"]]
        atomic_json(path, state)


def check_usage(gpu: Any) -> None:
    with quota_lock(gpu.uuid) as (_, state):
        memory = dict(gpu.process_memory_mib)
        own = 0
        for row in state["reservations"].values():
            if row.get('worker_pid') in memory:
                identity = process_identity(row['worker_pid'])
                if identity is None:
                    continue
                if identity != row.get('worker_start'):
                    raise RuntimeError('A GPU process PID was reused; its original reservation is retained.')
            used = memory.get(row.get("worker_pid"), 0)
            if used is None:
                raise RuntimeError("NVML returned unknown memory usage for a registered native worker.")
            own += used
            if row["mode"] == "shared" and used > row["quota"]["reservation_mib"]:
                raise MemoryError(f"Native worker {row['worker_pid']} used {used} MiB beyond its "
                                  f"{row['quota']['reservation_mib']} MiB GPU UUID reservation.")
        rows = list(state['reservations'].values())
        if rows and rows[0]['mode'] == 'shared':
            external = max(0, gpu.used_mib - own)
            reserved = sum(row['quota']['reservation_mib'] for row in rows)
            if reserved + external + max(row['safety_margin_mib'] for row in rows) > gpu.total_mib:
                raise MemoryError('External GPU load exhausted the registered native quotas and safety margin.')
