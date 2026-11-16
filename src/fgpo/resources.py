"""Cooperative Linux host-memory reservations before importing the model runtime.

The formal study loads unique saved parameters as bf16 and retains a bf16
reference plus three fp32 Adam tensors. Only safetensors headers are read here.
Reservations coordinate this user's study processes; MemAvailable also accounts
for memory already consumed by other work on the host.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import uuid


_MIB = 1024 ** 2
_GIB = 1024 ** 3
_FORMAT = "fgpo-host-memory-v1"


def _tensor_inventory(model_path):
    root = Path(model_path)
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        shards = {}
        for name, filename in weight_map.items():
            if not filename.endswith(".safetensors"):
                raise ValueError("Host memory estimation requires floating-point safetensors weights")
            shards.setdefault(filename, set()).add(name)
    elif (root / "model.safetensors").is_file():
        shards = {"model.safetensors": None}
    else:
        raise ValueError(f"Host memory estimation requires model.safetensors or its shard index: {root}")
    parameters, tensor_count, largest = 0, 0, 0
    for filename, names in shards.items():
        path = root / filename
        with path.open("rb") as handle:
            prefix = handle.read(8)
            header_size = int.from_bytes(prefix, "little")
            # Safetensors bounds its UTF-8 JSON header at 100 MB. No tensor
            # payload is read, mapped, or materialized by this estimator.
            if len(prefix) != 8 or not 0 < header_size <= min(100_000_000, os.fstat(handle.fileno()).st_size - 8):
                raise ValueError(f"Invalid safetensors header length: {path}")
            header = json.loads(handle.read(header_size))
        selected = set(header) - {"__metadata__"} if names is None else names
        for name in selected:
            tensor = header[name]
            shape = tensor["shape"]
            if (tensor["dtype"] not in {"BF16", "F16", "F32", "F64"}
                    or not isinstance(shape, list)
                    or any(type(size) is not int or size < 0 for size in shape)):
                raise ValueError(f"Host memory estimation requires unquantized floating-point tensors: {path}:{name}")
            count = math.prod(shape)
            parameters += count
            tensor_count += 1
            largest = max(largest, 2 * count)
    if parameters <= 0:
        raise ValueError(f"No model parameters in safetensors headers: {root}")
    return {"parameter_numel": parameters, "saved_tensor_count": tensor_count,
            "weight_shard_count": len(shards), "largest_bf16_tensor_bytes": largest}


def _configured_bytes(config, key, default, unit, *, allow_zero=False):
    value = float(config.get(key, default))
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{key} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    result = int(value * unit)
    if result == 0 and value != 0:
        raise ValueError(f"{key} must specify at least one byte")
    return result


def _estimate(config, model_gpu_count, resume_checkpoint):
    if type(model_gpu_count) is not int or model_gpu_count < 1:
        raise ValueError("model_gpu_count must be a positive integer")
    budget = _tensor_inventory(config["model_path"])
    largest = budget["largest_bf16_tensor_bytes"]
    shard = _configured_bytes(config, "checkpoint_shard_mib", 1024., _MIB)
    bucket = _configured_bytes(config, "optimizer_bucket_mib", 64., _MIB)
    runtime = _configured_bytes(config, "host_runtime_reserve_gib", 16., _GIB, allow_zero=True)
    gradients = 0
    if resume_checkpoint is not None:
        root = Path(resume_checkpoint)
        snapshot = json.loads((root / "CURRENT").read_text(encoding="utf-8"))["snapshot"]
        if not isinstance(snapshot, str) or Path(snapshot).name != snapshot or snapshot in {"", ".", ".."}:
            raise ValueError("Checkpoint CURRENT must name one snapshot directory")
        path = root / "snapshots" / snapshot / "gradients.pt"
        if path.is_file():
            gradients = path.stat().st_size
    budget.update(model_gpu_count=model_gpu_count, checkpoint_shard_bytes=shard,
                  optimizer_bucket_bytes=bucket, resume_gradients_file_bytes=gradients,
                  adam_and_reference_bytes=14 * budget["parameter_numel"],
                  serialization_temporary_bytes=max(2 * max(shard, largest), gradients + largest),
                  optimizer_transfer_bytes=4 * bucket * model_gpu_count,
                  optimizer_arithmetic_bytes=4 * bucket, runtime_reserve_bytes=runtime)
    budget["required_bytes"] = sum(budget[key] for key in (
        "adam_and_reference_bytes", "serialization_temporary_bytes", "optimizer_transfer_bytes",
        "optimizer_arithmetic_bytes", "runtime_reserve_bytes"))
    return budget


def _meminfo():
    values = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields[0] in {"MemTotal:", "MemAvailable:"}:
            values[fields[0][:-1]] = int(fields[1]) * 1024
    return values["MemTotal"], values["MemAvailable"]


def _process_memory(pid, uid, expected_start=None):
    descriptor = None
    try:
        # A directory fd stays attached to this process if its PID is recycled.
        descriptor = os.open(f"/proc/{pid}", os.O_RDONLY | os.O_DIRECTORY)
        if os.fstat(descriptor).st_uid != uid:
            return None
        with os.fdopen(os.open("stat", os.O_RDONLY, dir_fd=descriptor), encoding="utf-8") as handle:
            fields = handle.read().rsplit(")", 1)[1].split()
        start = int(fields[19])  # /proc/<pid>/stat field 22; state is field 3.
        if fields[0] in {"Z", "X", "x"} or (expected_start is not None and start != expected_start):
            return None
        with os.fdopen(os.open("status", os.O_RDONLY, dir_fd=descriptor), encoding="utf-8") as handle:
            status = {key: value.split() for key, value in
                      (line.split(":", 1) for line in handle if ":" in line)}
        if int(status["Uid"][0]) != uid or status["State"][0] in {"Z", "X", "x"}:
            return None
        return start, int(status["VmRSS"][0]) * 1024
    except (FileNotFoundError, ProcessLookupError):
        return None
    finally:
        if descriptor is not None:
            os.close(descriptor)


@contextmanager
def _locked(directory):
    with (directory / "allocation.lock").open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _active_claims(directory, uid):
    # Claim publication holds the same lock, so any surviving pending file
    # belongs to an interrupted writer, never to a concurrent live publication.
    for path in directory.glob("claim-*.pending"):
        path.unlink()
    active = []
    for path in directory.glob("claim-*.json"):
        claim = json.loads(path.read_text(encoding="utf-8"))
        if (claim.get("format") != _FORMAT
                or any(type(claim.get(key)) is not int or claim[key] <= 0
                       for key in ("pid", "required_bytes", "limit_bytes"))
                or type(claim.get("start_ticks")) is not int or claim["start_ticks"] < 0):
            raise ValueError(f"Invalid host memory reservation: {path}")
        process = _process_memory(claim["pid"], uid, claim["start_ticks"])
        if process is None:
            path.unlink()
        else:
            active.append((claim, process[1]))
    return active


@contextmanager
def reserve_host_memory(config, *, model_gpu_count, resume_checkpoint=None):
    """Reserve one cell's CPU peak and yield JSON-serializable budget provenance.

    Call before Engine import/model allocation and keep the context through cell
    teardown. A denied request raises immediately; no scheduler or sleep is used.
    resume_checkpoint is the checkpoint root containing CURRENT and snapshots/.
    """
    fraction = float(config.get("host_memory_fraction", .85))
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("host_memory_fraction must be finite and in (0, 1]")
    budget = _estimate(config, model_gpu_count, resume_checkpoint)
    uid, pid, identity = os.getuid(), os.getpid(), uuid.uuid4().hex
    directory = Path("/tmp") / f"fgpo-host-memory-{uid}"
    directory.mkdir(mode=0o700, exist_ok=True)
    if directory.is_symlink() or directory.stat().st_uid != uid:
        raise ValueError(f"Host reservation directory must belong to uid {uid}: {directory}")
    claim_path = directory / f"claim-{identity}.json"
    with _locked(directory):
        active = _active_claims(directory, uid)
        total, available = _meminfo()
        requested_limit = int(total * fraction)
        limit = min([requested_limit, *(claim["limit_bytes"] for claim, _ in active)])
        reserved = sum(claim["required_bytes"] for claim, _ in active)
        pending = sum(max(0, claim["required_bytes"] - rss) for claim, rss in active)
        required = budget["required_bytes"]
        if required + reserved > limit or required + pending > available:
            raise RuntimeError(
                "Host memory reservation refused: "
                f"request={required / _GIB:.2f} GiB; active reservations={reserved / _GIB:.2f} GiB; "
                f"cooperative limit={limit / _GIB:.2f} GiB; MemAvailable={available / _GIB:.2f} GiB; "
                f"active unmaterialized reservations={pending / _GIB:.2f} GiB; "
                f"available required={((required + pending) / _GIB):.2f} GiB. "
                "Reduce concurrent cells or configure host_memory_fraction/host_runtime_reserve_gib for this host.")
        process = _process_memory(pid, uid)
        if process is None:
            raise RuntimeError("Cannot read the reserving process's Linux memory identity")
        start, rss = process
        claim = {"format": _FORMAT, "pid": pid, "start_ticks": start,
                 "required_bytes": required, "limit_bytes": requested_limit}
        budget.update(format=_FORMAT, reservation_id=identity, pid=pid, start_ticks=start,
                      host_memory_fraction=fraction, requested_limit_bytes=requested_limit,
                      effective_limit_bytes=limit, host_mem_total_bytes=total,
                      host_mem_available_bytes_at_acquire=available, main_rss_bytes_at_acquire=rss,
                      other_reserved_bytes=reserved, other_unmaterialized_bytes=pending,
                      other_active_jobs=len(active), claim_path=str(claim_path))
        temporary = directory / f"claim-{identity}.pending"
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(claim, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, claim_path)
        finally:
            temporary.unlink(missing_ok=True)
    try:
        yield budget
    finally:
        with _locked(directory):
            claim_path.unlink(missing_ok=True)
            _active_claims(directory, uid)
