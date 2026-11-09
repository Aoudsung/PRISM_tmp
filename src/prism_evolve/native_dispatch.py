"""Model-free server orchestration for globally claimed, complete native cohorts."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import traceback
from typing import Any, Mapping

from .native_coordination import CoordinationClient, CoordinationUnavailable, atomic_json, cohort_key, encoded, immutable_json, now, sync_directory
from . import native_gpu_quota as quotas


class AdmissionTelemetryUnavailable(RuntimeError):
    """New admissions stop while existing authorizations remain valid."""


def build_execution(template: Mapping[str, Any], grant: Mapping[str, Any], parent: Path) -> tuple[Path, dict[str, Any]]:
    from .native_collection import _HOST_CONFIG_FIELDS
    origin = json.loads(grant["origin_manifest_text"])
    directory = parent / "executions" / grant["execution_id"]
    directory.mkdir(parents=True, exist_ok=False)
    manifest = deepcopy(origin)
    config = deepcopy(origin["config"])
    for key in _HOST_CONFIG_FIELDS:
        if key in template["config"]:
            config[key] = deepcopy(template["config"][key])
        else:
            config.pop(key, None)
    catalog = [*template["config"]["backbones"], *template["config"].get("execution_backbones", [])]
    model = next(item for item in catalog if item["id"] == grant["backbone_id"])
    config["backbones"] = [deepcopy(model)]
    config.pop('execution_backbones', None)
    config["scheduler"]["allowed_gpu_ids"] = [grant["gpu"]]
    config["scheduler"]["memory_fraction"] = grant["quota"]["allocator_fraction"]
    execution = {key: deepcopy(value) for key, value in grant.items() if key != "origin_manifest_text"}
    manifest.update(config=config, run_id=grant["execution_id"], manifest_id=grant["execution_id"], run_dir=str(directory),
                    host=socket.gethostname(), site_id=template["site_id"], created_at=now(),
                    source_root=template["source_root"],
                    runtime=template["runtime"], model_identities=[model],
                    execution_model_identities=[model],
                    execution=execution, registration_run_id=template["run_id"])
    bundle = deepcopy(grant["origin_bundle"])
    bundle.update(run_dir=str(directory), manifest_path=str(directory / "manifest.json"),
                  model_path=model["model_path"], site_id=template["site_id"], required_gpu=grant["gpu"],
                  manifest_id=grant["execution_id"])
    immutable_json(directory / "manifest.json", manifest)
    immutable_json(directory / "schedule.json", [bundle])
    return directory, bundle


def completion_payload(directory: Path, bundle: Mapping[str, Any], grant: Mapping[str, Any], return_code: int) -> dict[str, Any]:
    receipts, feedback, artifacts = {}, {}, {}
    for name in (f"{bundle['bundle_id']}.json", f"{bundle['bundle_id']}.error.json"):
        path = directory / 'model_runtime' / name
        if path.exists():
            artifacts[str(path.relative_to(directory))] = str(path)
    for cell in bundle["cells"]:
        folder = directory / "cells" / cell["cell_id"]
        for name in ("status.json", "test_records.json", "summary.json"):
            path = folder / name
            if path.exists():
                artifacts[str(path.relative_to(directory))] = str(path)
        if (folder / "status.json").exists():
            receipts[cell["cell_id"]] = json.loads((folder / "status.json").read_text())
    for benchmark in bundle["benchmarks"]:
        folder = directory / "feedback" / bundle["bundle_id"] / benchmark
        path = folder / "status.json"
        if path.exists():
            feedback[benchmark] = json.loads(path.read_text())
            for name in ("status.json", "measurement/feedback_change_protocol.json", "measurement/feedback_change_records.jsonl"):
                artifact = folder / name
                if artifact.exists():
                    artifacts[str(artifact.relative_to(directory))] = str(artifact)
    return {"job_key": grant["job_key"], "execution_id": grant["execution_id"], "return_code": return_code,
            "receipts": receipts, "feedback": feedback, "artifacts": artifacts}


def run_coordinated_schedule(bundles: list[dict[str, Any]], run_dir: Path, python_executable: str,
                             *, allowed_gpu_ids: list[int], max_workers: int, poll_seconds: float,
                             cpu_threads: int, expected_hostname: str, memory_fraction: float,
                             share_occupied_gpus: bool, safety_margin_mib: int,
                             max_workers_per_gpu: int, worker_memory_mib: int | None,
                             allocator_memory_mib: int | None, shutdown_grace_seconds: float) -> dict[str, Any]:
    from .native_scheduler import _inventory_record, _journal, _prepare_bundles, _scheduler_lock, nvidia_inventory
    if socket.gethostname() != expected_hostname:
        raise RuntimeError("The coordinated scheduler is on another host.")
    if (max_workers < 1 or max_workers_per_gpu < 1 or cpu_threads < 1
            or not 0 < poll_seconds <= 30 or safety_margin_mib < 0
            or not 0 < memory_fraction <= 1):
        raise ValueError("Invalid coordinated worker/resource settings.")
    run_dir = run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if not manifest["config"]["coordination"]["enabled"]:
        raise ValueError("Global claims must be enabled explicitly.")
    os.environ["PRISM_NATIVE_NVML_GPU_IDS"] = ",".join(map(str, allowed_gpu_ids))
    client = CoordinationClient(run_dir, manifest)
    state_path = run_dir / "scheduler_state.json"
    state = {"protocol": "native-v1", "host": expected_hostname, "status": "running", "jobs": {},
             "coordination": {"registered": False, "session_id": manifest["config"]["coordination"]["session_id"]}}
    running: dict[str, dict[str, Any]] = {}
    stop = threading.Event()
    client.stop_requested = stop
    handlers = {}
    worker_failed = False

    def observed_inventory():
        devices = nvidia_inventory()
        for device in devices:
            if any(used is None for pid, used in device.process_memory_mib if pid in device.native_worker_pids):
                raise AdmissionTelemetryUnavailable("NVML cannot report a native worker's GPU usage.")
        return devices

    def record_local_completion(execution_id, active, return_code):
        nonlocal worker_failed
        active["log"].close()
        payload = completion_payload(active["directory"], active["bundle"], active["grant"], return_code)
        pending = {"protocol": "native-cohort-claims-v1", "operation": "reconcile_finish",
                   "session_id": manifest["config"]["coordination"]["session_id"],
                   "site_id": manifest["site_id"], "host": expected_hostname,
                   "registration_manifest_id": client.manifest_id, "payload": payload}
        pending_path = active["directory"] / "completion-pending.json"
        if not pending_path.exists():
            immutable_json(pending_path, pending)
            sync_directory(pending_path.parent)
        local_complete = (return_code == 0
                          and set(payload["receipts"]) == {cell["cell_id"] for cell in active["bundle"]["cells"]}
                          and all(row.get("status") == "done" for row in payload["receipts"].values()))
        if (manifest["config"]["feedback_diagnostic"]["enabled"]
                and active["bundle"]["backbone_id"] == manifest["config"]["mechanism_backbone"]):
            local_complete = (local_complete and set(payload["feedback"]) == set(active["bundle"]["benchmarks"])
                              and all(row.get("status") == "done" for row in payload["feedback"].values()))
        worker_failed = worker_failed or not local_complete
        state["jobs"][execution_id].update(
            status="completed_pending_confirmation" if local_complete else "failed_pending_confirmation",
            return_code=return_code, ended_at=now(), local_execution_complete=local_complete,
            completion_pending=str(pending_path),
            worker_observed_wall_seconds=time.clock_gettime(time.CLOCK_MONOTONIC) - active['started'],
            reason="Local worker exited; coordinator confirmation is pending.")
        atomic_json(state_path, state)
        return payload
    with _scheduler_lock(run_dir):
        if state_path.exists():
            raise FileExistsError("Coordinated schedules require a new, unstarted run directory.")
        atomic_json(state_path, state)
        try:
            inventory = observed_inventory()
            result = client.request("register", {"manifest_text": (run_dir / "manifest.json").read_text(),
                                                   "schedule_text": (run_dir / "schedule.json").read_text(),
                                                   "gpus": [_inventory_record(gpu) for gpu in inventory]})
        except Exception as error:
            state.update(status="blocked", error=f"{type(error).__name__}: {error}", updated_at=now())
            state["coordination"].update(new_claims_enabled=False, error=state["error"])
            atomic_json(state_path, state)
            raise
        state["coordination"].update(registered=True, **result)
        atomic_json(state_path, state)
        if threading.current_thread() is threading.main_thread():
            for number in (signal.SIGINT, signal.SIGTERM):
                handlers[number] = signal.getsignal(number)
                signal.signal(number, lambda _number, _frame: stop.set())
        try:
            while not stop.is_set():
                for execution_id, active in list(running.items()):
                    return_code = active["process"].poll()
                    if return_code is None:
                        continue
                    observed_wall = time.clock_gettime(time.CLOCK_MONOTONIC) - active['started']
                    payload = record_local_completion(execution_id, active, return_code)
                    try:
                        result = client.request("finish", payload)
                        state["coordination"].update(result)
                        status = next(job["status"] for job in result["global"]["jobs"]
                                      if job["execution_id"] == execution_id)
                        state["jobs"][execution_id].update(status=status, return_code=return_code, ended_at=now(),
                                                          worker_observed_wall_seconds=observed_wall,
                                                          reason=None if status == "done" else "See worker log and real receipts.")
                        atomic_json(active["directory"] / "completion-confirmation.json", {
                            "execution_id": execution_id, "registry_status": status, "at": now(),
                            "source_completion": str(active["directory"] / "completion-pending.json"),
                        })
                    finally:
                        quotas.release(active["lease"])
                        del running[execution_id]
                    _journal(run_dir, "worker_finished", execution_id=execution_id, return_code=return_code,
                             status=state["jobs"][execution_id]["status"])
                    atomic_json(state_path, state)
                    if worker_failed:
                        raise RuntimeError("A worker failed or produced incomplete local results; no work will be retried.")
                try:
                    inventory = observed_inventory()
                except Exception as error:
                    raise AdmissionTelemetryUnavailable(f"GPU telemetry unavailable: {error}") from error
                state.update(last_gpu_inventory=[_inventory_record(gpu) for gpu in inventory],
                             inventory_at=now(), updated_at=now())
                _journal(run_dir, "gpu_inventory", devices=state["last_gpu_inventory"])
                atomic_json(state_path, state)
                for gpu in inventory:
                    quotas.check_usage(gpu)
                    for execution_id, active in running.items():
                        if active["lease"]["gpu_uuid"] == gpu.uuid:
                            used = dict(gpu.process_memory_mib).get(active["process"].pid)
                            state["jobs"][execution_id]["used_mib"] = used
                            if used is not None:
                                state['jobs'][execution_id]['gpu_peak_observed_mib'] = max(
                                    used, state['jobs'][execution_id].get('gpu_peak_observed_mib', 0))
                busy = [gpu.index for gpu in inventory if
                        sum(active["lease"]["gpu_uuid"] == gpu.uuid for active in running.values()) >= max_workers_per_gpu]
                result = client.request("status", {"busy_gpus": busy})
                state["coordination"].update(result)
                if result["global"]["failed"] or result['global'].get('halted_error'):
                    raise RuntimeError("The global cohort registry contains a failed execution; no work will be retried.")
                if result["global"]["global_complete"] and not running:
                    state["status"] = "done"
                    break
                required = max(int(model["params_b"] * 2300) for model in manifest["config"]["backbones"])
                blocked = set(busy)
                for gpu in inventory:
                    if gpu.index not in allowed_gpu_ids:
                        continue
                    while len(running) < max_workers and sum(
                            active["lease"]["gpu_uuid"] == gpu.uuid for active in running.values()) < max_workers_per_gpu:
                        lease = quotas.reserve(gpu, max_workers=max_workers_per_gpu, worker_memory_mib=worker_memory_mib,
                                               allocator_memory_mib=allocator_memory_mib, safety_margin_mib=safety_margin_mib,
                                               required_memory_mib=required, memory_fraction=memory_fraction,
                                               share_occupied_gpus=share_occupied_gpus)
                        if lease is None:
                            blocked.add(gpu.index)
                            break
                        grant, directory, log, claimed = None, None, None, False
                        try:
                            result = client.request("claim_next", {"slot": {
                                "gpu": gpu.index, "gpu_uuid": gpu.uuid, "reservation_id": lease["reservation_id"],
                                "quota": lease["quota"],
                            }})
                            state["coordination"]["global"] = result["global"]
                            grant = result["grant"]
                            if grant is None:
                                quotas.release(lease)
                                break
                            claimed = True
                            directory, bundle = build_execution(manifest, grant, run_dir)
                            result = client.request("bind_execution", {
                                "job_key": grant["job_key"], "execution_id": grant["execution_id"],
                                "manifest_text": (directory / "manifest.json").read_text(),
                                "schedule_text": (directory / "schedule.json").read_text(),
                            })
                            immutable_json(directory / "authorization.json", result["authorization"])
                            quotas.bind(lease, grant["execution_id"], result["authorization"]["manifest_id"])
                            prepared = _prepare_bundles([bundle], directory)[0]
                            environment = dict(os.environ)
                            environment.update({name: str(cpu_threads) for name in
                                                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
                            admission = {**_inventory_record(gpu), "shared_load": share_occupied_gpus,
                                         "reservation": lease, "execution_id": grant["execution_id"]}
                            environment.update(CUDA_VISIBLE_DEVICES=str(gpu.index), PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1",
                                               TOKENIZERS_PARALLELISM="false", PRISM_NATIVE_TORCH_THREADS=str(cpu_threads),
                                               PRISM_NATIVE_GPU_MEMORY_FRACTION=str(lease["quota"]["allocator_fraction"]),
                                               PRISM_NATIVE_NVML_GPU_IDS=str(gpu.index),
                                               PRISM_NATIVE_GPU_SHARE_OCCUPIED="1" if share_occupied_gpus else "0",
                                               PRISM_NATIVE_GPU_ADMISSION_JSON=json.dumps(admission, sort_keys=True),
                                               PYTHONPATH=str(Path(manifest["source_root"]) / "src"))
                            path = directory / "worker.log"
                            log = path.open("xb")
                            if stop.is_set():
                                raise InterruptedError("Scheduler stopped before the authorized worker was launched.")
                            process = subprocess.Popen([python_executable, "-m", "prism_evolve.native_runner",
                                                        "--worker-bundle", str(prepared["path"]), "--gpu", str(gpu.index)],
                                                       env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                            active = {"process": process, "log": log, "lease": lease, "directory": directory,
                                      "bundle": prepared["payload"], "grant": grant,
                                      "started": time.clock_gettime(time.CLOCK_MONOTONIC)}
                            running[grant["execution_id"]] = active
                            state["jobs"][grant["execution_id"]] = {
                                "bundle_id": bundle["bundle_id"], "status": "running", "pid": process.pid,
                                "pid_start": quotas.process_identity(process.pid),
                                "gpu": gpu.index, "gpu_uuid": gpu.uuid, "run_dir": str(directory), "log_path": str(path),
                                "origin_site": grant["origin_site"], "planned_gpu": grant["planned_gpu"],
                                "quota": lease["quota"], "cells": {cell["cell_id"]: {"status": "pending"} for cell in bundle["cells"]},
                                "started_at": now(), "execution_id": grant["execution_id"],
                            }
                            _journal(run_dir, "worker_started", execution_id=grant["execution_id"],
                                     bundle_id=bundle["bundle_id"], pid=process.pid, gpu=gpu.index, gpu_uuid=gpu.uuid,
                                     origin_site=grant["origin_site"], quota=lease["quota"])
                            atomic_json(state_path, state)
                        except Exception as error:
                            started = grant is not None and grant["execution_id"] in running
                            if not started:
                                if log is not None:
                                    log.close()
                                quotas.release(lease)
                            if claimed and not started and not client.unusable:
                                client.request("fail_claim", {"job_key": grant["job_key"], "execution_id": grant["execution_id"],
                                                              "reason": f"{type(error).__name__}: {error}"})
                            raise
                result = client.request("status", {"busy_gpus": sorted(blocked | {
                    gpu.index for gpu in inventory if sum(active["lease"]["gpu_uuid"] == gpu.uuid for active in running.values())
                    >= max_workers_per_gpu})})
                state["coordination"].update(result)
                atomic_json(state_path, state)
                stop.wait(poll_seconds)
        except (CoordinationUnavailable, AdmissionTelemetryUnavailable) as error:
            client.unusable = True
            telemetry_available = not isinstance(error, AdmissionTelemetryUnavailable)
            state.update(status="draining", error=f"{type(error).__name__}: {error}")
            if not telemetry_available:
                state["telemetry_error"] = state["error"]
            state["coordination"].update(control_state="disconnected" if isinstance(error, CoordinationUnavailable)
                                         else "telemetry_unavailable", new_claims_enabled=False,
                                         error=state["error"], disconnected_at=now())
            atomic_json(state_path, state)
            _journal(run_dir, "control_disconnected", reason=state["error"],
                     authorized_executions=list(running))
            try:
                if worker_failed:
                    raise RuntimeError("A worker failed before coordinator confirmation.")
                while running and not stop.is_set():
                    for execution_id, active in list(running.items()):
                        return_code = active["process"].poll()
                        if return_code is None:
                            continue
                        record_local_completion(execution_id, active, return_code)
                        quotas.release(active["lease"])
                        del running[execution_id]
                        _journal(run_dir, "worker_finished_pending_confirmation", execution_id=execution_id,
                                 return_code=return_code, status=state["jobs"][execution_id]["status"])
                        if worker_failed:
                            raise RuntimeError("An authorized worker failed while the control connection was unavailable.")
                    # Resource enforcement stays local; missing telemetry is explicit
                    # and does not revoke an already-bound execution authorization.
                    if telemetry_available:
                        try:
                            inventory = observed_inventory()
                        except Exception as telemetry_error:
                            telemetry_available = False
                            state["telemetry_error"] = f"{type(telemetry_error).__name__}: {telemetry_error}"
                        else:
                            state.update(last_gpu_inventory=[_inventory_record(gpu) for gpu in inventory], inventory_at=now())
                            for gpu in inventory:
                                quotas.check_usage(gpu)
                    state["updated_at"] = now()
                    atomic_json(state_path, state)
                    if running:
                        stop.wait(poll_seconds)
                state["status"] = "blocked" if stop.is_set() else "reconciliation_required"
            except Exception as local_error:
                state.update(status="failed", local_error=f"{type(local_error).__name__}: {local_error}",
                             traceback=traceback.format_exc())
                raise
        except InterruptedError as error:
            if not stop.is_set():
                state.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
                raise
            state.update(status="blocked", reason="Explicit stop received; original execution evidence retained.")
        except Exception as error:
            state.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            raise
        finally:
            for active in running.values():
                if active["process"].poll() is None:
                    try:
                        os.killpg(active["process"].pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            for execution_id, active in running.items():
                try:
                    active["process"].wait(timeout=shutdown_grace_seconds)
                except subprocess.TimeoutExpired:
                    os.killpg(active["process"].pid, signal.SIGKILL)
                    active["process"].wait()
                active["log"].close()
                payload = record_local_completion(execution_id, active, active["process"].returncode)
                quotas.release(active["lease"])
                state["jobs"][execution_id].update(status="blocked", return_code=active["process"].returncode,
                                                   reason="Scheduler stopped; execution and partial receipts retained.")
                if not client.unusable:
                    try:
                        client.request("finish", payload)
                    except Exception as error:
                        state["coordination"]["cleanup_error"] = f"{type(error).__name__}: {error}"
            if stop.is_set() and state["status"] != "failed":
                state["status"] = "blocked"
            for number, handler in handlers.items():
                signal.signal(number, handler)
            state["updated_at"] = now()
            atomic_json(state_path, state)
    return state
