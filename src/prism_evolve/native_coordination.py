"""Server-side, single-writer cohort claims, carried by the monitor's SSH/MCP bridge."""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any, Mapping
import uuid
import zlib
from urllib.parse import quote


PROTOCOL = "native-cohort-claims-v1"


class CoordinationUnavailable(RuntimeError):
    """The control channel is unusable; requests must not be resent automatically."""


def encoded(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pack(value: Any) -> str:
    return base64.b64encode(zlib.compress(encoded(value))).decode('ascii')


def unpack(value: str) -> Any:
    return json.loads(zlib.decompress(base64.b64decode(value, validate=True)))


def cohort_key(bundle: Mapping[str, Any]) -> str:
    return f"{quote(str(bundle['backbone_id']), safe='')}.s{bundle['seed']}.{'-'.join(bundle['benchmarks'])}"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded(value))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def immutable_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Duplicate coordination artifact: {path}")
    atomic_json(path, value)


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def mailbox_requests(run_dir: Path) -> list[dict[str, Any]]:
    mailbox = run_dir / "coordination" / "mailbox"
    return [json.loads(path.read_text()) for path in sorted((mailbox / "outbox").glob("*.json"))
            if not (mailbox / "inbox" / path.name).exists()]


def deliver_reply(run_dir: Path, reply: Mapping[str, Any]) -> str:
    request_id = str(reply["request_id"])
    if len(request_id) != 32 or any(character not in "0123456789abcdef" for character in request_id):
        raise ValueError("Invalid bridge request identity.")
    mailbox = run_dir / "coordination" / "mailbox"
    request = json.loads((mailbox / "outbox" / f"{request_id}.json").read_text())
    if request["request_id"] != reply["request_id"]:
        raise ValueError("Coordination reply belongs to another immutable request.")
    target = mailbox / "inbox" / f"{request_id}.json"
    if target.exists():
        return request_id
    immutable_json(target, dict(reply))
    return request_id


class CoordinationClient:
    def __init__(self, run_dir: Path, manifest: Mapping[str, Any]) -> None:
        self.run_dir = run_dir
        self.manifest = manifest
        self.manifest_id = manifest["manifest_id"]
        self.unusable = False
        self.stop_requested = None

    def request(self, operation: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self.unusable:
            raise CoordinationUnavailable("The coordination channel failed; no further requests will be attempted.")
        request_id = uuid.uuid4().hex
        envelope = {"protocol": PROTOCOL, "request_id": request_id,
                    "session_id": self.manifest["config"]["coordination"]["session_id"],
                    "site_id": self.manifest["site_id"], "host": socket.gethostname(),
                    "registration_manifest_id": self.manifest_id,
                    "operation": operation, "payload_z": pack(dict(payload))}
        mailbox = self.run_dir / "coordination" / "mailbox"
        immutable_json(mailbox / "outbox" / f"{request_id}.json", envelope)
        response = mailbox / "inbox" / f"{request_id}.json"
        deadline = time.clock_gettime(time.CLOCK_MONOTONIC) + 180
        while not response.exists():
            if self.stop_requested is not None and self.stop_requested.is_set():
                self.unusable = True
                raise InterruptedError("Coordination wait stopped; the request outcome is retained without resubmission.")
            if time.clock_gettime(time.CLOCK_MONOTONIC) >= deadline:
                self.unusable = True
                raise CoordinationUnavailable(f"Coordination request {request_id} has no reply; it will not be resubmitted.")
            time.sleep(0.1)
        try:
            reply = json.loads(response.read_text())
        except (OSError, ValueError) as error:
            self.unusable = True
            raise CoordinationUnavailable(f"Unreadable coordination reply for {request_id}: {error}") from error
        if not isinstance(reply, dict) or reply.get("request_id") != request_id:
            self.unusable = True
            raise CoordinationUnavailable("The coordination reply identity differs from the sent request.")
        if "error" in reply:
            self.unusable = True
            raise CoordinationUnavailable(reply["error"])
        try:
            return unpack(reply["result_z"])
        except Exception as error:
            self.unusable = True
            raise CoordinationUnavailable(f"Invalid coordination result for {request_id}: {type(error).__name__}") from error


@contextmanager
def registry_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def public_status(state: Mapping[str, Any]) -> dict[str, Any]:
    jobs = [{key: job.get(key) for key in (
        "job_key", "bundle_id", "backbone_id", "seed", "benchmarks", "origin_site", "planned_gpu",
        "status", "execution_id", "site_id", "host", "gpu", "gpu_uuid", "reason",
    )} for job in state["jobs"].values()]
    expected_sites = {work["site_id"] for work in state["contract"]["parameters"]["workload_plan"]}
    # Host A may register only the coordinator while all model work runs on B.
    registered = set(state["sources"])
    complete = (expected_sites <= registered <= expected_sites | {"host_a"}
                and all(job["status"] == "done" for job in state["jobs"].values()))
    return {"session_id": state["session_id"], "jobs": jobs, "global_complete": complete,
            "halted_error": state.get('halted_error'),
            "registered_sites": sorted(state['sources']),
            "expected_sites": sorted(expected_sites),
            "missing_sites": sorted(expected_sites - state['sources'].keys()),
            "failed": [job for job in jobs if job["status"] == "failed"],
            "pending": [job for job in jobs if job["status"] in {"pending", "unregistered"}]}


def _register(state: dict[str, Any] | None, request: Mapping[str, Any], registry_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, site = request["payload"], request["site_id"]
    manifest = json.loads(payload["manifest_text"])
    schedule = json.loads(payload["schedule_text"])
    config = manifest["config"]
    canonical = Path(config["coordination"]["registry_root"]) / request["session_id"] / "registry.json"
    if registry_path.resolve() != canonical.resolve():
        raise ValueError("A coordination session cannot be forked into another registry path.")
    if (manifest["site_id"] != site or manifest["host"] != request["host"]
            or manifest["manifest_id"] != request["registration_manifest_id"]
            or config["coordination"]["session_id"] != request["session_id"]):
        raise ValueError("Registration belongs to another run, host or coordination session.")
    if state is None:
        if site != "host_a" or request["host"] != socket.gethostname():
            raise ValueError("Host A must register its single-writer coordination session first.")
        jobs = {}
        for work in config["workload_plan"]:
            item = {"backbone_id": work["backbone"], "seed": work["seed"], "benchmarks": work["benchmarks"]}
            key = cohort_key(item)
            if key in jobs:
                raise ValueError("A complete cohort can have only one owner.")
            jobs[key] = {**item, "job_key": key, "origin_site": work["site_id"],
                         "planned_gpu": work["gpu"], "status": "unregistered"}
        state = {"protocol": PROTOCOL, "session_id": request["session_id"],
                 "contract": {"parameters": deepcopy(config), "data_splits": {
                     benchmark: {"splits": {split: row["tasks"] for split, row in data["splits"].items()}}
                     for benchmark, data in manifest["data_splits"].items()}},
                 "coordinator_configuration": config["coordination"],
                 "sources": {}, "jobs": jobs, "requests": {}, "created_at": now()}
    if state["session_id"] != request["session_id"] or site in state["sources"]:
        raise ValueError("This site is already registered or belongs to another session.")
    models = [*config["backbones"], *config.get("execution_backbones", [])]
    state["sources"][site] = {**dict(payload), "manifest_id": manifest["manifest_id"],
                              "run_id": manifest["run_id"], "host": request["host"],
                              "models": {model["id"]: model for model in models}, "busy_gpus": []}
    for bundle in schedule:
        job = state["jobs"][cohort_key(bundle)]
        if job["origin_site"] != site or job["status"] != "unregistered":
            raise ValueError("An origin cannot register another site's cohort twice.")
        job.update(status="pending", bundle_id=bundle["bundle_id"], origin_bundle=bundle,
                   origin_manifest_id=manifest["manifest_id"], origin_run_id=manifest["run_id"])
    return state, {"global": public_status(state)}


def _model_available(state: Mapping[str, Any], site: str, job: Mapping[str, Any]) -> bool:
    return job["backbone_id"] in state["sources"][site]["models"]


def _dispatch(state: dict[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    site, payload, operation = request["site_id"], request["payload"], request["operation"]
    node = state["sources"][site]
    if (state["session_id"] != request["session_id"] or node["host"] != request["host"]
            or node['manifest_id'] != request['registration_manifest_id']):
        raise ValueError("Coordination caller identity changed.")
    if operation == "status":
        node["busy_gpus"] = list(payload["busy_gpus"])
        status = public_status(state)
        free = set(json.loads(node['manifest_text'])['config']['scheduler']['allowed_gpu_ids']) - set(node['busy_gpus'])
        status["claimable"] = [job["job_key"] for job in state["jobs"].values()
                               if job["status"] == "pending" and _model_available(state, site, job) and free
                               and ((job['origin_site'] == site and job['planned_gpu'] in free)
                                    or (job['origin_site'] != site and job['planned_gpu'] in state['sources'][job['origin_site']]['busy_gpus']))]
        status['missing_backbone_capabilities'] = sorted({job['backbone_id'] for job in state['jobs'].values()
                                                        if job['status'] == 'pending' and not _model_available(state, site, job)})
        return {"global": status}
    if operation == "claim_next":
        status = public_status(state)
        if status["failed"] or status['halted_error']:
            raise RuntimeError("A formal cohort failed; further claims are stopped and no work is retried.")
        slot = payload["slot"]
        devices = {device["index"]: device for device in node["gpus"]}
        if slot["gpu"] not in devices or devices[slot["gpu"]]["uuid"] != slot["gpu_uuid"]:
            raise ValueError("Claimed GPU index/UUID differs from this registered node.")
        options = json.loads(node["manifest_text"])["config"]["scheduler"]
        if slot["gpu"] not in options["allowed_gpu_ids"]:
            raise ValueError("The claimed GPU is outside this node's registered resource pool.")
        if options.get("worker_memory_mib") is not None:
            if (slot["quota"]["mode"] != "shared"
                    or slot["quota"]["reservation_mib"] != options["worker_memory_mib"]
                    or slot["quota"]["allocator_mib"] != options["allocator_memory_mib"]):
                raise ValueError("The claimed shared quota differs from the registered resource configuration.")
        elif (slot["quota"]["mode"] != "exclusive"
              or slot["quota"]["allocator_fraction"] != options["memory_fraction"]):
            raise ValueError("The claimed single-worker quota changed.")
        candidates = []
        for job in state["jobs"].values():
            if job["status"] != "pending" or not _model_available(state, site, job):
                continue
            home = job["origin_site"] == site and job["planned_gpu"] == slot["gpu"]
            foreign_busy = (job["origin_site"] != site
                            and job["planned_gpu"] in state["sources"][job["origin_site"]]["busy_gpus"])
            if home or foreign_busy:
                candidates.append((not home, job["seed"], job["bundle_id"], job))
        if not candidates:
            return {"grant": None, "global": status}
        job = min(candidates, key=lambda entry: entry[:3])[-1]
        execution_id = uuid.uuid4().hex
        origin = state["sources"][job["origin_site"]]
        grant = {key: deepcopy(job[key]) for key in (
            "job_key", "bundle_id", "backbone_id", "seed", "benchmarks", "origin_site", "planned_gpu",
            "origin_bundle", "origin_manifest_id", "origin_run_id",
        )}
        grant.update(protocol=PROTOCOL, session_id=state["session_id"], execution_id=execution_id,
                     site_id=site, host=request["host"], gpu=slot["gpu"], gpu_uuid=slot["gpu_uuid"],
                     reservation_id=slot["reservation_id"], quota=slot["quota"], claimed_at=now(),
                     transferred=job["origin_site"] != site or job["planned_gpu"] != slot["gpu"],
                     source_pending_fence={"previous_status": "pending", "claim_request_id": request["request_id"]})
        job.update(status="claimed", execution_id=execution_id, site_id=site, host=request["host"],
                   gpu=slot["gpu"], gpu_uuid=slot["gpu_uuid"], grant=grant)
        return {"grant": {**deepcopy(grant), "origin_manifest_text": origin["manifest_text"]},
                "global": public_status(state)}
    job = state["jobs"][payload["job_key"]]
    if job.get("execution_id") != payload["execution_id"] or job.get("site_id") != site:
        raise ValueError("Execution does not own this complete cohort.")
    if operation in {"finish", "reconcile_finish"}:
        if job["status"] in {"done", "failed"}:
            if job.get("completion") != payload:
                raise ValueError("The recorded completion differs from the submitted recovery evidence.")
            return {"global": public_status(state), "completion_already_recorded": True}
        operation = "finish"
    if operation == "fail_claim":
        if job["status"] != "claimed":
            raise ValueError("A terminal cohort cannot be failed or reassigned again.")
        job.update(status="failed", reason=payload["reason"], finished_at=now())
        return {"global": public_status(state)}
    if operation == "bind_execution":
        if state.get('halted_error') or public_status(state)['failed']:
            raise RuntimeError('The coordination session is halted; no new worker will be authorized.')
        if job["status"] != "claimed" or "execution_manifest_text" in job:
            raise ValueError("A cohort execution can be bound only once.")
        manifest = json.loads(payload["manifest_text"])
        schedule = json.loads(payload["schedule_text"])
        if (manifest["manifest_id"] != job["execution_id"] or manifest["site_id"] != site
                or manifest["host"] != request["host"] or len(schedule) != 1
                or schedule[0]["bundle_id"] != job["bundle_id"] or schedule[0]["required_gpu"] != job["gpu"]):
            raise ValueError("The execution belongs to another claimed cohort or physical worker.")
        job.update(execution_manifest_text=payload["manifest_text"], execution_schedule_text=payload["schedule_text"],
                   execution_manifest_id=manifest["manifest_id"])
        return {"authorization": {"execution_id": job["execution_id"], "job_key": job["job_key"],
                                   "manifest_id": manifest["manifest_id"], "bundle_id": job["bundle_id"]},
                "global": public_status(state)}
    if operation == "finish":
        if job["status"] != "claimed" or "execution_manifest_id" not in job:
            raise ValueError("An unbound or completed execution cannot publish another completion.")
        receipts = payload["receipts"]
        expected = {cell["cell_id"] for cell in job["origin_bundle"]["cells"]}
        complete = payload["return_code"] == 0 and set(receipts) == expected
        required_artifacts = {f"cells/{cell_id}/{name}" for cell_id in expected
                              for name in ('status.json', 'test_records.json', 'summary.json')}
        required_artifacts.add(f"model_runtime/{job['bundle_id']}.json")
        complete = complete and required_artifacts <= payload["artifacts"].keys()
        for cell_id, receipt in receipts.items():
            if cell_id not in expected or any(receipt.get(key) != value for key, value in {
                "cell_id": cell_id, "manifest_id": job["execution_manifest_id"],
                "site_id": site, "host": request["host"], "physical_gpu": job["gpu"],
            }.items()):
                raise ValueError("A completion receipt has a different immutable execution identity.")
            complete = complete and receipt.get("status") == "done"
        manifest = json.loads(job["execution_manifest_text"])
        config = manifest['config']
        by_id = {cell['cell_id']: cell for cell in job['origin_bundle']['cells']}
        for cell_id, receipt in receipts.items():
            benchmark = by_id[cell_id]['benchmark']
            splits = config['appworld_test_splits'] if benchmark == 'appworld' else ['test']
            count = sum(len(manifest['data_splits'][benchmark]['splits'][split]['tasks']) for split in splits)
            complete = complete and receipt.get('num_evaluations') == count
        needs_feedback = (config["feedback_diagnostic"]["enabled"]
                          and job["backbone_id"] == config["mechanism_backbone"])
        if needs_feedback:
            complete = complete and set(payload["feedback"]) == set(job["benchmarks"])
            feedback_artifacts = {f"feedback/{job['bundle_id']}/{benchmark}/{name}"
                                  for benchmark in job["benchmarks"] for name in (
                                      'status.json', 'measurement/feedback_change_protocol.json',
                                      'measurement/feedback_change_records.jsonl')}
            complete = complete and feedback_artifacts <= payload["artifacts"].keys()
            for benchmark, receipt in payload["feedback"].items():
                if any(receipt.get(key) != value for key, value in {
                    "manifest_id": job["execution_manifest_id"], "host": request["host"],
                    "site_id": site, "physical_gpu": job["gpu"], "bundle_id": job["bundle_id"],
                    "benchmark": benchmark, "backbone": job["backbone_id"], "seed": job["seed"],
                }.items()):
                    raise ValueError("Feedback receipt differs from the authorized cohort.")
                complete = complete and receipt.get("status") == "done"
        job.update(status="done" if complete else "failed", finished_at=now(),
                   return_code=payload["return_code"], artifacts=payload["artifacts"],
                   completion=deepcopy(payload),
                   reason=None if complete else "Worker exit/receipts/feedback did not establish complete execution.")
        return {"global": public_status(state)}
    raise ValueError(f"Unknown coordination operation: {operation}")


def apply_request(path: Path, request: Mapping[str, Any]) -> dict[str, Any]:
    if request.get("protocol") != PROTOCOL:
        raise ValueError("Unknown coordination protocol.")
    request_id = request["request_id"]
    reply = {"request_id": request_id}
    with registry_lock(path):
        original = json.loads(path.read_text()) if path.exists() else None
        state = deepcopy(original)
        try:
            if state is not None and request_id in state["requests"]:
                recorded = state["requests"][request_id]
                if recorded["request"] != request or "reply" not in recorded:
                    raise ValueError("Duplicate coordination identity has changed content or no retained acknowledgement.")
                # Return the original receipt; no dispatch or ownership mutation.
                return deepcopy(recorded["reply"])
            decoded_request = {**request, "payload": unpack(request["payload_z"])}
            if request["operation"] == "register":
                state, result = _register(state, decoded_request, path)
            else:
                if state is None:
                    raise ValueError("Host A has not registered this new session.")
                result = _dispatch(state, decoded_request)
            reply["result_z"] = pack(result)
        except Exception as error:
            state = original
            reply["error"] = f"{type(error).__name__}: {error}"
            if state is not None:
                state['halted_error'] = reply['error']
        if state is not None:
            if request_id not in state["requests"]:
                state["requests"][request_id] = {"request": deepcopy(request),
                                                 "operation": request["operation"], "site_id": request["site_id"],
                                                 "at": now(), "error": reply.get("error"), "reply": deepcopy(reply)}
            if public_status(state)["global_complete"] and not state.get('halted_error') and "completed_at" not in state:
                state["completed_at"] = now()
            atomic_json(path, state)
            for directory in (path.parent, path.parent.parent, path.parent.parent.parent):
                sync_directory(directory)
            if "completed_at" in state:
                completed = {key: state[key] for key in (
                    "protocol", "session_id", "contract", "coordinator_configuration", "sources", "jobs", "completed_at",
                )}
                completed["sources"] = {site: {key: value for key, value in node.items() if key != "busy_gpus"}
                                        for site, node in state["sources"].items()}
                target = path.with_name("completed-registry.json")
                if not target.exists():
                    immutable_json(target, completed)
                sync_directory(target.parent)
    return reply


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--reconcile-completion", type=Path,
                        help="Operator-only registration of persisted worker results; never starts training.")
    arguments = parser.parse_args()
    if sys.platform != "linux" or socket.gethostname() != arguments.host:
        raise RuntimeError("The single-writer coordinator must run on the configured Host A server.")
    if arguments.serve and arguments.reconcile_completion is not None:
        parser.error("Choose the request server or one explicit completion reconciliation.")
    if arguments.reconcile_completion is not None:
        pending = json.loads(arguments.reconcile_completion.read_text())
        if pending.get("protocol") != PROTOCOL or pending.get("operation") != "reconcile_finish":
            raise ValueError("Expected an immutable completion-pending.json artifact.")
        registry = json.loads(arguments.registry.read_text())
        if registry["sources"]["host_a"]["host"] != socket.gethostname():
            raise RuntimeError("Completion reconciliation must run on the registered Host A coordinator.")
        canonical = Path(registry["coordinator_configuration"]["registry_root"]) / registry["session_id"] / "registry.json"
        if arguments.registry.resolve() != canonical.resolve():
            raise ValueError("Completion reconciliation cannot fork the authoritative registry.")
        job = registry["jobs"][pending["payload"]["job_key"]]
        directory = arguments.reconcile_completion.parent.resolve()
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["manifest_id"] != job["execution_manifest_id"]:
            raise ValueError("Completion belongs to another execution.")
        request = {key: pending[key] for key in (
            "protocol", "operation", "session_id", "site_id", "host", "registration_manifest_id")}
        request.update(request_id=uuid.uuid4().hex, payload_z=pack(pending["payload"]))
        reply = apply_request(arguments.registry, request)
        if "error" in reply:
            raise RuntimeError(reply["error"])
        result = unpack(reply["result_z"])
        job = next(item for item in result["global"]["jobs"]
                   if item["execution_id"] == pending["payload"]["execution_id"])
        acknowledgement = {"source_completion": str(arguments.reconcile_completion.resolve()),
                           "registry": str(arguments.registry.resolve()), "at": now(),
                           "execution_id": job["execution_id"], "registry_status": job["status"],
                           "reply": reply}
        atomic_json(arguments.reconcile_completion.with_name("completion-confirmation.json"), acknowledgement)
        sync_directory(arguments.reconcile_completion.parent)
        print(json.dumps({"execution_id": job["execution_id"], "registry_status": job["status"],
                          "global_complete": result["global"]["global_complete"]}, ensure_ascii=False))
    elif arguments.serve:
        for line in sys.stdin:
            print(json.dumps(apply_request(arguments.registry, json.loads(line)),
                             ensure_ascii=False, allow_nan=False), flush=True)
    else:
        request = json.load(sys.stdin)
        print(json.dumps(apply_request(arguments.registry, request), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
