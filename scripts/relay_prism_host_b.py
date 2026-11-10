"""Run on Host A: relay one immutable Host B formal run to its coordinator."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prism_evolve.native_coordination import apply_request, atomic_json, now, pack


REMOTE = """
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
manifest = root / 'manifest.json'
replies = json.load(sys.stdin)
if not manifest.exists():
    print(json.dumps({'preparing': True}))
    raise SystemExit(0)
m = json.loads(manifest.read_text())
sys.path.insert(0, str(pathlib.Path(m['source_root']) / 'src'))
from prism_evolve.native_coordination import deliver_reply, mailbox_requests
delivered = [deliver_reply(root, reply) for reply in replies]
state = root / 'pipeline_state.json'
print(json.dumps({'manifest': m, 'requests': mailbox_requests(root), 'delivered': delivered,
                  'pipeline': json.loads(state.read_text()) if state.exists() else None}))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--remote-run-dir", required=True)
    args = parser.parse_args()
    if sys.platform != "linux" or socket.gethostname() != "mgt01":
        raise RuntimeError("This relay must run on Host A (mgt01).")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
               "chenxiangwei@10.8.128.241",
               shlex.join(["python3", "-c", REMOTE, args.remote_run_dir])]
    registry = None
    replies = []
    while True:
        response = subprocess.run(command, input=json.dumps(replies), text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
        if response.returncode:
            raise RuntimeError(response.stderr[-2000:])
        report = json.loads(response.stdout)
        if report.get("preparing"):
            time.sleep(5)
            continue
        if not {item["request_id"] for item in replies} <= set(report["delivered"]):
            raise RuntimeError("Host B did not acknowledge the exact coordination replies.")
        manifest = report["manifest"]
        if manifest["host"] != "ubuntu" or manifest["run_dir"] != args.remote_run_dir:
            raise RuntimeError("Host B run identity changed.")
        if registry is None:
            config = manifest["config"]
            registry = Path(config["coordination"]["registry_root"]) / config["coordination"]["session_id"] / "registry.json"
            if not registry.exists():
                coordinator = deepcopy(manifest)
                coordinator.update(site_id="host_a", host=socket.gethostname(),
                                   run_id=args.run_dir.name, run_dir=str(args.run_dir),
                                   manifest_id="host_a/" + args.run_dir.name)
                coordinator["config"].update(site_id="host_a", execution_host=socket.gethostname(),
                                             backbones=[], execution_backbones=[])
                atomic_json(args.run_dir / "manifest.json", coordinator)
                atomic_json(args.run_dir / "schedule.json", [])
                request = {"protocol": "native-cohort-claims-v1", "request_id": uuid.uuid4().hex,
                           "session_id": config["coordination"]["session_id"], "site_id": "host_a",
                           "host": socket.gethostname(), "registration_manifest_id": coordinator["manifest_id"],
                           "operation": "register", "payload_z": pack({"manifest_text": json.dumps(coordinator),
                                                                        "schedule_text": "[]", "gpus": []})}
                reply = apply_request(registry, request)
                if "error" in reply:
                    raise RuntimeError(reply["error"])
        replies = [apply_request(registry, request) for request in report["requests"]]
        errors = [reply["error"] for reply in replies if "error" in reply]
        status = {"at": now(), "pending_replies": len(replies), "errors": errors,
                  "remote_pipeline": report["pipeline"], "registry": str(registry)}
        atomic_json(args.run_dir / "relay-status.json", status)
        if report["requests"]:
            print(json.dumps({"at": now(), "operations": [r["operation"] for r in report["requests"]],
                              "errors": errors}), flush=True)
        if errors:
            # Deliver the retained errors once before stopping the transport.
            subprocess.run(command, input=json.dumps(replies), text=True, check=True,
                           stdout=subprocess.DEVNULL, timeout=45)
            raise RuntimeError("; ".join(errors))
        if not replies and report["pipeline"] and report["pipeline"]["status"] in {"done", "failed", "blocked"}:
            print(json.dumps({"at": now(), "pipeline_status": report["pipeline"]["status"]}), flush=True)
            return
        time.sleep(0.25 if replies else 5)


if __name__ == "__main__":
    main()
