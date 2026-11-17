#!/usr/bin/env python3
"""Explicit GCS import, actor/data restore and physical-log publication.

Run only in the designated remote environment. Credentials are read from the
operator-refreshed FGPO_GCS_ACCESS_TOKEN_FILE, never command arguments or config.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlsplit

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from fgpo.checkpoint_gcs import GCSCheckpoint, GCSClient, StorageError
from fgpo.protocol import METHODS, resolve_method_config, require_protocol


def restore_asset(uri, destination):
    """Stream an existing trusted cold tar.zst asset directly into tar."""
    started = time.monotonic()
    parts = urlsplit(uri)
    if parts.scheme != "gs" or not parts.netloc or not parts.path.endswith(".tar.zst"):
        raise ValueError("Cold asset must be the existing gs://bucket/member.tar.zst")
    client = GCSClient(parts.netloc)
    descriptor = client.descriptor(client.metadata(parts.path.lstrip("/")))
    destination.mkdir(parents=True, exist_ok=True)
    child = subprocess.Popen(["tar", "--use-compress-program=zstd", "-xf", "-", "-C", str(destination)], stdin=subprocess.PIPE)
    try:
        with client.response("GET", client._url(descriptor["name"], alt="media", generation=descriptor["generation"]),
                             operation="restore_cold_asset") as response:
            received = 0
            while True:
                block = response.read(client.chunk_bytes)
                if not block:
                    break
                child.stdin.write(block)
                received += len(block)
            if received != descriptor["size"]:
                raise StorageError("restore_cold_asset", "short_read")
        child.stdin.close()
        code = child.wait()
        if code:
            raise RuntimeError(f"Cold asset extraction failed (tar status={code})")
    except BaseException:
        if child.poll() is None:
            child.kill()
        child.wait()
        raise
    return {"asset": uri, "destination": str(destination), "bytes": received, "seconds": time.monotonic() - started}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("import-local", "restore", "restore-final", "status", "pull", "sync-logs", "reclaim-completed"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", type=Path, required=True)
        sub.add_argument("--cell-dir", type=Path, required=command not in ("status", "pull"))
        sub.add_argument("--suite", choices=("comparison", "ablation"), default="comparison")
        sub.add_argument("--method", choices=sorted(METHODS), default="fgpo")
        sub.add_argument("--benchmark", choices=("alfworld", "appworld"), required=True)
        sub.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
        if command == "import-local":
            sub.add_argument("--activate-local-pointer", action="store_true",
                             help="Replace local CURRENT only after the old process has exited and migration is authorized")
        if command == "pull":
            sub.add_argument("--dest-dir", type=Path, default=None)
            sub.add_argument("--with-final-model", action="store_true")
    asset = commands.add_parser("restore-asset")
    asset.add_argument("--uri", required=True)
    asset.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "restore-asset":
        result = restore_asset(args.uri, args.destination)
    else:
        config = resolve_method_config(dict(json.loads(args.config.read_text()), suite=args.suite,
                                            benchmark=args.benchmark, method=args.method, seed=args.seed))
        require_protocol(config)
        if config.get("checkpoint_backend") != "gcs":
            raise ValueError("This CLI requires explicit checkpoint_backend=gcs")
        cell_dir = args.cell_dir if args.cell_dir is not None else (
            args.dest_dir if getattr(args, "dest_dir", None) is not None else Path("."))
        storage = GCSCheckpoint(config, cell_dir / "checkpoint")
        if args.command == "import-local":
            result = storage.import_local_snapshot(activate_local_pointer=args.activate_local_pointer)
        elif args.command == "restore":
            result = storage.restore_cell()
        elif args.command == "restore-final":
            result = storage.restore_final()
        elif args.command == "sync-logs":
            result = storage.publish_logs()
        elif args.command == "reclaim-completed":
            result = {"reclaimed": storage.reclaim_completed(cell_dir / "cell_summary.json")}
        elif args.command == "pull":
            dest = args.dest_dir if args.dest_dir is not None else cell_dir
            result = storage.pull(dest, with_final_model=args.with_final_model)
        else:
            # A clean missing CURRENT/FINAL is a status result, never a backend
            # fallback. Transport/auth errors still terminate the command.
            result = storage.status()
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except StorageError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from None
