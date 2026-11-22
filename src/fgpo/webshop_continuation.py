"""Preserve the complete WebShop V4 state before its 128-task evaluation."""
from __future__ import annotations

import argparse
from contextlib import closing, contextmanager
import json
import os
from pathlib import Path
import re
import shutil
import time

from . import checkpoint
from .checkpoint_gcs import configured
from .envs import make_env
from .protocol import (PROTOCOL, WEBSHOP_CONTINUATION_STUDIES, WEBSHOP_FIXED_MENU,
                       WEBSHOP_JOINT_OVERRIDES, WEBSHOP_JOINT_STUDY, WEBSHOP_PAIRED_STUDY,
                       WEBSHOP_RADIUS_OVERRIDES, WEBSHOP_RADIUS_STUDY,
                       require_protocol, require_training_identity, resolve_method_config)
from .runner import _file_lock, require_identity, write_json
from .webshop import PAIRED_TRAIN_ROUNDS, TRAIN_GROUPS, TRAIN_ROUNDS, VALIDATION_WORKERS


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_seed(cell):
    if cell.parent.name != "fgpo" or cell.parent.parent.name != "webshop" or cell.parents[2].name != "comparison":
        raise ValueError("Expected a comparison/webshop/fgpo/seed-N cell")
    if cell.name not in ("seed-0", "seed-1", "seed-2"):
        raise ValueError("The comparison uses seeds 0, 1 and 2")
    return int(cell.name[-1])


@contextmanager
def _staged_directory(destination):
    # Its caller holds the same lock as the eventual owner of destination.
    pending = destination.with_name(f".{destination.name}.webshop-incoming")
    if pending.exists():
        shutil.rmtree(pending)
    pending.mkdir()
    try:
        yield pending
        checkpoint._sync_directory(pending)
        os.replace(pending, destination)
        checkpoint._sync_directory(destination.parent)
    finally:
        if pending.exists():
            shutil.rmtree(pending)


def _read_state(directory):
    import torch

    return torch.load(directory / "state.pt", map_location="cpu", weights_only=False)


def _source_recipe(config, seed):
    if (config.get("protocol"), config.get("method"), config.get("benchmark"), config.get("seed")) != (
            PROTOCOL, "fgpo", "webshop", seed):
        raise ValueError("The source must be this seed's original WebShop V4 F-GPO cell")
    if (config.get("study_protocol") is not None or config["rounds"] != TRAIN_ROUNDS
            or WEBSHOP_FIXED_MENU not in config.get("adaptations", ()) or configured(config)):
        raise ValueError("The source must use the repaired WebShop menu, 60 rounds and a local checkpoint")


def _training_cut(path, ledger, version):
    if ledger["journal_path"] != str(path.resolve()):
        raise ValueError("Round-60 state names a different physical cost journal")
    if ledger["active_group_id"] is not None or ledger["active_round"] is not None:
        raise ValueError("A completed training cut cannot have an active round or group")
    # The original runner saves the completed state, records this publication,
    # then publishes training_summary before invoking official evaluation.
    with path.open("rb") as handle:
        handle.seek(ledger["durable_offset_bytes"])
        line = handle.readline()
        if line.endswith(b"\n"):
            row = json.loads(line)
            if (row["event_index"] == ledger["event_cursor"] + 1 and row["session_id"] == ledger["session_id"]
                    and row["role"] == "checkpoint.publish" and row["details"]["model_version"] == version):
                return handle.tell()
    raise ValueError("The original pre-evaluation checkpoint publication cost is absent")


def _copy_prefix(source, target, size):
    with source.open("rb") as incoming, target.open("xb") as outgoing:
        remaining = size
        block = b""
        while remaining:
            block = incoming.read(min(remaining, 1024 * 1024))
            if not block:
                raise ValueError(f"Journal is shorter than its committed cursor: {source}")
            outgoing.write(block)
            remaining -= len(block)
        if size and not block.endswith(b"\n"):
            raise ValueError(f"Journal cursor is not at a record boundary: {source}")
        outgoing.flush()
        os.fsync(outgoing.fileno())


def _pin_snapshot(source, archive, seed, training):
    root = source / "checkpoint"
    directory, metadata = checkpoint.snapshot_directory(root)
    if not re.fullmatch("[0-9a-f]{32}", directory.name):
        raise ValueError("The source checkpoint does not name a local committed snapshot")
    saved = _read_state(directory)
    trainer, ledger = saved["trainer"], saved["ledger"]
    state, source_config = trainer["state"], trainer["config"]
    if (not state.completed or state.phase != "complete" or state.round != TRAIN_ROUNDS
            or state.evaluation_completed):
        return None
    _source_recipe(source_config, seed)
    require_training_identity(_read_json(source / "config.json"), source_config)
    version = metadata["model_version"]
    if (version != training["optimizer_steps"] or metadata["base_model_path"] != source_config["model_path"]
            or not metadata["has_optimizer"] or not metadata["has_reference"] or metadata["has_teacher"]
            or not metadata.get("parameters_synced")):
        raise ValueError("Expected the exact completed V4 actor, Adam state and fixed reference")
    if (state.group_cursor or state.groups or state.reference_data is not None or state.advantages
            or state.credit_diagnostics or state.update_cursor is not None or state.published_batch_plan is not None):
        raise ValueError("Round-60 state still contains pending training work")
    if not all((directory / name).is_file() for name in ("optimizer.pt", "gradients.pt", "reference.pt")):
        # A rotation after CURRENT reads can remove a candidate while it is inspected.
        if _read_json(root / "CURRENT").get("snapshot") != directory.name:
            return None
        raise ValueError("The current optimizer, gradients or fixed reference are missing")
    tasks = _read_json(source / "official_tasks.json")
    if (len(tasks["train"]) != TRAIN_ROUNDS * TRAIN_GROUPS or tasks["train"] != state.task_order
            or len(tasks["test"]) != VALIDATION_WORKERS):
        raise ValueError("The source official task draw differs from its saved training order")
    cost_prefix = _training_cut(source / "costs.jsonl", ledger, version)
    receipt = {"protocol": PROTOCOL, "study_protocol": WEBSHOP_PAIRED_STUDY, "source_cell": str(source),
               "snapshot": directory.name, "seed": seed, "rounds": TRAIN_ROUNDS, "optimizer_steps": version,
               "training_log_bytes": state.training_log_offset, "cost_journal_bytes": cost_prefix,
               "serialized_cost_journal_bytes": ledger["durable_offset_bytes"]}
    with _staged_directory(archive) as pending:
        pinned_root = pending / "checkpoint"
        pinned = pinned_root / "snapshots" / directory.name
        checkpoint._link_tree(directory, pinned)
        if _read_json(root / "CURRENT")["snapshot"] != directory.name:
            # A rotating writer can unlink members before scandir sees them.
            raise FileNotFoundError("Source checkpoint rotated during the pin")
        checkpoint._sync_directory(pinned.parent)
        checkpoint._publish_pointer(pinned_root, directory.name)
        _copy_prefix(source / "training.jsonl", pending / "training.jsonl", state.training_log_offset)
        _copy_prefix(source / "costs.jsonl", pending / "costs.jsonl", cost_prefix)
        with (pending / "training.jsonl").open("rb") as handle:
            rows = [json.loads(line) for line in handle]
        if ([row["round"] for row in rows] != list(range(TRAIN_ROUNDS))
                or any(row["method"] != "fgpo" or row["protocol"] != PROTOCOL for row in rows)
                or rows[-1]["optimizer_steps"] != version):
            raise ValueError("The pinned training journal must contain the 60 committed rounds and cumulative Adam step")
        write_json(pending / "official_tasks.json", tasks, durable=True)
        write_json(pending / "pin.json", receipt, durable=True)
    return receipt


def pin(source, archive):
    source, archive = Path(source).resolve(), Path(archive).resolve()
    seed = _cell_seed(source)
    if archive.is_relative_to(source.parents[3]) or source.is_relative_to(archive):
        raise ValueError("The pin archive must live outside the original WebShop result tree")
    with _file_lock(archive.parent / f".{archive.name}.lock"):
        if archive.exists():
            receipt = _read_json(archive / "pin.json")
            if receipt["source_cell"] != str(source):
                raise ValueError("The archive already belongs to another source cell")
            checkpoint.snapshot_directory(archive / "checkpoint")
            return receipt
        while True:
            if (source / "config.json").is_file():
                _source_recipe(_read_json(source / "config.json"), seed)
            try:
                training = _read_json(source / "training_summary.json")
                if (training.get("status") == "complete" and training["rounds_completed"] == TRAIN_ROUNDS
                        and training["method"] == "fgpo" and training["benchmark"] == "webshop"
                        and training["protocol"] == PROTOCOL and training["seed"] == seed):
                    receipt = _pin_snapshot(source, archive, seed, training)
                    if receipt is not None:
                        return receipt
            except FileNotFoundError:
                # The original process can replace CURRENT and unlink the old
                # generation during a read/link. Retry from the pointer.
                pass
            if (source / ".finalization_complete.json").is_file():
                raise RuntimeError("The original cell finalized before its full pre-evaluation state was archived")
            time.sleep(1.)


def _require_source_completion(source, config, receipt):
    marker = _read_json(source / ".finalization_complete.json")
    summary = _read_json(source / "cell_summary.json")
    require_identity(summary, config)
    training, evaluation = summary["training"], summary["evaluation"]["test"]
    require_identity(evaluation, config)
    version = receipt["optimizer_steps"]
    if (marker["protocol"] != PROTOCOL or not marker["terminal_export_completed"]
            or marker["snapshot"] != summary["completed_checkpoint"]["snapshot"]
            or summary["status"] != "complete" or summary["finalization_stage"] != "finalized"
            or training["status"] != "complete" or training["rounds_completed"] != TRAIN_ROUNDS
            or training["optimizer_steps"] != version or evaluation["model_version"] != version
            or not evaluation["dataset_complete"] or evaluation["evaluated_count"] != VALIDATION_WORKERS):
        raise ValueError("The pinned source actor has not completed its original 128-task official evaluation")


def prepared_origin(target, config):
    target = Path(target).resolve()
    origin = _read_json(target / "continuation.json")
    if (origin["target_cell"] != str(target) or origin["study_protocol"] != config["study_protocol"]
            or origin["seed"] != config["seed"] or origin["rounds"] != TRAIN_ROUNDS
            or origin["target_rounds"] != PAIRED_TRAIN_ROUNDS):
        raise ValueError("The prepared continuation belongs to another cell or budget")
    return origin


def prepare(archive, target, config):
    archive, target = Path(archive).resolve(), Path(target).resolve()
    receipt = _read_json(archive / "pin.json")
    source = Path(receipt["source_cell"])
    seed = _cell_seed(target)
    if (seed != receipt["seed"] or target.is_relative_to(source.parents[3]) or target.is_relative_to(archive)
            or archive.is_relative_to(target)):
        raise ValueError("Prepare the corresponding seed in a separate 150-round result tree")
    options = resolve_method_config(dict(config, suite="comparison", method="fgpo", benchmark="webshop", seed=seed))
    study = options.get("study_protocol")
    if study not in WEBSHOP_CONTINUATION_STUDIES or configured(options):
        raise ValueError("The target requires a 150-round local WebShop continuation protocol")
    require_protocol(options)
    directory, metadata = checkpoint.snapshot_directory(archive / "checkpoint")
    saved = _read_state(directory)
    trainer = saved["trainer"]
    source_config = trainer["config"]
    state, ledger = trainer["state"], saved["ledger"]
    _source_recipe(source_config, seed)
    if ((receipt["protocol"], receipt["study_protocol"], receipt["rounds"])
            != (PROTOCOL, WEBSHOP_PAIRED_STUDY, TRAIN_ROUNDS)
            or directory.name != receipt["snapshot"] or metadata["model_version"] != receipt["optimizer_steps"]
            or not metadata["has_optimizer"] or not metadata["has_reference"] or metadata["has_teacher"]
            or not metadata.get("parameters_synced") or metadata["base_model_path"] != source_config["model_path"]
            or not state.completed or state.phase != "complete" or state.round != TRAIN_ROUNDS
            or state.evaluation_completed or state.group_cursor or state.groups or state.reference_data is not None
            or state.advantages or state.credit_diagnostics or state.update_cursor is not None
            or state.published_batch_plan is not None or state.training_log_offset != receipt["training_log_bytes"]
            or ledger["active_group_id"] is not None or ledger["active_round"] is not None
            or ledger["durable_offset_bytes"] != receipt["serialized_cost_journal_bytes"]
            or not all((directory / name).is_file() for name in ("optimizer.pt", "gradients.pt", "reference.pt"))):
        raise ValueError("The archive must contain the sealed round-60 pre-evaluation training state")
    # Reverse the declared migration only for comparison against the source.
    source_overrides = {"rounds": TRAIN_ROUNDS}
    if study in (WEBSHOP_JOINT_STUDY, WEBSHOP_RADIUS_STUDY):
        source_overrides.update(entropy_coef=.001, credit_transfer_max=.5)
    if study == WEBSHOP_RADIUS_STUDY:
        source_overrides.update(credit_radius=.10, credit_prior_weight=1.)
    prior = {key: value for key, value in options.items() if key != "study_protocol"}
    prior.update(source_overrides)
    prior["method_configs"] = {**prior.get("method_configs", {}),
        "fgpo": {**prior.get("method_configs", {}).get("fgpo", {}), **source_overrides}}
    require_training_identity(source_config, prior)
    if Path(source_config["webshop_data_root"]).resolve() != Path(options["webshop_data_root"]).resolve():
        raise ValueError("The continuation must use the same native WebShop catalog")
    _require_source_completion(source, source_config, receipt)
    with _file_lock(target.parent / f".{target.name}.lock", nonblocking=True):
        if target.exists():
            origin = prepared_origin(target, options)
            if origin["archive"] != str(archive) or origin["snapshot"] != receipt["snapshot"]:
                raise ValueError("The target was already prepared from another exact snapshot")
            require_training_identity(_read_json(target / "config.json"), options)
            return origin
        with _staged_directory(target) as pending:
            started = time.monotonic()
            with closing(make_env(options, "webshop", pending / "environments/catalog", seed=seed)) as catalog:
                tasks = {"train": catalog.list_tasks("train", train_rounds=PAIRED_TRAIN_ROUNDS),
                         "test": catalog.list_tasks("test")}
            old_tasks = _read_json(archive / "official_tasks.json")
            if (len(tasks["train"]) != PAIRED_TRAIN_ROUNDS * TRAIN_GROUPS
                    or tasks["train"][:TRAIN_ROUNDS * TRAIN_GROUPS] != old_tasks["train"]
                    or old_tasks["train"] != state.task_order or tasks["test"] != old_tasks["test"]):
                raise ValueError("The native draw does not extend the 960 original groups and preserve the 128 test tasks")
            state.task_order = tasks["train"]
            state.completed = False
            state.phase = "collect"
            state.evaluation_completed.clear()
            trainer["config"] = options
            saved["ledger"]["journal_path"] = str(target / "costs.jsonl")
            # Writable journals and the trainer cursor are new inodes. The
            # large immutable parameter snapshots keep their original storage.
            _copy_prefix(archive / "training.jsonl", pending / "training.jsonl", receipt["training_log_bytes"])
            _copy_prefix(archive / "costs.jsonl", pending / "costs.jsonl", receipt["cost_journal_bytes"])
            _bootstrap_checkpoint(directory, pending / "checkpoint", saved)
            write_json(pending / "official_tasks.json", tasks, durable=True)
            write_json(pending / "config.json", options, durable=True)
            origin = {**receipt, "study_protocol": study, "archive": str(archive), "target_cell": str(target),
                      "target_rounds": PAIRED_TRAIN_ROUNDS, "preparation_cpu_wall_seconds": time.monotonic() - started}
            if study == WEBSHOP_JOINT_STUDY:
                origin.update(training_overrides=WEBSHOP_JOINT_OVERRIDES, overrides_start_round=TRAIN_ROUNDS + 1)
            if study == WEBSHOP_RADIUS_STUDY:
                origin.update(variant="radius", overrides_start_round=TRAIN_ROUNDS + 1,
                              training_overrides={**WEBSHOP_RADIUS_OVERRIDES, "credit_radius": options["credit_radius"]})
            write_json(pending / "continuation.json", origin, durable=True)
        return origin


def _bootstrap_checkpoint(source, root, state):
    import torch

    target = root / "snapshots" / source.name
    target.mkdir(parents=True)
    for member in source.iterdir():
        if member.is_dir():
            checkpoint._link_tree(member, target / member.name)
        elif member.name != "state.pt":
            os.link(member, target / member.name)
    torch.save(state, target / "state.pt")
    checkpoint._sync_file(target / "state.pt")
    checkpoint._sync_directory(target)
    checkpoint._sync_directory(target.parent)
    checkpoint._publish_pointer(root, source.name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    watcher = subcommands.add_parser("pin", help="Wait for and archive one original seed's training cut")
    watcher.add_argument("--source-cell", type=Path, required=True)
    watcher.add_argument("--archive", type=Path, required=True)
    migrator = subcommands.add_parser("prepare", help="Create a 150-round cell from the pin after official evaluation")
    migrator.add_argument("--archive", type=Path, required=True)
    migrator.add_argument("--target-cell", type=Path, required=True)
    migrator.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "pin":
        receipt = pin(args.source_cell, args.archive)
    else:
        receipt = prepare(args.archive, args.target_cell, _read_json(args.config))
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
