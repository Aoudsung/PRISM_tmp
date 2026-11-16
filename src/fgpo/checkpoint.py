"""Exact new-study snapshots: model, Adam state, references, RNG and pending work.

Each CURRENT pointer names a complete snapshot. Unchanged model/optimizer files
are hard-linked during rollout collection; this is a storage transaction, with
no reconstruction of missing moments and no legacy checkpoint conversion.
"""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import random
import shutil
import uuid
import time

import numpy as np
import torch

FORMAT = "fgpo-exact-checkpoint-v1"
COMPLETED_FORMAT = "fgpo-completed-model-v1"


def seed_everything(seed,gpu_ids):
    random.seed(seed)
    np.random.seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    torch.random.set_rng_state(generator.get_state())
    for gpu in gpu_ids:
        with torch.cuda.device(gpu):
            torch.cuda.init()
            torch.cuda.manual_seed(seed)


def capture_rng(gpu_ids):
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch_cpu":torch.random.get_rng_state(),
            "cuda":[torch.cuda.get_rng_state(gpu) for gpu in gpu_ids],"gpu_ids":tuple(gpu_ids)}


def restore_rng(state,gpu_ids):
    if len(state["cuda"]) != len(gpu_ids):
        raise ValueError("Exact resume requires the saved number of CUDA RNG streams")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    for gpu,value in zip(gpu_ids,state["cuda"]):
        torch.cuda.set_rng_state(value,gpu)


def _link_tree(source,target):
    target.mkdir(parents=True,exist_ok=True)
    for child in source.iterdir():
        if child.is_dir():
            _link_tree(child,target/child.name)
        else:
            os.link(child,target/child.name)
    # Callers sync legacy source files once before linking them. Normal reuse
    # therefore only needs to flush these new directory entries.
    _sync_directory(target)


def snapshot_directory(root):
    root = Path(root)
    identity = json.loads((root/"CURRENT").read_text(encoding="utf-8"))
    directory = root/"snapshots"/identity["snapshot"]
    metadata = json.loads((directory/"metadata.json").read_text(encoding="utf-8"))
    if metadata["format"] == COMPLETED_FORMAT:
        raise ValueError("Completed model checkpoints retain no optimizer state and cannot resume incomplete training")
    if metadata["format"] != FORMAT:
        raise ValueError("Only exact F-GPO study checkpoints can resume this engine")
    return directory,metadata


def save(root,*,policy,optimizer,reference,teacher,trainer_state,ledger_state,rng_state,model_version):
    started = time.clock_gettime(time.CLOCK_MONOTONIC)
    root = Path(root)
    snapshots = root/"snapshots"
    snapshots.mkdir(parents=True,exist_ok=True)
    previous,previous_metadata = (snapshot_directory(root) if (root/"CURRENT").exists() else (None,None))
    if previous is not None:
        if not previous_metadata.get("parameters_synced",False):
            # Older v3 writers did not sync parameter files before publishing.
            # Adopt their complete snapshot durably once before reusing links.
            _sync_tree(previous)
        _sync_file(root/"CURRENT")
        _sync_directory(snapshots)
        _sync_directory(root)
        # One cell has one writer. Discard abandoned UUIDs before allocating
        # another full model/Adam generation, keeping CURRENT intact.
        _reclaim_superseded_snapshots(root,previous.name)
    identity = uuid.uuid4().hex
    directory = snapshots/identity
    directory.mkdir()
    same_model = previous_metadata is not None and previous_metadata["model_version"] == model_version
    try:
        if same_model:
            _link_tree(previous/"model",directory/"model")
        else:
            policy.save_model(directory/"model")
            _sync_tree(directory/"model")
        if optimizer is not None:
            if same_model:
                os.link(previous/"optimizer.pt",directory/"optimizer.pt")
            else:
                torch.save(optimizer.state_dict(),directory/"optimizer.pt")
                _sync_file(directory/"optimizer.pt")
            torch.save(optimizer.gradient_state(),directory/"gradients.pt")
            _sync_file(directory/"gradients.pt")
        if reference is not None:
            if previous is not None and (previous/"reference.pt").exists():
                os.link(previous/"reference.pt",directory/"reference.pt")
            else:
                torch.save(reference.weights,directory/"reference.pt")
                _sync_file(directory/"reference.pt")
        if teacher is not None:
            if same_model:
                os.link(previous/"teacher.pt",directory/"teacher.pt")
            else:
                torch.save(teacher.weights,directory/"teacher.pt")
                _sync_file(directory/"teacher.pt")
        policy.record("checkpoint","checkpoint.serialize_parameters",seconds=time.clock_gettime(time.CLOCK_MONOTONIC)-started,
                      details={"model_version":model_version,"reused_parameter_files":same_model})
        actual_ledger_state = {**ledger_state,**policy.ledger.state_dict()}
        if policy.ledger.path is not None:
            _sync_file(policy.ledger.path)
        torch.save({"trainer":trainer_state,"ledger":actual_ledger_state,"rng":rng_state},directory/"state.pt")
        _sync_file(directory/"state.pt")
        metadata = {"format":FORMAT,"model_version":model_version,"base_model_path":policy.model_path,
                    "parameters_synced":True,
                    "adaptation":policy.adaptation,"model":policy.metadata,
                    "has_optimizer":optimizer is not None,"has_reference":reference is not None,
                    "has_teacher":teacher is not None}
        (directory/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        _sync_file(directory/"metadata.json")
        _sync_directory(directory)
        _sync_directory(snapshots)
        _sync_directory(root)
        _sync_directory(root.parent)
        _publish_pointer(root,identity)
    except BaseException:
        _discard_unpublished(root,directory,identity)
        raise
    _reclaim_superseded_snapshots(root,identity)
    return directory


def load(root):
    directory,metadata = snapshot_directory(root)
    state = torch.load(directory/"state.pt",map_location="cpu",weights_only=False)
    state.update(metadata=metadata,model_path=str(directory/"model"))
    if metadata["has_optimizer"]:
        state["optimizer"] = torch.load(directory/"optimizer.pt",map_location="cpu",weights_only=False)
        state["gradients"] = torch.load(directory/"gradients.pt",map_location="cpu",weights_only=False)
    if metadata["has_reference"]:
        state["reference"] = torch.load(directory/"reference.pt",map_location="cpu",weights_only=False)
    if metadata["has_teacher"]:
        state["teacher"] = torch.load(directory/"teacher.pt",map_location="cpu",weights_only=False)
    return state


def _sync_directory(path):
    descriptor = os.open(path,os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_file(path):
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def _sync_tree(directory):
    for child in directory.iterdir():
        if child.is_dir():
            _sync_tree(child)
        else:
            _sync_file(child)
    _sync_directory(directory)


def _publish_pointer(root,identity):
    incoming = root/("CURRENT."+identity)
    with incoming.open("w",encoding="utf-8") as handle:
        handle.write(json.dumps({"snapshot":identity})+"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(incoming,root/"CURRENT")
    _sync_directory(root)


def _discard_unpublished(root,directory,identity):
    """Remove only this call's unfinished snapshot; never unlink CURRENT data."""
    pointer = root/"CURRENT"
    if pointer.exists() and json.loads(pointer.read_text(encoding="utf-8"))["snapshot"] == identity:
        # The atomic rename succeeded; a later directory-sync failure must
        # leave both complete snapshots available rather than remove CURRENT.
        return
    incoming = root/("CURRENT."+identity)
    if incoming.exists():
        incoming.unlink()
    if directory.exists():
        shutil.rmtree(directory)
    _sync_directory(directory.parent)
    _sync_directory(root)


def _reclaim_superseded_snapshots(root,current):
    """Collect this writer's other UUID directories while keeping durable CURRENT."""
    snapshots = root/"snapshots"
    for directory in snapshots.iterdir():
        name = directory.name
        if (name == current or len(name) != 32 or
                any(character not in "0123456789abcdef" for character in name)):
            continue
        if directory.is_symlink() or not directory.is_dir():
            continue
        shutil.rmtree(directory)
    _sync_directory(snapshots)


def prepare_completed(full_root,final_root,*,provenance):
    """Publish final deployment artifacts while the full resume tree still exists."""
    full_root,final_root = Path(full_root),Path(final_root)
    cell = final_root.parent.resolve()
    if (full_root.name!="checkpoint" or final_root.name!="final_checkpoint" or
        full_root.is_symlink() or final_root.is_symlink() or full_root.parent.resolve()!=cell):
        raise ValueError("Completed-cell finalization is confined to this cell's checkpoint directories")
    source,metadata = snapshot_directory(full_root)
    if not metadata.get("parameters_synced",False):
        _sync_tree(source)
    previous = None
    if (final_root/"CURRENT").exists():
        old = json.loads((final_root/"CURRENT").read_text(encoding="utf-8"))["snapshot"]
        previous = final_root/"snapshots"/old
        old_metadata = json.loads((previous/"metadata.json").read_text(encoding="utf-8"))
        if old_metadata["format"]!=COMPLETED_FORMAT or old_metadata["cell_root"]!=str(cell):
            raise ValueError("Existing final artifacts belong to a different completion contract")
        if not old_metadata.get("parameters_synced",False):
            _sync_tree(previous)
        _sync_file(final_root/"CURRENT")
        _sync_directory(previous.parent)
        _sync_directory(final_root)
        _reclaim_superseded_snapshots(final_root,old)
    identity = uuid.uuid4().hex
    directory = final_root/"snapshots"/identity
    directory.mkdir(parents=True)
    try:
        _link_tree(source/"model",directory/"model")
        # These files have already been synced by save(). Reuse their bytes;
        # the new metadata and hard-link directory entries still need a sync.
        os.link(source/"state.pt",directory/"state.pt")
        if metadata["has_teacher"]:
            os.link(source/"teacher.pt",directory/"ema_teacher.pt")
        completed_metadata = {**metadata,"format":COMPLETED_FORMAT,"cell_root":str(cell),
            "parameters_synced":True,
            "training_resume_supported":False,"has_optimizer":False,"has_reference":False,
            "has_teacher":False,"ema_teacher_retained":bool(metadata["has_teacher"]),
            "retain_training_artifacts":bool(provenance.get("retain_training_artifacts",True)),
            "source_full_snapshot":source.name,"state_contents":"config_trainer_provenance_rng_ledger",
            "official_results":"cell_summary.json and evaluation journals in the owning cell"}
        (directory/"metadata.json").write_text(json.dumps(completed_metadata,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        (directory/"provenance.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        _sync_file(directory/"metadata.json")
        _sync_file(directory/"provenance.json")
        _sync_directory(directory)
        _sync_directory(directory.parent)
        _sync_directory(final_root)
        _sync_directory(cell)
        _publish_pointer(final_root,identity)
    except BaseException:
        _discard_unpublished(final_root,directory,identity)
        raise
    _reclaim_superseded_snapshots(final_root,identity)
    return {"path":str(final_root.resolve()),"format":COMPLETED_FORMAT,"snapshot":identity,
            "source_full_snapshot":source.name,"training_resume_supported":False,
            "retain_training_artifacts":bool(provenance.get("retain_training_artifacts",True))}


def _reclaim_training_artifacts(cell):
    """Reclaim only the owning cell's declared optional raw training outputs."""
    reclaimed = False
    for name in ("groups","trajectories"):
        directory = cell/name
        if directory.is_symlink():
            raise ValueError(f"Raw artifact cleanup refuses linked directory {directory}")
        if directory.exists():
            if not directory.is_dir():
                raise ValueError(f"Raw artifact directory is not a directory: {directory}")
            shutil.rmtree(directory)
            reclaimed = True
    for path in cell.glob("reference-round-*.pt"):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Raw artifact cleanup requires a direct regular file: {path}")
        path.unlink()
        reclaimed = True
    if reclaimed:
        _sync_directory(cell)
    return reclaimed


def reclaim_completed_resume_state(summary_path):
    """Reclaim only the owning new cell's full state after durable completion."""
    summary_path = Path(summary_path)
    cell = summary_path.parent.resolve()
    with summary_path.open("rb") as handle:
        summary = json.load(handle)
        os.fsync(handle.fileno())
    artifact = summary.get("completed_checkpoint",{})
    if summary.get("status")!="complete" or artifact.get("format")!=COMPLETED_FORMAT:
        raise ValueError("Full state is retained until the completed-model cell summary is published")
    final_root = cell/"final_checkpoint"
    if final_root.is_symlink() or Path(artifact["path"]).resolve()!=final_root.resolve():
        raise ValueError("Completed summary does not name this cell's retained final checkpoint")
    pointer = json.loads((final_root/"CURRENT").read_text(encoding="utf-8"))
    if pointer["snapshot"]!=artifact["snapshot"]:
        raise ValueError("Completed summary and retained final snapshot differ")
    metadata = json.loads((final_root/"snapshots"/pointer["snapshot"]/"metadata.json").read_text(encoding="utf-8"))
    if (metadata["format"]!=COMPLETED_FORMAT or metadata["cell_root"]!=str(cell) or
        metadata["source_full_snapshot"]!=artifact["source_full_snapshot"]):
        raise ValueError("Completed checkpoint ownership differs from the published cell")
    retain_raw = bool(artifact.get("retain_training_artifacts",True))
    if retain_raw != bool(metadata.get("retain_training_artifacts",True)):
        raise ValueError("Completed summary and retained snapshot have different raw-artifact policies")
    _sync_directory(cell)
    full_root = cell/"checkpoint"
    reclaiming = cell/(".completed_resume_state."+artifact["source_full_snapshot"])
    if full_root.exists():
        if full_root.is_symlink() or reclaiming.exists():
            raise ValueError("Full state ownership differs from the published completion")
        source,_ = snapshot_directory(full_root)
        if source.name!=artifact["source_full_snapshot"]:
            raise ValueError("Full resume state has changed since completion; it remains retained")
        os.replace(full_root,reclaiming)
        _sync_directory(cell)
    reclaimed = False
    if reclaiming.is_symlink():
        raise ValueError("A linked checkpoint tree cannot be reclaimed by cell finalization")
    if reclaiming.exists():
        shutil.rmtree(reclaiming)
        _sync_directory(cell)
        reclaimed = True
    if not retain_raw:
        reclaimed = _reclaim_training_artifacts(cell) or reclaimed
    return reclaimed
