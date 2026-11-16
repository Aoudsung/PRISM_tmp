"""Rebase one copied, stopped F-GPO cell onto the authorized Host B resources.

Run remotely with the Host B interpreter, after copying the complete cell and
its unchanged costs.jsonl into /home/chenxiangwei/workspace/fgpo-reproduction-20261116.

Example:
  python scripts/relocate_fgpo_cell.py --cell /home/chenxiangwei/workspace/fgpo-reproduction-20261116/outputs/copied-full/alfworld/fgpo/seed-0 --resource-config /home/chenxiangwei/workspace/fgpo-reproduction-20261116/configs/host-b-resources.json --physical-gpu 0

The resource JSON supplies install_root, model_path, python_executable and the
benchmark's data/Python locations. A complete study config is also accepted;
only the explicit machine-resource fields below are consumed. Scientific
configuration, pending groups, sampled IDs, model/optimizer/reference files,
RNG streams, and physical-cost journal contents are never rewritten.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import uuid


TARGET_ROOT = Path('/home/chenxiangwei/workspace/fgpo-reproduction-20261116')
FORMAT = 'fgpo-exact-checkpoint-v1'
PATH_FIELDS = ('install_root', 'model_path', 'python_executable',
               'alfworld_python_executable', 'alfworld_data_root', 'appworld_data_root')
RESOURCE_FIELDS = ('activation_offload', 'execution_host', 'site_id')


def _write_json(path, value):
    with path.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _changes(old, new):
    return {key: {'old': old.get(key), 'new': new.get(key)}
            for key in old.keys() | new.keys() if old.get(key) != new.get(key)}


def _resource_overrides(resources, cell, physical_ids, benchmark):
    if Path(resources['install_root']).resolve() != TARGET_ROOT:
        raise ValueError('This relocation targets only the explicitly authorized Host B project root.')
    required = {'install_root', 'model_path', 'python_executable', benchmark + '_data_root'}
    if benchmark == 'alfworld':
        required.add('alfworld_python_executable')
    missing = required - resources.keys()
    if missing:
        raise ValueError('Host B resource config is missing: ' + ', '.join(sorted(missing)))
    result = {}
    for key in PATH_FIELDS:
        if key in resources:
            if not isinstance(resources[key], str) or not Path(resources[key]).is_absolute():
                raise ValueError(f'Host B {key} must be an explicit absolute path.')
            result[key] = resources[key]
    for key in RESOURCE_FIELDS:
        if key in resources:
            result[key] = deepcopy(resources[key])
    excluded = list(resources.get('excluded_gpu_ids', ()))
    if any(device in excluded for device in physical_ids):
        raise ValueError('The requested Host B allocation includes a device excluded by its resource config.')
    logical = list(range(len(physical_ids)))
    result.update(artifact_dir=str(cell / 'trajectories'), excluded_gpu_ids=excluded,
                  gpu_ids=logical, logical_gpu_ids=logical, physical_gpu_ids=list(physical_ids),
                  cuda_visible_devices=','.join(map(str, physical_ids)), cuda_device_order='PCI_BUS_ID',
                  gpu_id_semantics='Engine_gpu_ids_are_process_local_logical_indices')
    if 'gpu_free_memory_gib' in resources:
        measured = resources['gpu_free_memory_gib']
        result['gpu_free_memory_gib'] = {
            logical_id: measured[physical] if physical in measured else measured[str(physical)]
            for logical_id, physical in enumerate(physical_ids)}
    return result


def _rebase_config(original, overrides):
    updated = deepcopy(original)
    updated.update(deepcopy(overrides))
    # An A-side measurement must not masquerade as a B-side measurement. Without
    # an explicit B map, the existing policy measures its actual allocated GPUs.
    if 'gpu_free_memory_gib' not in overrides:
        updated.pop('gpu_free_memory_gib', None)
    method = original['method']
    selected = updated.get('method_configs', {}).get(method)
    if selected is not None:
        for key in tuple(selected):
            if key in overrides:
                selected[key] = deepcopy(overrides[key])
        if 'gpu_free_memory_gib' not in overrides:
            selected.pop('gpu_free_memory_gib', None)
    return updated


def relocate(cell, resources, physical_ids):
    cell = Path(cell).resolve()
    if not cell.is_relative_to(TARGET_ROOT) or cell == TARGET_ROOT:
        raise ValueError('Only the copied cell beneath the authorized Host B root may be modified.')
    if (cell / 'cell_summary.json').exists() or (cell / 'final_checkpoint').exists():
        raise ValueError('This utility relocates the stopped incomplete cells, not completed final-model artifacts.')
    if not physical_ids or len(set(physical_ids)) != len(physical_ids) or any(device not in (0, 1, 2) for device in physical_ids):
        raise ValueError('Select distinct physical devices from the three authorized Host B GPUs.')
    config_path = cell / 'config.json'
    original_config = json.loads(config_path.read_text(encoding='utf-8'))
    pointer = json.loads((cell / 'checkpoint' / 'CURRENT').read_text(encoding='utf-8'))
    identity = pointer['snapshot']
    if not isinstance(identity, str) or Path(identity).name != identity or identity in ('.', '..'):
        raise ValueError('CURRENT must name one snapshot directory in the copied cell.')
    snapshot = cell / 'checkpoint' / 'snapshots' / identity
    if snapshot.resolve().parent != (cell / 'checkpoint' / 'snapshots').resolve():
        raise ValueError('The copied snapshot must remain within its owning cell.')
    metadata = json.loads((snapshot / 'metadata.json').read_text(encoding='utf-8'))
    if metadata['format'] != FORMAT:
        raise ValueError('Relocation requires the full exact-resume F-GPO checkpoint format.')
    journal = cell / 'costs.jsonl'
    if journal.is_symlink() or not journal.is_file():
        raise FileNotFoundError('Copy the original physical-cost journal into this cell before relocation.')

    # Unpickling the DTOs imports their definitions; no Engine, model, environment
    # or CUDA operation is started. Only the small trainer/RNG state file is read.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    import torch
    state_path = snapshot / 'state.pt'
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    original_trainer_config = state['trainer']['config']
    for key in ('method', 'benchmark', 'seed'):
        if original_trainer_config[key] != original_config[key]:
            raise ValueError(f'Copied config and checkpoint differ at their cell identity {key}.')
    logical = tuple(range(len(physical_ids)))
    if len(state['rng']['cuda']) != len(physical_ids) or tuple(state['rng']['gpu_ids']) != logical:
        raise ValueError('The target allocation must preserve every saved logical CUDA RNG stream; none can be truncated or merged.')
    overrides = _resource_overrides(resources, cell, physical_ids, original_config['benchmark'])
    updated_config = _rebase_config(original_config, overrides)
    updated_trainer_config = _rebase_config(original_trainer_config, overrides)
    old_journal = state['ledger']['journal_path']
    state['trainer']['config'] = updated_trainer_config
    state['ledger']['journal_path'] = str(journal)

    # Completed split summaries have an operational model-path identity read by
    # runner.require_identity. Metric values and the actual evaluation rows stay
    # unchanged. Task IDs are relative dataset IDs in both envs, so
    # official_tasks and EngineState.task_order need no path substitution.
    summaries = []
    for split in original_config['evaluation_splits'][original_config['benchmark']]:
        path = cell / split / 'evaluation_summary.json'
        if path.is_file():
            original = json.loads(path.read_text(encoding='utf-8'))
            updated = dict(original, model_path=overrides['model_path'])
            summaries.append((path, original, updated))

    transaction = uuid.uuid4().hex
    pending_state = snapshot / ('.state.relocating.' + transaction)
    pending_config = cell / ('.config.relocating.' + transaction)
    pending_summaries = []
    try:
        torch.save(state, pending_state)
        with pending_state.open('rb') as handle:
            os.fsync(handle.fileno())
        _write_json(pending_config, updated_config)
        for path, original, updated in summaries:
            pending = path.parent / ('.evaluation-summary.relocating.' + transaction)
            _write_json(pending, updated)
            pending_summaries.append((pending, path))
        os.replace(pending_state, state_path)
        os.replace(pending_config, config_path)
        for pending, path in pending_summaries:
            os.replace(pending, path)
            _sync_directory(path.parent)
        _sync_directory(snapshot)
        _sync_directory(cell)
    finally:
        for pending in (pending_state, pending_config, *(p for p, _ in pending_summaries)):
            if pending.exists():
                pending.unlink()
    return {'cell': str(cell), 'snapshot': identity,
            'config.json': _changes(original_config, updated_config),
            'state.pt.trainer.config': _changes(original_trainer_config, updated_trainer_config),
            'state.pt.ledger.journal_path': {'old': old_journal, 'new': str(journal)},
            'evaluation_summary_model_paths': {
                str(path.relative_to(cell)): {'old': old.get('model_path'), 'new': new['model_path']}
                for path, old, new in summaries if old.get('model_path') != new['model_path']}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell', type=Path, required=True)
    parser.add_argument('--resource-config', type=Path, required=True)
    parser.add_argument('--physical-gpu', nargs='+', type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args(argv)
    resources = json.loads(args.resource_config.read_text(encoding='utf-8'))
    print(json.dumps(relocate(args.cell, resources, args.physical_gpu), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
