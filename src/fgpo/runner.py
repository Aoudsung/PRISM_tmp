"""Complete official F-GPO studies without resource-utility or tier substitutions."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
from statistics import fmean, stdev

from .envs import make_env, EnvironmentPool, ALFWORLD_TABLE_COLUMNS
from .metrics import official_summary
from .baseline_specs import PUBLISHED_SOURCES
from .records import CostLedger, monotonic_seconds
from .protocol import (METHODS, METHOD_ROLES, OFFICIAL_EVALUATION_SPLITS, OFFICIAL_HORIZONS, SUITE_METHODS, SUITE_BENCHMARKS,
                       WEBSHOP_CONTINUATION_STUDIES, study_identity, suite_methods,
                       method_benchmark_pairs, resolve_method_config, require_protocol,
                       require_training_identity)
from .textcraft import TEXTCRAFT_COUNTS


def write_json(path, value, *, durable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.incoming')
    with pending.open('w', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        if durable:
            handle.flush()
            os.fsync(handle.fileno())
    pending.replace(path)
    if durable:
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def require_identity(saved, expected):
    for key in ('protocol', 'method', 'benchmark', 'seed', 'model_path'):
        if saved[key] != expected[key]:
            raise ValueError(f'Existing cell identity differs at {key}; no work was restarted.')
    if saved.get('study_protocol') != expected.get('study_protocol'):
        raise ValueError('Existing cell belongs to a different study budget.')


def isolate_gpus(physical_ids, excluded_ids):
    physical = list(physical_ids)
    if not physical or len(set(physical)) != len(physical) or any(gpu < 0 for gpu in physical):
        raise ValueError('Select distinct nonnegative physical GPU IDs.')
    if any(gpu in excluded_ids for gpu in physical):
        raise ValueError('The physical allocation includes a GPU excluded by this server configuration.')
    mask = ','.join(map(str, physical))
    existing = os.environ.get('CUDA_VISIBLE_DEVICES')
    if existing is not None and existing != mask:
        raise ValueError('CUDA_VISIBLE_DEVICES differs from the explicit physical --gpu allocation.')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = mask
    return list(range(len(physical))), mask


def compute_allocation(physical_ids, model_physical_ids=None):
    allocated = list(physical_ids)
    compute = allocated if model_physical_ids is None else list(model_physical_ids)
    if not compute or len(set(compute)) != len(compute) or any(gpu not in allocated for gpu in compute):
        raise ValueError('--model-gpu must be a nonempty distinct physical subset of --gpu.')
    return compute, [allocated.index(gpu) for gpu in compute]


def _bind_compute_pair_cpu_affinity(config, compute_physical_ids):
    if len(compute_physical_ids) != 2:
        return None
    mapping = config.get('cpu_affinity_by_compute_gpu_pair', {})
    if not isinstance(mapping, dict):
        raise ValueError('cpu_affinity_by_compute_gpu_pair must map sorted physical GPU pairs to CPU lists.')
    pair = ','.join(map(str, sorted(compute_physical_ids)))
    if pair not in mapping:
        return None
    cpu_list = mapping[pair]
    if not isinstance(cpu_list, str) or not re.fullmatch(r'[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*', cpu_list):
        raise ValueError(f'Invalid CPU affinity list for compute GPU pair {pair}: {cpu_list!r}')
    if not hasattr(os, 'sched_getaffinity') or not hasattr(os, 'sched_setaffinity'):
        raise RuntimeError('Configured CPU affinity requires Linux sched_getaffinity and sched_setaffinity.')
    inherited = os.sched_getaffinity(0)
    selected = set()
    for item in cpu_list.split(','):
        first, _, last = item.partition('-')
        start, end = int(first), int(last or first)
        if end < start:
            raise ValueError(f'Descending CPU affinity range for compute GPU pair {pair}: {item}')
        selected.update(cpu for cpu in inherited if start <= cpu <= end)
    if not selected:
        raise ValueError(f'CPU affinity for compute GPU pair {pair} has no CPUs in the inherited mask.')
    os.sched_setaffinity(0, selected)
    return sorted(os.sched_getaffinity(0))


@contextmanager
def _file_lock(path, *, nonblocking=False, error_message=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle.fileno(), mode)
        except BlockingIOError as exc:
            raise RuntimeError(error_message or f'Resource is locked by another process: {path}') from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validate_suite_cells(cells, config, suite):
    allowed = set(method_benchmark_pairs(suite_methods(config, suite), SUITE_BENCHMARKS[suite]))
    if any(saved['protocol'] != config['protocol']
           or saved.get('study_protocol') != config.get('study_protocol')
           or (saved['benchmark'], saved['method']) not in allowed for saved in cells):
        raise ValueError('Existing suite summary belongs to another experiment definition')


def run_cell(config, output_dir, *, method, benchmark, seed, gpu_ids, model_gpu_ids=None, resume=False):
    """Reserve the full CPU training footprint before loading or sampling."""
    require_protocol(config)
    startup_started = monotonic_seconds()
    output = Path(output_dir)
    options = resolve_method_config(dict(config, method=method, benchmark=benchmark, seed=int(seed)))
    restore_receipt = None
    from .checkpoint_gcs import configured, GCSCheckpoint
    if configured(options):
        with _file_lock(output.parent / f'.{output.name}.lock', nonblocking=True,
                        error_message=f'Cell is already locked by another active worker: {output}'):
            storage = GCSCheckpoint(options, output / 'checkpoint')
            restore_receipt = storage.restore_cell(require_resume=resume, optional=True)
            if restore_receipt is None:
                resume = resume and output.exists()
    startup_receipt = {'started': startup_started, 'gcs_restore': restore_receipt}
    receipt_file = os.environ.get('FGPO_STARTUP_RECEIPT_FILE')
    if receipt_file and Path(receipt_file).is_file():
        startup_receipt['cold_asset_restore'] = json.loads(Path(receipt_file).read_text(encoding='utf-8'))
        startup_receipt['cold_asset_receipt_file'] = receipt_file
    arguments = dict(method=method, benchmark=benchmark, seed=seed, gpu_ids=gpu_ids,
                     model_gpu_ids=model_gpu_ids, resume=resume, startup_receipt=startup_receipt)
    if resume and (output / 'cell_summary.json').is_file():
        return _run_cell(config, output, **arguments)
    from .resources import reserve_host_memory
    compute = gpu_ids if model_gpu_ids is None else model_gpu_ids
    reserve_started = monotonic_seconds()
    with reserve_host_memory(options, model_gpu_count=len(compute),
                             resume_checkpoint=output / 'checkpoint' if resume and output.exists() else None) as allocation:
        startup_receipt['host_reserve_seconds'] = monotonic_seconds() - reserve_started
        return _run_cell(config, output, **arguments, resource_allocation=allocation)


def _run_cell(config, output_dir, *, method, benchmark, seed, gpu_ids, model_gpu_ids=None,
              resume=False, resource_allocation=None, startup_receipt=None):
    require_protocol(config)
    # Resolve physical allocation before importing any Engine/CUDA code.
    logical_ids, visible_mask = isolate_gpus(gpu_ids, config['excluded_gpu_ids'])
    compute_physical, compute_logical = compute_allocation(gpu_ids, model_gpu_ids)
    from .engine import Engine
    output = Path(output_dir)
    cell_config = resolve_method_config(dict(config,method=method,benchmark=benchmark,seed=int(seed)))
    if resource_allocation is not None:
        cell_config['host_memory_reservation'] = resource_allocation
    require_protocol(cell_config)
    cell_config.update(gpu_ids=logical_ids, physical_gpu_ids=list(gpu_ids),
                       logical_gpu_ids=logical_ids, cuda_visible_devices=visible_mask,
                       allocated_physical_gpu_ids=list(gpu_ids), allocated_logical_gpu_ids=logical_ids,
                       rng_physical_gpu_ids=list(gpu_ids), rng_logical_gpu_ids=logical_ids,
                       compute_physical_gpu_ids=compute_physical, compute_logical_gpu_ids=compute_logical,
                       model_placement_explicit=model_gpu_ids is not None,
                       cuda_device_order=os.environ['CUDA_DEVICE_ORDER'],
                       gpu_id_semantics='Engine_gpu_ids_are_process_local_logical_indices')
    if 'gpu_free_memory_gib' in cell_config:
        physical_memory = cell_config['gpu_free_memory_gib']
        cell_config['gpu_free_memory_gib'] = {
            logical: physical_memory[physical] if physical in physical_memory else physical_memory[str(physical)]
            for logical, physical in enumerate(gpu_ids)}
    with _file_lock(output.parent / f'.{output.name}.lock', nonblocking=True,
                    error_message=f'Cell is already locked by another active worker: {output}'):
        return _run_locked_cell(
            config, cell_config, output,
            method=method, benchmark=benchmark, seed=seed,
            Engine=Engine, resume=resume, startup_receipt=startup_receipt,
        )


_FINALIZATION_COSTS_SCOPE = (
    'complete_cell_including_official_evaluation_final_export_reclamation_and_final_publish_logs'
)
_TERMINAL_PUBLICATION_BOUNDARY = (
    'included_events_end_at_checkpoint_final_publish_logs_before_unmetered_terminal_summary_and_tail_export'
)


def _resume_pending_finalization(config, cell_config, output, complete_path, complete, *, storage, startup_receipt=None):
    from .checkpoint import _sync_file, reclaim_completed_resume_state

    marker_path = output / '.finalization_complete.json'
    is_finalized = complete.get('finalization_stage') == 'finalized'
    exported = marker_path.is_file()
    if not exported and storage is not None and is_finalized:
        with storage.open_snapshot(('CURRENT', 'FINAL'), verify_cell_root=True) as pinned:
            cur_manifest, _, _ = pinned['CURRENT']
            fin_manifest, _, _ = pinned['FINAL']
            if (cur_manifest is None and fin_manifest is not None
                    and bool(fin_manifest.get('metadata', {}).get('terminal_export_completed', False))):
                exported = True
    if is_finalized and exported:
        if not marker_path.is_file():
            write_json(marker_path, {
                'protocol': config['protocol'],
                'snapshot': complete.get('completed_checkpoint', {}).get('snapshot'),
                'terminal_export_completed': True,
            }, durable=True)
        return complete

    ledger = CostLedger(output / 'costs.jsonl', gpu_ids=cell_config['gpu_ids'])
    if startup_receipt is not None and not ledger._startup_credited:
        pre_seconds = monotonic_seconds() - startup_receipt['started']
        details = {'catalog_seconds': 0.0, 'resumed': True, 'finalization_recovery': True}
        for key in ('gcs_restore', 'host_reserve_seconds'):
            if startup_receipt.get(key) is not None:
                details[key] = startup_receipt[key]
        ledger.credit_startup(
            pre_engine_seconds=pre_seconds,
            startup_details=details,
            cold_asset_receipt=startup_receipt.get('cold_asset_restore'),
        )
        if startup_receipt.get('cold_asset_receipt_file'):
            Path(startup_receipt['cold_asset_receipt_file']).unlink(missing_ok=True)

    if storage is not None:
        saved_state = storage.load_final_state()
    else:
        import torch
        final_ptr = json.loads((output / 'final_checkpoint' / 'CURRENT').read_text(encoding='utf-8'))
        saved_state = torch.load(
            output / 'final_checkpoint' / 'snapshots' / final_ptr['snapshot'] / 'state.pt',
            map_location='cpu',
            weights_only=False,
        )
    ledger.load_state_dict(saved_state['ledger'])

    seen_roles = set()
    tail_offset = int(saved_state['ledger']['durable_offset_bytes'])
    if ledger.path is not None and ledger.path.is_file():
        with ledger.path.open('rb') as handle:
            handle.seek(tail_offset)
            for line in handle:
                if line.endswith(b'\n'):
                    seen_roles.add(json.loads(line).get('role'))

    if storage is not None:
        with storage.open_snapshot(('CURRENT',), verify_cell_root=True) as pinned:
            needs_reclaim = (output / 'checkpoint').exists() or pinned['CURRENT'][0] is not None
    else:
        needs_reclaim = (output / 'checkpoint').exists()

    if needs_reclaim:
        rec_start = monotonic_seconds()
        reclaimed = reclaim_completed_resume_state(
            complete_path, storage=storage, keep_journal_state=(storage is not None),
        )
        rec_details = {'reclaimed': reclaimed, 'cell': str(output)}
        if storage is not None and storage.last_transfer_stats is not None:
            rec_details['transfer'] = dict(storage.last_transfer_stats)
        ledger.record('checkpoint', 'checkpoint.reclaim_completed_resume_state',
                      seconds=monotonic_seconds() - rec_start, details=rec_details)

    if 'checkpoint.final_publish_logs' not in seen_roles:
        pub_start = monotonic_seconds()
        if storage is not None:
            pub_receipt = storage.publish_logs()
        else:
            if ledger.path is not None and ledger.path.is_file():
                _sync_file(ledger.path)
            pub_receipt = {'backend': 'local'}
        ledger.record('checkpoint', 'checkpoint.final_publish_logs',
                      seconds=monotonic_seconds() - pub_start, details=pub_receipt)

    if complete.get('finalization_stage') != 'finalized':
        complete['costs'] = ledger.snapshot()
        complete['costs_scope'] = _FINALIZATION_COSTS_SCOPE
        complete['terminal_publication_boundary'] = _TERMINAL_PUBLICATION_BOUNDARY
        complete['finalization_stage'] = 'finalized'
        write_json(complete_path, complete, durable=True)

    _sync_file(complete_path)
    if ledger.path is not None and ledger.path.is_file():
        _sync_file(ledger.path)
    if storage is not None:
        storage.publish_final_artifacts(complete_path)
    write_json(marker_path, {
        'protocol': config['protocol'],
        'snapshot': complete.get('completed_checkpoint', {}).get('snapshot'),
        'terminal_export_completed': True,
    }, durable=True)
    return complete


def _run_locked_cell(config, cell_config, output, *, method, benchmark, seed, Engine, resume, startup_receipt=None):
    resume_from = None
    catalog_seconds = 0.0
    study = study_identity(cell_config)
    continuation = {}
    if cell_config.get('study_protocol') in WEBSHOP_CONTINUATION_STUDIES and method == 'fgpo':
        from .webshop_continuation import prepared_origin
        if not resume:
            raise ValueError('The 150-round V4 arm requires a prepared round-60 continuation and --resume.')
        continuation['continuation'] = prepared_origin(output, cell_config)
    evaluation_splits = config.get('evaluation_splits', {}).get(benchmark, OFFICIAL_EVALUATION_SPLITS[benchmark])
    if output.exists():
        if not resume:
            raise FileExistsError(f'Existing cell requires explicit --resume: {output}')
        saved_config = json.loads((output / 'config.json').read_text())
        require_training_identity(saved_config, cell_config)
        complete_path = output / 'cell_summary.json'
        if complete_path.is_file():
            complete = json.loads(complete_path.read_text())
            require_identity(complete, cell_config)
            if complete.get('completed_checkpoint', {}).get('format') == 'fgpo-completed-model-v1':
                from .checkpoint_gcs import configured, GCSCheckpoint
                storage = GCSCheckpoint(cell_config, output / 'checkpoint') if configured(cell_config) else None
                return _resume_pending_finalization(
                    config, cell_config, output, complete_path, complete,
                    storage=storage, startup_receipt=startup_receipt,
                )
            return complete
        resume_from = output / 'checkpoint'
        if not (resume_from / 'CURRENT').is_file():
            raise FileNotFoundError(f'Existing incomplete cell has no actual checkpoint: {resume_from}')
        official_tasks = json.loads((output / 'official_tasks.json').read_text())
        train_tasks = official_tasks['train']
        evaluation_tasks = {split: official_tasks[split] for split in evaluation_splits}
        # Checkpoint retains prior execution provenance; config records this
        # explicit continuation's physical-to-logical allocation.
        write_json(output / 'config.json', cell_config)
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / 'config.json', cell_config)
        catalog_started = monotonic_seconds()
        catalog = make_env(cell_config, benchmark, output / 'environments/catalog', seed=seed)
        try:
            if benchmark == 'appworld' and method == 'g2po':
                train_tasks = [task for difficulty in cell_config['appworld_train_difficulties']
                               for task in catalog.list_tasks('train', difficulty=difficulty)]
            elif study and benchmark == 'webshop':
                train_tasks = catalog.list_tasks('train', train_rounds=cell_config['rounds'])
            else:
                train_tasks = catalog.list_tasks('train')
            evaluation_tasks = {split: catalog.list_tasks(split)
                                for split in evaluation_splits}
        finally:
            catalog.close()
        write_json(output / 'official_tasks.json', {'train': train_tasks, **evaluation_tasks})
        catalog_seconds = monotonic_seconds() - catalog_started

    if benchmark == 'textcraft' and (len(train_tasks) != TEXTCRAFT_COUNTS['train']
            or len(evaluation_tasks['val']) != TEXTCRAFT_COUNTS['val']):
        raise ValueError('TextCraft requires the original full train and val catalogs.')

    pre_engine_seconds = (monotonic_seconds() - startup_receipt['started']) if startup_receipt else catalog_seconds
    startup_details = {'catalog_seconds': catalog_seconds, 'resumed': resume_from is not None}
    if 'effective_cpu_affinity' in cell_config:
        startup_details['effective_cpu_affinity'] = cell_config['effective_cpu_affinity']
    cold_asset_restore = None
    cold_asset_receipt_file = None
    if startup_receipt is not None:
        for key in ('gcs_restore', 'host_reserve_seconds'):
            if startup_receipt.get(key) is not None:
                startup_details[key] = startup_receipt[key]
        cold_asset_restore = startup_receipt.get('cold_asset_restore')
        cold_asset_receipt_file = startup_receipt.get('cold_asset_receipt_file')
    engine_startup = {
        'pre_engine_seconds': pre_engine_seconds,
        'details': startup_details,
        'cold_asset_restore': cold_asset_restore,
    }
    trajectories = int(cell_config.get('trajectories', 8))
    pool_size = int(cell_config.get('environment_pool_size', trajectories))
    if benchmark == 'appworld':
        from .appworld_batch import appworld_eval_batch_size, appworld_fgpo_rollout_groups, appworld_hgpo_rollout_groups
        pool_size = max(pool_size, appworld_eval_batch_size(cell_config),
                        trajectories * appworld_fgpo_rollout_groups(cell_config),
                        trajectories * appworld_hgpo_rollout_groups(cell_config))
    env_factory = EnvironmentPool(cell_config, benchmark, output / 'environments/pool', size=pool_size)
    engine = None
    try:
        engine = Engine(cell_config, output, resume_from=resume_from, startup_receipt=engine_startup)
        if cold_asset_receipt_file:
            Path(cold_asset_receipt_file).unlink(missing_ok=True)
        # Each method executes its declared complete recipe once per study cell.
        training_path = output / 'training_summary.json'
        training = json.loads(training_path.read_text()) if resume_from and training_path.is_file() else None
        reusable_training = (training is not None and training.get('status') == 'complete'
            and engine.state.completed
            and training.get('rounds_completed') == engine.state.round
            and training.get('optimizer_steps') == engine.optimizer.steps
            and training.get('study_protocol') == cell_config.get('study_protocol')
            and all(training.get(key) == cell_config[key]
                    for key in ('protocol', 'method', 'benchmark', 'seed')))
        if not reusable_training:
            training = engine.train(train_tasks, env_factory)
        checkpoint = engine.save_checkpoint(output / 'checkpoint')
        write_json(training_path, training)
        if training.get('status') == 'interrupted':
            return {'status': 'interrupted', 'protocol':config['protocol'], 'method': method, 'benchmark': benchmark,
                    'seed': seed, 'model_path': cell_config['model_path'],
                    'checkpoint': str(checkpoint), 'training': training, **study, **continuation}
        scores = {}
        for split, tasks in evaluation_tasks.items():
            summary_path = output / split / 'evaluation_summary.json'
            if resume_from and summary_path.is_file():
                completed_evaluation = json.loads(summary_path.read_text())
                require_identity(completed_evaluation, cell_config)
                if completed_evaluation['split'] != split:
                    raise ValueError('Saved official evaluation split differs.')
                if completed_evaluation.get('model_version', engine.optimizer.steps) != engine.optimizer.steps:
                    raise ValueError('Saved official evaluation belongs to a different final model version.')
                if completed_evaluation['dataset_complete']:
                    scores[split] = completed_evaluation
                    continue
            evaluation = engine.evaluate(tasks, env_factory, split=split)
            records = evaluation['records']
            # Official evaluation is already finalized by the actor/Engine exactly once.
            summary = official_summary(benchmark, records, tasks)
            summary.update(protocol=config['protocol'],method=method, benchmark=benchmark, seed=seed,
                           model_path=cell_config['model_path'], split=split,
                           model_version=engine.optimizer.steps, **study)
            write_json(output / split / 'evaluation_records.json', records)
            write_json(summary_path, summary)
            if evaluation.get('status') == 'interrupted':
                checkpoint = engine.save_checkpoint(output / 'checkpoint')
                return {'status': 'interrupted', 'protocol':config['protocol'], 'method': method, 'benchmark': benchmark,
                        'seed': seed, 'model_path': cell_config['model_path'],
                        'checkpoint': str(checkpoint), 'evaluation': {**scores, split: summary},
                        **study, **continuation}
            scores[split] = summary
        completed_checkpoint = engine.prepare_completed_checkpoint(training, scores)
        result = {'status': 'complete', 'protocol':config['protocol'], 'method': method, 'benchmark': benchmark, 'seed': seed,
                  'method_role':METHOD_ROLES[method], 'suite':config['suite'],
                  'published_source':PUBLISHED_SOURCES.get(method),
                  'model_path': cell_config['model_path'],
                  'checkpoint': completed_checkpoint['path'], 'completed_checkpoint': completed_checkpoint,
                  'training_resume_supported': False, 'finalization_stage': 'preliminary',
                  'training': training, 'evaluation': scores, 'costs': engine.ledger.snapshot(),
                  'costs_scope':'complete_cell_including_official_evaluation_and_final_model_export',
                  'environment_pool':env_factory.statistics(), **study, **continuation}
        write_json(output / 'cell_summary.json', result, durable=True)
        engine.reclaim_completed_checkpoint(output / 'cell_summary.json')
        engine.publish_final_logs()
        result['costs'] = engine.ledger.snapshot()
        result['costs_scope'] = _FINALIZATION_COSTS_SCOPE
        result['terminal_publication_boundary'] = _TERMINAL_PUBLICATION_BOUNDARY
        result['finalization_stage'] = 'finalized'
        write_json(output / 'cell_summary.json', result, durable=True)
        engine.export_final_terminal_state(output / 'cell_summary.json')
        write_json(output / '.finalization_complete.json', {
            'protocol': config['protocol'],
            'snapshot': completed_checkpoint.get('snapshot'),
            'terminal_export_completed': True,
        }, durable=True)
        return result
    finally:
        try:
            env_factory.close()
        finally:
            if engine is not None:
                engine.close()


def aggregate_cells(cells):
    """Seed statistics from completed official splits, with explicit seed count."""
    grouped, categories = {}, {}
    seen = set()
    for cell in cells:
        if cell.get('status') != 'complete':
            continue
        for split, scores in cell['evaluation'].items():
            if not scores['dataset_complete']:
                continue
            if cell['benchmark'] in ('alfworld', 'webshop', 'textcraft'):
                identity = (cell['benchmark'], cell['method'], split, cell['seed'])
                if identity in seen:
                    raise ValueError(f'Duplicate seed in official split aggregation: {identity}')
                seen.add(identity)
            if cell['benchmark'] == 'alfworld':
                for category, item in scores.get('task_types', {}).items():
                    if category != 'All' and item['dataset_complete'] and item['SR'] is not None:
                        key = (cell['benchmark'], cell['method'], split, category)
                        categories.setdefault(key, []).append((cell['seed'], item['SR']))
                if scores.get('SR') is not None:
                    key = (cell['benchmark'], cell['method'], split, 'All')
                    categories.setdefault(key, []).append((cell['seed'], scores['SR']))
            elif cell['benchmark'] == 'textcraft':
                for depth, item in scores['recipe_depths'].items():
                    if item['dataset_complete'] and item['SR'] is not None:
                        key = (cell['benchmark'], cell['method'], split, depth)
                        categories.setdefault(key, []).append((cell['seed'], item['SR']))
            for metric in ('SR','TGC','SGC','Score'):
                if scores.get(metric) is not None:
                    key = (cell['benchmark'],cell['method'],split,metric)
                    grouped.setdefault(key,[]).append((cell['seed'],scores[metric]))
    def statistics(values):
        seeds = sorted(seed for seed, _ in values)
        return {'mean': fmean(value for _, value in values) if values else None,
                'sample_std': stdev(value for _, value in values) if len(values) > 1 else None,
                'seeds': seeds, 'seed_count': len(values), 'unit': 'percent',
                'three_seed_complete': seeds == [0, 1, 2]}

    rows = [{'benchmark':benchmark,'method':method,'method_role':METHOD_ROLES[method],
             'split':split,'metric':metric,'mean':fmean(value for _,value in values),
             'sample_std':stdev(value for _,value in values) if len(values)>1 else None,
             'seeds':sorted(seed for seed,_ in values),'seed_count':len(values),'unit':'percent'}
            for (benchmark,method,split,metric),values in sorted(grouped.items())]
    for row in rows:
        if row['benchmark'] == 'alfworld' and row['metric'] == 'SR':
            row['task_types'] = {
                category: statistics(categories.get((row['benchmark'], row['method'], row['split'], category), []))
                for category in ALFWORLD_TABLE_COLUMNS}
        elif row['benchmark'] == 'textcraft' and row['metric'] == 'SR':
            prefix = (row['benchmark'], row['method'], row['split'])
            row['recipe_depths'] = {key[3]:statistics(categories[key]) for key in sorted(categories)
                                    if key[:3] == prefix}
    return rows


def alfworld_table(aggregates):
    """Seven Table 1 percent columns from per-seed official success rates."""
    rows = [{'method': row['method'], 'split': row['split'],
             'task_types': row['task_types']}
            for row in aggregates if row['benchmark'] == 'alfworld'
            and row['metric'] == 'SR' and 'task_types' in row]
    lines = ['SR (%); dispersion is sample standard deviation across the listed seeds.', '',
             '| Method | Split | Seeds | ' + ' | '.join(ALFWORLD_TABLE_COLUMNS) + ' |',
             '| --- | --- | --- | ' + ' | '.join('---' for _ in ALFWORLD_TABLE_COLUMNS) + ' |']
    for row in rows:
        values = []
        for category in ALFWORLD_TABLE_COLUMNS:
            stats = row['task_types'][category]
            value = '—' if stats['mean'] is None else f"{stats['mean']:.2f}"
            if stats['sample_std'] is not None:
                value += f" ± {stats['sample_std']:.2f}"
            if stats['seeds'] != row['task_types']['All']['seeds']:
                value += ' (seeds: ' + ','.join(map(str, stats['seeds'])) + ')'
            values.append(value)
        seeds = ','.join(map(str, row['task_types']['All']['seeds']))
        lines.append('| ' + ' | '.join([row['method'], row['split'], seeds, *values]) + ' |')
    return {'columns': list(ALFWORLD_TABLE_COLUMNS), 'unit': 'percent',
            'dispersion': 'sample_std', 'rows': rows, 'markdown': '\n'.join(lines) + '\n'}


def aggregate_only(selected, suite_dir, config, suite, *, records_format):
    """Derive reports from completed receipts without touching terminal authority."""
    cells = []
    for benchmark, method, seed, output in selected:
        cell = json.loads((output / 'cell_summary.json').read_text())
        require_identity(cell, dict(config, benchmark=benchmark, method=method, seed=seed))
        if cell.get('suite', suite) != suite:
            raise ValueError(f'Report-only cell belongs to another suite: {output}')
        if cell['status'] != 'complete':
            raise ValueError(f'Report-only aggregation requires a completed cell: {output}')
        official_tasks = json.loads((output / 'official_tasks.json').read_text())
        evaluation = {}
        for split, saved in cell['evaluation'].items():
            path = (output / split / 'evaluation_records.json' if records_format == 'json'
                    else output / f'evaluation-{split}.jsonl')
            if records_format == 'json':
                records = json.loads(path.read_text())
            else:
                records = []
                with path.open('rb') as handle:
                    for line in handle:
                        if not line.endswith(b'\n'):
                            break
                        records.append(json.loads(line))
            for record in records:
                for key, value in (('protocol', cell['protocol']), ('benchmark', benchmark),
                                   ('method', method), ('seed', seed), ('split', split)):
                    if record[key] != value:
                        raise ValueError(f'Official receipt identity differs at {key}: {path}')
                if record.get('study_protocol') != cell.get('study_protocol'):
                    raise ValueError(f'Official receipt belongs to a different study budget: {path}')
                if 'model_version' in saved and record.get('model_version') != saved['model_version']:
                    raise ValueError(f'Official receipt model version differs: {path}')
            scores = official_summary(benchmark, records, official_tasks[split])
            for key in ('dataset_complete', 'official_task_count', 'evaluated_count',
                        'success_count', 'SR', 'TGC', 'SGC', 'Score', 'recipe_depths'):
                if key in saved and scores.get(key) != saved[key]:
                    raise ValueError(f'Derived report differs from terminal {key}: {path}')
            evaluation[split] = {**saved, **scores}
        cells.append({**cell, 'evaluation': evaluation})
    aggregates = aggregate_cells(cells)
    table = alfworld_table(aggregates)
    report = {'protocol': config['protocol'], 'suite': suite,
              'report_only': True, 'records_format': records_format,
              'cells': cells, 'aggregates': aggregates, 'alfworld_table1': table,
              **study_identity(config)}
    for cell, (_, _, _, output) in zip(cells, selected):
        write_json(output / 'reporting_summary.json', cell)
    write_json(suite_dir / 'study_reporting_summary.json', report)
    (suite_dir / 'study_reporting_table1.md').write_text(table['markdown'], encoding='utf-8')
    return report


def reused_fgpo_results(study_root, cells, config):
    """Read main-comparison receipts for the ablation table; never train them."""
    references,completed = [],[]
    for benchmark,seed in sorted({(cell['benchmark'],cell['seed']) for cell in cells}):
        relative = Path('..')/'comparison'/benchmark/'fgpo'/f'seed-{seed}'/'cell_summary.json'
        path = Path(study_root)/'ablation'/relative
        entry = {'method':'fgpo','benchmark':benchmark,'seed':seed,'path':str(relative),
                 'path_base':'study_summary_directory','reuse':'main_comparison_no_additional_training',
                 'status':'pending_main_comparison'}
        if path.is_file():
            reference = json.loads(path.read_text())
            expected = dict(config,method='fgpo',benchmark=benchmark,seed=seed)
            require_identity(reference,expected)
            if reference.get('status') == 'complete':
                entry.update(status='complete',evaluation=reference['evaluation'])
                completed.append(reference)
        references.append(entry)
    return references,aggregate_cells(completed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--suite', choices=tuple(SUITE_METHODS), default='comparison',
                        help='Published comparison or own ALFWorld ablations; stored separately.')
    parser.add_argument('--methods', nargs='+', choices=METHODS)
    parser.add_argument('--benchmarks', nargs='+', choices=tuple(OFFICIAL_HORIZONS))
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--gpu', nargs='+', type=int,
                        help='Physical GPU IDs; Engine uses isolated process-local logical indices.')
    parser.add_argument('--model-gpu', nargs='+', type=int,
                        help='Explicit physical model-compute subset of --gpu; all allocated RNG contexts remain intact.')
    parser.add_argument('--resume', action='store_true', help='Continue saved F-GPO study state explicitly.')
    parser.add_argument('--aggregate-only', action='store_true',
                        help='Read completed official receipts and write derived reports without model/environment work.')
    parser.add_argument('--evaluation-records-format', choices=('json', 'jsonl'), default='json',
                        help='Explicit report-only receipt source; jsonl reads newline-committed records.')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    require_protocol(config)
    plan = config['experiment_suites'][args.suite]
    methods = args.methods if args.methods is not None else plan['methods']
    benchmarks = args.benchmarks if args.benchmarks is not None else plan['benchmarks']
    seeds = args.seeds if args.seeds is not None else config['seeds']
    if (not methods or not benchmarks or not seeds
            or not set(methods).issubset(set(plan['methods']) & set(suite_methods(config, args.suite)))
            or not set(benchmarks).issubset(set(plan['benchmarks']) & set(SUITE_BENCHMARKS[args.suite]))
            or not set(seeds).issubset(config['seeds'])):
        raise ValueError('CLI selection must be a subset of the declared claim-scoped suite')
    if any(len(values) != len(set(values)) for values in (methods,benchmarks,seeds)):
        raise ValueError('Select each method, benchmark and seed only once')
    pairs = method_benchmark_pairs(methods, benchmarks)
    if not pairs or (args.methods is not None and set(methods) - {method for _, method in pairs}):
        raise ValueError('Each explicitly selected method requires a recipe for at least one selected benchmark')
    config = dict(config,suite=args.suite,methods=list(methods),benchmarks=list(benchmarks))
    require_protocol(config)
    suite_dir = args.output_dir / args.suite
    selected = [(benchmark, method, seed, suite_dir / benchmark / method / f'seed-{seed}')
                for benchmark, method in pairs for seed in seeds]
    if args.aggregate_only:
        report = aggregate_only(selected, suite_dir, config, args.suite,
                                records_format=args.evaluation_records_format)
        if benchmarks == ['textcraft']:
            print(json.dumps(report['aggregates'], ensure_ascii=False, indent=2))
        else:
            print(report['alfworld_table1']['markdown'], end='')
        return 0
    if args.gpu is None:
        parser.error('--gpu is required for training/evaluation')
    os.environ['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
    isolate_gpus(args.gpu, config['excluded_gpu_ids'])
    compute_physical, _ = compute_allocation(args.gpu, args.model_gpu)
    # Linux threads and child processes inherit the CLI thread's affinity.
    cpu_affinity = _bind_compute_pair_cpu_affinity(config, compute_physical)
    if cpu_affinity is not None:
        config['effective_cpu_affinity'] = cpu_affinity
    if args.resume:
        from .checkpoint_gcs import configured
        if configured(config):
            suite_dir.mkdir(parents=True, exist_ok=True)
        elif not suite_dir.is_dir():
            raise FileNotFoundError('Explicit --resume requires the existing selected suite directory.')
    else:
        suite_dir.mkdir(parents=True, exist_ok=True)
        existing = next((path for _, _, _, path in selected if path.exists()), None)
        if existing is not None:
            raise FileExistsError(f'Existing cell requires explicit --resume: {existing}')
    study_path = suite_dir / 'study_summary.json'
    lock_path = suite_dir / '.study_summary.lock'
    with _file_lock(lock_path):
        if study_path.is_file():
            _validate_suite_cells(json.loads(study_path.read_text())['cells'], config, args.suite)
    for benchmark, method, seed, output in selected:
        cell = run_cell(config, output, method=method, benchmark=benchmark,
                        seed=seed, gpu_ids=args.gpu, model_gpu_ids=args.model_gpu,
                        resume=args.resume)
        cell.update(method_role=METHOD_ROLES[method], suite=args.suite)
        with _file_lock(lock_path):
            cells = json.loads(study_path.read_text())['cells'] if study_path.is_file() else []
            _validate_suite_cells(cells, config, args.suite)
            position = next((index for index, saved in enumerate(cells)
                             if (saved['method'], saved['benchmark'], saved['seed'])
                             == (method, benchmark, seed)), None)
            if position is None:
                cells.append(cell)
            else:
                cells[position] = cell
            references, reference_aggregates = (reused_fgpo_results(args.output_dir, cells, config)
                if args.suite == 'ablation' else ([], []))
            aggregates = aggregate_cells(cells)
            table = alfworld_table(aggregates)
            write_json(study_path, {
                'protocol': config['protocol'], 'suite': args.suite, 'experiment_plan': plan,
                'cells': cells, 'aggregates': aggregates, 'alfworld_table1': table,
                'reference_cells': references, 'reference_aggregates': reference_aggregates,
                'design': 'docs/fgpo_design_and_derivation.md', **study_identity(config)})
            (suite_dir / 'study_table1.md').write_text(table['markdown'], encoding='utf-8')
        if cell['status'] == 'interrupted':
            return 130
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
