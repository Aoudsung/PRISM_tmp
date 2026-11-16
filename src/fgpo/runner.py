"""Complete official F-GPO studies without resource-utility or tier substitutions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import fmean, stdev

from .envs import make_env, EnvironmentPool
from .metrics import official_summary
from .baseline_specs import PUBLISHED_SOURCES
from .protocol import (METHODS, METHOD_ROLES, SUITE_METHODS, SUITE_BENCHMARKS,
                       resolve_method_config, require_protocol, require_training_identity)


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


def run_cell(config, output_dir, *, method, benchmark, seed, gpu_ids, model_gpu_ids=None, resume=False):
    """Reserve the full CPU training footprint before loading or sampling."""
    require_protocol(config)
    output = Path(output_dir)
    arguments = dict(method=method, benchmark=benchmark, seed=seed, gpu_ids=gpu_ids,
                     model_gpu_ids=model_gpu_ids, resume=resume)
    if resume and (output / 'cell_summary.json').is_file():
        return _run_cell(config, output, **arguments)
    from .resources import reserve_host_memory
    options = resolve_method_config(dict(config, method=method, benchmark=benchmark, seed=int(seed)))
    compute = gpu_ids if model_gpu_ids is None else model_gpu_ids
    with reserve_host_memory(options, model_gpu_count=len(compute),
                             resume_checkpoint=output / 'checkpoint' if resume and output.exists() else None) as allocation:
        return _run_cell(config, output, **arguments, resource_allocation=allocation)


def _run_cell(config, output_dir, *, method, benchmark, seed, gpu_ids, model_gpu_ids=None,
              resume=False, resource_allocation=None):
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
    resume_from = None
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
                from .checkpoint import reclaim_completed_resume_state
                reclaim_completed_resume_state(complete_path)
            return complete
        resume_from = output / 'checkpoint'
        if not (resume_from / 'CURRENT').is_file():
            raise FileNotFoundError(f'Existing incomplete cell has no actual checkpoint: {resume_from}')
        official_tasks = json.loads((output / 'official_tasks.json').read_text())
        train_tasks = official_tasks['train']
        evaluation_tasks = {split: official_tasks[split] for split in config['evaluation_splits'][benchmark]}
        # Checkpoint retains prior execution provenance; config records this
        # explicit continuation's physical-to-logical allocation.
        write_json(output / 'config.json', cell_config)
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / 'config.json', cell_config)
        catalog = make_env(cell_config, benchmark, output / 'environments/catalog', seed=seed)
        try:
            train_tasks = catalog.list_tasks('train')
            evaluation_tasks = {split: catalog.list_tasks(split)
                                for split in config['evaluation_splits'][benchmark]}
        finally:
            catalog.close()
        write_json(output / 'official_tasks.json', {'train': train_tasks, **evaluation_tasks})

    env_factory = EnvironmentPool(cell_config, benchmark, output / 'environments/pool',
                                  size=int(cell_config.get('environment_pool_size',cell_config.get('trajectories',8))))
    engine = None
    try:
        engine = Engine(cell_config, output, resume_from=resume_from)
        # Each method executes its declared complete recipe once per study cell.
        training_path = output / 'training_summary.json'
        training = json.loads(training_path.read_text()) if resume_from and training_path.is_file() else None
        reusable_training = (training is not None and training.get('status') == 'complete'
            and engine.state.completed
            and training.get('rounds_completed') == engine.state.round
            and training.get('optimizer_steps') == engine.optimizer.steps
            and all(training.get(key) == cell_config[key]
                    for key in ('protocol', 'method', 'benchmark', 'seed')))
        if not reusable_training:
            training = engine.train(train_tasks, env_factory)
        checkpoint = engine.save_checkpoint(output / 'checkpoint')
        write_json(training_path, training)
        if training.get('status') == 'interrupted':
            return {'status': 'interrupted', 'protocol':config['protocol'], 'method': method, 'benchmark': benchmark,
                    'seed': seed, 'model_path': cell_config['model_path'],
                    'checkpoint': str(checkpoint), 'training': training}
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
                           model_version=engine.optimizer.steps)
            write_json(output / split / 'evaluation_records.json', records)
            write_json(summary_path, summary)
            if evaluation.get('status') == 'interrupted':
                checkpoint = engine.save_checkpoint(output / 'checkpoint')
                return {'status': 'interrupted', 'protocol':config['protocol'], 'method': method, 'benchmark': benchmark,
                        'seed': seed, 'model_path': cell_config['model_path'],
                        'checkpoint': str(checkpoint), 'evaluation': {**scores, split: summary}}
            scores[split] = summary
        completed_checkpoint = engine.prepare_completed_checkpoint(training, scores)
        result = {'status': 'complete', 'protocol':config['protocol'], 'method': method, 'benchmark': benchmark, 'seed': seed,
                  'method_role':METHOD_ROLES[method], 'suite':config['suite'],
                  'published_source':PUBLISHED_SOURCES.get(method),
                  'model_path': cell_config['model_path'],
                  'checkpoint': completed_checkpoint['path'], 'completed_checkpoint': completed_checkpoint,
                  'training_resume_supported': False,
                  'training': training, 'evaluation': scores, 'costs': engine.ledger.snapshot(),
                  'costs_scope':'complete_cell_including_official_evaluation_and_final_model_export',
                  'environment_pool':env_factory.statistics()}
        write_json(output / 'cell_summary.json', result, durable=True)
        engine.reclaim_completed_checkpoint(output / 'cell_summary.json')
        return result
    finally:
        try:
            env_factory.close()
        finally:
            if engine is not None:
                engine.close()


def aggregate_cells(cells):
    """Seed statistics from completed official splits, with explicit seed count."""
    grouped = {}
    for cell in cells:
        if cell.get('status') != 'complete':
            continue
        for split, scores in cell['evaluation'].items():
            if not scores['dataset_complete']:
                continue
            for metric in ('SR','TGC','SGC'):
                if scores.get(metric) is not None:
                    key = (cell['benchmark'],cell['method'],split,metric)
                    grouped.setdefault(key,[]).append((cell['seed'],scores[metric]))
    return [{'benchmark':benchmark,'method':method,'method_role':METHOD_ROLES[method],
             'split':split,'metric':metric,'mean':fmean(value for _,value in values),
             'sample_std':stdev(value for _,value in values) if len(values)>1 else None,
             'seeds':sorted(seed for seed,_ in values),'seed_count':len(values),'unit':'percent'}
            for (benchmark,method,split,metric),values in sorted(grouped.items())]


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
    parser.add_argument('--benchmarks', nargs='+', choices=('alfworld', 'appworld'))
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--gpu', nargs='+', type=int, required=True,
                        help='Physical GPU IDs; Engine uses isolated process-local logical indices.')
    parser.add_argument('--model-gpu', nargs='+', type=int,
                        help='Explicit physical model-compute subset of --gpu; all allocated RNG contexts remain intact.')
    parser.add_argument('--resume', action='store_true', help='Continue saved F-GPO study state explicitly.')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    require_protocol(config)
    plan = config['experiment_suites'][args.suite]
    methods = args.methods if args.methods is not None else plan['methods']
    benchmarks = args.benchmarks if args.benchmarks is not None else plan['benchmarks']
    seeds = args.seeds if args.seeds is not None else config['seeds']
    if (not methods or not benchmarks or not seeds
            or not set(methods).issubset(set(plan['methods']) & set(SUITE_METHODS[args.suite]))
            or not set(benchmarks).issubset(set(plan['benchmarks']) & set(SUITE_BENCHMARKS[args.suite]))
            or not set(seeds).issubset(config['seeds'])):
        raise ValueError('CLI selection must be a subset of the declared claim-scoped suite')
    if any(len(values) != len(set(values)) for values in (methods,benchmarks,seeds)):
        raise ValueError('Select each method, benchmark and seed only once')
    config = dict(config,suite=args.suite,methods=list(methods),benchmarks=list(benchmarks))
    require_protocol(config)
    os.environ['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
    isolate_gpus(args.gpu, config['excluded_gpu_ids'])
    compute_allocation(args.gpu, args.model_gpu)
    suite_dir = args.output_dir / args.suite
    if args.resume:
        if not suite_dir.is_dir():
            raise FileNotFoundError('Explicit --resume requires the existing selected suite directory.')
    else:
        suite_dir.mkdir(parents=True, exist_ok=False)
    study_path = suite_dir / 'study_summary.json'
    cells = json.loads(study_path.read_text())['cells'] if args.resume and study_path.is_file() else []
    if any(saved['protocol'] != config['protocol'] or saved['method'] not in SUITE_METHODS[args.suite]
           or saved['benchmark'] not in SUITE_BENCHMARKS[args.suite] for saved in cells):
        raise ValueError('Existing suite summary belongs to another experiment definition')
    for benchmark in benchmarks:
        for method in methods:
            for seed in seeds:
                output = suite_dir / benchmark / method / f'seed-{seed}'
                cell = run_cell(config, output, method=method, benchmark=benchmark,
                                seed=seed, gpu_ids=args.gpu, model_gpu_ids=args.model_gpu,
                                resume=args.resume and output.exists())
                cell.update(method_role=METHOD_ROLES[method],suite=args.suite)
                position = next((index for index, saved in enumerate(cells)
                                 if (saved['method'], saved['benchmark'], saved['seed'])
                                 == (method, benchmark, seed)), None)
                if position is None:
                    cells.append(cell)
                else:
                    cells[position] = cell
                references,reference_aggregates = (reused_fgpo_results(args.output_dir,cells,config)
                    if args.suite=='ablation' else ([],[]))
                write_json(study_path, {
                    'protocol': config['protocol'], 'suite':args.suite,'experiment_plan':plan,
                    'cells': cells,'aggregates':aggregate_cells(cells),
                    'reference_cells':references,'reference_aggregates':reference_aggregates,
                    'design': 'docs/fgpo_design_and_derivation.md'})
                if cell['status'] == 'interrupted':
                    return 130
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
