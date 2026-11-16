"""Complete official F-GPO studies without resource-utility or tier substitutions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import uuid

from .envs import make_env
from .metrics import official_summary


METHODS = ('fgpo', 'fgpo_no_full_filtration', 'fgpo_no_befp', 'fgpo_no_dag_loo',
           'gigpo', 'bipace_q', 'self', 'skill_disco', 'native_qwen')
FGPO_METHODS = METHODS[:4]


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
    for key in ('method', 'benchmark', 'seed', 'model_path'):
        if saved[key] != expected[key]:
            raise ValueError(f'Existing cell identity differs at {key}; no work was restarted.')


def isolate_gpus(physical_ids, excluded_ids):
    physical = list(physical_ids)
    if not physical or len(set(physical)) != len(physical) or any(gpu < 0 for gpu in physical):
        raise ValueError('Select distinct nonnegative physical GPU IDs.')
    if any(gpu in excluded_ids for gpu in physical):
        raise ValueError('The physical allocation includes a known faulty GPU (0 or 2).')
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
    # Resolve physical allocation before importing any Engine/CUDA code.
    logical_ids, visible_mask = isolate_gpus(gpu_ids, config['excluded_gpu_ids'])
    compute_physical, compute_logical = compute_allocation(gpu_ids, model_gpu_ids)
    from .engine import Engine
    output = Path(output_dir)
    cell_config = dict(config, method=method, benchmark=benchmark, seed=int(seed))
    cell_config.update(config['method_configs'][method])
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
    output_limits = cell_config.pop('output_tokens_by_benchmark')
    cell_config['output_tokens'] = output_limits[benchmark]
    cell_config['skill_induction_tasks'] = config['skill_induction_tasks'][benchmark]
    resume_from = None
    if output.exists():
        if not resume:
            raise FileExistsError(f'Existing cell requires explicit --resume: {output}')
        saved_config = json.loads((output / 'config.json').read_text())
        require_identity(saved_config, cell_config)
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
            train_tasks = catalog.list_tasks('train') if method != 'native_qwen' else []
            evaluation_tasks = {split: catalog.list_tasks(split)
                                for split in config['evaluation_splits'][benchmark]}
        finally:
            catalog.close()
        write_json(output / 'official_tasks.json', {'train': train_tasks, **evaluation_tasks})

    def env_factory(task_id, initial_seed, trajectory_id):
        # Each invocation has its own process, files and AppWorld experiment identity.
        directory = output / 'environments' / (str(trajectory_id).replace('/', '_') + '-' + uuid.uuid4().hex)
        return make_env(cell_config, benchmark, directory, seed=int(initial_seed))

    engine = Engine(cell_config, output, resume_from=resume_from)
    try:
        # The method's own window/workflow is executed inside this single call;
        # offline skill induction is not wrapped in an outer round loop.
        training_path = output / 'training_summary.json'
        training = json.loads(training_path.read_text()) if resume_from and training_path.is_file() else None
        if training is None or training.get('status') != 'complete':
            training = engine.train(train_tasks, env_factory) if method != 'native_qwen' else {
                'status': 'complete', 'method': method, 'trained': False, 'initialization': 'frozen_base_qwen'}
        checkpoint = engine.save_checkpoint(output / 'checkpoint')
        write_json(training_path, training)
        if training.get('status') == 'interrupted':
            return {'status': 'interrupted', 'method': method, 'benchmark': benchmark,
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
                if completed_evaluation['dataset_complete']:
                    scores[split] = completed_evaluation
                    continue
            evaluation = engine.evaluate(tasks, env_factory, split=split)
            records = evaluation['records']
            # Official evaluation is already finalized by the actor/Engine exactly once.
            summary = official_summary(benchmark, records, tasks)
            summary.update(method=method, benchmark=benchmark, seed=seed,
                           model_path=cell_config['model_path'], split=split)
            write_json(output / split / 'evaluation_records.json', records)
            write_json(summary_path, summary)
            if evaluation.get('status') == 'interrupted':
                checkpoint = engine.save_checkpoint(output / 'checkpoint')
                return {'status': 'interrupted', 'method': method, 'benchmark': benchmark,
                        'seed': seed, 'model_path': cell_config['model_path'],
                        'checkpoint': str(checkpoint), 'evaluation': {**scores, split: summary}}
            scores[split] = summary
        completed_checkpoint = engine.prepare_completed_checkpoint(training, scores)
        result = {'status': 'complete', 'method': method, 'benchmark': benchmark, 'seed': seed,
                  'model_path': cell_config['model_path'],
                  'checkpoint': completed_checkpoint['path'], 'completed_checkpoint': completed_checkpoint,
                  'training_resume_supported': False,
                  'training': training, 'evaluation': scores, 'costs': engine.ledger.snapshot()}
        write_json(output / 'cell_summary.json', result, durable=True)
        engine.reclaim_completed_checkpoint(output / 'cell_summary.json')
        return result
    finally:
        engine.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
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
    os.environ['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
    isolate_gpus(args.gpu, config['excluded_gpu_ids'])
    compute_allocation(args.gpu, args.model_gpu)
    if args.resume:
        if not args.output_dir.is_dir():
            raise FileNotFoundError('Explicit --resume requires an existing study directory.')
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    study_path = args.output_dir / 'study_summary.json'
    cells = json.loads(study_path.read_text())['cells'] if args.resume and study_path.is_file() else []
    for benchmark in args.benchmarks if args.benchmarks is not None else config['benchmarks']:
        for method in args.methods if args.methods is not None else config['methods']:
            for seed in args.seeds if args.seeds is not None else config['seeds']:
                output = args.output_dir / benchmark / method / f'seed-{seed}'
                cell = run_cell(config, output, method=method, benchmark=benchmark,
                                seed=seed, gpu_ids=args.gpu, model_gpu_ids=args.model_gpu, resume=args.resume)
                position = next((index for index, saved in enumerate(cells)
                                 if (saved['method'], saved['benchmark'], saved['seed'])
                                 == (method, benchmark, seed)), None)
                if position is None:
                    cells.append(cell)
                else:
                    cells[position] = cell
                write_json(study_path, {
                    'protocol': config['protocol'], 'cells': cells,
                    'design': 'docs/fgpo_design_and_derivation.md'})
                if cell['status'] == 'interrupted':
                    return 130
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
