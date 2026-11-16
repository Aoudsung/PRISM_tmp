"""Four independent formal lanes, or an 840-second read-only progress window."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path('/apps/users/adminuser/fgpo-reproduction-20261116')
CONFIG = ROOT / 'configs/fgpo-qwen7b-official-20261116.json'
LANES = (
    ('full', [3], ['fgpo', 'fgpo_no_full_filtration', 'fgpo_no_befp', 'fgpo_no_dag_loo']),
    ('ppo', [6, 7], ['gigpo', 'bipace_q']),
    ('feedback', [4], ['self', 'skill_disco']),
    ('native', [5], ['native_qwen']),
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.incoming')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def tail(path, limit=8192):
    with path.open('rb') as stream:
        size = stream.seek(0, 2)
        start = max(0, size - limit)
        stream.seek(start)
        lines = stream.read().splitlines(keepends=True)
    if start and lines:
        lines = lines[1:]
    return [line.decode('utf-8', errors='replace').rstrip('\r\n')
            for line in lines if line.endswith(b'\n')]


def observe(supervisor_path):
    deadline = time.monotonic() + 840
    previous_bytes = {}
    shown_errors = set()
    while time.monotonic() < deadline:
        state = json.loads(supervisor_path.read_text())
        rows = []
        for item in state['lanes']:
            lane = Path(item['output_dir'])
            cost_bytes, completed, evaluation_rows = 0, 0, 0
            latest_training = None
            latest_time = -1
            for benchmark in ('alfworld', 'appworld'):
                for method in item['methods']:
                    for seed in (0, 1, 2):
                        cell = lane / benchmark / method / f'seed-{seed}'
                        completed += int((cell / 'cell_summary.json').is_file())
                        costs = cell / 'costs.jsonl'
                        if costs.is_file():
                            cost_bytes += costs.stat().st_size
                        training = cell / 'training.jsonl'
                        if training.is_file() and training.stat().st_mtime_ns > latest_time:
                            lines = tail(training, 65536)
                            if lines:
                                actual = json.loads(lines[-1])
                                latest_training = {'cell': str(cell), **{key: actual[key] for key in
                                    ('round', 'successes', 'trajectories', 'optimizer_steps') if key in actual}}
                                latest_time = training.stat().st_mtime_ns
                        splits = ('valid_unseen',) if benchmark == 'alfworld' else ('test_normal', 'test_challenge')
                        for split in splits:
                            evaluations = cell / f'evaluation-{split}.jsonl'
                            if evaluations.is_file():
                                with evaluations.open('rb') as stream:
                                    evaluation_rows += sum(line.endswith(b'\n') for line in stream)
            row = {key: item.get(key) for key in ('name', 'pid', 'pgid', 'gpu_ids', 'physical_gpu_ids',
                'logical_gpu_ids', 'cuda_visible_devices', 'status', 'exit_code', 'output_dir', 'log_path')}
            row.update(completed_cells=completed, official_evaluation_rows=evaluation_rows,
                       cost_bytes=cost_bytes,
                       cost_growth_bytes=cost_bytes-previous_bytes[item['name']] if item['name'] in previous_bytes else None,
                       latest_training=latest_training)
            previous_bytes[item['name']] = cost_bytes
            if item['status'] == 'failed' and item['name'] not in shown_errors:
                row['error'] = item.get('error')
                row['error_tail'] = tail(Path(item['log_path']))[-16:]
                shown_errors.add(item['name'])
            rows.append(row)
        pids = ','.join(str(item['pid']) for item in state['lanes'] if item['pid'] is not None)
        processes = subprocess.run(['ps', '-p', pids, '-o', 'pid,pgid,etime,pcpu,rss,stat'],
                                   capture_output=True, text=True).stdout if pids else ''
        print(json.dumps({'utc': utc(), 'supervisor_pid': state['supervisor_pid'],
                          'status': state['status'], 'lanes': rows, 'processes': processes},
                         ensure_ascii=False), flush=True)
        if state['lanes'] and state['status'] in ('completed', 'failed') and all(
                item['status'] in ('completed', 'failed') for item in state['lanes']):
            break
        time.sleep(min(20, max(0, deadline-time.monotonic())))
    return 0


def launch(config_path, output, selected_names, full_devices):
    config = json.loads(config_path.read_text())
    selected = [(name, list(full_devices) if name == 'full' else devices, methods)
                for name, devices, methods in LANES if name in selected_names]
    physical_ids = [device for _, devices, _ in selected for device in devices]
    if len(set(physical_ids)) != len(physical_ids) or any(
            device < 0 or device in config['excluded_gpu_ids'] for device in physical_ids):
        raise ValueError('Selected lanes require disjoint healthy physical GPU allocations.')
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), OMP_NUM_THREADS='16',
               MKL_NUM_THREADS='16', OPENBLAS_NUM_THREADS='16',
               TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    state = {'protocol': config['protocol'], 'supervisor_pid': os.getpid(),
             'started_utc': utc(), 'status': 'running', 'source_root': str(ROOT / 'src'),
             'output_dir': str(output), 'config': str(config_path),
             'selected_lanes': [name for name, _, _ in selected],
             'physical_allocations': {name: devices for name, devices, _ in selected},
             'lead_resource_snapshot': {'host': 'mgt01', 'cpu_cores': 144,
                 'mem_available_kib': 320374452,
                 'gpu_free_mib': {'3': 43192, '6': 35267, '7': 35599, '4': 35267, '5': 23560},
                 'excluded_gpu_ids': [0, 2]}, 'lanes': []}
    supervisor_path = output / 'supervisor.json'
    write_json(supervisor_path, state)
    children = []
    for name, devices, methods in selected:
        child_env = dict(env, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                         CUDA_VISIBLE_DEVICES=','.join(map(str, devices)))
        lane = output / name
        log_path = output / f'{name}.log'
        command = [config['python_executable'], '-u', '-m', 'fgpo.runner',
                   '--config', str(config_path), '--output-dir', str(lane),
                   '--methods', *methods, '--benchmarks', 'alfworld', 'appworld',
                   '--seeds', '0', '1', '2', '--gpu', *map(str, devices)]
        item = {'name': name, 'pid': None, 'pgid': None, 'gpu_ids': devices,
                'physical_gpu_ids': devices, 'logical_gpu_ids': list(range(len(devices))),
                'cuda_visible_devices': child_env['CUDA_VISIBLE_DEVICES'],
                'cuda_device_order': child_env['CUDA_DEVICE_ORDER'],
                'gpu_id_semantics': 'argv_gpu_ids_are_physical; Engine_gpu_ids_are_logical',
                'methods': methods, 'command': command, 'output_dir': str(lane),
                'log_path': str(log_path), 'started_utc': utc(), 'status': 'starting'}
        state['lanes'].append(item)
        write_json(output / f'{name}.command.json', {
            'argv': command, 'physical_gpu_ids': devices,
            'logical_gpu_ids': item['logical_gpu_ids'],
            'CUDA_VISIBLE_DEVICES': child_env['CUDA_VISIBLE_DEVICES'],
            'CUDA_DEVICE_ORDER': child_env['CUDA_DEVICE_ORDER']})
        log = log_path.open('x')
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=child_env, stdout=log,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       start_new_session=True)
        except OSError as error:
            log.write(str(error) + '\n')
            log.close()
            item.update(status='failed', exit_code=127, error=str(error), finished_utc=utc())
            (output / f'{name}.exit_status.txt').write_text('127\n')
        else:
            # start_new_session makes this actual child PID its process-group ID.
            item.update(pid=process.pid, pgid=process.pid, status='running')
            children.append((process, log, item))
        write_json(supervisor_path, state)
    with ThreadPoolExecutor(max_workers=len(selected)) as executor:
        waiting = {executor.submit(process.wait): (log, item) for process, log, item in children}
        for finished in as_completed(waiting):
            log, item = waiting[finished]
            result = finished.result()
            log.close()
            item.update(status='completed' if result == 0 else 'failed',
                        exit_code=result, finished_utc=utc())
            (output / f"{item['name']}.exit_status.txt").write_text(str(result) + '\n')
            write_json(supervisor_path, state)
    success = all(item['exit_code'] == 0 for item in state['lanes'])
    state.update(status='completed' if success else 'failed', finished_utc=utc())
    write_json(supervisor_path, state)
    (output / 'exit_status.txt').write_text('0\n' if success else '1\n')
    return 0 if success else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--lanes', nargs='+', choices=[name for name, _, _ in LANES],
                        default=[name for name, _, _ in LANES],
                        help='Launch only these disjoint lanes in a fresh output directory.')
    parser.add_argument('--full-gpu', nargs='+', type=int, default=[3],
                        help='Explicit physical GPU allocation for the selected full lane.')
    parser.add_argument('--watch', type=Path, metavar='SUPERVISOR_JSON')
    args = parser.parse_args()
    if args.watch is not None:
        return observe(args.watch.resolve())
    if args.output_dir is None:
        parser.error('--output-dir is required to launch; --watch only reads an existing supervisor.')
    return launch(args.config.resolve(), args.output_dir, args.lanes, args.full_gpu)


if __name__ == '__main__':
    raise SystemExit(main())
