"""Explicitly continue the transferred full/PPO lanes on Host B exactly once."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess


ROOT = Path('/home/chenxiangwei/workspace/fgpo-reproduction-20261116')
CONFIG = ROOT / 'configs/fgpo-host-b-official-20261116.json'
DATA = ROOT / 'outputs/fgpo-formal-20261116'
LANES = (
    ('full', [0], ['fgpo', 'fgpo_no_full_filtration', 'fgpo_no_befp', 'fgpo_no_dag_loo'],
     '/apps/users/adminuser/fgpo-reproduction-20261116/outputs/fgpo-formal-20261116/attempt04/full'),
    ('ppo', [1, 2], ['gigpo', 'bipace_q'],
     '/apps/users/adminuser/fgpo-reproduction-20261116/outputs/fgpo-formal-20261116/attempt03/ppo'),
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    pending = path.with_suffix(path.suffix + '.incoming')
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Fresh supervisor/log directory; transferred lane data remain separate.')
    parser.add_argument('--lanes', nargs='+', choices=['full', 'ppo'], default=['full', 'ppo'],
                        help='Explicit lanes to continue; unselected running lanes stay untouched.')
    args = parser.parse_args()
    selected = [lane for lane in LANES if lane[0] in args.lanes]
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), OMP_NUM_THREADS='16',
               MKL_NUM_THREADS='16', OPENBLAS_NUM_THREADS='16',
               TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    state = {'protocol': config['protocol'], 'supervisor_pid': os.getpid(),
             'started_utc': utc(), 'status': 'running', 'host': 'ubuntu',
             'source_root': str(ROOT / 'src'), 'config': str(config_path),
             'output_dir': str(output), 'selected_lanes': [lane[0] for lane in selected],
             'physical_allocations': {name: devices for name, devices, _, _ in selected},
             'compute_physical_allocations': {name: [2] if name == 'ppo' else devices
                                              for name, devices, _, _ in selected},
             'lead_resource_snapshot': {'cpu_cores': 152, 'available_ram_mib': 461098,
                 'gpu_model': 'RTX6000D', 'gpu_free_gb': {'0': 75, '1': 57, '2': 78},
                 'excluded_gpu_ids': []}, 'lanes': []}
    metadata = output / 'supervisor.json'
    write_json(metadata, state)
    children = []
    for name, devices, methods, origin in selected:
        compute_devices = [2] if name == 'ppo' else devices
        compute_logical = [devices.index(device) for device in compute_devices]
        lane = DATA / name
        child_env = dict(env, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                         CUDA_VISIBLE_DEVICES=','.join(map(str, devices)))
        command = [config['python_executable'], '-u', '-m', 'fgpo.runner',
                   '--config', str(config_path), '--output-dir', str(lane), '--resume',
                   '--methods', *methods, '--benchmarks', 'alfworld', 'appworld',
                   '--seeds', '0', '1', '2', '--gpu', *map(str, devices)]
        if name == 'ppo':
            command.extend(['--model-gpu', '2'])
        log_path = output / f'{name}.log'
        item = {'name': name, 'pid': None, 'pgid': None, 'status': 'starting',
                'started_utc': utc(), 'gpu_ids': devices, 'physical_gpu_ids': devices,
                'logical_gpu_ids': list(range(len(devices))),
                'allocated_physical_gpu_ids': devices,
                'allocated_logical_gpu_ids': list(range(len(devices))),
                'compute_physical_gpu_ids': compute_devices,
                'compute_logical_gpu_ids': compute_logical,
                'cuda_visible_devices': child_env['CUDA_VISIBLE_DEVICES'],
                'cuda_device_order': child_env['CUDA_DEVICE_ORDER'],
                'gpu_id_semantics': 'argv_physical; Engine_process_local_logical',
                'methods': methods, 'command': command, 'output_dir': str(lane),
                'log_path': str(log_path), 'transferred_from': origin, 'resume': True}
        state['lanes'].append(item)
        write_json(output / f'{name}.command.json', {
            'argv': command, 'physical_gpu_ids': devices,
            'logical_gpu_ids': item['logical_gpu_ids'],
            'allocated_physical_gpu_ids': devices,
            'allocated_logical_gpu_ids': item['allocated_logical_gpu_ids'],
            'compute_physical_gpu_ids': compute_devices,
            'compute_logical_gpu_ids': compute_logical,
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
            item.update(pid=process.pid, pgid=process.pid, status='running')
            children.append((process, log, item))
        write_json(metadata, state)
    with ThreadPoolExecutor(max_workers=len(selected)) as pool:
        waiting = {pool.submit(process.wait): (log, item) for process, log, item in children}
        for finished in as_completed(waiting):
            log, item = waiting[finished]
            result = finished.result()
            log.close()
            item.update(status='completed' if result == 0 else 'failed', exit_code=result,
                        finished_utc=utc())
            (output / f"{item['name']}.exit_status.txt").write_text(str(result) + '\n')
            write_json(metadata, state)
    success = all(item['exit_code'] == 0 for item in state['lanes'])
    state.update(status='completed' if success else 'failed', finished_utc=utc())
    write_json(metadata, state)
    (output / 'exit_status.txt').write_text('0\n' if success else '1\n')
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
