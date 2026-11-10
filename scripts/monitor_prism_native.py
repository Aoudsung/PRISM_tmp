#!/usr/bin/env python3
"""Monitor the three focused native experiment sites from the local terminal."""

import argparse
import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import sys
import traceback


PROJECT = Path(__file__).resolve().parents[1]
COLAB_RUNTIME = '/content/drive/MyDrive/prism-native-focused-20261105/monitor-runtime.json'
HOSTS = {
    'A': ('adminuser@10.8.128.25', 54678, 1, '/apps/users/adminuser/prism-exact-reproduction'),
    'B': ('chenxiangwei@10.8.128.241', 22, 2, '/home/chenxiangwei/workspace/long-horizon-decision'),
}
HANDOFF = Path('/private/tmp/prism-colab-mcp-local-handoff.txt')
MARKER = 'PRISM_MONITOR='

# Sent to the specified remote host or executed in the single monitoring cell.
SNAPSHOT = r'''
import json, runpy, socket
from pathlib import Path

if RUNTIME_FILE is not None:
    runtime_path = Path(RUNTIME_FILE)
    if not runtime_path.exists():
        print('PRISM_MONITOR=' + json.dumps({'host': socket.gethostname(), 'run_directory': RUN_DIRECTORY,
              'status': 'starting', 'jobs': [], 'coordination_requests': []}))
        raise SystemExit(0)
    runtime = json.loads(runtime_path.read_text())
    if RUN_DIRECTORY is not None and str(Path(RUN_DIRECTORY)) != str(Path(runtime['run_directory'])):
        raise RuntimeError('Runtime metadata belongs to another run directory.')
    if PIPELINE_PID is not None and PIPELINE_PID != runtime['pipeline_pid']:
        raise RuntimeError('The configured pipeline PID differs from the current server metadata.')
    if EXPECTED_HOST is not None and EXPECTED_HOST != runtime['host']:
        raise RuntimeError('Runtime metadata belongs to another configured host.')
    RUN_DIRECTORY = runtime['run_directory']
    PIPELINE_PID = runtime['pipeline_pid']
    PIPELINE_START = runtime['pipeline_start']
    EXPECTED_HOST = runtime['host']
if EXPECTED_HOST != socket.gethostname():
    raise RuntimeError('Runtime host changed; update the monitoring target before monitoring.')
if type(PIPELINE_PID) is not int or PIPELINE_PID <= 0:
    raise ValueError('Monitor pipeline_pid must be a positive integer.')
root = Path(RUN_DIRECTORY)
state_path = root / 'scheduler_state.json'
control = None
manifest_path = root / 'manifest.json'
delivered = []
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    if manifest['config'].get('coordination', {}).get('enabled'):
        control = runpy.run_path(str(Path(manifest['source_root']) / 'src/prism_evolve/native_coordination.py'))
        delivered = [control['deliver_reply'](root, reply) for reply in BRIDGE_REPLIES]

def alive(pid, expected_start=None):
    if not pid:
        return None
    path = Path('/proc') / str(pid) / 'stat'
    if not path.exists():
        return False
    fields = path.read_text().rsplit(') ', 1)[-1].split()
    return fields[0] != 'Z' and (expected_start is None or fields[19] == expected_start)

def tail(path):
    path = Path(path)
    if not path.exists():
        return f'Log file missing: {path}'
    with path.open('rb') as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 5000))
        return handle.read().decode('utf-8', errors='replace')

def learning_status(directory):
    path = directory / 'training' / 'training.jsonl'
    if not path.exists():
        return None
    lines = path.read_text().splitlines(keepends=True)
    committed = [line for line in lines if line.endswith('\n')]
    if not committed:
        return {'status': 'waiting_for_committed_training_record'}
    record = json.loads(committed[-1])
    return {key: record.get(key) for key in (
        'group', 'loss', 'gradient_surrogate_loss', 'updates', 'nonzero_gradient_steps', 'mean_return',
        'parameter_change_steps', 'parameters_changed', 'learning_status', 'learning_outcome', 'mean_native_success',
        'trajectory_count', 'trajectory_successes', 'public_error_stages',
        'optimizer_step_diagnostics', 'preference_status', 'partial', 'reason',
    )}

report = {'host': socket.gethostname(), 'run_directory': str(root),
          'pipeline_pid': PIPELINE_PID, 'pipeline_start': PIPELINE_START,
          'pipeline_alive': alive(PIPELINE_PID, PIPELINE_START), 'jobs': [],
          'coordination_delivered': delivered,
          'coordination_requests': control['mailbox_requests'](root) if control is not None else []}
if state_path.exists():
    state = json.loads(state_path.read_text())
    if state.get('host') != report['host']:
        raise RuntimeError('Scheduler state belongs to another host; check the selected run directory.')
    report.update(status=state.get('status'), updated_at=state.get('updated_at'),
                  coordination=state.get('coordination'), scheduler_error=state.get('error'),
                  gpu_inventory=state.get('last_gpu_inventory'), inventory_at=state.get('inventory_at'))
    for bundle, job in state.get('jobs', {}).items():
        job_root = Path(job.get('run_dir', str(root)))
        counts, active, failed, training_completed = {}, [], [], []
        for cell_id, previous in job.get('cells', {}).items():
            receipt = job_root / 'cells' / cell_id / 'status.json'
            cell = json.loads(receipt.read_text()) if receipt.exists() else previous
            status = cell.get('status', 'pending')
            counts[status] = counts.get(status, 0) + 1
            training_summary = job_root / 'cells' / cell_id / 'training_phase.json'
            if training_summary.exists():
                phase = json.loads(training_summary.read_text())
                training = phase.get('training', {})
                training_completed.append({'cell': cell_id, 'summary': {
                    'parameter_training': phase.get('parameter_training'),
                    **{key: training.get(key) for key in ('learning_outcome', 'learning_status', 'updates',
                        'nonzero_gradient_steps', 'parameter_change_steps', 'memory_update_status',
                        'memory_changed_episodes', 'stop_reason')},
                }})
            if status == 'running':
                active.append({'cell': cell_id, 'phase': cell.get('phase'),
                               'split': cell.get('split'), 'task_index': cell.get('task_index'),
                               'learning': learning_status(job_root / 'cells' / cell_id)})
            if status in ('failed', 'blocked'):
                failed.append({'cell': cell_id, 'status': status,
                               'reason': cell.get('error') or cell.get('reason'),
                               'traceback': cell.get('traceback')})
        item = {'bundle': job.get('bundle_id', bundle), 'execution_id': job.get('execution_id'),
                'origin_site': job.get('origin_site'), 'planned_gpu': job.get('planned_gpu'),
                'gpu_uuid': job.get('gpu_uuid'), 'quota': job.get('quota'), 'used_mib': job.get('used_mib'),
                'gpu_peak_observed_mib': job.get('gpu_peak_observed_mib'),
                'status': job.get('status'), 'gpu': job.get('gpu'),
                'pid': job.get('pid'),
                'pid_alive': alive(job.get('pid'), job.get('pid_start')), 'counts': counts, 'active': active,
                'training_completed': training_completed, 'failed': failed}
        if (item['status'] in ('failed', 'blocked') or failed
                or (item['status'] == 'running' and item['pid_alive'] is False
                    and set(counts) != {'done'})):
            item['reason'] = job.get('reason')
            item['log_tail'] = tail(job.get('log_path') or root / 'pipeline.log')
        report['jobs'].append(item)
else:
    report['status'] = 'starting' if report['pipeline_alive'] else 'missing_state'
if ((report['pipeline_alive'] is False and report.get('status') != 'done')
        or report['status'] in ('failed', 'blocked')):
    report['pipeline_log_tail'] = tail(root / 'pipeline.log')
print('PRISM_MONITOR=' + json.dumps(report, ensure_ascii=False))
'''


def snapshot_code(run_directory, pipeline_pid, runtime_file=None, expected_host=None, bridge_replies=()):
    return (f'RUN_DIRECTORY = {run_directory!r}\nPIPELINE_PID = {pipeline_pid!r}\n'
            'PIPELINE_START = None\n'
            f'RUNTIME_FILE = {runtime_file!r}\nEXPECTED_HOST = {expected_host!r}\n'
            f'BRIDGE_REPLIES = {list(bridge_replies)!r}\n' + SNAPSHOT)


async def ssh_process(label, command):
    host, port, password_index, directory = HOSTS[label]
    env = dict(os.environ, DISPLAY='prism-monitor', SSH_ASKPASS_REQUIRE='force',
               SSH_ASKPASS=str(PROJECT / 'deployment/prism_monitor_askpass.sh'),
               PRISM_SSH_DOCUMENT=str(PROJECT / 'SSH_Document.md'),
               PRISM_SSH_PASSWORD_INDEX=str(password_index))
    return await asyncio.create_subprocess_exec(
        'ssh', '-o', 'BatchMode=no', '-o', 'PreferredAuthentications=password',
        '-o', 'PubkeyAuthentication=no', '-o', 'NumberOfPasswordPrompts=1',
        '-o', 'ConnectTimeout=15', '-p', str(port), host, command,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, env=env, start_new_session=True,
        limit=4 * 1024 * 1024,
    )
async def host_snapshot(label, target, replies=()):
    directory = HOSTS[label][3]
    process = await ssh_process(label, f'cd {shlex.quote(directory)} && python3 -')
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(snapshot_code(target['run_directory'], target.get('pipeline_pid'),
                                              expected_host=target['host'], bridge_replies=replies,
                                              runtime_file=str(Path(target['run_directory']) / 'monitor-runtime.json')).encode()), timeout=25)
    except TimeoutError:
        process.kill()
        await process.wait()
        return {'connection_error': 'SSH 连接超时'}
    if process.returncode:
        return {'connection_error': stderr.decode(errors='replace')[-1200:]}
    report = extract_report(stdout.decode(errors='replace'))
    if report is None:
        raise RuntimeError(f'{label}: SSH output has no monitor report.')
    return report


async def start_coordinator(target):
    bootstrap = (
        "import json,os,pathlib,socket,sys; p=pathlib.Path(sys.argv[1]); "
        "m=json.loads((p/'manifest.json').read_text()); "
        "\nif socket.gethostname()!=sys.argv[2] or m['host']!=sys.argv[2]: "
        "raise RuntimeError('Coordinator host changed')\n"
        "os.chdir(m['config']['project_root']); "
        "os.environ['PYTHONPATH']=str(pathlib.Path(m['source_root'])/'src'); "
        "os.execv(m['config']['python_executable'], [m['config']['python_executable'], '-m', "
        "'prism_evolve.native_coordination', '--registry', "
        "str(pathlib.Path(m['config']['coordination']['registry_root'])/m['config']['coordination']['session_id']/'registry.json'), "
        "'--host', sys.argv[2], '--serve'])"
    )
    return await ssh_process('A', shlex.join(['python3', '-c', bootstrap, target['run_directory'], target['host']]))


async def relay_request(process, request):
    process.stdin.write((json.dumps(request, ensure_ascii=False, allow_nan=False) + '\n').encode())
    await process.stdin.drain()
    line = await asyncio.wait_for(process.stdout.readline(), timeout=60)
    if not line:
        error = (await process.stderr.read()).decode(errors='replace')
        raise RuntimeError('Server-side coordinator exited: ' + error[-3000:])
    reply = json.loads(line)
    if reply.get('request_id') != request['request_id']:
        raise ValueError('Coordinator response identity differs from the forwarded request.')
    return reply


async def relay_bridge(client, targets, reports, cell_id, bridge, runtime_path):
    process = bridge['process']
    identity = targets['A']['host'], targets['A']['run_directory']
    if process is not None and bridge['target'] != identity:
        raise RuntimeError('协调运行身份已改变；请主代理结束旧监控后启动新连接，不复用旧领取会话。')
    a = reports['A']
    ready = bool(a and ((a.get('coordination') or {}).get('registered')
                       or any(item['operation'] == 'register' for item in a.get('coordination_requests', []))))
    if process is None and ready:
        process = await start_coordinator(targets['A'])
        bridge.update(process=process, target=identity)
    if process is None:
        return reports, False
    processed = False
    for label in ('A', 'B', 'Colab'):
        report = reports[label]
        if not report or 'connection_error' in report:
            continue
        requests = report.get('coordination_requests', [])
        if not requests:
            continue
        replies = [await relay_request(process, request) for request in requests]
        if label == 'Colab':
            result = await client.call_tool_mcp('update_cell', {
                'cellId': cell_id, 'content': snapshot_code(None, None, runtime_file=runtime_path, bridge_replies=replies),
            }, timeout=30)
            if result.isError:
                raise RuntimeError('Colab rejected the data-only coordination bridge update.')
            result = await client.call_tool_mcp('run_code_cell', {'cellId': cell_id}, timeout=30)
            updated = extract_report(result)
            if updated is None:
                raise RuntimeError('The Colab bridge returned no monitoring report.')
        else:
            updated = await host_snapshot(label, targets[label], replies)
        if 'connection_error' in updated:
            raise RuntimeError(f"{label} coordination delivery failed: {updated['connection_error']}")
        if not set(reply['request_id'] for reply in replies) <= set(updated.get('coordination_delivered', [])):
            raise RuntimeError(f'{label} did not acknowledge its coordination responses; no requests will be repeated.')
        reports[label] = updated
        processed = True
        errors = [reply['error'] for reply in replies if 'error' in reply]
        if errors:
            raise RuntimeError('; '.join(errors))
    return reports, processed


def extract_report(value):
    if hasattr(value, 'model_dump'):
        value = value.model_dump()
    if isinstance(value, dict):
        # Colab returns notebook outputs inside JSON text and structuredContent.
        if value.get('isError'):
            raise RuntimeError(f"Colab MCP tool failed: {value.get('content')}")
        if value.get('structuredContent') is not None:
            return extract_report(value['structuredContent'])
        if 'outputs' in value:
            stdout = []
            for output in value['outputs']:
                if output.get('output_type') == 'error':
                    raise RuntimeError(f"Colab monitoring cell failed: {output.get('ename')}: "
                                       f"{output.get('evalue')}\n" + '\n'.join(output.get('traceback', [])))
                if output.get('output_type') == 'stream' and output.get('name') == 'stdout':
                    text = output['text']
                    stdout.append(''.join(text) if isinstance(text, list) else text)
            return extract_report(''.join(stdout))
        if value.get('type') == 'text':
            return extract_report(value['text'])
        for key in ('content', 'cells', 'result'):
            result = extract_report(value.get(key))
            if result is not None:
                return result
    elif isinstance(value, list):
        for item in value:
            result = extract_report(item)
            if result is not None:
                return result
    elif isinstance(value, str):
        reports = []
        for line in value.splitlines():
            if line.startswith(MARKER):
                report = json.loads(line[len(MARKER):])
                if (not isinstance(report, dict) or type(report.get('pipeline_alive')) is not bool
                        or not isinstance(report.get('jobs'), list)):
                    raise ValueError('Monitor output has an invalid report schema.')
                reports.append(report)
        if len(reports) > 1:
            raise ValueError('Monitor output contains multiple status reports.')
        if reports:
            return reports[0]
        if value.lstrip().startswith(('{', '[', '"')):
            return extract_report(json.loads(value))
    return None


def failures(report):
    if not report or 'connection_error' in report:
        return []
    found = []
    if (report.get('status') in ('failed', 'blocked')
            or report.get('pipeline_alive') is False and report.get('status') != 'done'):
        found.append({'reason': 'Pipeline failed or exited before completion',
                      'log_tail': report.get('pipeline_log_tail', '')})
    for job in report.get('jobs', []):
        if (job.get('status') in ('failed', 'blocked') or job.get('failed')
                or (job.get('status') == 'running' and job.get('pid_alive') is False
                    and set(job.get('counts', {})) != {'done'})):
            found.append(job)
    return found


def describe(label, report):
    if report is None:
        return f'{label}: 等待 Colab MCP 连接'
    if 'connection_error' in report:
        return f"{label}: 连接异常 {report['connection_error']}"
    jobs = []
    for job in report.get('jobs', []):
        active = ','.join(item['cell'] + (':' + item['phase'] if item.get('phase') else '')
                          + (f" {item.get('split')}#{item['task_index']}" if item.get('task_index') is not None else '')
                          + (f" learning={json.dumps(item['learning'], ensure_ascii=False)}" if item.get('learning') else '')
                          for item in job.get('active', []))
        jobs.append(f"{job['bundle']} GPU={job.get('gpu')} uuid={job.get('gpu_uuid')} {job.get('status')} "
                    f"execution={job.get('execution_id')} origin={job.get('origin_site')}/GPU{job.get('planned_gpu')} "
                    f"used_mib={job.get('used_mib')} peak_observed_mib={job.get('gpu_peak_observed_mib')} quota={job.get('quota')} "
                    f"pid={job.get('pid')} alive={job.get('pid_alive')} "
                    f"cells={job.get('counts')}" + (f' active={active}' if active else ''))
    inventory = ', '.join(
        f"GPU={gpu['index']} {gpu['name']} util={gpu['utilization_percent']}% "
        f"VRAM={gpu['used_mib']}/{gpu['total_mib']}MiB"
        for gpu in report.get('gpu_inventory') or []
    )
    coordination = report.get('coordination') or {}
    global_state = coordination.get('global') or {}
    claimable = global_state.get('claimable')
    pending = global_state.get('pending')
    return (f"{label}: host={report.get('host')} run={report.get('run_directory')} "
            f"{report.get('status')} pid={report.get('pipeline_pid')} "
            f"pipeline_alive={report.get('pipeline_alive')} state_at={report.get('updated_at')} "
            f"inventory_at={report.get('inventory_at')} {inventory} "
            f"claimable_count={len(claimable) if claimable is not None else None} "
            f"global_pending_count={len(pending) if pending is not None else None} "
            f"missing_backbones={global_state.get('missing_backbone_capabilities')} "
            f"scheduler_error={report.get('scheduler_error')} coordination_error={report.get('coordination_error')} " + ' | '.join(jobs))


class MonitorReporter:
    """Append complete snapshots; print only changes meaningful to the operator."""
    def __init__(self, path):
        self.path = path
        self.previous = {}
        self.first_events = set()

    def record(self, kind, **fields):
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'at': datetime.now().astimezone().isoformat(), 'kind': kind, **fields},
                                    ensure_ascii=False) + '\n')

    def event(self, kind, key, value):
        if self.previous.get((kind, key)) == value:
            return
        self.previous[kind, key] = value
        event = {'event': kind, 'identity': key, 'value': value}
        self.record('event', **event)
        print('PRISM_EVENT=' + json.dumps(event, ensure_ascii=False), flush=True)

    def reports(self, reports):
        self.record('snapshot', reports=reports)
        for site, report in reports.items():
            if report is None:
                continue
            self.event('site_status', site, {key: report.get(key) for key in
                       ('host', 'status', 'pipeline_alive', 'connection_error', 'scheduler_error', 'coordination_error')})
            for job in report.get('jobs', []):
                identity = site + ':' + job['bundle']
                self.event('worker_status', identity, {key: job.get(key) for key in ('pid', 'status', 'pid_alive')})
                for cell in job.get('active', []):
                    key = identity + ':' + cell['cell']
                    self.event('cell_phase', key, cell.get('phase'))
                    learning = cell.get('learning')
                    if learning and learning.get('group') is not None:
                        self.event('training_group', key, learning)
                        for name, observed in (
                            ('first_native_success', (learning.get('trajectory_successes') or 0) > 0),
                            ('first_parameter_update', (learning.get('parameter_change_steps') or 0) > 0),
                        ):
                            if observed and (name, key) not in self.first_events:
                                self.first_events.add((name, key))
                                self.event(name, key, learning)
                for cell in job.get('training_completed', []):
                    self.event('method_training_finished', identity + ':' + cell['cell'], cell['summary'])
            for issue in failures(report):
                summary = {key: issue.get(key) for key in ('bundle', 'status', 'reason', 'error')}
                summary['failed_cells'] = [{key: cell.get(key) for key in ('cell', 'status', 'reason')}
                                           for cell in issue.get('failed', [])]
                self.event('failure', site + ':' + str(issue.get('bundle', 'pipeline')), summary)


async def monitor(cell_id, interval, runtime_file, colab_runtime_file, reporter):
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport
    transport = StdioTransport(sys.executable, [str(PROJECT / 'deployment/prism_colab_mcp_entry.py')],
                               keep_alive=False)
    colab_connected = False
    bridge = {'process': None, 'target': None}
    all_complete_reported = False
    armed = False
    async with Client(transport) as client:
        print(f'Colab MCP 接入文件: {HANDOFF}（仅在本机复制到 Colab，不要发送到聊天）', flush=True)
        try:
            while True:
                targets = json.loads(runtime_file.read_text())
                if targets == {}:
                    if armed:
                        raise RuntimeError('运行中的监控目标被清空；协调连接不会自动重置。')
                    await client.list_tools()
                    await asyncio.sleep(1)
                    continue
                armed = True
                snapshots = await asyncio.gather(host_snapshot('A', targets['A']), host_snapshot('B', targets['B']),
                                                 return_exceptions=True)
                errors = [item for item in snapshots if isinstance(item, Exception)]
                a, b = [({'connection_error': f'{type(item).__name__}: {item}'}
                         if isinstance(item, Exception) else item) for item in snapshots]
                colab = None
                try:
                    names = {tool.name for tool in await client.list_tools()}
                    if 'run_code_cell' in names:
                        colab_connected = True
                        result = await client.call_tool_mcp('run_code_cell', {'cellId': cell_id}, timeout=30)
                        colab = extract_report(result)
                        if colab is None:
                            raise RuntimeError('Colab 返回中没有监控状态')
                        if 'coordination_requests' not in colab:
                            raise RuntimeError('Colab 专用监控单元缺少协调桥接代码；请部署 --print-colab-cell 生成的当前源码。')
                    elif colab_connected:
                        raise RuntimeError('Colab MCP 已断开；由主代理处理运行时及手工重连')
                except Exception as error:
                    errors.append(error)
                    colab = {'connection_error': f'{type(error).__name__}: {error}'}
                reports = {'A': a, 'B': b, 'Colab': colab}
                processed = False
                if not errors and not any(report and 'connection_error' in report for report in reports.values()):
                    try:
                        reports, processed = await relay_bridge(client, targets, reports, cell_id, bridge, colab_runtime_file)
                    except Exception as error:
                        errors.append(error)
                        reports['A']['coordination_error'] = f'{type(error).__name__}: {error}'
                a, b, colab = reports['A'], reports['B'], reports['Colab']
                reporter.reports(reports)
                if errors:
                    raise errors[0]
                if any(report and 'connection_error' in report for report in (a, b, colab)):
                    raise RuntimeError('三端监控连接失败；监控已停止，交主代理处理')
                if (not all_complete_reported and all(report and report.get('status') == 'done'
                        and (report.get('coordination') or {}).get('global', {}).get('global_complete')
                        for report in (a, b, colab))):
                    print('PRISM_ALL_COMPLETE=所有登记任务与反馈已完成；由主代理保存汇总并关闭 Colab GPU。', flush=True)
                    all_complete_reported = True
                await asyncio.sleep(0.25 if processed else interval)
        finally:
            coordinator = bridge['process']
            if coordinator is not None:
                coordinator.stdin.close()
                try:
                    await asyncio.wait_for(coordinator.wait(), timeout=5)
                except TimeoutError:
                    coordinator.kill()
                    await coordinator.wait()


async def monitor_servers(interval, runtime_file, reporter):
    """Relay the registered A/B experiment without starting a Colab connection."""
    bridge = {'process': None, 'target': None}
    try:
        while True:
            targets = json.loads(runtime_file.read_text())
            if targets == {}:
                await asyncio.sleep(1)
                continue
            snapshots = await asyncio.gather(host_snapshot('A', targets['A']), host_snapshot('B', targets['B']),
                                             return_exceptions=True)
            reports = {label: {'connection_error': f'{type(value).__name__}: {value}'}
                       if isinstance(value, Exception) else value
                       for label, value in zip(('A', 'B'), snapshots)}
            for label, report in reports.items():
                if 'connection_error' in report:
                    reporter.reports(reports)
                    raise RuntimeError(f"{label}: {report['connection_error']}")
            reports['Colab'] = None
            reports, processed = await relay_bridge(None, targets, reports, None, bridge, None)
            reporter.reports(reports)
            if all(reports[label].get('status') == 'done'
                   and (reports[label].get('coordination') or {}).get('global', {}).get('global_complete')
                   for label in ('A', 'B')):
                print('PRISM_ALL_COMPLETE=Host A/B 登记任务与反馈已完成。', flush=True)
                return
            await asyncio.sleep(0.25 if processed else interval)
    finally:
        process = bridge['process']
        if process is not None:
            process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                process.kill()
                await process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--colab-cell', help='Colab 中唯一监控单元格的 ID')
    parser.add_argument('--servers-only', action='store_true', help='仅监控并转发 Host A/B；不连接 Colab')
    parser.add_argument('--log-file', type=Path, help='追加保存完整原始状态和事件的JSONL；控制台仅报告重要变化')
    parser.add_argument('--print-colab-cell', action='store_true', help='输出监控与数据转发单元源码后退出')
    parser.add_argument('--colab-runtime-file', default=COLAB_RUNTIME, help='生成监控单元时嵌入的运行目标 JSON 路径')
    parser.add_argument('--runtime-file', type=Path, help='本机 A/B 目标 JSON：host、run_directory；PID 从服务器运行元数据读取。空对象用于连接准备。')
    parser.add_argument('--interval', type=int, default=5, help='刷新间隔（秒）；领取请求处理后立即继续读取新消息')
    arguments = parser.parse_args()
    if arguments.print_colab_cell:
        print(snapshot_code(None, None, runtime_file=arguments.colab_runtime_file))
        return
    if not arguments.servers_only and not arguments.colab_cell:
        parser.error('--colab-cell is required unless --print-colab-cell is used')
    if arguments.runtime_file is None:
        parser.error('--runtime-file is required for current A/B run directories, hosts and PIDs')
    if arguments.interval <= 0:
        parser.error('--interval must be positive')
    log_path = arguments.log_file or arguments.runtime_file.with_suffix('.monitor.jsonl')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    reporter = MonitorReporter(log_path)
    reporter.record('monitor_started', servers_only=arguments.servers_only)
    print('PRISM_MONITOR_LOG=' + str(log_path), flush=True)
    try:
        if arguments.servers_only:
            asyncio.run(monitor_servers(arguments.interval, arguments.runtime_file, reporter))
        else:
            asyncio.run(monitor(arguments.colab_cell, arguments.interval, arguments.runtime_file, arguments.colab_runtime_file, reporter))
    except Exception as error:
        reporter.record('exception', traceback=traceback.format_exc())
        reporter.event('monitor_failed', 'monitor', {'error': f'{type(error).__name__}: {error}'})
        print('PRISM_MONITOR_ERROR=' + json.dumps({
            'error': f'{type(error).__name__}: {error}',
            'action': '监控已停止，由主代理处理真实故障；实验不会自动重启。',
        }, ensure_ascii=False), file=sys.stderr, flush=True)
        raise
    finally:
        reporter.record('monitor_exited')
        if not arguments.servers_only:
            HANDOFF.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
