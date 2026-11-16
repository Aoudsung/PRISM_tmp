"""Official ALFWorld/AppWorld execution in independent environment processes.

Native initialization and scoring are extracted from the former native adapters;
no private state, reference answers, PRISM programs or resource utility enter feedback.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import threading
from typing import Any, Mapping
import uuid


@dataclass
class StepResult:
    observation: str
    done: bool
    success: bool
    metadata: dict[str, Any]

    @property
    def reward(self):
        return float(self.success)

    @property
    def info(self):
        return self.metadata


class ALFWorldEnv:
    benchmark = 'alfworld'
    SPLITS = {'train': 'train', 'dev': 'valid_seen', 'test': 'valid_unseen',
              'valid_seen': 'valid_seen', 'valid_unseen': 'valid_unseen'}

    def __init__(self, data_root, *, run_dir):
        import alfworld.agents.environment.alfred_tw_env as runtime
        import textworld
        self.runtime, self.textworld = runtime, textworld
        root = Path(data_root).resolve()
        self.root = root if root.name == 'json_2.1.1' else root / 'json_2.1.1'
        self.run_dir = Path(run_dir)
        self.world = None
        self.rows = {}
        self.state = None
        self.steps = 0
        self.done = False
        self.action_schema = {'benchmark': self.benchmark,
            'native_system_prompt': 'You act in the official ALFWorld text environment. Emit <think>...</think> followed by <action>one native command</action>. Use the disclosed admissible commands. The environment ends when the goal is satisfied.'}

    def list_tasks(self, split):
        directory = self.SPLITS[split]
        config = {'dataset': {'data_path': str(self.root / 'train'),
                             'eval_id_data_path': str(self.root / 'valid_seen'),
                             'eval_ood_data_path': str(self.root / 'valid_unseen'),
                             'num_train_games': -1, 'num_eval_games': -1},
                  'env': {'goal_desc_human_anns_prob': 0.0, 'task_types': [1, 2, 3, 4, 5, 6]}}
        native_split = {'train': 'train', 'valid_seen': 'eval_in_distribution',
                        'valid_unseen': 'eval_out_of_distribution'}[directory]
        loader = self.runtime.AlfredTWEnv(config, train_eval=native_split)
        rows = []
        for game in sorted(Path(p).resolve() for p in loader.game_files):
            task_id = game.relative_to(self.root).as_posix()
            row = {'task_id': task_id, 'scenario_id': game.parent.parent.name,
                   'split': split, 'native_split': directory}
            self.rows[task_id] = (row, game)
            rows.append(row)
        return rows

    def _observation(self):
        commands = list(self.state['admissible_commands']) if not self.done else []
        self.action_schema['available_actions'] = commands
        return json.dumps({'native_observation': str(self.state['feedback']),
                           'classic_text_action_state': {'available_actions': commands}}, ensure_ascii=False)

    def reset(self, task_id, initial_seed):
        self.close()
        if task_id not in self.rows:
            # The collector receives IDs from list_tasks; open that exact game
            # without rebuilding the full official catalog in every slot.
            game = (self.root / task_id).resolve()
            native_split = game.relative_to(self.root).parts[0]
            self.rows[task_id] = ({'task_id': task_id, 'scenario_id': game.parent.parent.name,
                                  'split': native_split, 'native_split': native_split}, game)
        row, game = self.rows[task_id]
        self.task_id = task_id
        random.seed(initial_seed)
        import numpy as np
        np.random.seed(initial_seed % (2**32))
        infos = self.textworld.EnvInfos(won=True, admissible_commands=True, extras=['gamefile'])
        self.world = self.textworld.start(str(game), request_infos=infos,
            wrappers=[partial(self.runtime.AlfredDemangler, shuffle=False), self.runtime.AlfredInfos])
        self.world.seed(int(initial_seed))
        self.state = self.world.reset()
        self.steps, self.done = 0, False
        match = re.search(r'Your task is to:\s*(.*)', str(self.state['feedback']))
        if match is None:
            raise ValueError('Official ALFWorld reset has no public instruction.')
        observation = self._observation()
        return {**row, 'benchmark': self.benchmark, 'instruction': match.group(1).strip(),
                'observation': observation, 'action_schema': deepcopy(self.action_schema)}

    def step_native(self, action):
        if self.world is None or self.done:
            raise RuntimeError('Native ALFWorld step requires a live reset episode.')
        if action.get('kind') == 'native_text':
            text = action['text']
        elif action.get('kind') == 'invalid':
            text = action.get('raw_text', action.get('text', ''))
        else:
            raise ValueError('ALFWorld requires the native_text action mapping.')
        valid = text in self.state['admissible_commands']
        self.state, _, native_done = self.world.step(text)
        self.steps += 1
        won = bool(self.state['won'])
        self.done = bool(native_done or won or self.steps >= 50)
        return StepResult(self._observation(), self.done, won,
                          {'valid': valid, 'step_failed': not valid, 'native_executed': True,
                           'native_admissible_commands': list(self.state['admissible_commands']),
                           'official_termination': bool(native_done or won), 'steps': self.steps})

    def finalize_evaluation(self):
        row, _ = self.rows[self.task_id]
        return {**row, 'benchmark': self.benchmark, 'success': bool(self.state['won']),
                'native_evaluator': 'ALFWorld.TextWorld.won', 'steps': self.steps}

    def close(self):
        if self.world is not None:
            self.world.close()
            self.world = None


class AppWorldEnv:
    benchmark = 'appworld'
    BOOTSTRAP = ('api_docs.show_app_descriptions', 'api_docs.show_api_descriptions',
                 'api_docs.show_api_doc', 'supervisor.complete_task')

    def __init__(self, data_root, *, run_dir):
        from freezegun import configure
        configure(extend_ignore_list=['torch', 'transformers', 'peft', 'accelerate', 'safetensors',
                                     'tokenizers', 'triton', 'huggingface_hub', 'xgrammar', 'tvm_ffi'])
        from appworld import AppWorld
        from appworld.common.path_store import path_store
        path_store.update_root(str(Path(data_root).resolve()))
        self.AppWorld = AppWorld
        self.run_dir = Path(run_dir)
        self.experiment_name = 'fgpo/' + self.run_dir.name + '-' + uuid.uuid4().hex
        self.world = None
        self.splits = {}
        self.steps = 0
        self.done = False
        self.action_schema = {'benchmark': self.benchmark, 'native_system_prompt':
            'Emit one Python code block. The persistent shell already provides apis. Call apis.<app>.<api>, print values you need to observe, discover documentation through api_docs, and submit with apis.supervisor.complete_task(). Variables persist across interactions. After completing all requested operations, submit without answer unless the task explicitly asks a question.'}

    def list_tasks(self, split):
        from appworld.task import load_task_ids, task_id_to_generator_id
        if split not in ('train', 'dev', 'test_normal', 'test_challenge'):
            raise ValueError(f'Unknown official AppWorld split: {split}')
        rows = [{'task_id': task, 'scenario_id': task_id_to_generator_id(task), 'split': split}
                for task in load_task_ids(split)]
        self.splits[split] = rows
        return rows

    def reset(self, task_id, initial_seed):
        self.close()
        self.task_id = task_id
        self.steps, self.done = 0, False
        self.world = self.AppWorld(task_id=task_id, experiment_name=self.experiment_name + '/' + uuid.uuid4().hex,
            random_seed=int(initial_seed), max_interactions=100, max_api_calls_per_interaction=1000,
            timeout_seconds=10, load_ground_truth=True, ground_truth_mode='minimal',
            raise_on_failure=True, raise_on_extra_parameters=True, show_api_response_schemas=True)
        task = self.world.task
        supervisor = task.supervisor
        public_supervisor = {key: supervisor.get(key) if isinstance(supervisor, Mapping)
                             else getattr(supervisor, key, None)
                             for key in ('first_name', 'last_name', 'email', 'phone_number')}
        docs = []
        for name in self.BOOTSTRAP:
            app, api = name.split('.')
            docs.append({**task.api_docs[app][api], 'app_name': app, 'api_name': api})
        observation = json.dumps({'supervisor': public_supervisor, 'datetime': str(task.datetime),
            'apps': dict(task.app_descriptions), 'bootstrap_apis': list(self.BOOTSTRAP),
            'bootstrap_api_docs': docs}, ensure_ascii=False)
        from appworld.task import task_id_to_generator_id
        return {'benchmark': self.benchmark, 'task_id': task_id,
                'scenario_id': task_id_to_generator_id(task_id), 'instruction': str(task.instruction),
                'observation': observation, 'action_schema': deepcopy(self.action_schema)}

    def step_native(self, action):
        if self.world is None or self.done:
            raise RuntimeError('Native AppWorld step requires a live reset episode.')
        self.steps += 1
        if action.get('kind') == 'invalid' or not isinstance(action.get('code'), str):
            observation = f"Action validation failed: {action.get('error', 'AppWorld requires native Python source text.')}"
            valid = False
        else:
            observation = str(self.world.execute(action['code']))
            valid = not observation.startswith('Execution failed')
        completed = bool(self.world.task_completed())
        self.done = bool(completed or self.steps >= 100)
        success = bool(self.world.evaluate().success) if completed else False
        return StepResult(observation, self.done, success,
                          {'valid': valid, 'step_failed': not valid, 'native_executed': action.get('kind') != 'invalid',
                           'official_termination': completed, 'steps': self.steps})

    def finalize_evaluation(self):
        from appworld.task import task_id_to_generator_id
        tracker = self.world.evaluate()
        scenario = task_id_to_generator_id(self.task_id)
        # Per-trajectory workers need the same official membership as the catalog.
        if not self.splits:
            for name in ('train', 'dev', 'test_normal', 'test_challenge'):
                self.list_tasks(name)
        split = next((name for name, rows in self.splits.items()
                      if any(row['task_id'] == self.task_id for row in rows)), None)
        members = [row['task_id'] for row in self.splits.get(split, []) if row['scenario_id'] == scenario]
        return {'benchmark': self.benchmark, 'task_id': self.task_id, 'scenario_id': scenario,
                'scenario_task_ids': members, 'split': split, 'success': bool(tracker.success),
                'submitted': bool(self.world.task_completed()), 'native_evaluator': 'AppWorld.evaluate',
                'stats': tracker.to_dict(stats_only=True), 'steps': self.steps}

    def close(self):
        if self.world is not None:
            self.world.close()
            self.world = None


class EnvironmentProcess:
    """One JSON channel and private working directory per official environment slot."""
    def __init__(self, python, benchmark, data_root, run_dir, *, seed=0):
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.seed = int(seed)
        self.action_schema = {}
        self.closed = False
        self.request_id = 0
        self.log = (self.run_dir / 'environment.log').open('a')
        env = dict(os.environ)
        env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
        self.process = subprocess.Popen([str(python), '-u', '-m', 'fgpo.envs', '--worker'],
            cwd=self.run_dir, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, text=True, encoding='utf-8', bufsize=1)
        try:
            self._rpc('initialize', benchmark=benchmark, data_root=str(data_root), run_dir=str(self.run_dir))
        except BaseException:
            self._shutdown()
            raise

    def _rpc(self, method, *args, **kwargs):
        if self.closed:
            raise RuntimeError('Environment worker is closed.')
        self.request_id += 1
        packet = {'id': self.request_id, 'method': method, 'args': args, 'kwargs': kwargs}
        self.process.stdin.write(json.dumps(packet, ensure_ascii=False) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'Environment process exited: {self.process.poll()}')
        response = json.loads(line)
        if response['id'] != self.request_id:
            raise RuntimeError('Out-of-order environment response.')
        if not response['ok']:
            raise RuntimeError(f"{response['error_type']}: {response['error']}")
        self.action_schema = response['action_schema']
        return response['result']

    def list_tasks(self, split):
        return self._rpc('list_tasks', split)

    def reset(self, task_id, initial_seed=None, *, seed=None):
        selected = initial_seed if initial_seed is not None else seed if seed is not None else self.seed
        return self._rpc('reset', task_id, int(selected))

    def step_native(self, action):
        return StepResult(**self._rpc('step_native', action))

    def finalize_evaluation(self):
        return self._rpc('finalize_evaluation')

    def release_episode(self):
        """Dispose episode state while keeping imported benchmark modules alive."""
        return self._rpc('release_episode')

    def _shutdown(self):
        self.closed = True
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for stream in (self.process.stdin, self.process.stdout, self.log):
            stream.close()

    def close(self):
        if not self.closed:
            try:
                self._rpc('close')
            finally:
                self._shutdown()


def make_env(config, benchmark, run_dir, *, seed=0):
    python = config['alfworld_python_executable'] if benchmark == 'alfworld' else config['python_executable']
    return EnvironmentProcess(python, benchmark, config[benchmark + '_data_root'], run_dir, seed=seed)


class _EnvironmentLease:
    """Exclusive episode ownership of one persistent process and RPC channel."""
    def __init__(self, pool, slot, environment):
        self._pool, self._slot, self._environment = pool, slot, environment
        self._released = False

    @property
    def action_schema(self):
        return self._environment.action_schema

    def reset(self, task_id, initial_seed=None, *, seed=None):
        if self._released:
            raise RuntimeError('Cannot reset a released environment lease.')
        return self._environment.reset(task_id, initial_seed, seed=seed)

    def step_native(self, action):
        if self._released:
            raise RuntimeError('Cannot step a released environment lease.')
        return self._environment.step_native(action)

    def finalize_evaluation(self):
        if self._released:
            raise RuntimeError('Cannot evaluate a released environment lease.')
        return self._environment.finalize_evaluation()

    def close(self):
        if not self._released:
            try:
                self._environment.release_episode()
            finally:
                self._released = True
                self._pool._release(self._slot)


class EnvironmentPool:
    """Bounded, lazy process pool reused across groups of a single study cell.

Every lease has exclusive access until its episode has been closed. The pool
never shares an AppWorld database or a mutable environment between trajectories.
Failures propagate to the caller; workers are not silently restarted.
"""
    def __init__(self, config, benchmark, run_dir, *, size=8):
        if int(size) < 1:
            raise ValueError('An environment pool needs at least one process slot.')
        self.config, self.benchmark = dict(config), benchmark
        self.run_dir, self.size = Path(run_dir), int(size)
        self._lock = threading.Lock()
        self._workers, self._busy = [], set()
        self._closed = False
        self.acquisitions = 0

    def __call__(self, task_id, initial_seed, trajectory_id):
        with self._lock:
            if self._closed:
                raise RuntimeError('The environment pool is closed.')
            slot = next((i for i in range(len(self._workers)) if i not in self._busy), None)
            if slot is None:
                if len(self._workers) >= self.size:
                    raise RuntimeError('Concurrent environment leases exceed the configured slot count.')
                slot = len(self._workers)
                self._workers.append(None)
            self._busy.add(slot)
            self.acquisitions += 1
            worker = self._workers[slot]
        if worker is None:
            try:
                worker = make_env(self.config, self.benchmark, self.run_dir / f'slot-{slot}', seed=initial_seed)
            except BaseException:
                self._release(slot)
                raise
            with self._lock:
                self._workers[slot] = worker
        return _EnvironmentLease(self, slot, worker)

    def _release(self, slot):
        with self._lock:
            self._busy.discard(slot)

    def statistics(self):
        with self._lock:
            started = sum(worker is not None for worker in self._workers)
            return {'scope': 'current_process_session',
                    'processes_started': started, 'episode_leases': self.acquisitions,
                    'process_reuses': max(0, self.acquisitions - started)}

    def close(self):
        with self._lock:
            self._closed = True
            workers = list(self._workers)
        first_error = None
        for worker in workers:
            if worker is not None:
                try:
                    worker.close()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
        if first_error is not None:
            raise first_error


def _worker():
    channel = sys.stdout
    environment = None
    for line in sys.stdin:
        packet = json.loads(line)
        method = packet['method']
        try:
            with redirect_stdout(sys.stderr):
                if method == 'initialize':
                    options = packet['kwargs']
                    cls = {'alfworld': ALFWorldEnv, 'appworld': AppWorldEnv}[options['benchmark']]
                    environment = cls(options['data_root'], run_dir=options['run_dir'])
                    result = None
                elif method == 'release_episode':
                    environment.close()
                    result = None
                elif method in ('list_tasks', 'reset', 'step_native', 'finalize_evaluation', 'close'):
                    result = getattr(environment, method)(*packet['args'], **packet['kwargs'])
                    if isinstance(result, StepResult):
                        result = asdict(result)
                else:
                    raise ValueError('Unknown environment RPC method.')
            response = {'id': packet['id'], 'ok': True, 'result': result,
                        'action_schema': environment.action_schema}
        except Exception as error:
            response = {'id': packet['id'], 'ok': False, 'error_type': type(error).__name__,
                        'error': str(error), 'action_schema': environment.action_schema if environment else {}}
        channel.write(json.dumps(response, ensure_ascii=False) + '\n')
        channel.flush()
        if method == 'close':
            return
    if environment is not None:
        environment.close()


if __name__ == '__main__':
    if sys.argv[1:] != ['--worker']:
        raise SystemExit('Use the official environment factory or --worker.')
    _worker()
