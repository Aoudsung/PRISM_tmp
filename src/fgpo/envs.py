"""Official benchmark execution in independent environment processes.

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
import shutil
import subprocess
import sys
import threading
from typing import Any, Mapping
import uuid

from .public import HORIZONS
from .textcraft import TEXTCRAFT_1P5B_FORMAT_STUDY, TextCraftEnv
from .webshop import TRAIN_ROUNDS, VALIDATION_SESSIONS, task_schedule


ALFWORLD_TASK_TYPES = {
    'pick_and_place_simple': 'Pick',
    'look_at_obj_in_light': 'Look',
    'pick_clean_then_place_in_recep': 'Clean',
    'pick_heat_then_place_in_recep': 'Heat',
    'pick_cool_then_place_in_recep': 'Cool',
    'pick_two_obj_and_place': 'Pick2',
}
ALFWORLD_TABLE_COLUMNS = (*ALFWORLD_TASK_TYPES.values(), 'All')


def alfworld_task_type(task_id):
    """Derive the Table 1 category from the official instance directory."""
    parts = str(task_id).split('/')
    prefix = parts[1].split('-', 1)[0] if len(parts) == 4 else None
    if prefix not in ALFWORLD_TASK_TYPES:
        raise ValueError(f'Unknown official task-type prefix in task_id: {task_id!r}')
    return ALFWORLD_TASK_TYPES[prefix]


@dataclass
class StepResult:
    observation: str
    done: bool
    success: bool
    metadata: dict[str, Any]
    native_reward: float | None = None

    @property
    def reward(self):
        return float(self.success) if self.native_reward is None else self.native_reward

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
        self.action_schema = {
            'benchmark': self.benchmark,
            'native_system_prompt': (
                'You act in the official ALFWorld text environment. '
                'First reason concisely enclosed within literal <think> and </think> tags, then output '
                'exactly one admissible command verbatim without quotes enclosed within literal <action> and </action> tags. '
                'The episode ends when the goal is satisfied or the 50-step horizon is reached.'
            ),
        }

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
        return json.dumps({'native_observation': str(self.state['feedback']), 'step': self.steps,
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


class WebShopEnv:
    benchmark = 'webshop'

    def __init__(self, data_root, *, run_dir):
        self.root = Path(data_root).resolve()
        sys.path.insert(0, str(self.root))
        from web_agent_site.envs import WebAgentTextEnv
        from web_agent_site.engine.engine import parse_action

        self.native_env, self.parse_action = WebAgentTextEnv, parse_action
        self.world, self.world_seed = None, None
        self.action_schema = {
            'benchmark': self.benchmark,
            'native_system_prompt': (
                'You act in the official WebShop text environment. Follow the shopping instruction. '
                'Reason concisely inside <think>...</think>, then put exactly one command '
                'inside <action>...</action>. Use search[keywords] where search is shown and '
                'click[label] for a listed button or product. Buy the item matching the instruction and its options.'
            ),
        }

    def _open(self, seed):
        if self.world is None or seed != self.world_seed:
            self.close()
            self.world = self.native_env(
                observation_mode='text_rich', seed=seed, human_goals=False, num_products=None,
                file_path=str(self.root / 'data/items_shuffle_1000.json'),
                attr_path=str(self.root / 'data/items_ins_v2_1000.json'),
            )
            self.world_seed = seed

    def list_tasks(self, split, *, seed, train_rounds=TRAIN_ROUNDS):
        self._open(seed)
        return task_schedule(split, seed, len(self.world.server.goals), train_rounds=train_rounds)

    def _observation(self, text):
        self.clickables = []
        commands = []
        if not self.done:
            available = self.world.get_available_actions()
            self.clickables = [label for label in available['clickables'] if label != 'search']
            if available['has_search_bar']:
                commands.append('search[<your query>]')
            commands.extend(f'click[{label}]' for label in self.clickables)
        self.action_schema.update(available_actions=commands, step=self.steps)
        return text

    def reset(self, task_id, initial_seed):
        split, worker_seed, session = task_id.split('/')
        session = int(session)
        if int(worker_seed) != initial_seed or split not in ('train', 'test'):
            raise ValueError(f'WebShop task and native worker seed differ: {task_id!r}')
        self._open(initial_seed)
        first, stop = ((VALIDATION_SESSIONS, len(self.world.server.goals)) if split == 'train'
                       else (0, VALIDATION_SESSIONS))
        if not first <= session < stop:
            raise ValueError(f'WebShop task lies outside the released split: {task_id!r}')
        self.task = {'task_id': task_id, 'split': split, 'environment_seed': initial_seed,
                     'session_index': session, 'benchmark': self.benchmark}
        text, _ = self.world.reset(session=session)
        self.steps, self.done, self.purchased, self.score = 0, False, False, 0.0
        return {**self.task, 'instruction': self.world.instruction_text,
                'observation': self._observation(text), 'action_schema': deepcopy(self.action_schema)}

    def step_native(self, action):
        if self.world is None or self.done:
            raise RuntimeError('Native WebShop step requires a live reset episode.')
        if action.get('kind') not in ('native_text', 'invalid'):
            raise ValueError('WebShop requires the native_text action mapping.')
        command = action['text']
        name, argument = self.parse_action(command)
        valid = bool((name == 'search' and argument) or
                     (name == 'click' and argument in self.clickables))
        text, reward, native_done, _ = self.world.step(command)
        self.steps += 1
        self.purchased = bool(native_done)
        self.score = float(reward)
        self.done = self.purchased or self.steps >= HORIZONS[self.benchmark]
        return StepResult(self._observation(text), self.done, self.purchased and self.score == 1.0,
                          {'valid': valid, 'step_failed': not valid, 'native_executed': True,
                           'official_termination': self.purchased, 'steps': self.steps})

    def finalize_evaluation(self):
        return {**self.task, 'success': self.purchased and self.score == 1.0, 'score': self.score,
                'purchased': self.purchased, 'native_evaluator': 'WebAgentTextEnv.step', 'steps': self.steps}

    def close(self):
        if self.world is not None:
            world, self.world = self.world, None
            try:
                world.close()
            finally:
                world.server.search_engine.close()


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
        self._experiment_output_dir = Path(path_store.experiment_outputs).resolve() / self.experiment_name
        self._episode_output_dir = None
        self.world = None
        self.splits = {}
        self.steps = 0
        self.done = False
        self.action_schema = {'benchmark': self.benchmark, 'native_system_prompt':
            'Emit one Python code block. The persistent shell already provides apis. Call apis.<app>.<api>, print values you need to observe, discover documentation through api_docs, and submit with apis.supervisor.complete_task(). Variables persist across interactions. After completing all requested operations, submit without answer unless the task explicitly asks a question.'}

    def list_tasks(self, split, *, difficulty=None):
        from appworld.task import load_task_ids, task_id_to_generator_id
        if split not in ('train', 'dev', 'test_normal', 'test_challenge'):
            raise ValueError(f'Unknown official AppWorld split: {split}')
        if difficulty is not None and (split != 'train' or difficulty not in (1, 2)):
            raise ValueError('G2PO AppWorld training uses official difficulties 1 and 2.')
        tasks = load_task_ids(split) if difficulty is None else load_task_ids(split, difficulty=difficulty)
        rows = [{'task_id': task, 'scenario_id': task_id_to_generator_id(task), 'split': split}
                for task in tasks]
        if difficulty is None:
            self.splits[split] = rows
        return rows

    def reset(self, task_id, initial_seed):
        self.close()
        self.task_id = task_id
        self.steps, self.done = 0, False
        episode_id = uuid.uuid4().hex
        episode_output_dir = self._experiment_output_dir / episode_id
        # Claim exactly one new episode directory before AppWorld can write it.
        # The official output root also covers versions with different task subpaths.
        episode_output_dir.mkdir(parents=True, exist_ok=False)
        self._episode_output_dir = episode_output_dir
        try:
            self.world = self.AppWorld(task_id=task_id, experiment_name=self.experiment_name + '/' + episode_id,
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
        except BaseException:
            self.close()
            raise

    def step_native(self, action):
        if self.world is None or self.done:
            raise RuntimeError('Native AppWorld step requires a live reset episode.')
        self.steps += 1
        author_error = action.get('author_error')
        if author_error is not None:
            observation = author_error
            valid = False
            executed = False
        elif action.get('kind') == 'invalid' or not isinstance(action.get('code'), str):
            observation = f"Action validation failed: {action.get('error', 'AppWorld requires native Python source text.')}"
            valid = False
            executed = action.get('kind') != 'invalid'
        else:
            observation = str(self.world.execute(action['code']))
            valid = not observation.startswith('Execution failed')
            executed = True
        completed = bool(self.world.task_completed())
        self.done = bool(completed or self.steps >= 100)
        success = bool(self.world.evaluate().success) if completed else False
        return StepResult(observation, self.done, success,
                          {'valid': valid, 'step_failed': not valid, 'native_executed': executed,
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
        elif self._episode_output_dir is not None:
            # A failed constructor can leave process-local DB/API/time state
            # without returning an object. Each worker owns only this world.
            self.AppWorld.close_all()
        if self._episode_output_dir is not None:
            # Both collectors copy official metrics before releasing the lease.
            # Remove only this instance's claimed episode, after closing its DBs.
            shutil.rmtree(self._episode_output_dir)
            self._episode_output_dir = None


class EnvironmentProcess:
    """One JSON channel and private working directory per official environment slot."""
    def __init__(self, python, benchmark, data_root, run_dir, *, seed=0, textcraft_format_example=False):
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.benchmark = benchmark
        self.seed = int(seed)
        self.action_schema = {}
        self.closed = False
        self.request_id = 0
        self.successful_resets = 0
        self.log = (self.run_dir / 'environment.log').open('a')
        env = dict(os.environ)
        env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
        if benchmark in ('webshop', 'textcraft'):
            env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        if benchmark == 'textcraft':
            env.update(PYTHONHASHSEED='0', PYTHONNOUSERSITE='1', RAY_USAGE_STATS_ENABLED='0')
        self.process = subprocess.Popen([str(python), '-u', '-m', 'fgpo.envs', '--worker'],
            cwd=self.run_dir, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, text=True, encoding='utf-8', bufsize=1)
        try:
            self._rpc('initialize', benchmark=benchmark, data_root=str(data_root), run_dir=str(self.run_dir),
                      textcraft_format_example=textcraft_format_example)
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

    def list_tasks(self, split, *, difficulty=None, train_rounds=TRAIN_ROUNDS):
        if self.benchmark == 'webshop':
            return self._rpc('list_tasks', split, seed=self.seed, train_rounds=train_rounds)
        if difficulty is None:
            return self._rpc('list_tasks', split)
        return self._rpc('list_tasks', split, difficulty=difficulty)

    def reset(self, task_id, initial_seed=None, *, seed=None):
        selected = initial_seed if initial_seed is not None else seed if seed is not None else self.seed
        result = self._rpc('reset', task_id, int(selected))
        self.successful_resets += 1
        return result

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
    python_key = benchmark + '_python_executable' if benchmark in ('alfworld', 'webshop', 'textcraft') else 'python_executable'
    python = config[python_key]
    return EnvironmentProcess(python, benchmark, config[benchmark + '_data_root'], run_dir, seed=seed,
                              textcraft_format_example=(benchmark == 'textcraft'
                                  and config.get('study_protocol') == TEXTCRAFT_1P5B_FORMAT_STUDY))


class _EnvironmentLease:
    """Exclusive episode ownership of one persistent process and RPC channel."""
    def __init__(self, pool, slot, environment):
        self._pool, self._slot, self._environment = pool, slot, environment
        self._released = False
        self._scored = False

    @property
    def action_schema(self):
        return self._environment.action_schema

    def reset(self, task_id, initial_seed=None, *, seed=None):
        if self._released:
            raise RuntimeError('Cannot reset a released environment lease.')
        self._scored = False
        return self._environment.reset(task_id, initial_seed, seed=seed)

    def step_native(self, action):
        if self._released:
            raise RuntimeError('Cannot step a released environment lease.')
        self._scored = False
        return self._environment.step_native(action)

    def finalize_evaluation(self):
        if self._released:
            raise RuntimeError('Cannot evaluate a released environment lease.')
        self._scored = False
        result = self._environment.finalize_evaluation()
        self._scored = True
        return result

    def close(self):
        if not self._released:
            released = False
            try:
                self._environment.release_episode()
                released = True
            finally:
                self._released = True
                self._pool._release(self._slot, scored=released and self._scored)


class EnvironmentPool:
    """Bounded, lazy process pool reused across groups of a single study cell.

Every lease has exclusive access until its episode has been closed. The pool
never shares an AppWorld database or a mutable environment between trajectories.
ALF study workers retire after one round's worth of successful resets at a
scored lease boundary.
Failures propagate to the caller without triggering turnover.
"""
    def __init__(self, config, benchmark, run_dir, *, size=8):
        if int(size) < 1:
            raise ValueError('An environment pool needs at least one process slot.')
        self.config, self.benchmark = dict(config), benchmark
        groups = self.config.get('groups') if benchmark == 'alfworld' else None
        self._resets_per_worker = int(groups) if groups is not None else None
        if self._resets_per_worker is not None and self._resets_per_worker < 1:
            raise ValueError('A training round requires at least one task group.')
        self.run_dir, self.size = Path(run_dir), int(size)
        self._lock = threading.Lock()
        self._workers, self._busy = [], set()
        self._closed = False
        self._processes_started = 0
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
            worker = self._workers[slot]
            if worker is not None:
                self.acquisitions += 1
        if worker is None:
            try:
                worker = make_env(self.config, self.benchmark, self.run_dir / f'slot-{slot}', seed=initial_seed)
            except BaseException:
                self._release(slot)
                raise
            with self._lock:
                self._workers[slot] = worker
                self._processes_started += 1
                self.acquisitions += 1
        return _EnvironmentLease(self, slot, worker)

    def _release(self, slot, *, scored=False):
        with self._lock:
            try:
                worker = self._workers[slot]
                if (not self._closed and scored and self._resets_per_worker is not None and
                        worker.successful_resets >= self._resets_per_worker):
                    # TextWorld pins each reset's unlinked planner library
                    # until process exit. Keep the slot busy until shutdown.
                    worker.close()
                    self._workers[slot] = None
            finally:
                self._busy.discard(slot)

    def statistics(self):
        with self._lock:
            return {'scope': 'current_process_session',
                    'processes_started': self._processes_started, 'episode_leases': self.acquisitions,
                    'process_reuses': self.acquisitions - self._processes_started}

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
                    cls = {'alfworld': ALFWorldEnv, 'appworld': AppWorldEnv,
                           'webshop': WebShopEnv, 'textcraft': TextCraftEnv}[options['benchmark']]
                    kwargs = {'format_example': True} if options.get('textcraft_format_example') else {}
                    environment = cls(options['data_root'], run_dir=options['run_dir'], **kwargs)
                    result = None
                elif method == 'release_episode':
                    environment.close()
                    result = None
                elif method in ('list_tasks', 'reset', 'step_native', 'finalize_evaluation', 'close'):
                    result = getattr(environment, method)(*packet['args'], **packet['kwargs'])
                    if method == 'step_native':
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
