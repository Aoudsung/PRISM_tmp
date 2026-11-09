"""Native benchmark bridges and public execution receipts for ``native-r3``.

Only explicitly public task fields and native execution feedback cross the
actor boundary. Benchmark answers and evaluation traces stay inside the bridge.
Heavy benchmark dependencies are imported when a bridge is constructed/reset.
"""

from __future__ import annotations

import ast
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from datetime import date, datetime
import importlib
import inspect
import json
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Mapping, Sequence
import uuid

from .native_core import BudgetMeter, NativeStep
from .native_semantics import public_document_quality


def _clock() -> float:
    # AppWorld freezes datetime and perf_counter inside the process.
    return time.clock_gettime(time.CLOCK_MONOTONIC)


def _public_json(value: Any) -> Any:
    """Copy native JSON values; serialize the benchmark's explicit date values."""
    if value is None or isinstance(value, (str, bool, int, float)):
        json.dumps(value, allow_nan=False)
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Native public JSON object keys must be strings.")
        return {key: _public_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_public_json(child) for child in value]
    raise TypeError(f"Native result is not a public JSON value: {type(value).__name__}")


def _receipt_observation(observation: str, receipts: Sequence[Mapping[str, Any]]) -> str:
    return json.dumps(
        {
            "feedback_type": "native_transition",
            "native_observation": observation,
            "r3_receipts": list(receipts),
        },
        ensure_ascii=False,
        allow_nan=False,
    )


def grouped_query_splits(
    queries: Sequence[Any], *, seed: int = 42, targets: tuple[int, int, int] = (180, 57, 90),
) -> dict[str, list[dict[str, str]]]:
    """Assign entire native task/template groups using only identifiers.

    Targets are desired query counts. Whole groups take priority over matching
    those counts; the actual allocation is published in ``split_manifest``.
    """
    if len(targets) != 3 or min(targets) < 0 or sum(targets) != len(queries):
        raise ValueError("Split targets must be nonnegative and sum to the query count.")
    groups: dict[str, list[str]] = {}
    seen: set[str] = set()
    for query in queries:
        qid = str(query["query_id"] if isinstance(query, Mapping) else query.query_id)
        sid = str(query["task_id"] if isinstance(query, Mapping) else query.task_id)
        if not qid or not sid or qid in seen:
            raise ValueError("Queries require unique query IDs and nonempty native task IDs.")
        seen.add(qid)
        groups.setdefault(sid, []).append(qid)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    names = ("train", "dev", "test")
    counts = [0, 0, 0]
    result: dict[str, list[dict[str, str]]] = {name: [] for name in names}
    for sid in keys:
        size = len(groups[sid])
        eligible = [i for i, target in enumerate(targets) if target > 0]
        chosen = min(
            eligible,
            key=lambda i: (
                sum((counts[j] + (size if i == j else 0) - targets[j]) ** 2 for j in range(3)),
                i,
            ),
        )
        name = names[chosen]
        result[name].extend(
            {"task_id": qid, "scenario_id": sid, "split": name}
            for qid in sorted(groups[sid])
        )
        counts[chosen] += size
    for rows in result.values():
        rows.sort(key=lambda row: row["task_id"])
    return result


class _AppWorldActionError(ValueError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


def _appworld_native_error(observation: str) -> str | None:
    """Classify the installed benchmark's actual feedback without rewriting it."""
    if not observation.startswith("Execution failed"):
        return None
    if "Syntax error in line:" in observation:
        return "syntax"
    if "No API named '" in observation or "No app named '" in observation:
        return "native_api_lookup"
    if (
        "Usage of the following module is not allowed:" in observation
        or "Usage of the following modules is not allowed:" in observation
    ):
        return "module_restriction"
    return "native_execution"


_APPWORLD_BOOTSTRAP_APIS = (
    "api_docs.show_app_descriptions",
    "api_docs.show_api_descriptions",
    "api_docs.show_api_doc",
    "supervisor.complete_task",
)


def _json_observations(text: str):
    """Read JSON objects actually printed in an observation, including nested docs."""
    decoder = json.JSONDecoder()
    cursor = 0
    while cursor < len(text):
        match = re.search(r"[\[{]", text[cursor:])
        if match is None:
            return
        start = cursor + match.start()
        try:
            value, end = decoder.raw_decode(text, start)
        except ValueError:
            cursor = start + 1
            continue
        yield from _json_objects(value)
        cursor = end


def _json_objects(value: Any):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _json_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _json_objects(child)


def _visible_read_information(observation: str, calls: Sequence[Mapping[str, Any]]) -> list[Any]:
    """Match printed values against successful native read returns only.

    The actor may print JSON, Python representations, or a scalar on its own
    line. Unprinted return fields and arbitrary local print output earn no
    information credit. Matching maximal subtrees avoids repeating all leaves
    when the entire return was printed.
    """
    def encode(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    visible: set[str] = set()

    def add(value: Any) -> None:
        if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
            try:
                visible.add(encode(value))
            except (TypeError, ValueError):
                return
            if isinstance(value, dict):
                for child in value.values():
                    add(child)
            elif isinstance(value, list):
                for child in value:
                    add(child)

    for value in _json_observations(observation):
        add(value)
    for text in [observation.strip(), *(line.strip() for line in observation.splitlines())]:
        if not text:
            continue
        add(text)
        try:
            add(ast.literal_eval(text))
        except (ValueError, SyntaxError, TypeError):
            pass

    evidence: dict[str, Any] = {}

    def match(value: Any) -> None:
        if value is None or value == "" or value == [] or value == {}:
            return
        key = encode(value)
        if key in visible:
            evidence[key] = value
        elif isinstance(value, dict):
            for child in value.values():
                match(child)
        elif isinstance(value, list):
            for child in value:
                match(child)

    for call in calls:
        if call.get("succeeded") and str(call.get("method", "")).upper() in {"GET", "HEAD"}:
            match(call.get("output"))
    return [evidence[key] for key in sorted(evidence)]


def _public_api_doc(value: Mapping[str, Any]) -> dict[str, Any] | None:
    if (
        not all(isinstance(value.get(key), str) for key in ("app_name", "api_name", "method", "path"))
        or not isinstance(value.get("parameters"), list)
    ):
        return None
    return deepcopy(dict(value))


def _document_identity(doc: Mapping[str, Any]) -> str:
    signature = {key: doc[key] for key in ("app_name", "api_name", "method", "path", "parameters")}
    return json.dumps(signature, sort_keys=True, ensure_ascii=False)


def _visible_observations(messages: Sequence[Mapping[str, Any]]):
    for message in messages:
        if message.get("role") != "user":
            continue
        text = str(message.get("content", ""))
        yield text
        for value in _json_observations(text):
            if "instruction" in value and "observation" in value:
                observation = value["observation"]
                yield observation if isinstance(observation, str) else json.dumps(observation, ensure_ascii=False)


def _public_value_evidence(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Retain quoted query values and native outputs without guessing their types."""
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for text in _visible_observations(messages):
        for value in _json_observations(text):
            item = None
            if isinstance(value.get("instruction"), str) and "observation" in value:
                instruction = value["instruction"]
                item = {
                    "source": "initial_query",
                    "text": instruction,
                    "quoted_values": re.findall(r"`([^`]+)`", instruction),
                }
            elif value.get("feedback_type") == "tool_call_result":
                item = {"source": "native_tool_feedback", "feedback": deepcopy(dict(value))}
            elif value.get("feedback_type") == "native_transition":
                item = {"source": "native_receipts", "receipts": deepcopy(value.get("r3_receipts", []))}
            if item is not None:
                identity = json.dumps(item, sort_keys=True, ensure_ascii=False)
                if identity not in seen:
                    seen.add(identity)
                    evidence.append(item)
    if not any(item["source"] == "initial_query" for item in evidence):
        initial = next((str(message.get("content", "")) for message in messages if message.get("role") == "user"), "")
        if initial:
            evidence.insert(
                0,
                {
                    "source": "initial_query",
                    "text": initial,
                    "quoted_values": re.findall(r"`([^`]+)`", initial),
                },
            )
    return evidence


def public_action_schema(schema: Mapping[str, Any], messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reconstruct the public ledger from precisely the observations being scored."""
    from .native_execution_state import ExecutionState
    from .native_program import build_frontier
    result = deepcopy(dict(schema))
    context = result["program_context"]
    visible = list(_visible_observations(messages))
    visible_text = set(visible)
    provenance = [source for source in context["sources"] if source["observation_text"] in visible_text]
    state = ExecutionState.load(context["initial_execution_state"])
    if context["benchmark"] == "appworld":
        sources = context["apis"]
        apis = {name: sources[name] for name in _APPWORLD_BOOTSTRAP_APIS}
        for source in provenance:
            for value in source["api_docs"]:
                doc = _public_api_doc(value)
                if doc is not None:
                    apis[f"{doc['app_name']}.{doc['api_name']}"] = doc
        context["apis"] = apis
    else:
        id_key = "tool_id" if context["use_tool_ids"] else "name"
        tools = {}
        for source in provenance:
            for tool in source["api_docs"]:
                tools[tool[id_key]] = deepcopy(tool)
        context["tools"] = list(tools.values())
    for source in provenance:
        for receipt in source.get("receipts", []):
            state.apply_receipt(receipt)
    context["execution_state"] = state.public_context()
    context["public_value_evidence"] = _public_value_evidence(messages)
    context["variables"] = []
    context["sources"] = provenance
    context["frontier"] = build_frontier(context)
    return result


class NativeAppWorldEnv:
    """Native AppWorld code execution with deterministic prefix-replay snapshots."""

    benchmark = "appworld"

    def __init__(
        self,
        data_root: str | Path,
        *,
        experiment_name: str,
        max_steps: int = 100,
        max_api_calls_per_interaction: int = 1000,
        timeout_seconds: int = 10,
        meter: BudgetMeter | None = None,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.experiment_name = experiment_name
        self.max_steps = int(max_steps)
        self.max_api_calls_per_interaction = int(max_api_calls_per_interaction)
        self.timeout_seconds = int(timeout_seconds)
        if min(self.max_steps, self.max_api_calls_per_interaction, self.timeout_seconds) <= 0:
            raise ValueError("Native AppWorld limits must be positive.")
        self.meter = meter
        self.world: Any = None
        self._actions: list[dict[str, Any]] = []
        self._observations: list[str] = []
        self._attempts = 0
        self._done = False
        self._task_id = ""
        self._seed = 0
        self._api_methods: dict[str, str] = {}
        self._public_apis: dict[str, dict[str, Any]] = {}
        self._public_context_sources: list[dict[str, Any]] = []
        self.execution_state: Any = None
        self._initial_execution_state: dict[str, Any] = {}
        self._replaying = False
        self._rng: Any = None
        self._np_rng: Any = None
        self.replayed_env_steps = 0
        self.snapshot_restore_seconds = 0.0
        self.last_native_result: dict[str, Any] | None = None
        self.action_schema: dict[str, Any] = {
            "benchmark": self.benchmark,
            "native_interface": "Emit one Python code block. The persistent shell already provides apis. Call apis.<app>.<api>, print values you need to observe, discover documentation through api_docs, and submit with apis.supervisor.complete_task().",
            "kind": "native-r3",
            "actor_contract": "One constrained JSON discover, call, map_call or finish action.",
            "runtime": "Selected actions compile to native apis.<app>.<api> calls. Task files belong to the file_system app.",
            "interaction": (
                "Discover exact public API documentation before selecting a call. Use references to actual "
                "task values or successful native returns for parameters. Authentication tokens come from "
                "login results and must be passed to every protected endpoint. Read returned JSON fields "
                "using their actual shape. Use observed collections and public response fields for loops, "
                "pagination and verification. Complete all requested task operations before finishing."
            ),
            "native_format": "The deterministic compiler emits native Python; the actor emits only the shared JSON action.",
            "I": "Discover public API names and full documentation through api_docs.",
            "E": "Call discovered business APIs, including authentication, reads, writes, or final submission.",
            "submission": (
                "After completing all required task operations, submit through "
                "apis.supervisor.complete_task(). For operation tasks, omit answer. "
                "Pass answer only if the task explicitly asks a question; its default is None. "
                "The native evaluator judges completion."
            ),
        }
        self.split_manifest: dict[str, Any] = {"benchmark": self.benchmark, "scheme": "official", "splits": {}}
        self.action_schema["native_system_prompt"] = self.action_schema["native_interface"]

    def _publish_program_context(self) -> None:
        from .native_program import CONTEXT_VERSION, build_frontier
        context = {
            "version": CONTEXT_VERSION,
            "benchmark": self.benchmark,
            "apis": deepcopy(self._public_apis),
            "variables": [],
            "sources": deepcopy(self._public_context_sources),
            "initial_execution_state": deepcopy(self._initial_execution_state),
            "execution_state": self.execution_state.public_context(),
        }
        context["frontier"] = build_frontier(context)
        self.action_schema = {**self.action_schema, "program_context": context}

    def _observe_public_catalog(
        self,
        observation: str,
        action: Mapping[str, Any],
        receipts: list[dict[str, Any]],
    ) -> None:
        docs = {}
        for receipt in receipts:
            if not receipt["succeeded"] or receipt["kind"] != "discover":
                continue
            for value in _json_objects(receipt["output"]):
                doc = _public_api_doc(value)
                if doc is not None:
                    self._public_apis[f"{doc['app_name']}.{doc['api_name']}"] = doc
                    docs[_document_identity(doc)] = doc
        self._public_context_sources.append({
            "observation_id": self._attempts,
            "observation_text": observation,
            "program_id": action.get("program_id"),
            "api_docs": list(docs.values()),
            "bindings": [],
            "receipts": deepcopy(receipts),
        })
        self._publish_program_context()

    def schema_for_history(self, messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return public_action_schema(self.action_schema, messages)

    def _load_runtime(self) -> tuple[Any, Any]:
        from freezegun import configure
        configure(extend_ignore_list=[
            "torch", "transformers", "peft", "accelerate", "safetensors",
            "tokenizers", "triton", "huggingface_hub", "xgrammar", "tvm_ffi",
        ])
        from appworld import AppWorld
        from appworld.common.path_store import path_store
        path_store.update_root(str(self.data_root))
        return AppWorld, path_store

    def list_tasks(self, split: str) -> list[dict[str, str]]:
        if split not in {"train", "dev", "test_normal", "test_challenge"}:
            raise ValueError(f"Unknown official AppWorld split: {split}")
        self._load_runtime()
        from appworld.task import load_task_ids, task_id_to_generator_id
        ids = list(load_task_ids(split))
        rows = [
            {"task_id": task_id, "scenario_id": task_id_to_generator_id(task_id), "split": split}
            for task_id in ids
        ]
        source = self.data_root / "data" / "datasets" / f"{split}.txt"
        self.split_manifest["splits"][split] = {"count": len(rows), "tasks": rows, "source": str(source)}
        return rows

    def training_reference(self, task_id: str | None = None) -> dict[str, Any]:
        """Private train-only teacher material; never included in environment feedback."""
        selected = self._task_id if task_id is None else str(task_id)
        if selected not in {task["task_id"] for task in self.list_tasks("train")}:
            raise ValueError("Ground-truth teacher references are restricted to registered training tasks.")
        from appworld.ground_truth import GroundTruth
        truth = GroundTruth.load(task_id=selected, mode="full")
        reference = {
            "solution_code": truth.solution_code,
            "compiled_solution_code": truth.compiled_solution_code,
        }
        if not reference["solution_code"].strip():
            raise ValueError("The official training task has no reference solution.")
        return {
            "benchmark": self.benchmark,
            "task_id": selected,
            "source": "AppWorld.GroundTruth.load(full)",
            "reference_id": f"appworld/train/{selected}",
            "reference": reference,
        }

    def training_state_fingerprint(self) -> dict[str, Any]:
        """Private train-only current database witness for fixed progress goals."""
        if self.world is None or self._task_id not in {
            task["task_id"] for task in self.list_tasks("train")
        }:
            raise ValueError("Training state witnesses require a registered train episode.")
        path = Path(self.world.output_db_home_path_on_disk) / "model_hashes.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def execute_training_reference(self, code: str) -> dict[str, Any]:
        if self.world is None or self._done:
            raise RuntimeError("Reference execution requires a fresh live train episode.")
        if self._task_id not in {task["task_id"] for task in self.list_tasks("train")}:
            raise ValueError("Reference execution is restricted to registered training tasks.")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("The official reference program is empty.")
        if self.meter is not None:
            self.meter.ensure_env_steps(1)
        started = _clock()
        self._attempts += 1
        try:
            with self._isolated_rng(), self._guard_readonly_requests(None) as requests:
                observation = str(self.world.execute(code))
            tracker = self.world.evaluate()
            self.last_native_result = {
                "task_id": self._task_id,
                "success": bool(tracker.success),
                "native_evaluator": "AppWorld.evaluate",
                "stats": tracker.to_dict(stats_only=True),
            }
            self._done = True
            return {
                "success": bool(tracker.success),
                "receipts": deepcopy(requests["calls"]),
                "observation": observation,
                "elapsed_seconds": _clock() - started,
                "execution_error": _appworld_native_error(observation),
                "native_result": deepcopy(self.last_native_result),
            }
        finally:
            if self.meter is not None:
                self.meter.record_env_step(_clock() - started)

    @contextmanager
    def _isolated_rng(self):
        import numpy as np
        prior_py, prior_np = random.getstate(), np.random.get_state()
        if self._rng is not None:
            random.setstate(self._rng)
            np.random.set_state(self._np_rng)
        try:
            yield
        finally:
            self._rng, self._np_rng = random.getstate(), np.random.get_state()
            random.setstate(prior_py)
            np.random.set_state(prior_np)

    def reset(self, task_id: str, seed: int) -> dict[str, Any]:
        import numpy as np
        AppWorld, _ = self._load_runtime()
        self.close()
        self._task_id, self._seed = str(task_id), int(seed)
        self._rng = random.Random(self._seed).getstate()
        self._np_rng = np.random.RandomState(self._seed % (2**32)).get_state()
        self._actions, self._observations = [], []
        self._attempts, self._done = 0, False
        self.last_native_result = None
        episode_name = f"{self.experiment_name}/episode-{uuid.uuid4().hex}"
        with self._isolated_rng():
            self.world = AppWorld(
                task_id=self._task_id,
                experiment_name=episode_name,
                random_seed=self._seed,
                max_interactions=self.max_steps,
                max_api_calls_per_interaction=self.max_api_calls_per_interaction,
                timeout_seconds=self.timeout_seconds,
                load_ground_truth=True,
                ground_truth_mode="minimal",
                raise_on_failure=True,
                raise_on_extra_parameters=True,
                show_api_response_schemas=True,
            )
        task = self.world.task
        self._api_methods = {
            f"{app}.{api}": str(doc["method"])
            for app, apis in task.api_docs.items()
            for api, doc in apis.items()
        }
        self._public_apis = {}
        for name in _APPWORLD_BOOTSTRAP_APIS:
            app, api = name.split(".")
            doc = _public_api_doc({**task.api_docs[app][api], "app_name": app, "api_name": api})
            if doc is None:
                raise ValueError(f"The official bootstrap API has no public signature: {name}")
            self._public_apis[name] = doc
        self._public_context_sources = []
        supervisor = task.supervisor
        public_supervisor = {
            key: supervisor.get(key) if isinstance(supervisor, Mapping) else getattr(supervisor, key, None)
            for key in ("first_name", "last_name", "email", "phone_number")
        }
        observation = json.dumps(
            {
                "supervisor": public_supervisor,
                "datetime": str(task.datetime),
                "apps": dict(task.app_descriptions),
                "bootstrap_apis": list(_APPWORLD_BOOTSTRAP_APIS),
                "bootstrap_api_docs": list(self._public_apis.values()),
            },
            ensure_ascii=False,
        )
        from .native_execution_state import ExecutionState
        self.execution_state = ExecutionState.from_task(
            self.benchmark, self._task_id, str(task.instruction), observation
        )
        self._initial_execution_state = self.execution_state.dump()
        self._publish_program_context()
        from appworld.task import task_id_to_generator_id
        return {
            "task_id": self._task_id,
            "scenario_id": task_id_to_generator_id(self._task_id),
            "instruction": str(task.instruction),
            "observation": observation,
            "action_schema": deepcopy(self.action_schema),
            "system_prompt": (
                "Complete the task in the native AppWorld environment using the shared JSON action contract. "
                "The public ledger contains actual task values and native API results. Select documented "
                "APIs, bind required inputs by reference, and verify the requested task operations before finish."
            ),
        }

    def _code(self, action: Mapping[str, Any]) -> str:
        code = action.get("code")
        if (
            action.get("kind") != "code"
            or not isinstance(code, str)
            or not isinstance(action.get("r3_program"), Mapping)
        ):
            raise ValueError("AppWorld requires a compiled native-r3 action.")
        return code

    @contextmanager
    def _guard_readonly_requests(
        self,
        mode: str | None,
        action: Mapping[str, Any] | None = None,
        *, native: bool = False, probe: bool = False,
    ):
        """Capture real results in memory; meter them after native execution."""
        requester = self.world.requester
        original = requester.request
        original_response_to_json = requester.response_to_json
        signature = inspect.signature(original)
        data_parameter = next(
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind == inspect.Parameter.VAR_KEYWORD
        )
        plan = list(action["receipt_plan"]) if action is not None and not native else None
        interceptions: list[str] = []
        calls: list[dict[str, Any]] = []
        active: list[dict[str, Any]] = []
        replay = self._replaying

        def response_to_json(response: Any, *args: Any, **kwargs: Any):
            if active:
                active[-1]["status_code"] = int(response.status_code)
            return original_response_to_json(response, *args, **kwargs)

        def guarded(_app_name: str, _api_name: str, *args: Any, **kwargs: Any):
            name = f"{_app_name}.{_api_name}"
            method = self._api_methods.get(name, "").upper()
            kind = (
                "discover"
                if _app_name == "api_docs"
                else ("finish" if name == "supervisor.complete_task" else "call")
            )
            actual_mode = "I" if kind == "discover" else "E"
            if probe and (method not in {"GET", "HEAD"} or kind == "finish"):
                interceptions.append(name)
                raise ValueError(f"EnvProbe requires a read-only API; {name} uses {method}.")
            if mode is not None and actual_mode != mode:
                interceptions.append(name)
                raise ValueError(f"Native-r3 {kind} belongs to {actual_mode}; selected mode is {mode}.")
            bound = signature.bind(_app_name, _api_name, *args, **kwargs)
            bound.apply_defaults()
            arguments = _public_json(bound.arguments[data_parameter])
            planned: dict[str, Any] = {}
            if plan is not None:
                if len(calls) >= len(plan):
                    raise ValueError("The compiled action requested an unplanned native call.")
                planned = plan[len(calls)]
                explicit = planned["arguments"]
                parameters = {
                    parameter["name"]: parameter for parameter in self._public_apis[name]["parameters"]
                }
                explicit_matches = all(
                    key in arguments and arguments[key] == value for key, value in explicit.items()
                )
                extra_matches = all(
                    key in parameters
                    and not parameters[key].get("required", False)
                    and "default" in parameters[key]
                    and value == parameters[key]["default"]
                    for key, value in arguments.items()
                    if key not in explicit
                )
                if planned["tool"] != name or not explicit_matches or not extra_matches:
                    raise ValueError("The compiled call differs from its exact public binding plan.")
            call_id = f"step_{self._attempts:05d}_call_{len(calls):04d}"
            event = {
                "call_id": call_id,
                "name": name,
                "tool": name,
                "method": method,
                "arguments": arguments,
                "output": None,
                "succeeded": False,
                "status": None,
                "observation_id": self._attempts,
                "bind": planned.get("bind", call_id),
                "kind": kind,
                "rejection_kind": "",
            }
            for key in ("batch_bind", "batch_index", "batch_size"):
                if key in planned:
                    event[key] = planned[key]
            if name in self._public_apis:
                event["public_doc"] = deepcopy(self._public_apis[name])
            event["value_status"] = public_document_quality(event.get("public_doc", {}))
            event["usable_for_binding"] = False
            calls.append(event)
            active.append(event)
            try:
                result = original(_app_name, _api_name, *args, **kwargs)
                event["status"] = event.get("status_code")
                event["output"] = _public_json(result)
                event["succeeded"] = event.get("status_code") == 200
                event["usable_for_binding"] = bool(event["succeeded"] and kind == "call")
                if not event["succeeded"] and kind == "call":
                    event["rejection_kind"] = "terminal_rejection"
                if event["succeeded"] and kind == "discover" and not native:
                    new_count = 0
                    for value in _json_objects(result):
                        doc = _public_api_doc(value)
                        if doc is not None:
                            full_name = f"{doc['app_name']}.{doc['api_name']}"
                            if full_name not in self._public_apis:
                                new_count += 1
                            self._public_apis[full_name] = doc
                    event["new_tool_count"] = new_count
                return result
            except Exception as error:
                event["error"] = f"{type(error).__name__}: {error}"
                if kind == "call":
                    event["rejection_kind"] = "terminal_rejection"
                raise
            finally:
                active.pop()
                if not native:
                    self.execution_state.apply_receipt(event)

        requester.request = guarded
        requester.response_to_json = response_to_json
        try:
            yield {"interceptions": interceptions, "calls": calls}
        finally:
            requester.request = original
            requester.response_to_json = original_response_to_json
            if self.meter is not None and calls:
                self.meter.record_api_calls(
                    len(calls),
                    successful=sum(int(call["succeeded"]) for call in calls),
                    replay=replay,
                )

    def _step_unmetered(self, action: Mapping[str, Any]) -> NativeStep:
        if self.world is None:
            raise RuntimeError("Call reset before step.")
        if self._done:
            raise RuntimeError("Cannot step a terminal native episode.")
        self._attempts += 1
        native = action.get("_execution_contract") == "official-native-v1"
        validation_stage = action.get(
            "validation_stage",
            "action_parse" if action.get("kind") == "invalid" else "action_contract",
        )
        policy_error_kind = None
        actual_mode = None
        try:
            if action.get("mode") is not None and not native:
                requested_mode = str(action["mode"])
                if requested_mode not in {"I", "E"}:
                    validation_stage = "policy_contract"
                    raise _AppWorldActionError("invalid_mode", "The explicit policy mode must be I or E.")
                actual_mode = requested_mode
            if action.get("kind") == "invalid":
                raise ValueError(action.get("error", "The model did not emit one native action."))
            code = action.get("code") if native else self._code(action)
            if not isinstance(code, str):
                raise ValueError("AppWorld requires native Python source text.")
            if not native and not isinstance(action.get("receipt_plan"), list):
                raise ValueError("A compiled native-r3 action requires its exact receipt plan.")
        except (TypeError, ValueError) as exc:
            if isinstance(exc, _AppWorldActionError):
                validation_stage = "policy_contract"
                policy_error_kind = exc.stage
            observation = f"Action validation failed: {exc}"
            self._actions.append(deepcopy(dict(action)))
            self._observations.append(observation)
            self._done = self._attempts >= self.max_steps
            return NativeStep(
                observation,
                self._done,
                False,
                {
                    "mode": actual_mode or action.get("mode"),
                    "step_failed": True,
                    "native_api_calls": 0,
                    "api_calls": 0,
                    "attempted_api_calls": 0,
                    "successful_api_calls": 0,
                    "successful_business_calls": 0,
                    "successful_documentation_calls": 0,
                    "successful_submission_calls": 0,
                    "native_execution_succeeded": False,
                    "validation_stage": validation_stage,
                    "validation_error": str(exc),
                    "policy_error_kind": policy_error_kind,
                    "requested_mode": action.get("mode"),
                    "mode_guard_blocked": False,
                    "native_executed": False,
                    "submitted": False,
                    "native_success": None,
                    "information_action_executed": False,
                    "receipts": [],
                    "termination": "environment_step_limit" if self._done else "running",
                    "failure_reason": "environment_step_limit" if self._done else None,
                    "native_transition": {
                        "state_changed": False,
                        "database_changed": False,
                        "new_information_candidate": False,
                        "hard_bad": True,
                        "final_success": False,
                    },
                },
            )
        hash_path = Path(self.world.output_db_home_path_on_disk) / "model_hashes.json"
        before_state = json.loads(hash_path.read_text(encoding="utf-8"))
        before = len(self.world.requester.requests)
        with self._isolated_rng(), self._guard_readonly_requests(
            actual_mode, action, native=native, probe=bool(action.get("_probe", False)),
        ) as requests:
            native_observation = str(self.world.execute(code))
        native_error = _appworld_native_error(native_observation)
        interceptions = requests["interceptions"]
        native_calls = requests["calls"]
        successful_calls = [call for call in native_calls if call["succeeded"]]
        successful_docs = sum(call["name"].startswith("api_docs.") for call in successful_calls)
        successful_submissions = sum(call["name"] == "supervisor.complete_task" for call in successful_calls)
        mode_guard_blocked = bool(interceptions)
        empty_code = not code.strip()
        if mode_guard_blocked and native_error == "native_execution":
            native_error = "mode_guard_blocked"
        observation = native_observation if native else _receipt_observation(native_observation, native_calls)
        visible_information = _visible_read_information(native_observation, native_calls) if native else []
        if native:
            # A Python assignment does not expose its return to the actor. Only
            # documents actually printed in the official observation are public.
            docs = []
            for value in visible_information:
                for item in _json_objects(value):
                    doc = _public_api_doc(item)
                    if doc is not None:
                        self._public_apis[f"{doc['app_name']}.{doc['api_name']}"] = doc
                        docs.append(doc)
            self._public_context_sources.append({"observation_id": self._attempts,
                "observation_text": observation, "program_id": None,
                "api_docs": docs, "bindings": [], "receipts": []})
            self._publish_program_context()
        else:
            self._observe_public_catalog(observation, action, native_calls)
        api_calls = len(self.world.requester.requests) - before
        read_api_calls = sum(
            request["method"].upper() in {"GET", "HEAD"}
            for request in self.world.requester.requests[before:]
        )
        database_changed = before_state != json.loads(hash_path.read_text(encoding="utf-8"))
        self._actions.append(deepcopy(dict(action)))
        self._observations.append(observation)
        completed = bool(self.world.task_completed())
        success = False
        if completed:
            tracker = self.world.evaluate()
            success = bool(tracker.success)
            self.last_native_result = {
                "task_id": self._task_id,
                "success": success,
                "native_evaluator": "AppWorld.evaluate",
                "stats": tracker.to_dict(stats_only=True),
            }
        self._done = completed or self._attempts >= self.max_steps
        step_failed = native_error is not None or mode_guard_blocked or empty_code
        return NativeStep(
            observation,
            self._done,
            success,
            {
                "mode": actual_mode,
                "step_failed": step_failed,
                "native_api_calls": api_calls,
                "submitted": completed,
                "api_calls": len(native_calls),
                "attempted_api_calls": len(native_calls),
                "successful_api_calls": len(successful_calls),
                "successful_documentation_calls": successful_docs,
                "successful_submission_calls": successful_submissions,
                "successful_business_calls": len(successful_calls) - successful_docs - successful_submissions,
                "native_read_api_calls": read_api_calls,
                "information_action_executed": (
                    (native or actual_mode == "I")
                    and (bool(visible_information) if native else successful_docs > 0)
                    and not step_failed
                    and not database_changed
                    and not completed
                ),
                "receipts": [] if native else deepcopy(native_calls),
                "native_executed": True,
                "native_execution_succeeded": not step_failed,
                "validation_stage": (
                    "mode_guard" if mode_guard_blocked else ("empty_code" if empty_code else "native")
                ),
                "native_error_kind": native_error or ("empty_code" if empty_code else None),
                "mode_guard_blocked": mode_guard_blocked,
                "mode_guard_interceptions": list(interceptions),
                "empty_code": empty_code,
                "native_success": success if completed else None,
                "failure_reason": (
                    "native_task_failed"
                    if completed and not success
                    else "environment_step_limit"
                    if self._done and not completed
                    else None
                ),
                "native_transition": {
                    "state_changed": database_changed,
                    "database_changed": database_changed,
                    "new_information_candidate": bool(visible_information) if native else any(call["output"] is not None for call in successful_calls),
                    "public_information_evidence": visible_information if native else None,
                    "public_information_signature": json.dumps(visible_information, ensure_ascii=False,
                        sort_keys=True, separators=(",", ":")) if native else None,
                    "hard_bad": native_error is not None or mode_guard_blocked or empty_code,
                    "final_success": success,
                },
                "termination": "submitted" if completed else ("environment_step_limit" if self._done else "running"),
            },
        )

    def step(self, action: Mapping[str, Any]) -> NativeStep:
        if self.meter is not None:
            self.meter.ensure_env_steps(1)
        started = _clock()
        try:
            result = self._step_unmetered(action)
            return NativeStep(
                result.observation,
                result.done,
                result.success,
                {**result.metadata, "environment_seconds": _clock() - started},
            )
        finally:
            if self.meter is not None:
                self.meter.record_env_step(_clock() - started)

    def step_native(self, action: Mapping[str, Any], *, probe: bool = False) -> NativeStep:
        """Execute one official Python interaction without a PRISM compiler."""
        transported = deepcopy(dict(action))
        transported.update(_execution_contract="official-native-v1", _probe=bool(probe))
        return self.step(transported)

    def finalize_evaluation(self) -> dict[str, Any]:
        """Private official scoring for every exit, including budget exhaustion."""
        if self.world is None:
            raise RuntimeError("Final scoring requires a reset AppWorld episode.")
        from importlib.metadata import version
        from appworld.task import task_id_to_generator_id
        tracker = self.world.evaluate()
        scenario = task_id_to_generator_id(self._task_id)
        split = None
        members = []
        for name, definition in self.split_manifest["splits"].items():
            rows = definition["tasks"]
            if any(row["task_id"] == self._task_id for row in rows):
                split = name
                members = [row["task_id"] for row in rows if row["scenario_id"] == scenario]
                break
        from appworld.common.constants import DATA_VERSION
        self.last_native_result = {
            "benchmark": self.benchmark, "task_id": self._task_id,
            "scenario_id": scenario, "scenario_task_ids": members, "split": split,
            "package_version": version("appworld"), "data_version": DATA_VERSION,
            "success": bool(tracker.success), "submitted": bool(self.world.task_completed()),
            "native_evaluator": "AppWorld.evaluate", "stats": tracker.to_dict(stats_only=True),
        }
        return deepcopy(self.last_native_result)

    def save_snapshot(self) -> dict[str, Any]:
        if self.world is None or self._done:
            raise RuntimeError("Snapshots require a live nonterminal episode.")
        started = _clock()
        result = {
            "benchmark": self.benchmark,
            "task_id": self._task_id,
            "seed": self._seed,
            "actions": deepcopy(self._actions),
            "observations": list(self._observations),
            "public_apis": deepcopy(self._public_apis),
            "public_context_sources": deepcopy(self._public_context_sources),
            "execution_state": self.execution_state.dump(),
            "initial_execution_state": deepcopy(self._initial_execution_state),
        }
        if self.meter is not None:
            self.meter.record_snapshot_save(_clock() - started)
        return result

    def restore_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        if snapshot.get("benchmark") != self.benchmark:
            raise ValueError("Snapshot belongs to another benchmark.")
        actions = list(snapshot["actions"])
        observations = list(snapshot["observations"])
        if len(actions) != len(observations):
            raise ValueError("Incomplete replay snapshot.")
        if self.meter is not None:
            self.meter.ensure_env_steps(len(actions))
        started, replayed = _clock(), 0
        try:
            self.reset(str(snapshot["task_id"]), int(snapshot["seed"]))
            self._replaying = True
            for action, expected in zip(actions, observations):
                replayed += 1
                result = self._step_unmetered(action)
                if result.observation != expected or result.done:
                    raise RuntimeError("Native AppWorld prefix replay diverged; no counterfactual pair was accepted.")
            if (
                self._public_apis != snapshot["public_apis"]
                or self._public_context_sources != snapshot["public_context_sources"]
                or self.execution_state.dump() != snapshot["execution_state"]
            ):
                raise RuntimeError("Native prefix replay changed the public program context.")
        finally:
            self._replaying = False
            elapsed = _clock() - started
            self.replayed_env_steps += replayed
            self.snapshot_restore_seconds += elapsed
            if self.meter is not None:
                self.meter.record_restore(elapsed, replayed)

    def close(self) -> None:
        if self.world is not None:
            self.world.close()
            self.world = None


def _load_planbench(repo_root: Path) -> Any:
    source = (repo_root / "src").resolve()
    existing = sys.modules.get("env")
    if existing is not None:
        paths = [Path(path).resolve() for path in getattr(existing, "__path__", [])]
        if source / "env" not in paths:
            raise RuntimeError("Another 'env' package is loaded; use an isolated PlanBench worker process.")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return importlib.import_module("env.runtime.runner")


def _planbench_retrieval_schema() -> dict[str, Any]:
    variants = []
    for name in ("by_output_info", "by_inputs", "by_io_info"):
        fields: dict[str, Any] = {"meta_tool": {"const": name}}
        if name in {"by_output_info", "by_io_info"}:
            fields["output_information"] = {"type": "string", "minLength": 1}
        if name in {"by_inputs", "by_io_info"}:
            fields["input_information"] = {
                "type": "array",
                "minItems": 1,
                "maxItems": 2,
                "items": {"type": "string", "minLength": 1},
            }
        variants.append({
            "type": "object",
            "properties": fields,
            "required": list(fields),
            "additionalProperties": False,
        })
    return {"anyOf": variants}


class NativePlanBenchEnv:
    """Thin stateful bridge over the official PlanBench-XL runtime handlers."""

    benchmark = "planbench_xl"

    def __init__(
        self,
        repo_root: str | Path,
        *,
        output_dir: str | Path,
        max_steps: int = 100,
        split_seed: int = 42,
        split_targets: tuple[int, int, int] = (180, 57, 90),
        run_config_path: str | Path | None = None,
        meter: BudgetMeter | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.meter = meter
        self.replayed_env_steps = 0
        self.snapshot_restore_seconds = 0.0
        self.last_native_result: dict[str, Any] | None = None
        self._runtime = _load_planbench(self.repo_root)
        from env.core import types as types
        from env.domains.executor import DomainToolExecutor
        from env.events.controller import EventController
        from env.events.noisy import NoisyToolAugmenter
        from env.retriever.semantic import SemanticRetriever
        from env.runtime.prompts import PromptManager
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        data_root = self.repo_root / "src" / "data"
        retail = data_root / "retail"
        if run_config_path is not None:
            from env.core.config import load_config
            config = load_config(Path(run_config_path), cli_overrides={"runtime.max_steps": max_steps})
            config.output.output_dir = self.output_dir
        else:
            config = types.RunnerConfig(
                run_id="native-r3",
                domain="retail",
                base_config_path=self.repo_root / "src/env/config/base/env.yaml",
                run_config_path=self.output_dir / "native-environment.json",
                data=types.DataConfig(
                    **{
                        field: retail / filename
                        for field, filename in {
                            "task_file": "tasks.json",
                            "query_file": "queries.json",
                            "baseline_tools_file": "baseline_tools.json",
                            "noisy_tools_file": "noisy_tools.json",
                            "blocker_tools_file": "blocker_tools.json",
                            "datatypes_file": "datatypes.json",
                            "database_file": "database.json",
                            "paths_set_catalog_file": "paths_set_catalog.json",
                        }.items()
                    }
                ),
                model=types.ModelProfile(
                    "metered-native-policy",
                    "local",
                    "local",
                    "metered-native-policy",
                    types.AuthConfig(),
                    types.RequestConfig(max_retries=0),
                ),
                retriever=types.RetrieverConfig(embedding_model="hashing_v1"),
                noise=types.NoiseConfig(mode="append_noisy_siblings", max_total_tools=30),
                query_sample=types.QuerySampleConfig(seed=split_seed),
                blocker=types.BlockerConfig(seed=split_seed),
                runtime=types.RuntimeConfig(max_steps=max_steps, max_concurrency=1, max_resume_attempts=0),
                prompt=types.PromptConfig(prompt_dir=self.repo_root / "src/env/prompt"),
                output=types.OutputConfig(root_dir=self.output_dir, output_dir=self.output_dir),
                logging=types.LoggingConfig(console=False),
                merged_config={"protocol": "native-r3"},
            )
        self.config = config
        baseline = self._runtime.load_all_baseline_tools(data_root)
        datatypes = self._runtime.load_all_datatypes(data_root)
        noisy, noisy_domains = self._runtime.load_noisy_tools_file(config.data.noisy_tools_file, domain=config.domain)
        blocker, blocker_domains = self._runtime.load_all_blocker_tools(data_root)
        replacements = None
        if config.blocker.enable_block:
            from env.events.blocker import generate_blocker_replacements_by_task
            block_args = asdict(config.blocker)
            block_args.pop("enable_block")
            replacements = generate_blocker_replacements_by_task(
                paths_set_catalog=self._runtime.load_paths_set_catalog(config.data.paths_set_catalog_file),
                baseline_tools_path=config.data.baseline_tools_file,
                tasks_path=config.data.task_file,
                **block_args,
            )

        class InMemoryRunner(self._runtime.EnvRunner):
            def _mark_turn_completed(self, *args: Any, **kwargs: Any) -> None:
                return None

            def _finalize_query(
                self,
                query: Any,
                state: Any,
                final_answer: Any,
                failure_reason_detail: str | None,
            ) -> dict[str, Any]:
                result = {
                    "query_id": query.query_id,
                    "status": "success" if failure_reason_detail is None else "failed",
                    "stop_reason": "normal",
                    "final_answer": final_answer,
                    "steps": state.total_step_count,
                    "trace": deepcopy(state.steps_trace),
                    "final_state": deepcopy(self._agent_state_snapshot(state)),
                }
                if failure_reason_detail is not None:
                    result["failure_reason_detail"] = failure_reason_detail
                return result

        self.runner = InMemoryRunner(
            config=config,
            llm_client=None,
            retriever=SemanticRetriever(baseline, datatypes, config.retriever.embedding_model),
            event_controller=EventController(blocker_domains, config.blocker.enable_block, replacements),
            noisy_tool_augmenter=NoisyToolAugmenter(
                noisy_domains, config.noise.mode, max_total_tools=config.noise.max_total_tools
            ),
            tool_executor=DomainToolExecutor(self._runtime.load_all_databases(data_root)),
            prompt_manager=PromptManager(config.prompt.prompt_dir),
            tool_registry={**baseline, **noisy, **blocker},
        )
        self._queries = {query.query_id: query for query in self._runtime.load_queries(config.data.query_file)}
        self._splits = grouped_query_splits(list(self._queries.values()), seed=split_seed, targets=split_targets)
        self.split_manifest = {
            "benchmark": self.benchmark,
            "scheme": "native-task-id-groups",
            "seed": split_seed,
            "target_query_counts": dict(zip(("train", "dev", "test"), split_targets)),
            "splits": {name: {"count": len(rows), "tasks": deepcopy(rows)} for name, rows in self._splits.items()},
            "query_source": str(config.data.query_file),
            "task_source": str(config.data.task_file),
            "condition": {
                "blocker": asdict(config.blocker),
                "noise": asdict(config.noise),
                "runtime": asdict(config.runtime),
                "retriever": asdict(config.retriever),
            },
        }
        self.action_schema = {
            "benchmark": self.benchmark,
            "native_interface": "Emit one official XML action: <retrieve_tools>{JSON}</retrieve_tools>, <tool_call>{JSON}</tool_call>, or <final_answer>answer</final_answer>. Retrieval supports by_output_info (output_information), by_inputs (input_information: one or two phrases), and by_io_info (both). Choose your own information phrases. Call discovered tools with their documented names or tool IDs and arguments.",
            "kind": "native-r3",
            "actor_contract": "One constrained JSON discover, call or finish action.",
            "I": "Discover public tools that produce an unmet input concept.",
            "E": "Call a discovered business tool using committed values, or submit an actually obtained target value.",
            "native_format": (
                "The actor emits only the shared JSON action. Its deterministic compiler emits one official "
                "retrieve_tools, tool_call or final_answer XML action for the native evaluator."
            ),
            "input_values": (
                "Use values from the public task query and actual tool feedback. Match every required parameter "
                "to its documented business meaning. An identifier for one record type is not an identifier for "
                "another record type. Retrieve and call upstream tools to obtain missing values, preserving "
                "all reliability qualifications in each tool description."
            ),
        }
        self._state: Any = None
        self.action_schema["native_system_prompt"] = self.action_schema["native_interface"]
        self._query: Any = None
        self._history: list[dict[str, str]] = []
        self._done = False
        self._seed = 0
        self._public_tools: dict[str, dict[str, Any]] = {}
        self._public_context_sources: list[dict[str, Any]] = []
        self.execution_state: Any = None
        self._initial_execution_state: dict[str, Any] = {}

    def _publish_program_context(self) -> None:
        from .native_program import CONTEXT_VERSION, build_frontier
        context = {
            "version": CONTEXT_VERSION,
            "benchmark": self.benchmark,
            "tools": deepcopy(list(self._public_tools.values())),
            "variables": [],
            "use_tool_ids": self.runner.use_tool_ids,
            "retrieval_schema": _planbench_retrieval_schema(),
            "sources": deepcopy(self._public_context_sources),
            "initial_execution_state": deepcopy(self._initial_execution_state),
            "execution_state": self.execution_state.public_context(),
        }
        context["frontier"] = build_frontier(context)
        self.action_schema = {**self.action_schema, "program_context": context}

    def schema_for_history(self, messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return public_action_schema(self.action_schema, messages)

    def _observe_public_tools(self, observation: str) -> tuple[list[dict[str, Any]], int]:
        id_key = "tool_id" if self.runner.use_tool_ids else "name"
        docs = []
        new_count = 0
        for value in _json_observations(observation):
            if value.get("feedback_type") != "retrieve_tools_result":
                continue
            for tool in value["retrieval_result"]["tools"]:
                public = {
                    "name": tool["name"],
                    "description": str(tool.get("description", "")),
                    "surface_output": deepcopy(tool.get("surface_output")),
                    "arguments_schema": deepcopy(tool["parameters"]),
                }
                if "tool_id" in tool:
                    public["tool_id"] = tool["tool_id"]
                if tool[id_key] not in self._public_tools:
                    new_count += 1
                self._public_tools[tool[id_key]] = public
                docs.append(deepcopy(public))
        return docs, new_count

    def list_tasks(self, split: str) -> list[dict[str, str]]:
        return deepcopy(self._splits[split])

    def training_reference(self, task_id: str | None = None) -> dict[str, Any]:
        """Private registered-train answer and released tool-dependency solution."""
        selected = (
            self._query.query_id
            if task_id is None and self._query is not None
            else str(task_id)
            if task_id is not None
            else ""
        )
        if selected not in {task["task_id"] for task in self._splits["train"]}:
            raise ValueError("Ground-truth teacher references are restricted to registered training tasks.")
        query = self._queries[selected]
        tasks = json.loads(Path(self.config.data.task_file).read_text(encoding="utf-8"))
        task = next(item for item in tasks if item["task_id"] == query.task_id)
        path = task["one_available_path"]
        if not path:
            raise ValueError("The official training task has no released reference dependency path.")
        reference = {
            "correct_answer": query.correct_answer,
            "input_values": query.input_values,
            "input_datatypes": list(query.input_datatypes),
            "target_datatype": query.target_datatype,
            "one_available_path": list(path),
            "reference_tools": [deepcopy(self.runner.tool_registry[name]) for name in path],
        }
        return {
            "benchmark": self.benchmark,
            "task_id": selected,
            "source": "PlanBench-XL.QuerySpec/tasks.json.one_available_path/tool_registry",
            "reference_id": f"planbench_xl/train/{selected}",
            "reference": reference,
        }

    def reset(self, task_id: str, seed: int) -> dict[str, Any]:
        query = self._queries[str(task_id)]
        self._query, self._seed, self._done = query, int(seed), False
        self.last_native_result = None
        self._public_tools = {}
        self._public_context_sources = []
        initial = self.runner._initial_query_progress(query)
        self._state = self.runner._load_agent_state(initial, query)
        self._history = deepcopy(initial["resume_checkpoint"]["effective_history"])
        from .native_execution_state import ExecutionState
        self.execution_state = ExecutionState.from_task(self.benchmark, query.query_id, query.query_text, "")
        self._initial_execution_state = self.execution_state.dump()
        self._publish_program_context()
        return {
            "task_id": query.query_id,
            "scenario_id": query.task_id,
            "instruction": query.query_text,
            "observation": "",
            "system_prompt": (
                "Solve the native PlanBench task with one shared JSON action at a time. "
                "Discover public tool schemas, bind required parameters to actual task values or tool "
                "returns with compatible business meaning, and finish using an obtained target value."
            ),
            "action_schema": deepcopy(self.action_schema),
        }

    @staticmethod
    def _raw_action(action: Mapping[str, Any]) -> str:
        kind = action.get("kind")
        if "native_payload" in action and kind in {"retrieve_tools", "tool_call", "final_answer"}:
            return f"<{kind}>{action['native_payload']}</{kind}>"
        if "native_payload" in action:
            return str(action["raw_text"])
        if kind == "final_answer":
            return f"<final_answer>{action.get('answer', '')}</final_answer>"
        if kind == "retrieve_tools":
            payload = action.get("arguments", action.get("request", {}))
        elif kind == "tool_call":
            payload = {key: action[key] for key in ("name", "tool_name", "tool_id", "arguments") if key in action}
        else:
            return str(action.get("raw_text", action.get("text", "")))
        return f"<{kind}>{json.dumps(payload, ensure_ascii=False, allow_nan=False)}</{kind}>"

    def _step_unmetered(self, action: Mapping[str, Any]) -> NativeStep:
        if self._state is None or self._done:
            raise RuntimeError("PlanBench step requires a live reset episode.")
        from env.runtime.parsing import extract_action
        action = deepcopy(dict(action))
        raw = self._raw_action(action)
        kind, content = extract_action(raw)
        if action.get("kind") == "invalid":
            kind, content = None, None
        planned = None
        public_tool = None
        request_arguments: dict[str, Any] = {}
        if kind is not None:
            try:
                if not isinstance(action.get("r3_program"), Mapping):
                    raise ValueError("PlanBench requires a compiled native-r3 action.")
                plan = action.get("receipt_plan")
                if kind == "final_answer" and plan == []:
                    planned = {"tool": "final_answer", "arguments": {}, "bind": ""}
                elif isinstance(plan, list) and len(plan) == 1:
                    planned = plan[0]
                else:
                    raise ValueError("PlanBench compiles one native operation per turn.")
                if kind in {"retrieve_tools", "tool_call"}:
                    payload = json.loads(content)
                    if not isinstance(payload, dict):
                        raise ValueError("The native request payload must be a JSON object.")
                    request_arguments = payload if kind == "retrieve_tools" else payload["arguments"]
                    if kind == "tool_call":
                        id_key = "tool_id" if self.runner.use_tool_ids else "name"
                        public_tool = self._public_tools[payload[id_key]]
                        if planned["tool"] != public_tool["name"]:
                            raise ValueError("The native tool differs from its public binding plan.")
                    if planned["arguments"] != request_arguments:
                        raise ValueError("The native parameters differ from their exact public binding plan.")
            except (KeyError, TypeError, ValueError) as error:
                action.update(kind="invalid", validation_stage="action_contract", error=str(error))
                kind, content = None, None
        mode = "I" if kind == "retrieve_tools" else "E"
        self._state.total_step_count += 1
        turn = self._state.total_step_count
        native_failed = False
        rejection_kind = ""
        retrievals_before = self._state.retrieval_exec_count
        state_before = (
            deepcopy(self._state.trusted_values_by_datatype),
            deepcopy(self._state.untrusted_values_by_datatype),
            set(self._state.discovered_tool_names),
        )
        new_information_candidate = False
        retrieval_fact = None
        tool_fact = None
        public_docs: list[dict[str, Any]] = []
        receipts: list[dict[str, Any]] = []
        mode_error = (
            str(action["error"])
            if action.get("validation_stage") == "mode"
            else "The explicit policy mode must be I or E."
            if action.get("mode") is not None and action["mode"] not in {"I", "E"}
            else f"Native {kind} belongs to mode {mode}; sampled mode was {action['mode']}."
            if kind is not None and action.get("mode") is not None and action["mode"] != mode
            else None
        )
        if mode_error is not None:
            feedback = "Mode constraint violation: " + mode_error
            self._state.steps_trace.append({
                "step_id": turn,
                "action": "mode_rejected",
                "raw_output": raw,
                "error": mode_error,
            })
            self._history = self.runner._append_turn_messages(self._history, raw, feedback)
            result = {"done": False, "history": self._history, "result": None}
        elif kind is None:
            self._state.label_error_cnt += 1
            feedback = self.runner.prompt_manager.render(
                "error_label_format.txt",
                current_attempt=self._state.label_error_cnt,
                max_label_errors=self.config.runtime.max_label_errors,
            )
            if action.get("error"):
                feedback += "\nAction parser: " + str(action["error"])
            self._state.steps_trace.append({"step_id": turn, "action": "others", "raw_output": raw})
            self._history = self.runner._append_turn_messages(self._history, raw, feedback)
            result = {
                "done": self._state.label_error_cnt >= self.config.runtime.max_label_errors,
                "history": self._history,
                "result": None,
            }
            if result["done"]:
                result["result"] = self.runner._finalize_query(
                    self._query, self._state, None, "exceeded_max_label_errors"
                )
        else:
            self._state.label_error_cnt = 0
            handler = {
                "retrieve_tools": self.runner._handle_retrieve,
                "tool_call": self.runner._handle_tool_call,
                "final_answer": self.runner._handle_final_answer,
            }[kind]
            previous_len = len(self._history)
            previous_trace_len = len(self._state.steps_trace)
            result = handler(self._query, raw, content or "", turn, self._state, self._history)
            current_traces = self._state.steps_trace[previous_trace_len:]
            if len(current_traces) != 1 or current_traces[0].get("step_id") != turn:
                raise RuntimeError("Native PlanBench handler did not record exactly this action.")
            current_trace = current_traces[0]
            if kind == "retrieve_tools":
                native_failed = "retrieval_result" not in current_trace
                new_information_candidate = (
                    not native_failed and current_trace["retrieval_result"]["tool_count"] > 0
                )
                if not native_failed:
                    retrieval_fact = {"tool_count": current_trace["retrieval_result"]["tool_count"]}
            elif kind == "tool_call":
                native_failed = (
                    "tool_result" not in current_trace
                    or current_trace["tool_result"]["success"] is False
                    or (
                        current_trace["tool_result"]["tool_type"] == "baseline"
                        and current_trace["tool_result"]["output_value"] is None
                    )
                )
                if native_failed:
                    rejection_kind = (
                        "input_precondition"
                        if "tool_result" not in current_trace
                        else "terminal_rejection"
                    )
                new_information_candidate = (
                    not native_failed and current_trace["tool_result"]["output_value"] is not None
                )
                if "tool_result" in current_trace:
                    tool_fact = {
                        "success": not native_failed,
                        "output_value": deepcopy(current_trace["tool_result"]["output_value"]),
                    }
            self._history = result["history"]
            feedbacks = [
                message["content"]
                for message in self._history[previous_len:]
                if message["role"] != "assistant"
            ]
            feedback = feedbacks[-1] if feedbacks else ""
            new_tool_count = 0
            if kind == "retrieve_tools":
                public_docs, new_tool_count = self._observe_public_tools(feedback)
            public_output = None
            if kind == "tool_call" and tool_fact is not None:
                public_output = deepcopy(tool_fact["output_value"])
            elif kind == "retrieve_tools" and not native_failed:
                result_object = next(
                    value
                    for value in _json_observations(feedback)
                    if value.get("feedback_type") == "retrieve_tools_result"
                )
                public_output = deepcopy(result_object["retrieval_result"])
            receipt = {
                "call_id": f"step_{turn:05d}_call_0000",
                "tool": planned["tool"],
                "arguments": _public_json(request_arguments),
                "output": _public_json(public_output),
                "succeeded": not native_failed,
                "status": "success" if not native_failed else "failed",
                "observation_id": turn,
                "bind": planned.get("bind", f"step_{turn:05d}_call_0000"),
                "kind": {"retrieve_tools": "discover", "tool_call": "call", "final_answer": "finish"}[kind],
                "rejection_kind": rejection_kind,
            }
            if kind == "retrieve_tools":
                receipt["new_tool_count"] = new_tool_count
            if public_tool is not None:
                receipt["public_doc"] = deepcopy(public_tool)
            receipt["value_status"] = public_document_quality(receipt.get("public_doc", {}))
            receipt["usable_for_binding"] = bool(receipt["succeeded"] and receipt["kind"] == "call")
            self.execution_state.apply_receipt(receipt)
            receipts.append(receipt)
        self._done = bool(result["done"])
        if not self._done and turn >= self.config.runtime.max_steps:
            self._done = True
            result["result"] = self.runner._finalize_query(self._query, self._state, None, "exceeded_max_steps")
        if self._done:
            self.last_native_result = deepcopy(result["result"])
        success = bool(self.last_native_result and self.last_native_result["status"] == "success")
        executed = mode_error is None and kind is not None
        attempted_calls = int(executed and kind in {"retrieve_tools", "tool_call"})
        successful_retrievals = int(
            executed
            and kind == "retrieve_tools"
            and not native_failed
            and self._state.retrieval_exec_count > retrievals_before
        )
        successful_business = int(executed and kind == "tool_call" and not native_failed)
        step_failed = mode_error is not None or kind is None or native_failed
        if self.meter is not None:
            self.meter.record_api_calls(attempted_calls, successful=successful_retrievals + successful_business)
        feedback = _receipt_observation(feedback, receipts)
        self._public_context_sources.append({
            "observation_id": turn,
            "observation_text": feedback,
            "program_id": action.get("program_id"),
            "api_docs": public_docs,
            "bindings": [],
            "receipts": deepcopy(receipts),
        })
        self._publish_program_context()
        return NativeStep(
            feedback,
            self._done,
            success,
            {
                "mode": action.get("mode") or (mode if kind is not None else None),
                "submitted": mode_error is None and kind == "final_answer",
                "native_action": kind,
                "step_failed": step_failed,
                "native_executed": executed,
                "native_execution_succeeded": executed and not native_failed,
                "native_api_calls": attempted_calls,
                "api_calls": attempted_calls,
                "attempted_api_calls": attempted_calls,
                "successful_api_calls": successful_retrievals + successful_business,
                "successful_business_calls": successful_business,
                "successful_retrieval_calls": successful_retrievals,
                "successful_documentation_calls": 0,
                "successful_submission_calls": int(executed and kind == "final_answer" and not native_failed),
                "information_action_executed": successful_retrievals > 0,
                "receipts": deepcopy(receipts),
                "native_success": success if self._done else None,
                "native_transition": {
                    "state_changed": state_before
                    != (
                        self._state.trusted_values_by_datatype,
                        self._state.untrusted_values_by_datatype,
                        self._state.discovered_tool_names,
                    ),
                    "database_changed": None,
                    "new_information_candidate": new_information_candidate,
                    "retrieval_result": retrieval_fact,
                    "tool_result": tool_fact,
                    "hard_bad": mode_error is not None or kind is None or native_failed,
                    "final_success": success,
                },
                "failure_reason": self.last_native_result.get("failure_reason_detail") if self._done else None,
                "validation_stage": (
                    "mode"
                    if mode_error is not None
                    else (action.get("validation_stage", "action_parse") if kind is None else "native")
                ),
                "validation_error": mode_error or action.get("error"),
                "mode_guard_blocked": mode_error is not None,
                "termination": (
                    self.last_native_result.get("failure_reason_detail", "submitted")
                    if self._done
                    else "running"
                ),
            },
        )

    def step(self, action: Mapping[str, Any]) -> NativeStep:
        if self.meter is not None:
            self.meter.ensure_env_steps(1)
        started = _clock()
        try:
            result = self._step_unmetered(action)
            return NativeStep(
                result.observation,
                result.done,
                result.success,
                {**result.metadata, "environment_seconds": _clock() - started},
            )
        finally:
            if self.meter is not None:
                self.meter.record_env_step(_clock() - started)

    def step_native(self, action: Mapping[str, Any], *, probe: bool = False) -> NativeStep:
        """Use the released parser and handlers on unrestricted native actions.

        The pinned retail executor is a read-only lookup over database rows;
        business tool calls therefore also qualify as environment probes.
        """
        if self._state is None or self._done:
            raise RuntimeError("PlanBench step requires a live reset episode.")
        if self.meter is not None:
            self.meter.ensure_env_steps(1)
        started = _clock()
        try:
            from env.runtime.parsing import extract_action
            raw = self._raw_action(action)
            kind, content = extract_action(raw)
            if action.get("kind") == "invalid":
                kind = None
            self._state.total_step_count += 1
            turn = self._state.total_step_count
            before_trace = len(self._state.steps_trace)
            before_history = len(self._history)
            before_tools = set(self._state.discovered_tool_names)
            before_values = deepcopy(self._state.trusted_values_by_datatype)
            if probe and kind == "final_answer":
                feedback = "A read-only environment probe cannot submit a final answer."
                self._state.steps_trace.append({"step_id": turn, "action": "probe_rejected", "raw_output": raw})
                result = {"done": False, "result": None,
                          "history": self.runner._append_turn_messages(self._history, raw, feedback)}
                kind = None
            elif kind is None:
                self._state.label_error_cnt += 1
                feedback = self.runner.prompt_manager.render("error_label_format.txt",
                    current_attempt=self._state.label_error_cnt,
                    max_label_errors=self.config.runtime.max_label_errors)
                self._state.steps_trace.append({"step_id": turn, "action": "others", "raw_output": raw})
                exhausted = self._state.label_error_cnt >= self.config.runtime.max_label_errors
                result = {"done": exhausted, "history": self.runner._append_turn_messages(self._history, raw, feedback),
                          "result": self.runner._finalize_query(self._query, self._state, None,
                              "exceeded_max_label_errors") if exhausted else None}
            else:
                self._state.label_error_cnt = 0
                handler = {"retrieve_tools": self.runner._handle_retrieve,
                           "tool_call": self.runner._handle_tool_call,
                           "final_answer": self.runner._handle_final_answer}[kind]
                result = handler(self._query, raw, content or "", turn, self._state, self._history)
            self._history = result["history"]
            observations = [item["content"] for item in self._history[before_history:] if item["role"] != "assistant"]
            feedback = observations[-1] if observations else ""
            traces = self._state.steps_trace[before_trace:]
            trace = traces[-1] if traces else {}
            tool_result = trace.get("tool_result", {})
            retrieval_ok = kind == "retrieve_tools" and "retrieval_result" in trace
            call_ok = kind == "tool_call" and bool(tool_result.get("success"))
            docs, new_tools = self._observe_public_tools(feedback) if retrieval_ok else ([], 0)
            self._public_context_sources.append({"observation_id": turn,
                "observation_text": feedback, "program_id": None, "api_docs": docs,
                "bindings": [], "receipts": []})
            self._publish_program_context()
            self._done = bool(result["done"])
            if not self._done and turn >= self.config.runtime.max_steps:
                self._done = True
                result["result"] = self.runner._finalize_query(self._query, self._state, None, "exceeded_max_steps")
            if self._done:
                self.last_native_result = deepcopy(result["result"])
            success = bool(self.last_native_result and self.last_native_result["status"] == "success")
            attempted = int(kind in {"retrieve_tools", "tool_call"})
            if self.meter is not None:
                self.meter.record_api_calls(attempted, successful=int(retrieval_ok or call_ok))
            failed = kind is None or (kind == "retrieve_tools" and not retrieval_ok) or (kind == "tool_call" and not call_ok)
            public_output = tool_result.get("output_value") if call_ok else None
            output_signature = (json.dumps(_public_json(public_output), ensure_ascii=False, sort_keys=True,
                                           separators=(",", ":")) if public_output is not None else None)
            return NativeStep(feedback, self._done, success, {
                "native_action": kind, "submitted": kind == "final_answer", "native_executed": kind is not None,
                "step_failed": failed, "native_execution_succeeded": not failed,
                "native_api_calls": attempted, "api_calls": attempted, "attempted_api_calls": attempted,
                "successful_api_calls": int(retrieval_ok or call_ok), "successful_business_calls": int(call_ok),
                "successful_retrieval_calls": int(retrieval_ok), "successful_documentation_calls": 0,
                "successful_submission_calls": int(kind == "final_answer" and success),
                "information_action_executed": bool(retrieval_ok or (probe and call_ok)),
                "receipts": [], "new_tool_count": new_tools,
                "native_success": success if self._done else None,
                "native_transition": {"state_changed": before_tools != self._state.discovered_tool_names or before_values != self._state.trusted_values_by_datatype,
                    "new_information_candidate": bool(retrieval_ok or output_signature is not None),
                    "public_information_signature": output_signature, "hard_bad": failed, "final_success": success},
                "environment_seconds": _clock() - started,
                "termination": self.last_native_result.get("failure_reason_detail", "submitted") if self._done else "running",
            })
        finally:
            if self.meter is not None:
                self.meter.record_env_step(_clock() - started)

    def finalize_evaluation(self) -> dict[str, Any]:
        """Private per-query official metrics; never sent as actor feedback."""
        if self._state is None:
            raise RuntimeError("Final scoring requires a reset PlanBench episode.")
        result = deepcopy(self.last_native_result) if self.last_native_result is not None else self.runner._finalize_query(
            self._query, self._state, None, "external_budget_or_controller_stop")
        evaluator = importlib.import_module("env.evaluate")
        query = evaluator.QueryRecord(self._query.query_id, self._query.task_id, self._query.correct_answer,
                                      set(self._query.input_datatypes), self._query.target_datatype)
        grounded = evaluator.datatype_check_passed_for_result(result, query)
        success = bool(grounded and self.runner.answer_judge.is_correct(query.correct_answer, result.get("final_answer")))
        noisy = evaluator.build_noisy_tool_name_set(Path(self.config.data.noisy_tools_file))
        metrics = evaluator.trace_metrics(result, query, noisy)
        paths = self._runtime.load_paths_set_catalog(self.config.data.paths_set_catalog_file)
        truth = evaluator.ground_truth_datatypes_for_task(query.task_id, paths, self.runner.tool_registry)
        executed = set(metrics["executed_datatypes"])
        metrics["executed_gt_datatype_precision"] = evaluator.safe_ratio(len(executed & truth), len(executed))
        metrics["executed_gt_datatype_recall"] = evaluator.safe_ratio(len(executed & truth), len(truth))
        return {"benchmark": self.benchmark, "task_id": query.query_id, "scenario_id": query.task_id,
                "success": success, "native_evaluator": "PlanBench-XL.env.evaluate",
                "split": next((name for name, rows in self._splits.items() if any(row["task_id"] == query.query_id for row in rows)), None),
                "split_scheme": "custom_task_grouped", "metrics": metrics, "result": result}

    def save_snapshot(self) -> dict[str, Any]:
        if self._state is None or self._done:
            raise RuntimeError("Snapshots require a live nonterminal episode.")
        started = _clock()
        result = {
            "benchmark": self.benchmark,
            "query_id": self._query.query_id,
            "seed": self._seed,
            "state": deepcopy(self._state),
            "history": deepcopy(self._history),
            "public_tools": deepcopy(self._public_tools),
            "public_context_sources": deepcopy(self._public_context_sources),
            "execution_state": self.execution_state.dump(),
            "initial_execution_state": deepcopy(self._initial_execution_state),
        }
        if self.meter is not None:
            self.meter.record_snapshot_save(_clock() - started)
        return result

    def restore_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        if snapshot.get("benchmark") != self.benchmark:
            raise ValueError("Snapshot belongs to another benchmark.")
        if self.meter is not None:
            self.meter.ensure_env_steps(0)
        started = _clock()
        self._query = self._queries[str(snapshot["query_id"])]
        self._seed = int(snapshot["seed"])
        self._state, self._history = deepcopy(snapshot["state"]), deepcopy(snapshot["history"])
        self._public_tools = deepcopy(snapshot["public_tools"])
        self._public_context_sources = deepcopy(snapshot["public_context_sources"])
        from .native_execution_state import ExecutionState
        self.execution_state = ExecutionState.load(snapshot["execution_state"])
        self._initial_execution_state = deepcopy(snapshot["initial_execution_state"])
        self._publish_program_context()
        self._done, self.last_native_result = False, None
        elapsed = _clock() - started
        self.snapshot_restore_seconds += elapsed
        if self.meter is not None:
            self.meter.record_restore(elapsed, 0)

    def close(self) -> None:
        self._state, self._query, self._history = None, None, []
