"""Train-only full native demonstrations and fixed task-progress predicates.

Reference choices live in this module. Student prefixes contain only the public
task, real documentation discoveries, committed native returns and the common
native-r3 action contract. Evaluation workers never import this module.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import random
import re
from typing import Any, Callable, Mapping, Sequence

import torch

from .native_core import (
    BudgetExhausted,
    BudgetMeter,
    NativeEvent,
    NativeTrajectory,
    append_interaction,
    initial_messages,
    monotonic_seconds,
)
from .native_execution_state import (
    binding_table,
    canonical,
    pointer_get,
    reference_for_value,
    semantic_key,
    walk_values,
)
from .native_program import (
    PROGRAM_VERSION,
    _leaf_semantic,
    _semantic_index,
    build_frontier,
    next_binding_name,
    public_value_candidates,
)
from .native_semantics import PublicSemanticIndex


class DemonstrationError(ValueError):
    """A fixed reference could not produce a complete public-contract example."""


@dataclass
class NativeDemonstration:
    task_id: str
    environment_seed: int
    trajectory: NativeTrajectory
    reference_id: str
    initial_fingerprint: dict[str, Any] = field(default_factory=dict)
    target_fingerprint: dict[str, Any] = field(default_factory=dict)


def _parameters(doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    if isinstance(doc.get("parameters"), list):
        return [dict(parameter) for parameter in doc["parameters"]]
    schema = doc.get("arguments_schema", doc.get("parameters"))
    if not isinstance(schema, Mapping) or not isinstance(schema.get("properties"), Mapping):
        raise DemonstrationError("The real public tool document has no argument schema.")
    required = set(schema.get("required", []))
    return [
        {**value, "name": name, "required": name in required}
        for name, value in schema["properties"].items()
    ]


def _public_tool(context: Mapping[str, Any], name: str) -> dict[str, Any] | None:
    if context["benchmark"] == "appworld":
        return context["apis"].get(name)
    return next((tool for tool in context["tools"] if tool["name"] == name), None)


def _argument_expression(
    context: Mapping[str, Any],
    name: str,
    parameter: Mapping[str, Any],
    value: Any,
) -> dict[str, Any]:
    """Map a reference-selected value to currently available student evidence."""
    index = _semantic_index(context)
    candidates = public_value_candidates(context, parameter, name, semantic_index=index)
    matching = [
        candidate["expression"]
        for candidate in candidates
        if canonical(candidate["value"]) == canonical(value)
    ]
    if matching:
        return matching[0]
    documented = list(parameter.get("enum", []))
    if "default" in parameter:
        documented.append(parameter["default"])
    control = str(parameter["name"]) in {
        "page_index", "page_size", "page", "offset", "limit", "recursive",
        "overwrite", "exist_ok", "show_hidden", "include_hidden", "ascending", "descending",
    }
    if any(canonical(value) == canonical(item) for item in documented):
        return {"literal": value}
    if control and isinstance(value, (int, float, bool)):
        return {"literal": value}
    if value is None and not parameter.get("required", False):
        return {"literal": None}
    try:
        return reference_for_value(context, value, semantic_type=index.parameter_semantic(parameter, name))
    except ValueError as error:
        raise DemonstrationError(f"No public expression for {name}.{parameter['name']}: {error}") from error


class _PublicReplay:
    def __init__(
        self,
        policy: Any,
        env: Any,
        task: Mapping[str, Any],
        context: Mapping[str, Any],
        config: Mapping[str, Any],
        meter: BudgetMeter,
    ) -> None:
        from .native_training import _limits
        self.policy, self.env, self.config, self.meter = policy, env, config, meter
        self.messages = initial_messages(context)
        self.started = monotonic_seconds()
        self.trajectory = NativeTrajectory(
            str(task["task_id"]),
            str(context["scenario_id"]),
            budget_seconds=_limits(config).max_wall_seconds,
            metadata={
                "environment_seed": int(task["environment_seed"]),
                "actor_contract": PROGRAM_VERSION,
                "demonstration": True,
                "success_counting_unit": "teacher_replay_only",
            },
        )

    def schema(self) -> dict[str, Any]:
        return self.env.schema_for_history(self.messages)

    def step(self, program: Mapping[str, Any]) -> NativeEvent:
        from .native_training import _decision_messages, _limits, _public_error
        from .native_policy import NoFeasibleAction
        from jsonschema import ValidationError
        if self.trajectory.termination != "unfinished":
            raise DemonstrationError("A reference attempted to act after native termination.")
        if len(self.trajectory.events) >= _limits(self.config).max_env_steps:
            raise DemonstrationError("Reference replay reached the fixed episode turn limit.")
        self.trajectory.elapsed_seconds = monotonic_seconds() - self.started
        if self.trajectory.elapsed_seconds >= self.trajectory.budget_seconds:
            raise DemonstrationError("Reference replay reached the fixed episode time limit.")
        before = _decision_messages(self.messages, self.trajectory, _limits(self.config), self.meter)
        schema = self.env.schema_for_history(before)
        target = deepcopy(dict(program))
        target.pop("goals", None)
        target.pop("goal_id", None)
        if target.get("kind") in {"discover", "call", "map_call"}:
            identifier = next_binding_name(schema["program_context"])
            if "bind" in target and target["bind"] != identifier:
                raise DemonstrationError(
                    "A reference action selected a binding outside the shared fresh-name contract."
                )
            target["bind"] = identifier
        try:
            generation = self.policy.teacher_forced_generation(
                before,
                schema,
                target,
                gated=True,
                role="native_reference_public_target",
            )
        except (ValueError, ValidationError, NoFeasibleAction) as error:
            raise DemonstrationError(
                f"Reference action is outside the shared public actor contract: {error}"
            ) from error
        began = monotonic_seconds()
        step = self.env.step(generation.action)
        self.trajectory.elapsed_seconds = monotonic_seconds() - self.started
        event = NativeEvent(
            before,
            generation,
            step.observation,
            monotonic_seconds() - began,
            metadata={
                "native_step": deepcopy(step.metadata),
                "native_done": bool(step.done),
                "native_terminal_success": bool(step.done and step.success),
                "public_error": _public_error(step.metadata, step.observation),
                "elapsed_after": self.trajectory.elapsed_seconds,
            },
            receipts=deepcopy(step.metadata.get("receipts", [])),
        )
        self.trajectory.events.append(event)
        append_interaction(self.messages, generation, step)
        if step.done:
            self.trajectory.success = (
                bool(step.success) and self.trajectory.elapsed_seconds <= self.trajectory.budget_seconds
            )
            self.trajectory.termination = "native_terminal" if self.trajectory.success else "native_failure"
        if step.metadata.get("step_failed"):
            raise DemonstrationError("A reference action failed in the native replay; its actual receipt is retained.")
        return event

    def finish_record(self) -> NativeTrajectory:
        from .native_training import program_execution_summary
        self.trajectory.elapsed_seconds = monotonic_seconds() - self.started
        self.trajectory.metadata["final_messages"] = deepcopy(self.messages)
        self.trajectory.metadata["program_outcomes"] = program_execution_summary(self.trajectory.events)
        self.trajectory.metadata["resource_usage"] = self.meter.snapshot()
        return self.trajectory


def _discover_appworld(replay: _PublicReplay, tool_name: str) -> None:
    """Teacher selects public discovery actions; the runtime makes every call."""
    app_name, api_name = tool_name.split(".", 1)
    while _public_tool(replay.schema()["program_context"], tool_name) is None:
        context = replay.schema()["program_context"]
        candidates = build_frontier(context)["discovery_queries"]
        exact = {"tool": "api_docs.show_api_doc", "arguments": {"app_name": app_name, "api_name": api_name}}
        app = {"tool": "api_docs.show_api_descriptions", "arguments": {"app_name": app_name}}
        catalog = {"tool": "api_docs.show_app_descriptions", "arguments": {}}
        chosen = next(
            (item for wanted in (exact, app, catalog) for item in candidates if canonical(item) == canonical(wanted)),
            None,
        )
        if chosen is None:
            raise DemonstrationError(f"Public documentation does not expose the reference API {tool_name}.")
        replay.step({
            "kind": "discover",
            "tool": chosen["tool"],
            "arguments": {key: {"literal": value} for key, value in chosen["arguments"].items()},
        })


def _translated(value: Any, correspondence: Mapping[str, Any]) -> Any:
    key = canonical(value)
    if key in correspondence:
        return deepcopy(correspondence[key])
    if isinstance(value, Mapping):
        return {key: _translated(item, correspondence) for key, item in value.items()}
    if isinstance(value, list):
        return [_translated(item, correspondence) for item in value]
    return deepcopy(value)


def _replay_appworld(replay: _PublicReplay, reference_receipts: Sequence[Mapping[str, Any]]) -> None:
    correspondence: dict[str, Any] = {}
    for reference_call in reference_receipts:
        if not reference_call["succeeded"]:
            raise DemonstrationError("The successful reference includes a failed API call outside the action contract.")
        name = str(reference_call["tool"])
        if name == "supervisor.complete_task":
            args = reference_call["arguments"]
            program: dict[str, Any] = {"kind": "finish"}
            if args.get("answer") is not None:
                program["answer"] = reference_for_value(
                    replay.schema()["program_context"],
                    _translated(args["answer"], correspondence),
                )
            replay.step(program)
            break
        _discover_appworld(replay, name)
        context = replay.schema()["program_context"]
        doc = _public_tool(context, name)
        if doc is None:
            raise DemonstrationError("Reference replay lost the just-observed public API document.")
        supplied = reference_call["arguments"]
        parameters = {item["name"]: item for item in _parameters(doc)}
        arguments = {
            key: _argument_expression(context, name, parameters[key], _translated(value, correspondence))
            for key, value in supplied.items()
            if key in parameters
        }
        for key, parameter in parameters.items():
            if parameter.get("required", False) and key not in arguments:
                raise DemonstrationError(f"The real reference receipt omitted required native argument {name}.{key}.")
        event = replay.step({
            "kind": "discover" if name.startswith("api_docs.") else "call",
            "tool": name,
            "arguments": arguments,
        })
        actual = [receipt for receipt in event.receipts if receipt["tool"] == name and receipt["succeeded"]]
        if len(actual) != 1:
            raise DemonstrationError("Reference replay requires exactly one real API receipt for each selected call.")
        new_output = actual[0]["output"]
        for path, old_value in walk_values(reference_call["output"]):
            correspondence[canonical(old_value)] = deepcopy(pointer_get(new_output, path))
        if replay.trajectory.termination != "unfinished":
            break
    if replay.trajectory.termination == "unfinished":
        replay.step({"kind": "finish"})


def _reference_input_semantics(tool: str, index: PublicSemanticIndex) -> set[str]:
    document = index.documents.get(tool)
    if document is not None:
        return {
            index.parameter_semantic(parameter, tool)
            for parameter in _parameters(document)
            if parameter.get("required", False)
        }
    operation = str(tool).rsplit(".", 1)[-1].lower()
    parts = operation.split("_from_", 1)
    if len(parts) != 2:
        return set()
    source = re.sub(r"_(?:realtime|enterprise|swift|max|edge|nano)$", "", parts[1])
    return {index.resolve(part) for part in source.split("_and_") if part}


def _retrieval_affinity(
    query: Mapping[str, Any],
    tool: str,
    context: Mapping[str, Any],
) -> tuple[int, int, int, int, int]:
    index = _semantic_index(context)
    inputs = {index.resolve(str(item)) for item in query.get("input_information", [])}
    known = {
        _leaf_semantic(binding, path, value, index)
        for binding in binding_table(context).values()
        if binding.get("source") in {"task_span", "native_return"}
        and binding.get("usable_for_binding", True)
        for path, value in walk_values(binding["value"])
        if value is not None
        and not isinstance(value, (Mapping, list))
        and _leaf_semantic(binding, path, value, index)
    }
    required = _reference_input_semantics(tool, index)
    matched = inputs & required & known
    full_input_edge = bool(required) and required <= inputs and inputs <= known
    phrase = str(query.get("output_information", ""))
    produced = index.produced_semantics(tool)
    exact_output = bool(phrase) and index.resolve(phrase) in produced
    strategy = query.get("meta_tool")
    if full_input_edge and exact_output and strategy == "by_io_info":
        priority = 5
    elif full_input_edge and strategy in {"by_inputs", "by_io_info"}:
        priority = 4
    elif matched and strategy in {"by_inputs", "by_io_info"}:
        priority = 3
    elif exact_output:
        priority = 2
    elif inputs and inputs <= known:
        priority = 1
    else:
        priority = 0
    return (
        priority,
        len(matched),
        -len(inputs - required),
        int(exact_output),
        -len(re.findall(r"[a-z0-9]+", phrase.lower())),
    )


def _replay_planbench(replay: _PublicReplay, reference: Mapping[str, Any]) -> None:
    path = reference["one_available_path"]
    if not isinstance(path, list) or not path or not all(isinstance(name, str) for name in path):
        raise DemonstrationError("PlanBench training reference must contain an ordered list of tool names.")
    last_binding = None
    for name in path:
        while _public_tool(replay.schema()["program_context"], name) is None:
            context = replay.schema()["program_context"]
            queries = build_frontier(context)["discovery_queries"]
            if not queries:
                raise DemonstrationError(f"Public discovery exhausted before reference tool {name} became visible.")
            chosen = max(queries, key=lambda item: _retrieval_affinity(item, name, context))
            replay.step({"kind": "discover", "arguments": deepcopy(chosen)})
        context = replay.schema()["program_context"]
        doc = _public_tool(context, name)
        if doc is None:
            raise DemonstrationError("Reference replay lost the just-observed public tool document.")
        index = _semantic_index(context)
        arguments = {}
        for parameter in _parameters(doc):
            if not parameter.get("required", False):
                continue
            candidates = public_value_candidates(context, parameter, name, semantic_index=index)
            if not candidates:
                raise DemonstrationError(
                    f"The reference tool {name}.{parameter['name']} has no public compatible value."
                )
            # In a linear PlanBench path, prefer the top-scored candidate from the
            # immediately preceding tool binding when available, otherwise the top
            # soft-ranked candidate in public_value_candidates.
            selected = None
            if last_binding is not None:
                from_last = [
                    cand
                    for cand in candidates
                    if cand["expression"].get("ref") == last_binding
                ]
                if from_last:
                    selected = from_last[0]
            if selected is None:
                selected = candidates[0]
            arguments[parameter["name"]] = deepcopy(selected["expression"])
        event = replay.step({"kind": "call", "tool": name, "arguments": arguments})
        actual = [
            receipt
            for receipt in event.receipts
            if receipt.get("tool") == name and receipt.get("succeeded")
        ]
        if len(actual) != 1 or not actual[0].get("bind"):
            raise DemonstrationError("A reference call must commit one successful public result binding.")
        last_binding = str(actual[0]["bind"])
        if last_binding not in binding_table(replay.env.execution_state):
            raise DemonstrationError("A reference call's returned binding was not committed to public state.")
    if last_binding is None:
        raise DemonstrationError("The reference path produced no native target value.")
    replay.step({"kind": "finish", "answer": {"ref": last_binding, "path": ""}})


def _flatten_fingerprint(value: Mapping[str, Any], path: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    flattened = {}
    for key, child in value.items():
        current = (*path, str(key))
        if isinstance(child, Mapping):
            flattened.update(_flatten_fingerprint(child, current))
        else:
            flattened[current] = deepcopy(child)
    return flattened


@dataclass
class NativeProgressSupervisor:
    """Fixed successful-train predicates; no actor input depends on this state."""
    task_id: str
    benchmark: str
    predicates: list[dict[str, Any]]

    @classmethod
    def from_demonstration(cls, demonstration: NativeDemonstration) -> "NativeProgressSupervisor":
        if not demonstration.trajectory.success:
            raise ValueError("Only a complete native-successful demonstration defines training progress.")
        events = demonstration.trajectory.events
        calls = [
            receipt
            for event in events
            for receipt in event.receipts
            if receipt.get("succeeded") and receipt.get("kind") not in {"discover", "finish"}
        ]
        predicates: dict[str, dict[str, Any]] = {}
        write_apps = set()
        for receipt in calls:
            tool = receipt["tool"]
            writing = receipt.get("method") in {"POST", "PUT", "PATCH", "DELETE"}
            authentication = tool.endswith(".login") or tool.endswith(".logout")
            if writing and not authentication:
                write_apps.add(tool.split(".", 1)[0])
                continue
            args = {
                key: deepcopy(value)
                for key, value in receipt.get("arguments", {}).items()
                if semantic_key(key) != "access_token"
            }
            predicate = {
                "kind": "native_value",
                "tool": tool,
                "arguments": args,
                "authentication": authentication,
                "output": None if authentication else deepcopy(receipt["output"]),
            }
            predicates[canonical(predicate)] = predicate
        before = _flatten_fingerprint(demonstration.initial_fingerprint)
        after = _flatten_fingerprint(demonstration.target_fingerprint)
        for path, target in after.items():
            if before.get(path) == target:
                continue
            if not any(app in path or any(part.startswith(app + ".") for part in path) for app in write_apps):
                continue
            predicate = {"kind": "current_model_hash", "path": list(path), "target": target}
            predicates[canonical(predicate)] = predicate
        fixed = list(predicates.values())
        if not fixed:
            raise DemonstrationError("The successful reference yielded no task-related progress predicates.")
        weight = 1.0 / len(fixed)
        for predicate in fixed:
            predicate["weight"] = weight
        return cls(
            demonstration.task_id,
            str(events[0].generation.metadata["program_context"]["benchmark"]),
            fixed,
        )

    def score(self, env: Any) -> float:
        state = env.execution_state
        if state is None or state.task_id != self.task_id:
            raise ValueError("Train progress must match the current registered task state.")
        receipts = state.receipts
        fingerprint = (
            env.training_state_fingerprint()
            if any(item["kind"] == "current_model_hash" for item in self.predicates)
            else {}
        )
        fulfilled = 0.0
        for predicate in self.predicates:
            satisfied = False
            if predicate["kind"] == "current_model_hash":
                current: Any = fingerprint
                for part in predicate["path"]:
                    if not isinstance(current, Mapping) or part not in current:
                        current = None
                        break
                    current = current[part]
                satisfied = current == predicate["target"]
            else:
                for receipt in receipts:
                    if not receipt.get("succeeded") or receipt["tool"] != predicate["tool"]:
                        continue
                    args = {
                        key: value
                        for key, value in receipt.get("arguments", {}).items()
                        if semantic_key(key) != "access_token"
                    }
                    if canonical(args) != canonical(predicate["arguments"]):
                        continue
                    if predicate["authentication"]:
                        output = receipt.get("output")
                        satisfied = (
                            isinstance(output, Mapping)
                            and isinstance(output.get("access_token"), str)
                            and bool(output["access_token"])
                        )
                    else:
                        satisfied = canonical(receipt.get("output")) == canonical(predicate["output"])
                    if satisfied:
                        break
            if satisfied:
                fulfilled += predicate["weight"]
        return min(1.0, max(0.0, fulfilled))

    def definition(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "benchmark": self.benchmark,
            "visibility": "train_only",
            "source": "successful_native_training_demonstration",
            "effect_granularity": "current_changed_model_hash",
            "predicates": deepcopy(self.predicates),
        }


def collect_native_demonstrations(
    policy: Any,
    environment_factory: Callable[..., Any],
    tasks: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    meter: BudgetMeter,
    output_dir: str | Path,
) -> tuple[list[NativeDemonstration], dict[str, NativeProgressSupervisor], dict[str, Any]]:
    """One fixed full reference attempt per registered train task, without gates."""
    from .native_training import _limits, _meters, _preserve_actor_rng, _write_json, public_trace
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    demonstrations, supervisors, records = [], {}, []
    for index, task in enumerate(tasks):
        if task.get("split") != "train" or "environment_seed" not in task:
            raise ValueError("Demonstrations require explicit registered train IDs and fixed environment seeds.")
        task_id, seed = str(task["task_id"]), int(task["environment_seed"])
        record: dict[str, Any] = {
            "task_id": task_id,
            "environment_seed": seed,
            "visibility": "train_only",
            "reference_success": None,
            "demo_success": False,
            "actor_success_count": 0,
            "attempts": 1,
            "status": "collecting",
        }
        reference_env = environment_factory(f"reference-{index:03d}", reference_execution=True)
        replay_env = None
        replay = None
        initial_fingerprint: dict[str, Any] = {}
        reference_receipts: list[dict[str, Any]] = []
        episode_meter = BudgetMeter(_limits(config), parent=meter)
        try:
            with _meters(policy, reference_env, episode_meter), _preserve_actor_rng():
                reference_env.reset(task_id, seed)
                reference = reference_env.training_reference(task_id)
                record["reference_id"] = reference["reference_id"]
                if reference_env.benchmark == "appworld":
                    initial_fingerprint = reference_env.training_state_fingerprint()
                    capture = reference_env.execute_training_reference(
                        reference["reference"]["compiled_solution_code"] + "\nsolution(apis, requester)\n"
                    )
                    record["reference_success"] = bool(capture["success"])
                    reference_receipts = capture["receipts"]
                    _write_json(output / "private_reference" / f"task-{index:03d}.json", capture)
                    if not capture["success"]:
                        raise DemonstrationError("Official compiled reference did not achieve native success.")
            reference_env.close()
            reference_env = None
            replay_env = environment_factory(f"demonstration-{index:03d}", reference_execution=False)
            replay_meter = BudgetMeter(_limits(config), parent=meter)
            with _meters(policy, replay_env, replay_meter), _preserve_actor_rng():
                context = replay_env.reset(task_id, seed)
                if replay_env.benchmark == "appworld":
                    initial_fingerprint = replay_env.training_state_fingerprint()
                replay = _PublicReplay(policy, replay_env, task, context, config, replay_meter)
                if replay_env.benchmark == "appworld":
                    _replay_appworld(replay, reference_receipts)
                else:
                    _replay_planbench(replay, reference["reference"])
                trajectory = replay.finish_record()
                if not trajectory.success:
                    raise DemonstrationError("Full public reference replay did not pass the native terminal evaluator.")
                target_fingerprint = (
                    replay_env.training_state_fingerprint() if replay_env.benchmark == "appworld" else {}
                )
                demonstration = NativeDemonstration(
                    task_id,
                    seed,
                    trajectory,
                    reference["reference_id"],
                    initial_fingerprint,
                    target_fingerprint,
                )
                supervisor = NativeProgressSupervisor.from_demonstration(demonstration)
                demonstrations.append(demonstration)
                supervisors[task_id] = supervisor
                record.update(
                    status="native_successful_demonstration",
                    demo_success=True,
                    reference_success=True,
                    steps=len(trajectory.events),
                )
                _write_json(output / "private_progress" / f"task-{index:03d}.json", supervisor.definition())
        except (DemonstrationError, BudgetExhausted) as error:
            record.update(
                status="reference_data_failure",
                failure_type=type(error).__name__,
                reason=str(error),
            )
        finally:
            if replay is not None:
                trajectory = replay.finish_record()
                if trajectory.termination == "unfinished":
                    trajectory.termination = "reference_reconstruction_stopped"
                _write_json(output / "public_traces" / f"task-{index:03d}.json", public_trace(trajectory))
            if reference_env is not None:
                reference_env.close()
            if replay_env is not None:
                replay_env.close()
            records.append(record)
            _write_json(output / "demonstration_records.json", records)
    summary = {
        "fixed_reference_attempts": len(tasks),
        "successful_demonstrations": len(demonstrations),
        "failed_reference_attempts": len(tasks) - len(demonstrations),
        "success_counting_unit": "teacher_only",
        "actor_terminal_successes": 0,
        "records": records,
        "usage": meter.snapshot(),
    }
    _write_json(output / "demonstration_summary.json", summary)
    return demonstrations, supervisors, summary


def train_warmstart(
    policy: Any,
    demonstrations: Sequence[NativeDemonstration],
    config: Mapping[str, Any],
    meter: BudgetMeter,
    output_dir: str | Path,
    *,
    seed: int,
) -> dict[str, Any]:
    """Full-trajectory action supervision with the same constrained likelihood."""
    from .native_training import _append_json, _step_optimizer, _write_json
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    epochs, learning_rate = int(config["warmstart_epochs"]), float(config["warmstart_lr"])
    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("Warmstart epochs and learning rate must be positive.")
    if not demonstrations:
        raise DemonstrationError("The fixed full-reference attempts produced no native-successful SFT dataset.")
    if any(not item.trajectory.success for item in demonstrations):
        raise ValueError("SFT consumes only complete native-successful training demonstrations.")
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    initial = [parameter.detach().clone() for parameter in parameters]
    old_meter = policy.meter
    updates, trained_examples, total_loss = 0, 0, 0.0
    randomizer = random.Random(seed)
    policy.meter = meter
    policy.train()
    try:
        for epoch in range(epochs):
            ordering = list(demonstrations)
            randomizer.shuffle(ordering)
            for demonstration in ordering:
                events = demonstration.trajectory.events
                denominator = sum(
                    len(event.generation.response_ids)
                    + int(event.generation.metadata.get("gate_active", False))
                    for event in events
                )
                if denominator == 0:
                    raise DemonstrationError("A complete demonstration contains no teacher target tokens.")
                optimizer.zero_grad(set_to_none=True)
                started, loss_total = monotonic_seconds(), 0.0
                try:
                    for event in events:
                        loss = -policy.score(event.generation, role="warmstart_supervision") / denominator
                        if not torch.isfinite(loss.detach()).all():
                            raise FloatingPointError("Nonfinite native-r3 supervised likelihood.")
                        loss_total += float(loss.detach().item())
                        loss.backward()
                        del loss
                    diagnostics = _step_optimizer(policy, optimizer, meter, config, started)
                except BaseException:
                    optimizer.zero_grad(set_to_none=True)
                    raise
                updates += int(diagnostics["parameters_changed"])
                trained_examples += len(events)
                total_loss += loss_total
                _append_json(
                    output / "warmstart.jsonl",
                    {
                        "epoch": epoch,
                        "task_id": demonstration.task_id,
                        "loss": loss_total,
                        "action_examples": len(events),
                        "tokens": denominator,
                        **diagnostics,
                    },
                )
    finally:
        policy.meter = old_meter
    delta = (
        sum(
            float((parameter.detach() - before).float().square().sum().item())
            for parameter, before in zip(parameters, initial)
        )
        ** 0.5
    )
    del initial
    policy.freeze_warmstart_reference()
    checkpoint = output / "warmstart_adapter"
    policy.save_adapter(checkpoint)
    policy.eval()
    summary = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "successful_demonstrations": len(demonstrations),
        "action_examples_processed": trained_examples,
        "parameter_change_steps": updates,
        "parameter_delta_l2": delta,
        "summed_loss": total_loss,
        "status": "supervision_completed",
        "reference_adapter": "warmstart_reference",
        "checkpoint": str(checkpoint),
        "actor_contract": PROGRAM_VERSION,
        "reference_visibility": "train_only",
        "usage": meter.snapshot(),
    }
    _write_json(output / "warmstart_summary.json", summary)
    return summary
