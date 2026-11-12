"""Real public-entry execution for R3 B/R/P, with current-state bindings."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Callable

from .native_core import BudgetExhausted, BudgetMeter, Resources, monotonic_seconds
from .native_program_execution import ProgramYield
from .native_program_ir import check_type
from .native_trace_artifacts import canonical_bytes, public_observation, write_json_exclusive
from .program_hypothesis_r3 import (Documents, ProgramProposal, RepairContext, TrialRecord,
    WorkLeases, append, measured, native_catalog, pointer, select, short_index, strip_catalog)


def type_schema(name: str) -> dict[str, Any]:
    return {"Text": {"type": "string"}, "Ref": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
            "Number": {"type": "number"}, "Truth": {"type": "string", "enum": ["T", "F", "U"]},
            "Bool": {"type": "boolean"}, "RefSet": {"type": "array", "maxItems": 100,
                "items": {"anyOf": [{"type": "string"}, {"type": "integer"}]}}}[name]


class PublicRuntime:
    def __init__(self, policy, env, meter: BudgetMeter, leases: WorkLeases,
                 output: Path, record: TrialRecord, context: dict[str, Any],
                 experience: RepairContext | None, seed: int):
        self.policy, self.env, self.meter, self.leases = policy, env, meter, leases
        self.output, self.record, self.seed = output, record, seed
        self.goal = str(context["instruction"])
        self.observation = str(context["observation"])
        self.history: list[dict[str, Any]] = []
        self.values: dict[str, Any] = {}
        self.experience = experience
        self.done = False
        self.last_step = None
        self.serial = 0
        self.fuel = 0
        self.in_program = False
        # The empty-experience initial distribution has exactly these inputs in
        # all arms. Persist the actual reset inputs, then compare matched arms.
        record.initial_policy = {"goal": self.goal,
            "observation": strip_catalog(public_observation(self.observation)),
            "tool_index": short_index(native_catalog(self.schema())) if env is not None else [],
            "seed": seed, "actor_entry": "PublicRuntime.actor", "frozen_raw_base": True,
            "sampling": "temperature_1_top_p_1_tool_then_selected_native_arguments"} if env is not None else {}

    @property
    def costs_path(self):
        return self.output / "outer_costs.jsonl"

    def available(self) -> bool:
        return (not self.done and not self.meter.exploration_stopped and
                self.meter.remaining_env_steps > 0 and
                self.meter.elapsed_seconds < self.meter.limits.max_wall_seconds)

    def schema(self):
        return self.env.schema_for_history([{"role": "user", "content": self.observation}])

    def roots(self) -> dict[str, Any]:
        return {"goal": self.goal, "observation": strip_catalog(public_observation(self.observation)),
                "last_return": self.history[-1]["observation"] if self.history else None,
                "history": self.history, "values": self.values}

    def work_view(self):
        with measured(self.costs_path, self.record.trial_id, "projection"):
            docs = Documents()
            history = [{"action": row["action"], "observation": docs.add(row["observation"])}
                       for row in self.history]
            view = {"goal": docs.add(self.goal),
                    "observation": docs.add(strip_catalog(public_observation(self.observation))),
                    "history": history, "documents": docs.documents,
                    "tool_index": short_index(native_catalog(self.schema()))}
        return view

    def model_call(self, messages, role, max_output=512, schema=None, greedy=False):
        import torch
        from jsonschema import Draft202012Validator
        self.meter._ensure()
        tokens = len(self.policy._tokenize_messages(messages))
        if tokens + max_output > self.policy.max_context_tokens:
            raise ProgramYield("ContextLimit")
        self.meter.ensure_model_call(tokens, max_output, role)
        remaining_wall = self.meter.limits.max_wall_seconds - self.meter.elapsed_seconds
        lease = self.leases.reserve(self.record.trial_id, role, Resources(remaining_wall, tokens + max_output, 0))
        before_tokens = self.meter.total_model_tokens
        started = monotonic_seconds()
        try:
            contract = None
            if schema is not None:
                with measured(self.costs_path, self.record.trial_id, "schema_contract_hash"):
                    contract = self.policy._text_decoder_engine().native_contract({
                        "native_action_schemas_by_mode": {"I": schema}, "mode_actions": {"I": [], "E": []}}, None)
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                # Auxiliary work cannot advance the actor's sampling RNG stream.
                offset = 900000 if "semantic" in role else 0
                torch.manual_seed(self.seed + len(self.history) * 13 + offset + self.serial)
                torch.cuda.manual_seed_all(self.seed + len(self.history) * 13 + offset + self.serial)
                with self.policy.frozen_reference():
                    generation = self.policy.complete(messages, role=role, max_new_tokens=max_output,
                                                      greedy=greedy, program_contract=contract)
            self.serial += 1
            append(self.output / "raw_generations.jsonl", {"trial_id": self.record.trial_id,
                "role": role, "text": generation.text, "metadata": generation.metadata,
                "prompt_token_ids": generation.prompt_ids, "response_token_ids": generation.response_ids})
            if generation.metadata.get("output_limit_reached") or generation.metadata.get("wall_deadline_reached"):
                raise ProgramYield("IncompleteGeneration")
            value = json.loads(generation.text)
            if schema is not None:
                Draft202012Validator(schema).validate(value)
            return value, generation
        finally:
            self.leases.settle(lease, Resources(monotonic_seconds() - started,
                self.meter.total_model_tokens - before_tokens, 0))

    def actor(self) -> tuple[str, list[str]]:
        catalog = native_catalog(self.schema())
        if not catalog:
            raise ProgramYield("NoPublicAction")
        view = self.work_view()
        payload = {"current": view}
        if self.experience is not None:
            payload["experience"] = self.experience.work_view
            payload["experience_scope"] = ("same source posthoc repair" if self.record.target_identity == "source_repair"
                else "different source instance; use only CURRENT target public facts for arguments")
        messages = [{"role": "system", "content":
            "Select one generic native tool name to advance the current goal. The complete short public tool index is supplied; "
            "the selected tool's native argument schema is expanded next. Return only the JSON tool-name string. "
            "Use current public history and available experience; missing evidence calls for information acquisition."},
            {"role": "user", "content": canonical_bytes(payload).decode()}]
        tool, name_gen = self.model_call(messages, "shared.actor.tool", 128,
                                         {"type": "string", "enum": list(catalog)})
        # Exactly the selected native schema, without branch-order coupling.
        messages = [{"role": "system", "content":
            "Generate arguments for the selected native tool using CURRENT target goal and public observations. "
            "Return only one JSON arguments object matching the expanded schema. Never copy entity IDs from a different instance."},
            {"role": "user", "content": canonical_bytes({**payload, "selected_tool": tool,
                    "native_schema": catalog[tool]["arguments"]}).decode()}]
        arguments, arg_gen = self.model_call(messages, "shared.actor.arguments", 512, catalog[tool]["arguments"])
        action = self.render(tool, arguments, catalog)
        return action, [name_gen.metadata["model_call_id"], arg_gen.metadata["model_call_id"]]

    @staticmethod
    def render(tool, arguments, catalog):
        from jsonschema import Draft202012Validator
        if tool not in catalog:
            raise ProgramYield("UndisclosedCurrentTool")
        selected = catalog[tool]
        Draft202012Validator(selected["arguments"]).validate(arguments)
        if selected["kind"] == "json":
            return canonical_bytes({"name": tool, "arguments": arguments}).decode()
        if selected["kind"] == "search":
            return "search[" + arguments["query"] + "]"
        return arguments["command"]

    def snapshot(self, kind):
        started = monotonic_seconds()
        with measured(self.costs_path, self.record.trial_id, "snapshot_export", nested=True):
            snapshot = self.env.save_snapshot()
        path = self.output / "snapshots" / self.record.trial_id / (str(len(self.record.snapshots)) + ".json")
        with measured(self.costs_path, self.record.trial_id, "snapshot_serialization"):
            write_json_exclusive(path, snapshot)
        self.record.snapshots.append({"kind": kind, "path": str(path), "history_length": len(self.history),
                                      "wall_seconds": monotonic_seconds() - started})
        return snapshot

    def restore(self, snapshot, reason):
        with measured(self.costs_path, self.record.trial_id, "restore_outer", nested=True):
            self.env.restore_snapshot(snapshot)
        self.record.recoveries.append({"kind": "restore", "reason": reason,
                                       "history_length": len(self.history)})

    def native(self, action, origin, node=None, call_ids=None):
        if not self.available():
            raise ProgramYield("BudgetOrNativeTerminal")
        self.meter.ensure_env_steps(1)
        before = self.observation
        snapshot = None
        # A real recovery point is saved once per program call; fallback effects
        # are never silently rolled back and retries consume the same ledger.
        if origin == "program":
            snapshot = self.snapshot("before_program_call")
        upper_wall = self.meter.limits.max_wall_seconds - self.meter.elapsed_seconds
        upper_steps = min(self.meter.remaining_env_steps,
                          2 + len(snapshot.get("actions", [])) if snapshot is not None else 1)
        lease = self.leases.reserve(self.record.trial_id, "native_call", Resources(upper_wall, 0, upper_steps))
        started, before_steps = monotonic_seconds(), self.meter.env_steps
        try:
            for attempt in range(2):
                try:
                    step = self.env.step_native({"kind": "native_text", "text": action})
                    break
                except BudgetExhausted:
                    raise
                except Exception as exc:
                    append(self.output / "native_failures.jsonl", {"trial_id": self.record.trial_id,
                        "action": action, "attempt": attempt, "error": f"{type(exc).__name__}: {exc}"})
                    self.record.recoveries.append({"kind": "native_exception", "attempt": attempt,
                                                  "action": action, "error": str(exc)})
                    if attempt or snapshot is None:
                        raise ProgramYield("NativeExecutionError") from exc
                    self.restore(snapshot, "native_exception_retry")
                    self.record.recoveries.append({"kind": "retry", "action": action})
            self.observation, self.done, self.last_step = step.observation, bool(step.done), step
            public_return = strip_catalog(public_observation(step.observation))
            self.history.append({"action": action, "observation": public_return})
            event = {"index": len(self.record.events), "origin": origin, "node": node,
                     "model_call_ids": call_ids or [], "action": action,
                     "observation_before": before, "observation_after": step.observation,
                     "native_receipt": asdict(step), "history_index": len(self.history) - 1}
            self.record.events.append(event)
            # Incremental durable write retains failures even if an environment exits.
            append(self.output / "trial_events" / (self.record.trial_id + ".jsonl"), event)
            return public_return
        finally:
            self.leases.settle(lease, Resources(monotonic_seconds() - started, 0,
                                                self.meter.env_steps - before_steps))

    def resolve_expression(self, value):
        if isinstance(value, dict) and set(value) == {"var"}:
            name, _, suffix = value["var"].partition("/")
            if name not in self.values or self.values[name] is None:
                raise ProgramYield("MissingCurrentBinding:" + name)
            return pointer(self.values[name], "/" + suffix if suffix else "")
        if isinstance(value, dict):
            return {key: self.resolve_expression(child) for key, child in value.items()}
        if isinstance(value, list):
            return [self.resolve_expression(child) for child in value]
        raise ProgramYield("UngroundedArgument")

    def semantic(self, fields):
        props = {name: {"anyOf": [type_schema(spec["type"]), {"type": "null"}]}
                 for name, spec in fields.items()}
        schema = {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
        roots = self.roots()
        messages = [{"role": "system", "content":
            "Bind the entire requested semantic batch from CURRENT target public goal, observation, history and runtime values. "
            "Use set-level reasoning for finite currently disclosed collections. Return exactly the requested JSON fields. "
            "Missing truth evidence is U; missing other data is null. Native terminal success cannot be inferred. "
            "A command must be one currently disclosed complete native command. No source-instance lookup is available."},
            {"role": "user", "content": canonical_bytes({"current": self.work_view(), "values": self.values,
                "fields": fields, "schema": schema}).decode()}]
        values, generation = self.model_call(messages, "program.semantic_batch", 1024, schema, greedy=True)
        row = {"fields": fields, "values": values, "call_id": generation.metadata["model_call_id"],
               "history_length": len(self.history), "current_public_roots": roots}
        self.record.semantic_batches.append(row)
        for name, spec in fields.items():
            value = values[name]
            if value is None and spec["type"] == "Truth":
                value = "U"
            if value is None:
                raise ProgramYield("UnknownSemanticBinding:" + name)
            check_type(value, spec["type"])
            self.values[name] = value
            self.record.bindings.append({"value_id": name, "value": value, "origin": "semantic_batch",
                "call_id": generation.metadata["model_call_id"], "history_length": len(self.history)})

    def program(self, proposal: ProgramProposal):
        from jsonschema import ValidationError
        self.fuel = proposal.fuel
        self.in_program = True
        def block(nodes):
            for node in nodes:
                if self.fuel <= 0 or not self.available():
                    raise ProgramYield("FuelOrBudgetOrTerminal")
                self.fuel -= 1
                op, args = node["op"], node.get("args", {})
                if op == "Bind":
                    ref = args["source"]
                    try:
                        value = select(self.roots()[ref["root"]], ref)
                        check_type(value, args["type"])
                    except (KeyError, TypeError, ValueError, IndexError) as exc:
                        raise ProgramYield("InvalidatedCurrentSource") from exc
                    self.values[args["name"]] = value
                    self.record.bindings.append({"value_id": args["name"], "value": value,
                        "origin": "current_public_source", "reference": ref, "history_length": len(self.history)})
                elif op == "SemanticBatch":
                    self.semantic(args["fields"])
                elif op == "Frontier":
                    value = self.resolve_expression(args["items"])
                    check_type(value, "RefSet")
                    self.values[args["name"]] = list(dict.fromkeys(value))
                    self.record.bindings.append({"value_id": args["name"], "value": self.values[args["name"]],
                        "origin": "current_finite_frontier", "complete_collection": False,
                        "history_length": len(self.history)})
                elif op == "Loop":
                    frontier = self.values.get(args["frontier"])
                    check_type(frontier, "RefSet")
                    for index, item in enumerate(tuple(frontier)):
                        if index >= args["fuel"] or self.fuel <= 0:
                            raise ProgramYield("LoopFuelExhausted")
                        self.fuel -= 1
                        self.values[args["item"]] = item
                        block(node.get("body", []))
                    self.values.pop(args["item"], None)
                elif op == "If3":
                    label = self.values.get(args["predicate"], "U")
                    if label not in {"T", "F", "U"}:
                        label = "U"
                    block(node["branches"][label])
                elif op == "Call":
                    tool = args["tool"] if isinstance(args["tool"], str) else self.resolve_expression(args["tool"])
                    arguments = self.resolve_expression(args["arguments"])
                    action = self.render(tool, arguments, native_catalog(self.schema()))
                    self.native(action, "program", node["id"])
                elif op == "Return":
                    raise ProgramYield("Return")
                elif op == "Yield":
                    raise ProgramYield(args.get("reason", "ExplicitYield"))
        try:
            block(proposal.ast)
            reason = "ProgramEnd"
        except (ProgramYield, BudgetExhausted, ValueError, KeyError, TypeError, IndexError, ValidationError) as exc:
            reason = str(exc)
        finally:
            self.in_program = False
        self.record.local_exits.append({"reason": reason, "history_length": len(self.history),
                                        "fuel_remaining": self.fuel})

    def finish(self):
        while self.available():
            try:
                action, calls = self.actor()
                if self.record.arm == "P":
                    self.record.fallback.append({"history_length": len(self.history), "action": action,
                                                  "model_call_ids": calls})
                self.native(action, "frozen_actor", call_ids=calls)
            except (ProgramYield, BudgetExhausted) as exc:
                self.record.termination = str(exc)
                break
        if self.record.termination == "unfinished":
            self.record.termination = "native_terminal" if self.done else "resource_limit"
        with measured(self.costs_path, self.record.trial_id, "native_terminal_evaluation"):
            official = dict(self.env.finalize_evaluation())
        self.record.native_terminal = {"done": self.done, "official": official,
            "last_step_success": bool(self.last_step is not None and self.last_step.success)}
        # Unknown/nonterminal value is exactly zero, with no progress proxy.
        self.record.terminal_y = float(self.done and bool(official.get("success", False)))
        self.record.utility = self.record.terminal_y * max(0.0,
            1 - self.meter.elapsed_seconds / self.meter.limits.max_wall_seconds)
        self.record.costs = self.meter.snapshot()
        self.record.costs["online_rho_satisfied"] = False
