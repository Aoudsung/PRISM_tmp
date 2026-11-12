"""Gate 0 source-aligned execution of the shared public program IR.

Callbacks supply only the current public prefix, real accounted semantic calls,
and budget checks. This module has no model, environment or posterior imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Callable, Mapping

from .native_program_ir import Node, ProgramCandidate, Source, check_type


@dataclass
class ExecutionResult:
    exit: str
    actions: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    values: dict[str, Any] = field(default_factory=dict)
    semantic_batches: int = 0
    unknown_reason: str | None = None


class ProgramYield(RuntimeError):
    pass


def render_action(template: str, values: Mapping[str, Any]) -> str:
    def substitute(match):
        name = match.group(1)
        if name not in values:
            raise ProgramYield("MissingBinding")
        value = values[name]
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            raise ProgramYield("InvalidBindingType")
        return str(value)
    if template.startswith("{"):
        def quoted(match):
            name = match.group(1)
            if name not in values:
                raise ProgramYield("MissingBinding")
            return json.dumps(values[name], ensure_ascii=False, allow_nan=False)
        template = re.sub(r'"\$\{([a-zA-Z0-9_]+)\}"', quoted, template)
    action = re.sub(r"\$\{([a-zA-Z0-9_]+)\}", substitute, template)
    if "${" in action:
        raise ProgramYield("MalformedBinding")
    if action.startswith("{"):
        try:
            value = json.loads(action)
        except ValueError as exc:
            raise ProgramYield("InvalidBoundJSON") from exc
        if not isinstance(value, dict) or set(value) != {"name", "arguments"}:
            raise ProgramYield("InvalidBoundToolCall")
    return action


def execute(candidate: ProgramCandidate, sources: Mapping[str, Source], *,
            resolve_source: Callable[[Source, list[dict[str, Any]]], Any],
            call_native: Callable[[str, str, list[dict[str, Any]]], Any],
            action_is_bound: Callable[[str, list[dict[str, Any]]], bool],
            semantic_batch: Callable[[dict[str, Any], list[dict[str, Any]], str], dict[str, Any]],
            budget_available: Callable[[], bool]) -> ExecutionResult:
    result = ExecutionResult("UnknownYield")

    def resolve(source_id):
        source = sources[source_id]
        if source.event_index > len(result.history):
            raise ProgramYield("FutureSource")
        value = resolve_source(source, result.history)
        if value is None:
            raise ProgramYield("InvalidatedSource")
        try:
            check_type(value, source.value_type)
        except ValueError as exc:
            raise ProgramYield("InvalidatedSourceType") from exc
        return value

    def needed(nodes: tuple[Node, ...]) -> set[str]:
        reads = set()
        for node in nodes:
            reads.update(node.reads + node.cache_refs)
            if node.op == "If3" and node.reads and result.values.get(node.reads[0]) in {"T", "F", "U"}:
                reads.update(needed(node.branches[result.values[node.reads[0]]]))
            else:
                for branch in node.branches.values():
                    reads.update(needed(branch))
            reads.update(needed(node.body))
        return reads

    def run(nodes: tuple[Node, ...]) -> None:
        for index, node in enumerate(nodes):
            if not budget_available():
                raise ProgramYield("BudgetUnknownYield")
            if node.op == "Bind":
                value = resolve(node.sources[0])
                check_type(value, node.args["type"])
                result.values[node.defines[0]] = value
            elif node.op == "Frontier":
                # A snapshot of the currently disclosed finite set, with no hidden pagination.
                result.values[node.defines[0]] = [resolve(source) for source in node.sources]
            elif node.op == "Loop":
                frontier = result.values.get(node.reads[0])
                if not isinstance(frontier, list):
                    raise ProgramYield("UnknownFrontier")
                for item in tuple(frontier):
                    if not budget_available():
                        raise ProgramYield("BudgetUnknownYield")
                    result.values[node.args["item_value"]] = item
                    run(node.body)
                result.values.pop(node.args["item_value"], None)
            elif node.op == "Predicate":
                values = [resolve(source) for source in node.sources]
                operator = node.args["operator"]
                label = "U"
                if operator == "exists" and len(values) == 1:
                    label = "T"
                elif operator == "equals" and len(values) == 2:
                    label = "T" if type(values[0]) is type(values[1]) and values[0] == values[1] else "F"
                elif operator == "contains" and len(values) == 2 and all(isinstance(value, str) for value in values):
                    label = "T" if values[1] in values[0] else "F"
                elif operator == "number_le" and len(values) == 2 and all(type(value) in (int, float) for value in values):
                    label = "T" if values[0] <= values[1] else "F"
                result.values[node.defines[0]] = label
            elif node.op == "If3":
                label = result.values.get(node.reads[0], "U")
                if label not in {"T", "F", "U"}:
                    label = "U"
                run(node.branches[label])
            elif node.op == "SemanticBatch":
                live = needed(nodes[index + 1:])
                fields = {name: spec for name, spec in node.args.get("fields", {}).items() if name in live}
                if not fields:
                    continue
                values = semantic_batch(fields, list(result.history), candidate.semantic_schema_version)
                if set(values) != set(fields):
                    raise ProgramYield("IncompleteSemanticBatch")
                for name, value in values.items():
                    check_type(value, fields[name]["type"])
                    result.values[name] = value
                result.semantic_batches += 1
            elif node.op == "Call":
                action = render_action(node.args["template"], result.values)
                if not action_is_bound(action, result.history):
                    raise ProgramYield("MissingCurrentActionBinding")
                observation = call_native(action, node.node_id, list(result.history))
                result.actions.append(action)
                result.history.append({"action": action, "observation": observation})
                for value_id in node.defines:
                    result.values[value_id] = observation
            elif node.op == "Return":
                result.exit = "Return"
                raise ProgramYield("Return")
            elif node.op == "Yield":
                raise ProgramYield(str(node.args.get("reason", "ExplicitYield")))
            else:
                raise ValueError("Unregistered executable operation")
    try:
        for name, source_id in candidate.parameter_sources.items():
            value = resolve(source_id)
            check_type(value, candidate.entry_parameters[name])
            result.values[name] = value
        run(candidate.ast)
        result.exit = "Continue"
    except ProgramYield as exc:
        if result.exit != "Return":
            result.exit = "UnknownYield"
            result.unknown_reason = str(exc)
    return result
