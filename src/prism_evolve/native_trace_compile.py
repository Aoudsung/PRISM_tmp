"""Deterministic Gate 0 compilation with at most two frozen-7B annotations.

Source discovery never depends on terminal success. Models select stable source
IDs and controlled labels; they cannot supply executable ASTs or repair calls.
The original prompts, outputs and every rejection remain immutable artifacts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from .native_core import BudgetExhausted, BudgetLimits, BudgetMeter, monotonic_seconds
from .native_program_ir import (
    Effect, IR_VERSION, Node, ProgramCandidate, SEMANTIC_SCHEMA_VERSION, Source,
    liveness, validate,
)
from .native_trace_artifacts import canonical_bytes, sha256, write_json_exclusive

ANNOTATION_SCHEMA = {
    "parameters": "[{name,type:Text|Ref|Number,source_id}]",
    "bindings": "[{event_index,source_id,parameter,action_substring,argument_path(optional JSON pointer /arguments/field)}]", 
    "structures": "[{op:Frontier|Loop|If3,event_index,source_ids,predicate_id,body_events}]",
}
PREDICATE_SCHEMA = {
    "predicates": "[{predicate_id,operator:exists|equals|contains|number_le,source_ids,label:T|F|U}]",
}


@dataclass(frozen=True)
class CompileMaterial:
    public_view: dict[str, Any]
    documents: dict[str, str]
    sources: dict[str, Source]
    input_payload: dict[str, Any]


def executed_action(event: Mapping[str, Any]) -> str:
    action = event["action"]
    if action.get("kind") != "native_text" or not isinstance(action.get("text"), str):
        raise ValueError("Source event has no executable native text action")
    return action["text"]


def _argument_values(action: str) -> list[str]:
    try:
        value = json.loads(action)
    except ValueError:
        words = re.findall(r"[\w.-]+", action)
        # Pure lexical intersection candidates; no domain-specific entity rules.
        return sorted({" ".join(words[i:i + size]) for size in range(1, 6)
                       for i in range(max(0, len(words) - size + 1))})
    leaves = []
    def visit(item):
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, (str, int, float)) and not isinstance(item, bool):
            leaves.append(item if isinstance(item, str) else canonical_bytes(item).decode("utf-8"))
    visit(value.get("arguments", {}) if isinstance(value, dict) else {})
    return sorted(set(leaves))


def build_material(view: dict[str, Any], end_event_exclusive: int) -> CompileMaterial:
    """Keep every native effect in the source prefix and exact original span maps.

    Complete native schemas are retained. Oversized closed inputs fail the
    registered context ceiling; this function never silently trims a catalog,
    a native effect predecessor or a public return needed by a binding.
    """
    events = view["events"][:end_event_exclusive]
    if not events:
        raise ValueError("Empty source window")
    documents = {"goal": view["goal"]}
    sources: dict[str, Source] = {}
    source_rows, public_events = [], []
    all_atoms = sorted(set(atom for event in events for atom in _argument_values(executed_action(event))))

    def spans(document_id: str, text: str, event_index: int, kind: str) -> list[str]:
        documents[document_id] = text
        ids = []
        values = [text] + [atom for atom in all_atoms if atom and atom != text and atom in text]
        for atom in values:
            offset = text.find(atom)
            start = len(text[:offset].encode("utf-8"))
            source_id = document_id + f":{start}:{start + len(atom.encode('utf-8'))}"
            value, value_type = atom, "Text"
            if atom != text and re.fullmatch(r"-?(?:0|[1-9]\d*)(?:\.\d+)?", atom):
                value, value_type = json.loads(atom), "Number"
            source = Source(source_id, event_index, kind, value_type, value, document_id,
                            start, start + len(atom.encode("utf-8")))
            sources[source_id] = source
            ids.append(source_id)
        return ids

    spans("goal", view["goal"], -1, "goal")
    for index, event in enumerate(events):
        before = event["obs_before"]
        ids = spans(f"e{index:05d}.before.native", str(before["native_observation"]), index, "public_return")
        actions = before["classic_text_action_state"].get("available_actions", [])
        for action_index, action in enumerate(actions):
            ids.extend(spans(f"e{index:05d}.before.action{action_index}", action, index, "ActionRef"))
        schema = event["schema"]
        public_events.append({"event_id": event["event_id"], "source_index": index,
            "obs_before_document": f"e{index:05d}.before.native", "action": executed_action(event),
            "actor_response": event["response"],
            "obs_after": event["obs_after"]["native_observation"],
            "current_source_ids": ids, "schema": schema})
    for source in sources.values():
        source.verify(documents)
        source_rows.append(asdict(source))
    payload = {"goal": view["goal"], "events": public_events, "sources": source_rows,
               "source_projection": "contiguous_native_prefix_with_exact_utf8_span_maps",
               "public_contract": view["public_contract"], "ir_version": IR_VERSION}
    return CompileMaterial({**view, "events": events}, documents, sources, payload)


def on_episode_trace_closed(view: dict[str, Any], extract_windows: Callable, *,
                            budget_available: bool) -> dict[str, Any]:
    windows = extract_windows(view)
    unique = {sha256(canonical_bytes(window)): window for window in windows}
    return {"status": "NO_ELIGIBLE_WINDOW" if not unique else
            "READY_FOR_COMPILATION" if budget_available else "DEFERRED_BY_BUDGET",
            "windows": list(unique.values()), "native_success_increment": 0}


def parse_object(text: str, expected_keys: set[str]) -> dict[str, Any]:
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("Output must be exactly the registered JSON object")
    if any(not isinstance(item, list) for item in value.values()):
        raise ValueError("Annotation object fields must be arrays")
    return value


def prove_predicates(labels: Mapping[str, Any], sources: Mapping[str, Source]) -> dict[str, dict[str, Any]]:
    """Independently recompute controlled predicates; absent evidence stays U."""
    predicates = {}
    for row in labels["predicates"]:
        if set(row) != {"predicate_id", "operator", "source_ids", "label"}:
            raise ValueError("Unexpected predicate fields")
        name, operator = row["predicate_id"], row["operator"]
        if not isinstance(name, str) or name in predicates or row["label"] not in {"T", "F", "U"}:
            raise ValueError("Invalid predicate identity or label")
        refs = row["source_ids"]
        if not isinstance(refs, list) or any(ref not in sources for ref in refs):
            raise ValueError("Unknown predicate source")
        values = [sources[ref].value for ref in refs]
        actual = "U"
        if operator == "exists":
            if len(values) == 1:
                actual = "T"  # absence is not derivable from a partial public list
        elif operator == "equals":
            if len(values) == 2:
                actual = "T" if type(values[0]) is type(values[1]) and values[0] == values[1] else "F"
        elif operator == "contains":
            if len(values) == 2 and all(isinstance(value, str) for value in values):
                actual = "T" if values[1] in values[0] else "F"
        elif operator == "number_le":
            if len(values) == 2 and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
                actual = "T" if values[0] <= values[1] else "F"
        else:
            raise ValueError("Unregistered predicate operator")
        if row["label"] != actual:
            raise ValueError("Predicate label lacks matching public evidence")
        predicates[name] = {**row, "proved_label": actual}
    return predicates


def construct_candidate(material: CompileMaterial, annotation: Mapping[str, Any], labels: Mapping[str, Any],
                        *, source_class: str) -> tuple[ProgramCandidate, list[dict[str, Any]]]:
    sources = material.sources
    parameters, parameter_sources = {}, {}
    for row in annotation["parameters"]:
        if set(row) != {"name", "type", "source_id"} or not re.fullmatch(r"[a-z][a-z0-9_]*", row["name"]):
            raise ValueError("Invalid parameter declaration")
        if row["name"] in parameters or row["source_id"] not in sources:
            raise ValueError("Duplicate parameter or missing source")
        parameters[row["name"]] = row["type"]
        parameter_sources[row["name"]] = row["source_id"]
    if not parameters:
        raise ValueError("Literal-only recording is not a parameterized candidate")
    predicates = prove_predicates(labels, sources)
    event_bindings: dict[int, list[dict[str, Any]]] = {}
    for row in annotation["bindings"]:
        if (not {"event_index", "source_id", "parameter", "action_substring"} <= set(row)
                or set(row) - {"event_index", "source_id", "parameter", "action_substring", "argument_path"}):
            raise ValueError("Unexpected binding fields")
        index = row["event_index"]
        if type(index) is not int or not 0 <= index < len(material.public_view["events"]):
            raise ValueError("Binding event outside source window")
        if row["source_id"] not in sources or row["parameter"] not in parameters:
            raise ValueError("Binding lacks a declared parameter/source")
        source = sources[row["source_id"]]
        source_text = source.value if isinstance(source.value, str) else canonical_bytes(source.value).decode("utf-8")
        if source.event_index > index or (source_text != row["action_substring"] and source_class != "INTERFACE_REPAIR_HYPOTHESIS"):
            raise ValueError("Future or fabricated action binding")
        if source.kind not in {"goal", "public_return", "ActionRef"}:
            raise ValueError("Unregistered binding source class")
        event_bindings.setdefault(index, []).append(row)
    if not event_bindings:
        raise ValueError("No grounded parameter consumption")
    nodes = []
    for index, event in enumerate(material.public_view["events"]):
        failed_call = bool(re.search(r"not found|traceback|\berror\b|invalid|missing required|nothing happens|cannot|failed",
                                    str(event["obs_after"]["native_observation"]), re.I))
        source_action = executed_action(event)
        text = source_action
        reads, refs, repaired = [], [], False
        try:
            tool_call = json.loads(source_action)
        except ValueError:
            tool_call = None
        json_bindings = {}
        for binding in event_bindings.get(index, []):
            substring = binding["action_substring"]
            parameter = binding["parameter"]
            entry_source = sources[parameter_sources[parameter]]
            source = sources[binding["source_id"]]
            source_text = source.value if isinstance(source.value, str) else canonical_bytes(source.value).decode("utf-8")
            repaired = repaired or source_text != substring
            if source.value != entry_source.value or entry_source.event_index > 0:
                # Return-bound roles are local ValueIds, never future-valued entry parameters.
                local = f"{parameter}_at_{index}"
                nodes.append(Node(f"bind_{index}_{parameter}", "Bind", Effect.PURE_TOTAL,
                                  defines=(local,), sources=(source.source_id,), args={"type": parameters[parameter]}))
                parameter = local
            pointer = binding.get("argument_path")
            if pointer is not None:
                if (not isinstance(tool_call, dict) or not isinstance(pointer, str)
                        or not re.fullmatch(r"/arguments/[^/]+", pointer)):
                    raise ValueError("Binding requires a direct public JSON argument pointer")
                key = pointer.split("/")[-1].replace("~1", "/").replace("~0", "~")
                if key in json_bindings:
                    raise ValueError("Duplicate argument binding")
                arguments = tool_call.get("arguments", {})
                original = arguments.get(key)
                original_text = original if isinstance(original, str) else canonical_bytes(original).decode("utf-8")
                if key not in arguments or original_text != substring:
                    raise ValueError("Argument pointer differs from the recorded action")
                json_bindings[key] = (parameter, source.value)
            else:
                if not substring or text.count(substring) != 1:
                    raise ValueError("Text binding must identify one exact action span")
                text = text.replace(substring, "${" + parameter + "}", 1)
            reads.append(parameter)
            refs.append(source.source_id)
        if json_bindings:
            if any(binding.get("argument_path") is None for binding in event_bindings.get(index, [])):
                raise ValueError("JSON bindings must all use explicit argument pointers")
            bound = {**tool_call, "arguments": dict(tool_call["arguments"])}
            for key, (parameter, value) in json_bindings.items():
                bound["arguments"][key] = "${" + parameter + "}"
            text = canonical_bytes(bound).decode("utf-8")
            for key, (parameter, value) in json_bindings.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    text = text.replace(json.dumps("${" + parameter + "}"), "${" + parameter + "}")
        if failed_call and not (source_class == "INTERFACE_REPAIR_HYPOTHESIS" and repaired):
            raise ValueError("An original failed call cannot be admitted as a witnessed executable action")
        if not refs:
            # A witnessed literal action is retained for its unknown native effect.
            refs = [source.source_id for source in sources.values() if source.event_index == index
                    and source.kind == "ActionRef" and source.value == source_action][:1]
        if not refs:
            # JSON calls are grounded in the actual public call/return event. The
            # action text is evidence of occurrence, never proof that its IDs exist.
            document = f"e{index:05d}.witness_action"
            material.documents[document] = source_action
            source_id = document + ":whole"
            sources[source_id] = Source(source_id, index, "witness_action", "Text", source_action, document,
                                       0, len(source_action.encode("utf-8")))
            refs = [source_id]
        nodes.append(Node(f"call_{index}", "Call", Effect.NATIVE_CALL, reads=tuple(reads),
                          sources=tuple(refs), touches_history=True,
                          args={"template": text, "source_event": index}, hypothesis=repaired))
    # Control annotations refer to the same source events; deterministic folds
    # require a witnessed body and never create hidden API or entity rules.
    for structure in reversed(annotation["structures"]):
        op = structure.get("op")
        index = structure.get("event_index")
        if type(index) is not int or not 0 <= index <= len(material.public_view["events"]):
            raise ValueError("Structure event outside source window")
        if op == "If3":
            predicate = predicates.get(structure.get("predicate_id"))
            if predicate is None:
                raise ValueError("If3 lacks a registered predicate")
            value_id = "predicate_" + predicate["predicate_id"]
            pred_node = Node(value_id, "Predicate", Effect.PURE_TOTAL, defines=(value_id,),
                             sources=tuple(predicate["source_ids"]), args=predicate)
            position = next((i for i, node in enumerate(nodes) if node.args.get("source_event") == index), len(nodes))
            suffix = tuple(nodes[position:])
            node = Node(f"if3_{index}", "If3", Effect.CONTROL, reads=(value_id,), hypothesis=True,
                        branches={"T": suffix, "F": (Node(f"false_yield_{index}", "Yield", Effect.CONTROL,
                                    args={"reason": "AlternativeBranchUntested"}, hypothesis=True),),
                                  "U": (Node(f"unknown_yield_{index}", "Yield", Effect.CONTROL,
                                    args={"reason": "UnknownPredicate"}, hypothesis=True),)})
            nodes = nodes[:position] + [pred_node, node]
        elif op in {"Loop", "Frontier"}:
            refs = structure.get("source_ids")
            if not isinstance(refs, list) or not refs or len(refs) != len(set(refs)):
                raise ValueError("Frontier requires distinct explicit public sources")
            if any(ref not in sources or sources[ref].event_index > index for ref in refs):
                raise ValueError("Frontier borrows undisclosed sources")
            values = [sources[ref].value for ref in refs]
            if any(not isinstance(value, (str, int)) or isinstance(value, bool) for value in values):
                raise ValueError("Frontier elements must be finite public Refs")
            frontier_id = f"frontier_{index}"
            frontier = Node(frontier_id, "Frontier", Effect.PURE_TOTAL, defines=(frontier_id,),
                            sources=tuple(refs), args={"complete_collection": False})
            if op == "Frontier":
                position = next((i for i, node in enumerate(nodes) if node.args.get("source_event") == index), len(nodes))
                nodes.insert(position, frontier)
                continue
            iterations = structure.get("body_events")
            if (not isinstance(iterations, list) or len(iterations) != len(refs) or len(iterations) < 2
                    or len(set(canonical_bytes(value) for value in values)) < 2):
                raise ValueError("Loop needs at least two distinct-binding witnessed iterations")
            if any(not isinstance(body, list) or not body or any(type(i) is not int for i in body) for body in iterations):
                raise ValueError("Loop body_events must list each witnessed iteration")
            event_indices = [i for body in iterations for i in body]
            if event_indices != list(range(index, index + len(event_indices))):
                raise ValueError("Loop fold cannot discard intervening native effects")
            width = len(iterations[0])
            if any(len(body) != width for body in iterations):
                raise ValueError("Witnessed loop iterations differ in structure")
            item_id = f"loop_item_{index}"
            normalized_bodies = []
            chosen_nodes = []
            for iteration_index, body_indices in enumerate(iterations):
                body_nodes = [node for node in nodes if node.op == "Call" and node.args.get("source_event") in body_indices]
                if len(body_nodes) != width:
                    raise ValueError("Loop fold overlaps another control region")
                normalized = []
                for body_node in body_nodes:
                    event_index = body_node.args["source_event"]
                    bindings = event_bindings.get(event_index, [])
                    matching = [binding for binding in bindings if binding["action_substring"] ==
                                (values[iteration_index] if isinstance(values[iteration_index], str)
                                 else canonical_bytes(values[iteration_index]).decode("utf-8"))]
                    if len(matching) != 1:
                        raise ValueError("Each loop iteration must consume its own bound Ref")
                    parameter = matching[0]["parameter"]
                    template = body_node.args["template"]
                    template = re.sub(r"\$\{" + re.escape(parameter) + r"(?:_at_\d+)?\}",
                                      "${" + item_id + "}", template)
                    reads = tuple(item_id if read == parameter or read.startswith(parameter + "_at_") else read
                                  for read in body_node.reads)
                    normalized.append((template, reads))
                normalized_bodies.append(normalized)
                chosen_nodes.append(body_nodes)
            if any(body != normalized_bodies[0] for body in normalized_bodies[1:]):
                raise ValueError("Different bindings do not share a control structure")
            body = tuple(Node(f"loop_{index}_call_{offset}", "Call", Effect.NATIVE_CALL,
                              reads=reads, sources=chosen_nodes[0][offset].sources, touches_history=True,
                              args={**chosen_nodes[0][offset].args, "template": template})
                         for offset, (template, reads) in enumerate(normalized_bodies[0]))
            positions = [i for i, node in enumerate(nodes) if node.op == "Call" and node.args.get("source_event") in event_indices]
            first, last = min(positions), max(positions)
            region = nodes[first:last + 1]
            if any(node.op not in {"Call", "Bind"} for node in region):
                raise ValueError("Loop fold would discard a control effect")
            loop = Node(f"loop_{index}", "Loop", Effect.CONTROL, reads=(frontier_id,), body=body,
                        args={"finite_frontier": True, "item_value": item_id,
                              "witness_iterations": iterations, "source_ids": refs})
            nodes = nodes[:first] + [frontier, loop] + nodes[last + 1:]
        else:
            raise ValueError("Unregistered structure")
    ast, _, report = liveness(tuple(nodes))
    from .native_program_ir import walk
    entry_parameters = {name: value_type for name, value_type in parameters.items()
                        if sources[parameter_sources[name]].event_index <= 0}
    candidate = ProgramCandidate("", ast, entry_parameters,
        {name: parameter_sources[name] for name in entry_parameters},
        tuple(event["event_id"] for event in material.public_view["events"]),
        tuple(node.node_id for node in walk(ast) if node.hypothesis), source_class).seal()
    validate(candidate, sources, material.documents)
    return candidate, report


def compile_window(policy: Any, material: CompileMaterial, output: Path, case_id: str, *,
                   lease_seconds: float, source_class: str) -> dict[str, Any]:
    """Exactly the two preregistered roles, with no repair or stronger teacher."""
    if policy.meter is None or not 0 < lease_seconds <= 60:
        raise ValueError("Compilation requires an accounted lease no longer than 60 seconds")
    started = monotonic_seconds()
    parent_meter = policy.meter
    lease = BudgetMeter(BudgetLimits(1, max(1, min(18432, parent_meter.remaining_model_tokens)), lease_seconds),
                        parent=parent_meter)
    policy.meter = lease
    deadline = min(started + lease_seconds, parent_meter.started + parent_meter.limits.max_wall_seconds)
    result = {"case_id": case_id, "status": "REJECTED_SOURCE_OR_TYPE", "call_ids": [],
              "native_execution_count": 0, "native_success_count": 0}
    values = []
    try:
        for role, schema in (("compiler.annotate_trace", ANNOTATION_SCHEMA),
                             ("compiler.label_predicates", PREDICATE_SCHEMA)):
            if monotonic_seconds() >= deadline or policy.meter.exploration_stopped:
                raise BudgetExhausted("compile_lease")
            payload = material.input_payload if not values else {
                "goal": material.public_view["goal"], "sources": material.input_payload["sources"],
                "bound_annotation": values[0]}
            prompt = [{"role": "system", "content":
                "Return exactly one JSON object matching the controlled annotation schema. Use only supplied stable source IDs. "
                "Do not generate an AST or invent identifiers. Missing predicate evidence must be U.\n" + json.dumps(schema)},
                {"role": "user", "content": canonical_bytes(payload).decode("utf-8")}]
            prompt_ids = policy._tokenize_messages(prompt)
            if len(prompt_ids) > 8192:
                raise ValueError("INPUT_SOURCE_CLOSURE_EXCEEDS_8192_TOKENS")
            write_json_exclusive(output / "compiler_inputs" / (case_id + "." + role + ".json"), {
                "role": role, "messages": prompt, "prompt_token_ids": list(prompt_ids),
                "semantic_schema_version": SEMANTIC_SCHEMA_VERSION})
            first_call = len(lease.model_calls)
            try:
                with policy.frozen_reference():
                    generation = policy.complete(prompt, role=role, max_new_tokens=1024, greedy=True)
            except BaseException:
                calls = lease.model_calls[first_call:]
                result["call_ids"].extend(call.call_id for call in calls)
                write_json_exclusive(output / "compiler_outputs" / (case_id + "." + role + ".json"), {
                    "role": role, "text": None, "parse_status": "CALL_FAILED",
                    "calls": [asdict(call) for call in calls]})
                raise
            call_id = generation.metadata.get("model_call_id")
            result["call_ids"].append(call_id)
            raw = {"role": role, "text": generation.text, "call_id": call_id,
                   "metadata": generation.metadata, "parse_status": "NOT_PARSED"}
            # Raw output is committed before parsing, so malformed output survives.
            write_json_exclusive(output / "compiler_outputs" / (case_id + "." + role + ".json"), raw)
            if monotonic_seconds() > deadline:
                policy.meter.exploration_stopped = True
                raise TimeoutError("LEASE_OVERRUN")
            values.append(parse_object(generation.text, {"parameters", "bindings", "structures"}
                                       if not values else {"predicates"}))
        candidate, pruning = construct_candidate(material, values[0], values[1], source_class=source_class)
        write_json_exclusive(output / "asts" / (case_id + ".json"), asdict(candidate))
        write_json_exclusive(output / "provenance" / (case_id + ".json"), {
            "sources": {key: asdict(value) for key, value in material.sources.items()},
            "documents": material.documents, "independent_public_evidence_review": "REQUIRED"})
        write_json_exclusive(output / "pruning_reports" / (case_id + ".json"), pruning)
        result.update(status="AWAITING_INDEPENDENT_PUBLIC_REVIEW", version=candidate.version)
    except BaseException as exc:
        result.update(status="BUDGET_TRUNCATED" if isinstance(exc, (BudgetExhausted, TimeoutError)) else
                      "REJECTED_SOURCE_OR_TYPE" if isinstance(exc, (ValueError, KeyError, TypeError)) else "COMPILER_ERROR",
                      error=f"{type(exc).__name__}: {exc}")
        if not isinstance(exc, Exception):
            raise
    finally:
        result["elapsed_seconds"] = monotonic_seconds() - started
        result["lease_overrun"] = result["elapsed_seconds"] > lease_seconds
        if result["lease_overrun"] or lease.exploration_stopped:
            parent_meter.exploration_stopped = True
        policy.meter = parent_meter
        write_json_exclusive(output / "compiler_outputs" / (case_id + ".result.json"), result)
    return result
