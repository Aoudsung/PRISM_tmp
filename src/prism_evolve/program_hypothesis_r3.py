"""R3 public work views, hypothesis IR, and next-work research leases.

R2 artifacts and witness admission remain untouched. R3 admits an explicitly
untested control program; native terminal evaluation supplies its only value.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Mapping
import uuid

from .native_core import BudgetExhausted, Resources, monotonic_seconds
from .native_program_ir import Effect
from .native_trace_artifacts import canonical_bytes, public_view, write_json_exclusive

VERSION = "prism-program-hypothesis-r3"
TYPES = {"Text", "Ref", "Number", "Truth", "RefSet", "Bool"}
EFFECTS = {"Bind": Effect.PURE_TOTAL, "Frontier": Effect.PURE_TOTAL,
           "SemanticBatch": Effect.AUX_CALL, "Call": Effect.NATIVE_CALL,
           "Loop": Effect.CONTROL, "If3": Effect.CONTROL,
           "Yield": Effect.CONTROL, "Return": Effect.CONTROL}


def append(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_bytes(value).decode() + "\n")


@dataclass
class RepairContext:
    episode_id: str
    benchmark: str
    task_id: str
    source_path: str
    source_role: str
    work_view: dict[str, Any]
    documents: dict[str, Any]
    catalogs: dict[str, Any]
    public_contract: str = VERSION


@dataclass
class ProgramProposal:
    episode_id: str
    status: str
    anchor_event: int = 0
    gap: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    ast: list[dict[str, Any]] = field(default_factory=list)
    fuel: int = 100
    call_ids: list[str] = field(default_factory=list)
    hypothesis: str = "HYPOTHESIS"
    error: str | None = None
    source_closure: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TrialRecord:
    trial_id: str
    benchmark: str
    target_task_id: str
    target_split: str
    target_identity: str
    arm: str
    source_episode: str
    experience_episodes: list[str]
    seed: int
    anchor_event: int
    events: list[dict[str, Any]] = field(default_factory=list)
    bindings: list[dict[str, Any]] = field(default_factory=list)
    semantic_batches: list[dict[str, Any]] = field(default_factory=list)
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    recoveries: list[dict[str, Any]] = field(default_factory=list)
    local_exits: list[dict[str, Any]] = field(default_factory=list)
    fallback: list[dict[str, Any]] = field(default_factory=list)
    replay: dict[str, Any] = field(default_factory=dict)
    native_terminal: dict[str, Any] = field(default_factory=dict)
    costs: dict[str, Any] = field(default_factory=dict)
    initial_policy: dict[str, Any] = field(default_factory=dict)
    termination: str = "unfinished"
    terminal_y: float = 0.0
    utility: float = 0.0


class WorkLeases:
    """Independent research allocation; these leases never certify online rho.

    Only the next operation is reserved. Trial admission follows a candidate.
    There is no speculative trial reservation during a proposal call.
    """
    def __init__(self, path: Path):
        self.path = path
        self.active: dict[str, dict[str, Any]] = {}

    def reserve(self, owner: str, kind: str, upper: Resources) -> str:
        identity = uuid.uuid4().hex
        row = {"lease_id": identity, "owner": owner, "kind": kind,
               "upper": asdict(upper), "budget_class": "independent_research",
               "online_rho_satisfied": False, "status": "RESERVED"}
        self.active[identity] = row
        append(self.path, row)
        return identity

    def settle(self, identity: str, actual: Resources, status: str = "SETTLED") -> None:
        row = self.active.pop(identity)
        append(self.path, {**row, "actual": asdict(actual),
                          "status": status if actual.fits_fraction(Resources(**row["upper"]), 1) else "LEASE_OVERRUN"})

    def cancel(self, identity: str, reason: str) -> None:
        row = self.active.pop(identity)
        append(self.path, {**row, "status": "CANCELLED", "reason": reason,
                          "actual": asdict(Resources(0, 0, 0))})


@contextmanager
def measured(path: Path, owner: str, kind: str, **extra):
    started = monotonic_seconds()
    status = "OK"
    try:
        yield
    except BaseException:
        status = "ERROR"
        raise
    finally:
        append(path, {"cost_event_id": uuid.uuid4().hex, "charged_round_id": owner,
                      "kind": kind, "status": status,
                      "elapsed_seconds": monotonic_seconds() - started, **extra})


def pointer(value: Any, path: str) -> Any:
    if path == "":
        return value
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("A source pointer must be an RFC6901 JSON pointer")
    for token in path[1:].split("/"):
        key = token.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def select(value: Any, reference: Mapping[str, Any]) -> Any:
    value = pointer(value, reference.get("pointer", ""))
    if "span" in reference:
        start, end = reference["span"]
        if not isinstance(value, str) or type(start) is not int or type(end) is not int:
            raise ValueError("Span requires text and integer UTF-8 byte offsets")
        data = value.encode("utf-8")
        if not 0 <= start <= end <= len(data):
            raise ValueError("Source span outside document")
        value = data[start:end].decode("utf-8")
    return value


def native_catalog(schema: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Derive every tool from the public native schema, including untried APIs."""
    catalog: dict[str, dict[str, Any]] = {}
    def visit(value):
        if "anyOf" in value or "oneOf" in value:
            for child in value.get("anyOf", value.get("oneOf", [])):
                visit(child)
            return
        props = value.get("properties", {})
        name = props.get("name", {}).get("const")
        if name is not None:
            catalog[name] = {"kind": "json", "schema": value,
                             "arguments": props["arguments"]}
    for value in schema.get("native_action_schemas_by_mode", {}).values():
        visit(value)
    if catalog:
        return dict(sorted(catalog.items()))
    commands = list(dict.fromkeys(schema.get("available_actions", []) +
                    [action for values in schema.get("mode_actions", {}).values() for action in values]))
    for command in commands:
        name = command.split("[", 1)[0] if "[" in command else command.split(" ", 1)[0]
        catalog.setdefault(name, {"kind": "text", "commands": []})["commands"].append(command)
    if schema.get("search_available"):
        catalog["search"] = {"kind": "search", "arguments": {"type": "object",
            "properties": {"query": {"type": "string", "minLength": 1,
                                      "pattern": "^[^\\[\\]\\r\\n]+$"}},
            "required": ["query"], "additionalProperties": False}}
    for name, tool in catalog.items():
        if tool["kind"] == "text":
            tool["arguments"] = {"type": "object", "properties": {
                "command": {"type": "string", "enum": tool["commands"]}},
                "required": ["command"], "additionalProperties": False}
    return dict(sorted(catalog.items()))


def short_index(catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    # A fixed, complete short index; arguments are disclosed after name selection.
    return [{"name": name, "parameters": list(row["arguments"].get("properties", {})),
             "required": row["arguments"].get("required", [])} for name, row in catalog.items()]


def strip_catalog(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: strip_catalog(child) for key, child in value.items()
                if key not in {"native_tools", "native_action_schemas_by_mode"}}
    if isinstance(value, list):
        return [strip_catalog(child) for child in value]
    return value


class Documents:
    """Intern repeated public documents without enumerating substring products."""
    def __init__(self):
        self.documents: dict[str, Any] = {}
        self._identities: dict[bytes, str] = {}

    def add(self, value: Any) -> str:
        identity = canonical_bytes(value)
        if identity not in self._identities:
            name = f"d{len(self.documents):04d}"
            self._identities[identity] = name
            self.documents[name] = value
        return self._identities[identity]


def make_context(trace: dict[str, Any], row: dict[str, Any], source_path: str) -> RepairContext:
    view = public_view(trace)
    docs, catalogs, catalog_ids, events = Documents(), {}, {}, []
    goal = docs.add(view["goal"])
    for index, event in enumerate(view["events"]):
        catalog = native_catalog(event["schema"])
        identity = canonical_bytes(catalog)
        if identity not in catalog_ids:
            name = f"c{len(catalogs):03d}"
            catalog_ids[identity] = name
            catalogs[name] = catalog
        events.append({"event": index, "before": docs.add(strip_catalog(event["obs_before"])),
                       "action": event["action"], "after": docs.add(strip_catalog(event["obs_after"])),
                       "catalog": catalog_ids[identity]})
    work = {"goal": goal, "events": events, "documents": docs.documents,
            "catalogs": {key: short_index(value) for key, value in catalogs.items()},
            "source_rule": "Use document_id with exact UTF-8 span or JSON pointer; no source combinations are pre-enumerated",
            "scope": "complete public training episode; source posthoc diagnosis only"}
    return RepairContext(row["episode_id"], row["benchmark"], row["task_id"],
                         source_path, "source_posthoc", work, docs.documents, catalogs)


def validate_proposal(proposal: ProgramProposal, context: RepairContext) -> None:
    if proposal.status in {"NO_PROPOSAL", "NO_REUSABLE_PROGRAM"}:
        if proposal.ast:
            raise ValueError("A negative proposal must not contain a program")
        return
    if proposal.status != "PROPOSAL" or not proposal.ast:
        raise ValueError("Expected one main PROPOSAL or an explicit negative outcome")
    if type(proposal.anchor_event) is not int or not 0 <= proposal.anchor_event < len(context.work_view["events"]):
        raise ValueError("Anchor outside source episode")
    if type(proposal.fuel) is not int or not 1 <= proposal.fuel <= 100:
        raise ValueError("Program fuel must be in [1,100]")
    if not proposal.evidence:
        raise ValueError("A proposal needs public evidence")
    proposal.source_closure = []
    for ref in proposal.evidence:
        document = context.documents[ref["document_id"]]
        value = select(document, ref)
        proposal.source_closure.append({"reference": ref, "value": value})
    public_tools = {name for catalog in context.catalogs.values() for name in catalog}
    if not set(proposal.required_tools) <= public_tools:
        raise ValueError("Required API absent from public catalog")
    ids: set[str] = set()
    def expression(value, defined):
        if isinstance(value, dict) and set(value) == {"var"}:
            if value["var"].split("/", 1)[0] not in defined:
                raise ValueError("Non-dominating runtime argument")
        elif isinstance(value, dict):
            for child in value.values():
                expression(child, defined)
        elif isinstance(value, list):
            for child in value:
                expression(child, defined)
        else:
            raise ValueError("Call/Frontier arguments must bind current runtime values, never source literals")
    def block(nodes, defined, depth=0):
        if depth > 12 or not isinstance(nodes, list):
            raise ValueError("Invalid AST nesting")
        defined = set(defined)
        for node in nodes:
            if not isinstance(node, dict) or set(node) - {"id", "op", "args", "body", "branches", "effect", "hypothesis"}:
                raise ValueError("Unregistered AST fields")
            name, op = node["id"], node["op"]
            if not isinstance(name, str) or name in ids or op not in EFFECTS:
                raise ValueError("Invalid node identity or operation")
            ids.add(name)
            node["effect"] = EFFECTS[op].value
            node["hypothesis"] = "HYPOTHESIS"
            args = node.get("args", {})
            outputs = []
            if op == "Bind":
                if args["type"] not in TYPES or args["source"]["root"] not in {"goal", "observation", "last_return", "history", "values"}:
                    raise ValueError("Bind must select current public state")
                outputs = [args["name"]]
            elif op == "SemanticBatch":
                if not args.get("fields"):
                    raise ValueError("Empty semantic batch")
                for spec in args["fields"].values():
                    if spec["type"] not in TYPES or not isinstance(spec["instruction"], str):
                        raise ValueError("Invalid semantic field")
                outputs = list(args["fields"])
            elif op == "Frontier":
                expression(args["items"], defined)
                outputs = [args["name"]]
            elif op == "Call":
                tool = args["tool"]
                if isinstance(tool, str):
                    if tool not in public_tools:
                        raise ValueError("Call uses an undisclosed API")
                else:
                    expression(tool, defined)
                expression(args["arguments"], defined)
            elif op == "Loop":
                if args["frontier"] not in defined or type(args["fuel"]) is not int or not 1 <= args["fuel"] <= 100:
                    raise ValueError("Loop needs a finite Frontier and explicit fuel")
                block(node.get("body", []), defined | {args["item"]}, depth + 1)
            elif op == "If3":
                if args["predicate"] not in defined or set(node.get("branches", {})) != {"T", "F", "U"}:
                    raise ValueError("If3 requires a runtime predicate and T/F/U")
                for branch in node["branches"].values():
                    block(branch, defined, depth + 1)
            if len(set(outputs)) != len(outputs) or any(not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in defined for name in outputs):
                raise ValueError("Invalid or duplicate ValueId")
            defined.update(outputs)
        return defined
    block(proposal.ast, set())
    if len(ids) > 100:
        raise ValueError("AST exceeds the bounded IR")


IR_INSTRUCTION = """Return one JSON object, without markdown. Discover the goal/behavior gap yourself from this complete public training episode. Choose one main reusable control-program hypothesis, or status NO_PROPOSAL/NO_REUSABLE_PROGRAM. A single observed behavior may motivate a Loop; no multi-iteration witness is required. Public catalog APIs not previously executed are available. Do not infer terminal success from observation wording.
You may finish in this call: {status:PROPOSAL,anchor_event:int,gap:string,evidence:[{document_id,pointer:JSON_pointer,span:[start_utf8,end_utf8] (optional)}],required_tools:[native_tool_name],fuel:1..100,ast:[node]}. Alternatively request {status:LOCATE,anchor_event:int,evidence:[references],required_tools:[names],gap:string} for a second and final call with expanded schemas. Negative status objects may include gap and empty ast.
A node is {id:unique_string,op:Bind|SemanticBatch|Frontier|Call|Loop|If3|Return|Yield,args:object,body:[nodes] (Loop only),branches:{T:[nodes],F:[nodes],U:[nodes]} (If3 only).
Bind args={name,type:Text|Ref|Number|Truth|RefSet|Bool,source:{root:goal|observation|last_return|history|values,pointer:JSON_pointer,span:[UTF8_start,UTF8_end] optional}}. This source is resolved against CURRENT target state, never stored source values. SemanticBatch args={fields:{value_id:{type,instruction}}}; one real model call extracts the whole requested set of fields from CURRENT target goal/history/observation and bound values. Missing truth is U, other missing values are null and yield. Use set-level semantic fields rather than per-item calls when possible. Frontier args={name,items:{var:value_id}} snapshots a finite currently disclosed RefSet; it never proves hidden completeness. Loop args={frontier:value_id,item:scoped_value_id,fuel:positive_int}; body runs once for each current Ref. If3 args={predicate:value_id}; all T/F/U branches execute as specified. Call args={tool:native_tool_name or {var:value_id},arguments:{native_arg:{var:value_id}}}. For text tools use arguments.command containing a complete CURRENT public command; search uses arguments.query. All argument leaves must reference runtime variables, optionally /JSON/pointer after the ValueId, including numeric data. Tool names alone may be literal. Call updates the current observation/last_return/history. Return leaves the local program; Yield args={reason:string} transfers to the same frozen fallback. All branches, calls and loops are HYPOTHESIS. No Python, shell outside a public API, eval, hidden evaluator access, cached source entity values or invented tools. Preserve unknown native effects. Use only these operations. Evidence references close only the sources actually selected, using short document IDs rather than invented source IDs. Source posthoc details are diagnostic; a transferable program must rebind all data using the current target's public state.
"""


def parse_proposal(value: dict[str, Any], context: RepairContext, calls: list[str]) -> ProgramProposal:
    proposal = ProgramProposal(context.episode_id, value["status"], value.get("anchor_event", 0),
        value.get("gap", ""), value.get("evidence", []), value.get("required_tools", []),
        value.get("ast", []), value.get("fuel", 100), calls)
    validate_proposal(proposal, context)
    return proposal
