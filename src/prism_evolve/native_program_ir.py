"""Source-proven Gate 0–2 IR: deterministic types, effects and strict replay.

Minimal conventions approved for the 2026-11-12 contract: Frontier enumerates
only a finite, currently disclosed Ref set. Loop binds each such Ref once and
checks budget at every iteration. Invalid sources, unavailable budgets and
Unknown predicates yield control. If3 has exactly T/F/U branches. Enumeration
never certifies completion of a hidden collection. All native effects default
to unknown; only dead PURE_TOTAL computations may be statically deleted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Iterable, Mapping

from .native_trace_artifacts import canonical_bytes, sha256

IR_VERSION = "prism-public-program-ir-v1"
SEMANTIC_SCHEMA_VERSION = "prism-public-predicates-v1"


class Effect(str, Enum):
    PURE_TOTAL = "PURE_TOTAL"
    AUX_CALL = "AUX_CALL"
    NATIVE_CALL = "NATIVE_CALL"
    CONTROL = "CONTROL"


class Truth(str, Enum):
    T = "T"
    F = "F"
    U = "U"


REPLAY_STATUSES = frozenset({"EXACT_WITNESS_REPLAY", "PREFIX_CONSISTENT", "DIVERGED_AT_EVENT",
    "SOURCE_INCOMPLETE", "REJECTED_SOURCE_OR_TYPE", "BUDGET_TRUNCATED"})


@dataclass(frozen=True)
class Source:
    source_id: str
    event_index: int  # -1 is the public goal; other sources are observations before that action
    kind: str
    value_type: str
    value: Any
    document_id: str
    start_utf8: int
    end_utf8: int

    def verify(self, documents: Mapping[str, str]) -> None:
        if self.document_id not in documents:
            raise ValueError(f"Missing source document: {self.source_id}")
        data = documents[self.document_id].encode("utf-8")
        if not 0 <= self.start_utf8 <= self.end_utf8 <= len(data):
            raise ValueError(f"Invalid span: {self.source_id}")
        text = data[self.start_utf8:self.end_utf8].decode("utf-8")
        expected = self.value if isinstance(self.value, str) else canonical_bytes(self.value).decode("utf-8")
        if text != expected:
            raise ValueError(f"Source span differs from original: {self.source_id}")
        check_type(self.value, self.value_type)


def check_type(value: Any, name: str) -> None:
    valid = {"Text": isinstance(value, str), "Ref": isinstance(value, (str, int)) and not isinstance(value, bool),
             "Number": isinstance(value, (int, float)) and not isinstance(value, bool),
             "Bool": isinstance(value, bool), "Truth": value in ("T", "F", "U"),
             "RefSet": isinstance(value, list) and all(isinstance(v, (str, int)) and not isinstance(v, bool) for v in value)}
    if name not in valid or not valid[name]:
        raise ValueError(f"Value is not {name}")


@dataclass(frozen=True)
class Node:
    node_id: str
    op: str
    effect_class: Effect
    reads: tuple[str, ...] = ()
    defines: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    touches_history: bool = False
    rng_scope: str = "none"
    cache_refs: tuple[str, ...] = ()
    args: dict[str, Any] = field(default_factory=dict)
    body: tuple[Node, ...] = ()
    branches: dict[str, tuple[Node, ...]] = field(default_factory=dict)
    hypothesis: bool = False


@dataclass(frozen=True)
class ProgramCandidate:
    version: str
    ast: tuple[Node, ...]
    entry_parameters: dict[str, str]
    parameter_sources: dict[str, str]
    local_witnesses: tuple[str, ...]
    unexecuted_nodes: tuple[str, ...]
    source_class: str
    status: str = "UNTESTED"
    semantic_schema_version: str = SEMANTIC_SCHEMA_VERSION
    parent_version: str | None = None
    native_execution_count: int = 0
    native_success_count: int = 0

    def identity_payload(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("version", "status", "native_execution_count", "native_success_count"):
            value.pop(key)
        return value

    def seal(self) -> ProgramCandidate:
        return replace(self, version=sha256(canonical_bytes(self.identity_payload())))


def candidate_from_dict(value: Mapping[str, Any]) -> ProgramCandidate:
    def node(row: Mapping[str, Any]) -> Node:
        fields = dict(row)
        fields["effect_class"] = Effect(fields["effect_class"])
        for name in ("reads", "defines", "sources", "cache_refs"):
            fields[name] = tuple(fields.get(name, ()))
        fields["body"] = tuple(node(child) for child in fields.get("body", ()))
        fields["branches"] = {key: tuple(node(child) for child in branch)
                              for key, branch in fields.get("branches", {}).items()}
        return Node(**fields)
    fields = dict(value)
    fields["ast"] = tuple(node(row) for row in fields["ast"])
    fields["local_witnesses"] = tuple(fields["local_witnesses"])
    fields["unexecuted_nodes"] = tuple(fields["unexecuted_nodes"])
    return ProgramCandidate(**fields)


def walk(nodes: Iterable[Node]) -> Iterable[Node]:
    for node in nodes:
        yield node
        yield from walk(node.body)
        for branch in node.branches.values():
            yield from walk(branch)


def validate(candidate: ProgramCandidate, sources: Mapping[str, Source], documents: Mapping[str, str]) -> None:
    if candidate.version != candidate.seal().version:
        raise ValueError("Program version mismatch")
    if candidate.source_class not in {"LOCAL_WITNESS", "INTERFACE_REPAIR_HYPOTHESIS"}:
        raise ValueError("Unknown candidate source class")
    if set(candidate.entry_parameters) != set(candidate.parameter_sources):
        raise ValueError("Every entry parameter needs a source")
    for name, source_id in candidate.parameter_sources.items():
        source = sources[source_id]
        source.verify(documents)
        check_type(source.value, candidate.entry_parameters[name])
        if source.event_index > 0:
            raise ValueError("Entry parameter borrows a future observation")
    node_ids = [node.node_id for node in walk(candidate.ast)]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Duplicate node identity")
    expected_effects = {"Bind": Effect.PURE_TOTAL, "Frontier": Effect.PURE_TOTAL,
        "Predicate": Effect.PURE_TOTAL, "SemanticBatch": Effect.AUX_CALL,
        "Call": Effect.NATIVE_CALL, "If3": Effect.CONTROL, "Loop": Effect.CONTROL,
        "Return": Effect.CONTROL, "Yield": Effect.CONTROL}

    def block(nodes: tuple[Node, ...], defined: set[str], cursor: int) -> tuple[set[str], int]:
        defined = set(defined)
        for node in nodes:
            if node.op not in expected_effects or node.effect_class != expected_effects[node.op]:
                raise ValueError(f"Invalid effect class: {node.node_id}")
            if not set(node.reads + node.cache_refs) <= defined:
                raise ValueError(f"Non-dominating value/cache reference: {node.node_id}")
            if len(set(node.defines)) != len(node.defines) or set(node.defines) & defined:
                raise ValueError("Values must have unique definitions")
            for source_id in node.sources:
                source = sources[source_id]
                source.verify(documents)
                if source.event_index > cursor:
                    raise ValueError(f"Future source: {node.node_id}")
            if node.op == "Call":
                if not node.sources or not node.touches_history:
                    raise ValueError("Native calls need provenance and a history effect")
                if node.args.get("source_event") != cursor and not node.hypothesis:
                    raise ValueError("Witness calls must retain contiguous native effect context")
                cursor += 1
            elif node.op == "If3":
                if set(node.branches) != {"T", "F", "U"} or not node.reads:
                    raise ValueError("If3 requires a predicate and exactly T/F/U branches")
                outputs = [block(branch, defined, cursor) for branch in node.branches.values()]
                defined |= set.intersection(*(out[0] for out in outputs))
                cursor = min(out[1] for out in outputs)
            elif node.op == "Loop":
                if not node.reads or node.args.get("finite_frontier") is not True:
                    raise ValueError("Loop requires a finite current Frontier")
                item = node.args.get("item_value")
                if not isinstance(item, str) or item in defined:
                    raise ValueError("Loop needs a scoped item ValueId")
                block(node.body, defined | {item}, cursor)
                iterations = node.args.get("witness_iterations", [])
                if iterations:
                    flat = [event for iteration in iterations for event in iteration]
                    if flat != list(range(cursor, cursor + len(flat))):
                        raise ValueError("Loop source interval is not contiguous")
                    cursor += len(flat)
            elif node.op == "SemanticBatch":
                if node.rng_scope != "auxiliary" or node.touches_history:
                    raise ValueError("Auxiliary calls require independent RNG and no actor history writes")
            elif node.op == "Frontier":
                if not node.sources or node.args.get("complete_collection") is not False:
                    raise ValueError("Frontier cannot claim exhaustive hidden traversal")
            defined.update(node.defines)
        return defined, cursor

    block(candidate.ast, set(candidate.entry_parameters), 0)
    hypotheses = {node.node_id for node in walk(candidate.ast) if node.hypothesis}
    if hypotheses != set(candidate.unexecuted_nodes):
        raise ValueError("All unexecuted nodes must be explicit grammar hypotheses")


def liveness(nodes: tuple[Node, ...], live_out: frozenset[str] = frozenset(), *,
             continuation_reads_history: bool = True) -> tuple[tuple[Node, ...], frozenset[str], list[dict[str, Any]]]:
    """Backward may-liveness; branches union and loops take the least fixed point."""
    live, kept, report = set(live_out), [], []
    for node in reversed(nodes):
        outgoing = set(live)
        if node.op == "If3":
            branches, incoming = {}, set()
            for label, branch in node.branches.items():
                pruned, branch_live, notes = liveness(branch, frozenset(outgoing),
                    continuation_reads_history=continuation_reads_history)
                branches[label] = pruned
                incoming.update(branch_live)
                report.extend(notes)
            node = replace(node, branches=branches)
            live |= incoming
        elif node.op == "Loop":
            loop_live = set(outgoing)
            while True:
                body, body_live, _ = liveness(node.body, frozenset(loop_live),
                    continuation_reads_history=continuation_reads_history)
                incoming = outgoing | (set(body_live) - {node.args.get("item_value")})
                if incoming <= loop_live:
                    break
                loop_live |= incoming
            body, _, notes = liveness(node.body, frozenset(loop_live),
                continuation_reads_history=continuation_reads_history)
            node = replace(node, body=body)
            live |= loop_live
            report.extend(notes)
        dead = not (set(node.defines) & outgoing)
        delete = node.effect_class == Effect.PURE_TOTAL and dead
        skip_aux = node.effect_class == Effect.AUX_CALL and dead
        if delete or skip_aux:
            report.append({"node_id": node.node_id, "decision": "DELETE_PURE_TOTAL" if delete else "SKIP_UNDEMANDED_AUX",
                           "reason": "no live result consumer", "cost_already_incurred_is_retained": True})
            continue
        if node.touches_history and continuation_reads_history:
            live.add("$history")
        live = set(node.reads + node.cache_refs) | (live - set(node.defines))
        report.append({"node_id": node.node_id, "decision": "KEEP",
                       "reason": "unknown native effect/history" if node.effect_class == Effect.NATIVE_CALL else "live value or control"})
        kept.append(node)
    return tuple(reversed(kept)), frozenset(live), report


def native_deletion_version(candidate: ProgramCandidate, deleted_node_id: str) -> ProgramCandidate:
    """Deleting a native action changes identity and invalidates all inherited evidence.

    Rebinding and source/type validation are required on the returned UNTESTED
    version; later ActionRefs from the removed return will fail validation.
    """
    found = False
    def remove(nodes: tuple[Node, ...]) -> tuple[Node, ...]:
        nonlocal found
        result = []
        for node in nodes:
            if node.node_id == deleted_node_id:
                if node.effect_class != Effect.NATIVE_CALL:
                    raise ValueError("Native deletion requires a native action")
                found = True
                continue
            result.append(replace(node, body=remove(node.body),
                                  branches={key: remove(branch) for key, branch in node.branches.items()}))
        return tuple(result)
    ast = remove(candidate.ast)
    if not found:
        raise ValueError("Missing deletion target")
    return replace(candidate, ast=ast, version="", parent_version=candidate.version, status="UNTESTED",
                   native_execution_count=0, native_success_count=0).seal()


@dataclass
class ReplayResult:
    status: str
    source_events_consumed: int
    first_divergence_event: int | None = None
    reason: str = ""
    native_execution_count: int = 0
    native_success_count: int = 0


def strict_source_replay(actions: Iterable[str], read_source_event: Callable[[int], Mapping[str, Any]],
                         *, source_length: int, max_steps: int) -> ReplayResult:
    """Read a source return only after its action matches; never expose future observations.

    The action iterator must be lazy. Its next action can use only source returns
    already read by this function, never a preloaded counterfactual suffix.
    """
    consumed = 0
    for action in actions:
        if consumed >= max_steps:
            return ReplayResult("BUDGET_TRUNCATED", consumed, reason="replay step limit")
        if consumed >= source_length:
            return ReplayResult("SOURCE_INCOMPLETE", consumed, reason="program requests another source action")
        event = read_source_event(consumed)
        if action != event["action"]:
            return ReplayResult("DIVERGED_AT_EVENT", consumed, consumed, "action mismatch; return not delivered")
        # read_source_event supplies action metadata only. The matched return is
        # delivered separately to prevent exposing it before the comparison.
        deliver = event.get("deliver_observation")
        if deliver is not None:
            deliver()
        consumed += 1
    return ReplayResult("EXACT_WITNESS_REPLAY" if consumed == source_length else "PREFIX_CONSISTENT", consumed)
