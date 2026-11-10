"""PRISM DCC v2 (Decision-Contrast Closure) unified mathematical and execution kernel.

Implements the systematic revision contract (`docs/prism-dcc-v2-design.md`):
1. One control objective: official native terminal utility U_B(Y, C) = Y * max(0, 1 - C / B).
   Public responses (structured fields, resource deltas, termination category, and observed
   terminal outcome) supervise the response model K_phi(x_{t+1} | g, h_t, a_t, beta_t);
   no hand-crafted mid-episode progress integral is used as reward or value.
2. Normalized context-conditioned residual response ensemble:
   p_{phi,m}(x_i | c, x_{<i}) = softmax(ell_0(c, x_{<i}) + r_{phi,m}(c, x_{<i}))_{x_i},
   with zero-residual initialization preserving the frozen backbone prior K_{phi_0} = p_0,
   explicit (state x context x outcome) emission conditioning, and cluster-level bootstrap
   negative log-likelihood over unique physical transitions.
3. Typed executable observation/control programs (`Call`, `Read`, `Seq`, `Case`, `Stop`)
   with reversible entity role bindings, step-by-step interpreter state (`ProgramExecutionState`),
   def-use dependency `Delete`, and scope-unified `Concat`.
4. Unified long-horizon candidate evaluation Q_m^rho(g, h, beta) = E_{K_{phi,m}, rho -> pi_0}[U_B(Y, C)]
   averaged across ensemble members before argmax selection, with exact finite-table EVSI
   retained strictly as a zero-modulation decision-theoretic reference.
5. Witness-guided program discovery driven by observed terminal action differences d_real
   (kept strictly separate from model estimates d_model), single-label-view isolation for
   Label-Shuffled controls, and complete candidate construction/inspection budget accounting.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import random
import re
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DCC_CONTRACT_VERSION = "prism-dcc-v2-unified-r2"
TAIL_POLICY_VERSION = "frozen-prior-tail-v1"
GRAMMAR_VERSION = "typed-ast-v2"
DEFAULT_ENSEMBLE_SIZE = 8
DEFAULT_MAX_PROBE_STEPS = 3
DEFAULT_CONTINUATION_STEPS = 4
DEFAULT_PROPOSAL_BUDGET = 3
DEFAULT_MAX_CANDIDATE_CHECKS = 24
STATE_DIM = 16
CONTEXT_DIM = 64
OUTCOME_BINS = 8
TERMINATION_CATEGORIES = ("live", "native_terminal", "task_budget_terminal")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _short_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class ASTNode:
    """Typed AST node in the executable program grammar: Call, Read, Seq, Case, Stop."""

    op: str  # "Call" | "Read" | "Seq" | "Case" | "Stop"
    tool_or_action: str = ""
    typed_arguments: dict[str, str] = field(default_factory=dict)
    bind_return_to: str = ""
    variable: str = ""
    public_field_path: str = ""
    children: tuple["ASTNode", ...] = ()
    condition_field: str = ""
    condition_equals: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "tool_or_action": self.tool_or_action,
            "typed_arguments": dict(self.typed_arguments),
            "bind_return_to": self.bind_return_to,
            "variable": self.variable,
            "public_field_path": self.public_field_path,
            "children": [c.to_dict() for c in self.children],
            "condition_field": self.condition_field,
            "condition_equals": self.condition_equals,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ASTNode":
        return cls(
            op=str(data.get("op", "Stop")),
            tool_or_action=str(data.get("tool_or_action", "")),
            typed_arguments={str(k): str(v) for k, v in dict(data.get("typed_arguments", {})).items()},
            bind_return_to=str(data.get("bind_return_to", "")),
            variable=str(data.get("variable", "")),
            public_field_path=str(data.get("public_field_path", "")),
            children=tuple(cls.from_dict(c) for c in data.get("children", ())),
            condition_field=str(data.get("condition_field", "")),
            condition_equals=str(data.get("condition_equals", "")),
        )


@dataclass(frozen=True)
class DefUseEdge:
    """Data-dependency edge linking a step index that produces a variable/field to its consumer."""

    producer_step: int
    variable_name: str
    field_path: str
    consumer_step: int | None  # None means consumed by final program readout


@dataclass(frozen=True)
class ExecutableProgram:
    """Typed executable observation/control program e = (rho_e, chi_e) with depth l <= 3."""

    program_id: str
    benchmark: str
    step_templates: tuple[str, ...]
    readout_fields: tuple[str, ...]
    origin: str = "starter"
    parent_ids: tuple[str, ...] = ()
    expected_cost: float = 1.0
    ast_nodes: tuple[ASTNode, ...] = ()
    variable_scope: tuple[str, ...] = ()
    role_bindings: dict[str, str] = field(default_factory=dict)
    def_use_edges: tuple[DefUseEdge, ...] = ()
    case_condition_field: str = ""
    case_condition_value: str = ""
    case_true_template: str = ""
    case_false_template: str = ""

    def __post_init__(self) -> None:
        if not self.ast_nodes and self.step_templates:
            nodes: list[ASTNode] = []
            scope: list[str] = []
            edges: list[DefUseEdge] = []
            for idx, tmpl in enumerate(self.step_templates):
                var_name = f"ret_{idx}"
                scope.append(var_name)
                slots = re.findall(r"\{([a-zA-Z0-9_]+)\}", tmpl)
                arg_map = {slot: f"${slot}" for slot in slots}
                nodes.append(
                    ASTNode(
                        op="Call",
                        tool_or_action=tmpl,
                        typed_arguments=arg_map,
                        bind_return_to=var_name,
                    )
                )
                if idx < len(self.readout_fields):
                    fld = self.readout_fields[idx]
                    nodes.append(ASTNode(op="Read", variable=var_name, public_field_path=fld))
                    edges.append(
                        DefUseEdge(
                            producer_step=idx,
                            variable_name=var_name,
                            field_path=fld,
                            consumer_step=None,
                        )
                    )
            for extra_idx in range(len(self.step_templates), len(self.readout_fields)):
                fld = self.readout_fields[extra_idx]
                src_step = max(0, len(self.step_templates) - 1)
                var_name = f"ret_{src_step}"
                nodes.append(ASTNode(op="Read", variable=var_name, public_field_path=fld))
                edges.append(
                    DefUseEdge(
                        producer_step=src_step,
                        variable_name=var_name,
                        field_path=fld,
                        consumer_step=None,
                    )
                )
            if self.case_condition_field and (self.case_true_template or self.case_false_template):
                nodes.append(
                    ASTNode(
                        op="Case",
                        condition_field=self.case_condition_field,
                        condition_equals=self.case_condition_value,
                        children=(
                            ASTNode(op="Call", tool_or_action=self.case_true_template)
                            if self.case_true_template else ASTNode(op="Stop"),
                            ASTNode(op="Call", tool_or_action=self.case_false_template)
                            if self.case_false_template else ASTNode(op="Stop"),
                        ),
                    )
                )
            nodes.append(ASTNode(op="Stop"))
            object.__setattr__(self, "ast_nodes", tuple(nodes))
            if not self.variable_scope:
                object.__setattr__(self, "variable_scope", tuple(scope))
            if not self.def_use_edges:
                object.__setattr__(self, "def_use_edges", tuple(edges))

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "benchmark": self.benchmark,
            "step_templates": list(self.step_templates),
            "readout_fields": list(self.readout_fields),
            "origin": self.origin,
            "parent_ids": list(self.parent_ids),
            "expected_cost": float(self.expected_cost),
            "ast_nodes": [node.to_dict() for node in self.ast_nodes],
            "variable_scope": list(self.variable_scope),
            "role_bindings": dict(self.role_bindings),
            "def_use_edges": [asdict(e) for e in self.def_use_edges],
            "case_condition_field": self.case_condition_field,
            "case_condition_value": self.case_condition_value,
            "case_true_template": self.case_true_template,
            "case_false_template": self.case_false_template,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutableProgram":
        steps = tuple(str(s) for s in data.get("step_templates", ()))
        fields = tuple(str(f) for f in data.get("readout_fields", ()))
        ast_nodes = tuple(ASTNode.from_dict(n) for n in data.get("ast_nodes", ()))
        edges = tuple(
            DefUseEdge(
                producer_step=int(e["producer_step"]),
                variable_name=str(e["variable_name"]),
                field_path=str(e["field_path"]),
                consumer_step=int(e["consumer_step"]) if e.get("consumer_step") is not None else None,
            )
            for e in data.get("def_use_edges", ())
        )
        return cls(
            program_id=str(data.get("program_id") or _short_hash((data.get("benchmark", ""), steps, fields))),
            benchmark=str(data.get("benchmark", "alfworld")),
            step_templates=steps,
            readout_fields=fields,
            origin=str(data.get("origin", "starter")),
            parent_ids=tuple(str(p) for p in data.get("parent_ids", ())),
            expected_cost=float(data.get("expected_cost", max(1, len(steps)))),
            ast_nodes=ast_nodes,
            variable_scope=tuple(str(v) for v in data.get("variable_scope", ())),
            role_bindings={str(k): str(v) for k, v in dict(data.get("role_bindings", {})).items()},
            def_use_edges=edges,
            case_condition_field=str(data.get("case_condition_field", "")),
            case_condition_value=str(data.get("case_condition_value", "")),
            case_true_template=str(data.get("case_true_template", "")),
            case_false_template=str(data.get("case_false_template", "")),
        )


# Backward-compatible alias for existing serialized libraries and imports
BoundedObservationProgram = ExecutableProgram


@dataclass
class ProgramExecutionState:
    """Step-by-step runtime interpreter state preserving residual continuation across replanning."""

    program: ExecutableProgram
    program_counter: int = 0
    bound_variables: dict[str, str] = field(default_factory=dict)
    observed_readouts: dict[str, str] = field(default_factory=dict)
    executed_actions: list[str] = field(default_factory=list)
    binding_failed: bool = False
    completed: bool = False

    def remaining_templates(self) -> tuple[str, ...]:
        if self.completed or self.binding_failed:
            return ()
        rem = list(self.program.step_templates[self.program_counter :])
        if not rem and self.program.case_condition_field:
            cond_val = self.observed_readouts.get(self.program.case_condition_field, "unknown")
            if cond_val != "unknown":
                branch_tmpl = (
                    self.program.case_true_template
                    if cond_val == self.program.case_condition_value
                    else self.program.case_false_template
                )
                if branch_tmpl:
                    rem.append(branch_tmpl)
        return tuple(rem)

    def residual_program(self) -> ExecutableProgram | None:
        rem = self.remaining_templates()
        if not rem:
            return None
        return ExecutableProgram(
            program_id=f"{self.program.program_id}:pc{self.program_counter}",
            benchmark=self.program.benchmark,
            step_templates=rem,
            readout_fields=self.program.readout_fields,
            origin=self.program.origin,
            parent_ids=(self.program.program_id,),
            expected_cost=float(len(rem)),
            role_bindings=dict(self.bound_variables),
            case_condition_field=self.program.case_condition_field,
            case_condition_value=self.program.case_condition_value,
            case_true_template=self.program.case_true_template,
            case_false_template=self.program.case_false_template,
        )

    def clone(self) -> "ProgramExecutionState":
        return deepcopy(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program.program_id,
            "program_counter": self.program_counter,
            "bound_variables": dict(self.bound_variables),
            "observed_readouts": dict(self.observed_readouts),
            "executed_actions": list(self.executed_actions),
            "binding_failed": self.binding_failed,
            "completed": self.completed,
            "remaining_templates": list(self.remaining_templates()),
        }


@dataclass(frozen=True)
class PublicTransition:
    """Supervised physical transition (h_t, a_t, x_{t+1}) in a measurement cluster."""

    transition_id: str
    task_id: str
    episode_id: str
    cluster_id: str
    benchmark: str
    step_index: int
    goal_text: str
    pre_history_ref: str
    pre_public_fields: dict[str, str]
    action_text: str
    action_template: str
    role_bindings: dict[str, str]
    raw_response: str
    typed_response: dict[str, str]
    resource_delta: dict[str, float]
    remaining_budget: dict[str, float]
    termination: str  # "live" | "native_terminal" | "task_budget_terminal"
    native_outcome: bool | None  # None when live or interrupted by global job budget

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MeasurementRecord:
    """Paired intervention record separating observed terminal d_real from model estimate d_model."""

    record_id: str
    benchmark: str
    task_id: str
    goal_category: str
    anchor_signature: str
    program_id: str
    step_templates: tuple[str, ...]
    readout_fields: tuple[str, ...]
    public_readout_z: dict[str, str]
    z_parity_bit: int
    continuation_pair_key: str
    u_score: float
    v_score: float
    u_cost: int
    v_cost: int
    u_done: bool
    v_done: bool
    action_diff_d: float
    outcome_bin: int
    surprise: float
    rebound_trace_templates: tuple[str, ...]
    rebound_trace_fields: tuple[str, ...]
    post_q_checkpoint_id: str = ""
    u_policy_id: str = "u_continuation"
    v_policy_id: str = "v_continuation"
    tail_policy_version: str = TAIL_POLICY_VERSION
    factual_tail_ref: str = ""
    counterfactual_tail_ref: str = ""
    u_native_outcome: bool | None = None
    v_native_outcome: bool | None = None
    d_real: float | None = None
    d_model: float | None = None
    d_source: str = "observed_native_terminal"
    role_bindings: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Explicit contract alias
Measurement = MeasurementRecord


@dataclass(frozen=True)
class PlanningDecision:
    """Traceable planning record linking candidate Q evaluations to the executed concrete action."""

    decision_id: str
    task_id: str
    step_index: int
    candidate_ids: tuple[str, ...]
    candidate_actions: tuple[str, ...]
    predicted_q_values: tuple[float, ...]
    member_q_values: tuple[tuple[float, ...], ...]
    model_version: str
    tail_policy_version: str
    planning_compute_cost: dict[str, Any]
    selected_candidate_id: str
    executed_action: str
    residual_program_state: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProposalRun:
    """Budget-accounted proposal audit record for matched fixed-D and closed-loop comparisons."""

    run_id: str
    benchmark: str
    dataset_version: str
    model_version: str
    grammar_version: str
    record_order: tuple[str, ...]
    call_index: int
    candidates_constructed: int
    candidates_checked: int
    candidates_scored: int
    label_view_mode: str
    label_view_values: tuple[float, ...]
    unique_witness_pairs: int
    cumulative_witness_visits: int
    added_program_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PublicBenchmarkBinder:
    """Public field extractor, reversible entity role binder, and program template interpreter.

    Strictly separates:
      - Public response extraction (supervising K_phi and updating intra-task state)
      - Official native terminal utility U_B(Y, C) (the sole control and evaluation objective)
    """

    @classmethod
    def starter_programs(cls, benchmark: str) -> list[ExecutableProgram]:
        name = benchmark.split(":", 1)[0]
        if name == "alfworld":
            specs = [
                (("look",), ("room_receptacles", "target_object_visible")),
                (("inventory",), ("inventory_object", "object_treatment_state")),
                (("open {target_receptacle}",), ("open_receptacle_state", "target_object_visible")),
                (("examine {target_receptacle}",), ("open_receptacle_state", "object_treatment_state")),
            ]
        elif name == "webshop":
            specs = [
                (("search[{goal_query}]",), ("page_type", "price_rating_match")),
                (("click[{candidate_asin}]",), ("current_asin", "selected_options")),
                (("click[attributes]", "click[< prev]"), ("current_asin", "inspected_attributes")),
                (("click[description]", "click[< prev]"), ("current_asin", "inspected_attributes")),
            ]
        elif name == "loca":
            specs = [
                (("inspect_schema:{primary_read_tool}",), ("last_tool_family", "prerequisite_status")),
                (("read_public_resource:{primary_read_tool}",), ("bound_identifiers", "prerequisite_status")),
                (("verify_workspace:{secondary_read_tool}",), ("bound_identifiers", "workspace_artifact_state")),
            ]
        else:
            specs = [
                (("probe_step_1",), ("public_field_0", "public_field_1")),
                (("probe_step_2",), ("public_field_1", "public_field_2")),
            ]
        result = []
        for steps, fields in specs:
            pid = _short_hash((name, steps, fields))
            result.append(
                ExecutableProgram(
                    program_id=pid,
                    benchmark=name,
                    step_templates=steps,
                    readout_fields=fields,
                    origin="starter",
                    expected_cost=float(len(steps)),
                )
            )
        return result

    @classmethod
    def extract_task_instruction(
        cls,
        messages: Sequence[Mapping[str, Any]],
        fallback: str = "",
    ) -> str:
        """Extract the actual public task goal g from message history, skipping system prompts."""
        if not messages:
            return fallback
        for msg in messages:
            role = str(msg.get("role", "")).lower()
            content = str(msg.get("content", "")).strip()
            if role == "user" and content:
                return content
        # If only one message exists and is not tagged user, inspect last non-system message
        for msg in reversed(messages):
            if str(msg.get("role", "")).lower() != "system":
                content = str(msg.get("content", "")).strip()
                if content:
                    return content
        return fallback or str(messages[0].get("content", ""))

    @classmethod
    def extract_role_bindings(
        cls,
        benchmark: str,
        task_instruction: str,
        events: Sequence[Any],
        action_schema: Mapping[str, Any],
    ) -> dict[str, str]:
        """Construct a reversible role-binding table mapping symbolic slots to concrete entities."""
        name = benchmark.split(":", 1)[0]
        obs_texts = [str(getattr(ev, "observation", "")) for ev in events]
        joined_obs = "\n".join(obs_texts[-6:]) if obs_texts else ""
        bindings: dict[str, str] = {}

        if name == "alfworld":
            goal_lower = task_instruction.lower()
            goal_nouns = [
                tok for tok in re.findall(r"\b([a-z]{3,})\b", goal_lower)
                if tok not in {"put", "the", "and", "in", "on", "to", "from", "with", "look", "find", "clean", "heat", "cool", "examine", "turn", "under", "some", "two"}
            ]
            if goal_nouns:
                bindings["target_object_type"] = goal_nouns[0]
            if len(goal_nouns) > 1:
                bindings["goal_receptacle_type"] = goal_nouns[-1]

            mode_actions = action_schema.get("mode_actions", {})
            all_actions = list(mode_actions.get("I", ())) + list(mode_actions.get("E", ()))
            if not all_actions and action_schema.get("available_actions"):
                all_actions = [str(a) for a in action_schema.get("available_actions", ())]

            receptacles = sorted(set(re.findall(r"\b([a-z]+\s+\d+)\b", joined_obs.lower())))
            for idx, rec in enumerate(receptacles[:4]):
                bindings[f"receptacle_{idx + 1}"] = rec
            if "goal_receptacle_type" in bindings:
                matching_rec = [
                    r for r in receptacles if r.startswith(bindings["goal_receptacle_type"])
                ]
                if matching_rec:
                    bindings["target_receptacle"] = matching_rec[0]
            if "target_receptacle" not in bindings:
                for act in all_actions:
                    m_open = re.match(r"^(?:open|go to|examine)\s+([a-z]+\s+\d+)$", str(act).strip().lower())
                    if m_open:
                        bindings["target_receptacle"] = m_open.group(1)
                        break
            if "target_object_type" in bindings:
                for act in all_actions:
                    m_take = re.match(r"^take\s+([a-z]+\s+\d+)\s+from\s+([a-z]+\s+\d+)$", str(act).strip().lower())
                    if m_take and m_take.group(1).startswith(bindings["target_object_type"]):
                        bindings["target_object"] = m_take.group(1)
                        bindings["source_receptacle"] = m_take.group(2)
                        break
            return bindings

        if name == "webshop":
            words = [
                w for w in re.findall(r"[a-z0-9]+", task_instruction.lower())
                if len(w) > 2 and w not in {"instruction", "find", "buy", "with", "and", "the", "for", "price", "lower", "than", "dollars"}
            ][:6]
            if words:
                bindings["goal_query"] = " ".join(words)
            clickables = [str(c) for c in action_schema.get("available_actions", ())]
            if not clickables:
                mode_actions = action_schema.get("mode_actions", {})
                clickables = [str(c) for c in (*mode_actions.get("I", ()), *mode_actions.get("E", ()))]
            asins: list[str] = []
            for c in clickables:
                m_asin = re.search(r"click\[(b0[0-9a-z]{8})\]", c.lower())
                if m_asin and m_asin.group(1) not in asins:
                    asins.append(m_asin.group(1))
            if asins:
                bindings["candidate_asin"] = asins[0]
                for idx, asin in enumerate(asins[:3]):
                    bindings[f"candidate_asin_{idx + 1}"] = asin
            return bindings

        # LOCA / structured tool environments
        prog_ctx = action_schema.get("program_context", action_schema)
        tools = prog_ctx.get("tools", action_schema.get("tools", ()))
        read_tools: list[str] = []
        write_tools: list[str] = []
        for tool in tools if isinstance(tools, Sequence) else ():
            t_name = str(tool.get("name", "")) if isinstance(tool, Mapping) else str(tool)
            if not t_name:
                continue
            if any(k in t_name.lower() for k in ("get", "list", "read", "search", "inspect", "find", "query", "status")):
                read_tools.append(t_name)
            else:
                write_tools.append(t_name)
        if read_tools:
            bindings["primary_read_tool"] = read_tools[0]
            bindings["secondary_read_tool"] = read_tools[min(1, len(read_tools) - 1)]
        if write_tools:
            bindings["primary_write_tool"] = write_tools[0]
        for ev in events[-4:]:
            obs_str = str(getattr(ev, "observation", ""))
            for key_name, val in re.findall(r'"(id|path|file|token|key|record_id)"\s*:\s*"([^"]{1,64})"', obs_str):
                bindings[f"bound_{key_name}"] = val
        return bindings

    @classmethod
    def extract_public_fields(
        cls,
        benchmark: str,
        task_instruction: str,
        events: Sequence[Any],
        action_schema: Mapping[str, Any],
    ) -> dict[str, str]:
        """Extract structured public fields from the observed history without guessing unobserved facts."""
        name = benchmark.split(":", 1)[0]
        if not events:
            if name == "alfworld":
                return {
                    "room_receptacles": "unknown",
                    "open_receptacle_state": "unknown",
                    "inventory_object": "unknown",
                    "object_treatment_state": "unknown",
                    "target_object_visible": "unknown",
                }
            if name == "webshop":
                return {
                    "page_type": "search_home" if action_schema.get("search_available") else "unknown",
                    "current_asin": "unknown",
                    "selected_options": "none",
                    "inspected_attributes": "none",
                    "price_rating_match": "unknown",
                }
            return {
                "last_tool_family": "none",
                "bound_identifiers": "count_0",
                "prerequisite_status": "unknown",
                "workspace_artifact_state": "unknown",
            }

        obs_texts = [str(getattr(ev, "observation", "")) for ev in events]
        joined_obs = "\n".join(obs_texts[-4:]) if obs_texts else ""
        all_obs_lower = "\n".join(obs_texts).lower()
        goal_lower = task_instruction.lower()

        if name == "alfworld":
            goal_tokens = [
                tok for tok in re.findall(r"[a-z]{3,}", goal_lower)
                if tok not in {"put", "the", "and", "in", "on", "to", "from", "with", "look", "find", "some", "two"}
            ]
            visible_receptacles = sorted(set(re.findall(r"\b([a-z]+\s+\d+)\b", joined_obs.lower())))
            open_state = "unknown"
            if "is open" in joined_obs.lower() or "you open" in joined_obs.lower():
                open_state = (
                    "open_with_items"
                    if ("in it, you see" in joined_obs.lower() and "nothing" not in joined_obs.lower())
                    else "open_empty"
                )
            elif "is closed" in joined_obs.lower():
                open_state = "closed"

            # Track held object across full episode events accurately
            held_item = "none"
            for ev in events:
                obs_l = str(getattr(ev, "observation", "")).lower()
                act_l = str(getattr(getattr(ev, "generation", None), "text", "") or getattr(ev, "action", {}).get("text", "")).lower()
                picked = re.findall(r"(?:you pick up|you are carrying:)\s+([^\.\n]+)", obs_l)
                if picked and "nothing" not in picked[-1]:
                    held_item = picked[-1].strip()
                elif act_l.startswith("take ") and ("you pick up" in obs_l or "you take" in obs_l):
                    held_item = act_l.replace("take ", "").split(" from ")[0].strip()
                elif (act_l.startswith("put ") or act_l.startswith("move ")) and ("you put" in obs_l or "you move" in obs_l):
                    held_item = "none"

            treatment = "raw"
            for tag, phrase in (
                ("clean", "you clean"),
                ("hot", "you heat"),
                ("cool", "you cool"),
                ("examined", "you turn on"),
            ):
                if phrase in all_obs_lower:
                    treatment = tag
                    break
            target_visible = (
                "yes"
                if (goal_tokens and any(tok in joined_obs.lower() for tok in goal_tokens[:2]))
                else "no"
            )
            return {
                "room_receptacles": ",".join(visible_receptacles[:4]) or "none",
                "open_receptacle_state": open_state,
                "inventory_object": held_item[:32],
                "object_treatment_state": treatment,
                "target_object_visible": target_visible,
            }

        if name == "webshop":
            clickables = [str(c) for c in action_schema.get("available_actions", ())]
            if not clickables:
                mode_actions = action_schema.get("mode_actions", {})
                clickables = [str(c) for c in (*mode_actions.get("I", ()), *mode_actions.get("E", ()))]
            has_search = bool(action_schema.get("search_available", False))
            asin_matches = re.findall(r"\bb0[0-9a-z]{8}\b", joined_obs.lower())
            current_asin = asin_matches[-1] if asin_matches else "none"
            page_type = (
                "search_home" if has_search and not asin_matches
                else "search_results" if "next >" in " ".join(clickables).lower()
                else "item_subpage" if "< prev" in " ".join(clickables).lower() and "buy now" not in " ".join(clickables).lower()
                else "item_detail"
            )
            chosen_opts = [
                str(getattr(ev, "action", {}).get("text", "") or getattr(getattr(ev, "generation", None), "text", ""))
                for ev in events
                if str(getattr(ev, "action", {}).get("text", "") or getattr(getattr(ev, "generation", None), "text", "")).lower().startswith("click[")
                and "buy now" not in str(getattr(ev, "action", {}).get("text", "") or getattr(getattr(ev, "generation", None), "text", "")).lower()
                and not re.search(r"click\[(?:b0[0-9a-z]{8}|description|features|attributes|reviews|< prev|next >|back to search)\]", str(getattr(ev, "action", {}).get("text", "") or getattr(getattr(ev, "generation", None), "text", "")).lower())
            ]
            inspected = [
                sub for sub in ("description", "features", "attributes", "reviews")
                if any(
                    f"click[{sub}]" in str(getattr(ev, "action", {}).get("text", "") or getattr(getattr(ev, "generation", None), "text", "")).lower()
                    for ev in events
                )
            ]
            goal_words = set(re.findall(r"[a-z0-9]+", goal_lower))
            obs_words = set(re.findall(r"[a-z0-9]+", joined_obs.lower()))
            overlap = len(goal_words & obs_words) / max(1, len(goal_words))
            return {
                "page_type": page_type,
                "current_asin": current_asin,
                "selected_options": ",".join(chosen_opts[-2:]) or "none",
                "inspected_attributes": ",".join(inspected) or "none",
                "price_rating_match": "high" if overlap >= 0.45 else "low",
            }

        tool_families = []
        bound_ids = []
        had_error = False
        artifact_state = "unchanged"
        for ev in events[-4:]:
            act_text = str(getattr(ev, "generation", None).text if getattr(ev, "generation", None) else "")
            obs_str = str(getattr(ev, "observation", ""))
            tool_families.append(act_text[:32] or "call")
            for m in re.findall(r'"(?:id|path|file|token|key|record_id)"\s*:\s*"([^"]{1,32})"', obs_str):
                bound_ids.append(m)
            if "error" in obs_str.lower() or "exception" in obs_str.lower() or "invalid" in obs_str.lower():
                had_error = True
            if not had_error and ("success" in obs_str.lower() or "written" in obs_str.lower() or "updated" in obs_str.lower()):
                artifact_state = "verified_write"
        return {
            "last_tool_family": tool_families[-1] if tool_families else "none",
            "bound_identifiers": f"count_{min(4, len(set(bound_ids)))}",
            "prerequisite_status": "error" if had_error else ("ready" if bound_ids else "pending"),
            "workspace_artifact_state": artifact_state,
        }

    @staticmethod
    def native_terminal_utility(
        native_outcome: bool | None,
        cost_used: float,
        cost_budget: float,
    ) -> float | None:
        """Compute the sole control objective U_B(Y, C) = Y * max(0, 1 - C / B).

        Returns None when native_outcome is unobserved (live or job-budget truncated).
        """
        if native_outcome is None:
            return None
        if not native_outcome:
            return 0.0
        if cost_budget <= 0:
            return 0.0
        ratio = max(0.0, min(1.0, float(cost_used) / float(cost_budget)))
        return float(max(0.0, 1.0 - ratio))

    @classmethod
    def public_goal_readout(
        cls,
        benchmark: str,
        task_instruction: str,
        events: Sequence[Any],
        action_schema: Mapping[str, Any],
        *,
        terminal_success: bool = False,
        within_budget: bool = True,
        cost_used: float = 0.0,
        cost_budget: float = 1.0,
    ) -> float:
        """Return official terminal utility only; mid-episode hand-crafted progress is removed."""
        if not within_budget or not terminal_success:
            return 0.0
        val = cls.native_terminal_utility(True, cost_used, max(1e-8, cost_budget))
        return float(val if val is not None else 0.0)

    @classmethod
    def encode_response_outcome_bin(
        cls,
        pre_fields: Mapping[str, str],
        post_fields: Mapping[str, str],
        *,
        had_error: bool = False,
        terminal_done: bool = False,
        terminal_success: bool = False,
    ) -> int:
        """Unified 8-category response schema shared identically by training, filtering, and rollout."""
        changed_fields = sum(1 for k, v in post_fields.items() if pre_fields.get(k, "unknown") != v)
        info_bit = 1 if changed_fields > 0 else 0
        entity_bound = any(
            post_fields.get(k, "unknown") not in ("unknown", "none", "count_0")
            for k in ("inventory_object", "current_asin", "bound_identifiers", "target_object_visible")
        )
        bind_bit = 1 if (entity_bound or terminal_success) else 0
        valid_bit = 0 if (had_error or (terminal_done and not terminal_success)) else 1
        return int((4 * info_bit) + (2 * bind_bit) + valid_bit)

    @staticmethod
    def abstract_action_to_template(
        benchmark: str,
        action_text: str,
        role_bindings: Mapping[str, str] | None = None,
    ) -> str:
        """Rebind concrete entity IDs in an executed action to reversible symbolic role slots."""
        name = benchmark.split(":", 1)[0]
        text = action_text.strip()
        inv_bindings = {v.lower(): k for k, v in (role_bindings or {}).items() if v}

        if name == "alfworld":
            lowered = text.lower()
            for entity_val, slot_name in sorted(inv_bindings.items(), key=lambda x: len(x[0]), reverse=True):
                if entity_val in lowered:
                    lowered = lowered.replace(entity_val, f"{{{slot_name}}}")
            lowered = re.sub(r"\b([a-z]+)\s+\d+\b", r"{\1_id}", lowered)
            return lowered[:64]

        if name == "webshop":
            if text.lower().startswith("search["):
                return "search[{goal_query}]"
            if re.search(r"click\[b0[0-9a-z]{8}\]", text.lower()):
                return "click[{candidate_asin}]"
            return text[:64]

        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "name" in obj:
                tool_name = str(obj["name"])
                args = obj.get("arguments", {})
                arg_keys = sorted(args.keys()) if isinstance(args, dict) else []
                return f"{tool_name}({','.join(arg_keys[:3])})"
        except Exception:
            pass
        return text[:64]

    @classmethod
    def enumerate_admissible_actions(
        cls,
        benchmark: str,
        action_schema: Mapping[str, Any],
        task_instruction: str = "",
        role_bindings: Mapping[str, str] | None = None,
    ) -> list[str]:
        """Return the unified full native action support without partitioning or dropping I/E classes."""
        name = benchmark.split(":", 1)[0]
        mode_actions = action_schema.get("mode_actions", {})
        candidates: list[str] = []
        for act in (*mode_actions.get("I", ()), *mode_actions.get("E", ())):
            s = str(act).strip()
            if s and s not in candidates:
                candidates.append(s)
        for act in action_schema.get("available_actions", ()):
            s = str(act).strip()
            if s and s not in candidates:
                candidates.append(s)
        if name == "webshop" and action_schema.get("search_available"):
            query = (role_bindings or {}).get("goal_query", "")
            if not query and task_instruction:
                words = [w for w in re.findall(r"[a-z0-9]+", task_instruction.lower()) if len(w) > 2][:6]
                query = " ".join(words) or "product"
            if query:
                search_act = f"search[{query}]"
                if search_act not in candidates:
                    candidates.insert(0, search_act)
        return candidates

    @classmethod
    def instantiate_template_action(
        cls,
        benchmark: str,
        template: str,
        action_schema: Mapping[str, Any],
        task_instruction: str,
        role_bindings: Mapping[str, str] | None = None,
    ) -> str | None:
        """Bind a symbolic step template to a concrete admissible action.

        Returns None on binding failure; never silently substitutes an unrelated default action.
        """
        name = benchmark.split(":", 1)[0]
        bindings = dict(role_bindings or {})
        if not bindings:
            bindings = cls.extract_role_bindings(benchmark, task_instruction, (), action_schema)
        all_candidates = cls.enumerate_admissible_actions(benchmark, action_schema, task_instruction, bindings)

        # Direct match
        if template in all_candidates:
            return template

        # Substitute known role bindings into template slots
        resolved = template
        for slot in re.findall(r"\{([a-zA-Z0-9_]+)\}", template):
            if slot in bindings and bindings[slot]:
                resolved = resolved.replace(f"{{{slot}}}", bindings[slot])
        if resolved in all_candidates:
            return resolved
        for cand in all_candidates:
            if cand.lower() == resolved.lower():
                return cand

        if name == "alfworld":
            verb_prefix = template.split("{", 1)[0].strip().lower()
            if not verb_prefix:
                return None
            slot_matches = re.findall(r"\{([a-z0-9_]+)\}", template.lower())
            matching: list[str] = []
            for cand in all_candidates:
                cand_l = cand.lower()
                if not cand_l.startswith(verb_prefix):
                    continue
                if slot_matches:
                    slot_type = slot_matches[0].replace("_id", "").replace("target_", "").replace("source_", "")
                    if slot_type and slot_type not in ("receptacle", "object") and slot_type not in cand_l:
                        continue
                matching.append(cand)
            return matching[0] if matching else None

        if name == "webshop":
            if template.startswith("search[") and action_schema.get("search_available"):
                query = bindings.get("goal_query", "")
                if not query:
                    words = [w for w in re.findall(r"[a-z0-9]+", task_instruction.lower()) if len(w) > 2][:6]
                    query = " ".join(words)
                return f"search[{query}]" if query else None
            if template == "click[{candidate_asin}]":
                target_asin = bindings.get("candidate_asin", "")
                for cand in all_candidates:
                    if target_asin and f"click[{target_asin}]" == cand.lower():
                        return cand
                asin_clicks = [c for c in all_candidates if re.search(r"click\[b0[0-9a-z]{8}\]", c.lower())]
                return asin_clicks[0] if asin_clicks else None
            for cand in all_candidates:
                if cand.lower() == template.lower():
                    return cand
            return None

        # LOCA / structured tool environment
        prog_ctx = action_schema.get("program_context", action_schema)
        tools = prog_ctx.get("tools", action_schema.get("tools", ()))
        tool_names = [
            str(t.get("name", "")) if isinstance(t, Mapping) else str(t)
            for t in (tools if isinstance(tools, Sequence) else ())
        ]
        for slot_key in ("primary_read_tool", "secondary_read_tool", "primary_write_tool"):
            if f"{{{slot_key}}}" in resolved and bindings.get(slot_key):
                resolved = resolved.replace(f"{{{slot_key}}}", bindings[slot_key])
        tool_head = resolved.split(":", 1)[-1].split("(", 1)[0].strip()
        if tool_head in tool_names:
            args_payload: dict[str, Any] = {}
            for k, v in bindings.items():
                if k.startswith("bound_"):
                    args_payload[k.replace("bound_", "")] = v
            return json.dumps({"name": tool_head, "arguments": args_payload}, ensure_ascii=False, sort_keys=True)
        return None

    @classmethod
    def step_program_interpreter(
        cls,
        state: ProgramExecutionState,
        action_schema: Mapping[str, Any],
        task_instruction: str,
        events: Sequence[Any],
    ) -> tuple[str | None, ProgramExecutionState]:
        """Bind and advance one step of `ExecutableProgram`, preserving residual state for replanning."""
        next_state = state.clone()
        if next_state.completed or next_state.binding_failed:
            return None, next_state

        live_bindings = cls.extract_role_bindings(
            next_state.program.benchmark, task_instruction, events, action_schema
        )
        merged_bindings = {**live_bindings, **next_state.program.role_bindings, **next_state.bound_variables}
        next_state.bound_variables = merged_bindings
        next_state.observed_readouts = cls.extract_public_fields(
            next_state.program.benchmark, task_instruction, events, action_schema
        )

        rem = next_state.remaining_templates()
        if not rem:
            next_state.completed = True
            return None, next_state

        current_template = rem[0]
        concrete_action = cls.instantiate_template_action(
            next_state.program.benchmark,
            current_template,
            action_schema,
            task_instruction,
            merged_bindings,
        )
        if concrete_action is None:
            next_state.binding_failed = True
            return None, next_state

        next_state.program_counter += 1
        next_state.executed_actions.append(concrete_action)
        if next_state.program_counter >= len(next_state.program.step_templates) and not next_state.program.case_condition_field:
            next_state.completed = True
        return concrete_action, next_state


class ControlledResponseEnsemble(nn.Module):
    """Normalized context-conditioned residual response ensemble K_{phi,m}(x | g, h, a, beta).

    Parameterizes:
      p_{phi,m}(x_i | c, x_{<i}) = softmax(ell_0(c, x_{<i}) + r_{phi,m}(c, x_{<i}))_{x_i}
    where ell_0 comes from the frozen backbone prior (when available) and r_{phi,m} is the
    trainable side network initialized at zero residual so K_{phi_0, m} == p_0.

    Predicts:
      - Public response category distribution (8 unified categories)
      - Step resource cost ratio in (0, 1)
      - Termination distribution over ("live", "native_terminal", "task_budget_terminal")
      - Conditional native terminal success probability P_m(Y = 1 | g, h, a, beta, pi_0)
    """

    def __init__(
        self,
        ensemble_size: int = DEFAULT_ENSEMBLE_SIZE,
        state_dim: int = STATE_DIM,
        outcome_bins: int = OUTCOME_BINS,
        context_dim: int = CONTEXT_DIM,
    ) -> None:
        super().__init__()
        self.ensemble_size = int(ensemble_size)
        self.state_dim = int(state_dim)
        self.outcome_bins = int(outcome_bins)
        self.context_dim = int(context_dim)

        # Context encoder mapping (g, h_t, a_t, beta_t, frozen_backbone_summary) -> context_dim
        self.context_encoder = nn.Sequential(
            nn.Linear(self.context_dim, self.context_dim),
            nn.LayerNorm(self.context_dim),
            nn.SiLU(),
            nn.Linear(self.context_dim, self.context_dim),
        )
        # State-context-transition interaction: (context_dim) -> (M * S * S)
        self.transition_residual_proj = nn.Linear(
            self.context_dim, self.ensemble_size * self.state_dim * self.state_dim
        )
        # Explicit state-context-outcome emission interaction: (context_dim) -> (M * S * O)
        # Fixes the constant-shift softmax cancellation bug (I11).
        self.emission_residual_proj = nn.Linear(
            self.context_dim, self.ensemble_size * self.state_dim * self.outcome_bins
        )
        # Per-member base transition and emission parameters
        self.transition_logits = nn.Parameter(
            0.02 * torch.randn(self.ensemble_size, self.state_dim, self.state_dim)
        )
        self.emission_logits = nn.Parameter(
            0.02 * torch.randn(self.ensemble_size, self.state_dim, self.outcome_bins)
        )
        # Residual heads for termination (3 classes), cost ratio, and terminal outcome Y under tail pi_0
        self.termination_head = nn.Linear(
            self.state_dim + self.context_dim, self.ensemble_size * len(TERMINATION_CATEGORIES)
        )
        self.cost_head = nn.Linear(self.state_dim + self.context_dim, self.ensemble_size)
        self.terminal_outcome_head = nn.Linear(
            self.state_dim + self.context_dim + self.outcome_bins, self.ensemble_size
        )

        # Zero-initialize residual projections so K_{phi_0} preserves the base prior at initialization
        nn.init.zeros_(self.transition_residual_proj.weight)
        nn.init.zeros_(self.transition_residual_proj.bias)
        nn.init.zeros_(self.emission_residual_proj.weight)
        nn.init.zeros_(self.emission_residual_proj.bias)
        nn.init.zeros_(self.termination_head.weight)
        nn.init.zeros_(self.termination_head.bias)
        nn.init.zeros_(self.cost_head.weight)
        nn.init.zeros_(self.cost_head.bias)
        nn.init.zeros_(self.terminal_outcome_head.weight)
        nn.init.zeros_(self.terminal_outcome_head.bias)

        # Uniform member weights (used for finite-model reference or uniform bootstrap averaging)
        self.register_buffer(
            "mechanism_weights",
            torch.full((self.ensemble_size,), 1.0 / self.ensemble_size, dtype=torch.float32),
        )
        # Save complete initial state dict for exact phi_0 evaluation and reset (fixes E04)
        self._initial_state_tensors: dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in self.state_dict().items()
        }

    def snapshot_initial_parameters(self) -> None:
        """Record the exact initial parameter snapshot phi_0 across all layers."""
        self._initial_state_tensors = {
            k: v.detach().clone() for k, v in self.state_dict().items()
        }

    def restore_initial_parameters(self) -> None:
        """Restore all parameters and buffers to their exact phi_0 snapshot."""
        if self._initial_state_tensors:
            dev = self.transition_logits.device
            restored = {k: v.to(dev).clone() for k, v in self._initial_state_tensors.items()}
            self.load_state_dict(restored, strict=True)

    def reset_intra_task_belief(self, device: torch.device | None = None) -> torch.Tensor:
        """Reset intra-task state s_m(h_0) at the beginning of a new task episode."""
        dev = device or self.transition_logits.device
        return torch.full(
            (self.ensemble_size, self.state_dim),
            1.0 / self.state_dim,
            device=dev,
            dtype=torch.float32,
        )

    @classmethod
    def encode_context_vector(
        cls,
        *,
        goal_text: str,
        public_fields: Mapping[str, str],
        action_text: str,
        remaining_budget_ratio: float = 1.0,
        prior_action_logprob: float = 0.0,
        prior_success_logit: float = 0.0,
        program: ExecutableProgram | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Encode c = (g, h_t, a_t, beta_t, ell_0) into a 64-dim structured semantic context vector."""
        vec = np.zeros(CONTEXT_DIM, dtype=np.float32)
        goal_tokens = set(re.findall(r"[a-z0-9]{3,}", goal_text.lower()))
        act_tokens = set(re.findall(r"[a-z0-9]{2,}", action_text.lower()))
        field_vals = " ".join(f"{k}:{v}" for k, v in sorted(public_fields.items())).lower()
        field_tokens = set(re.findall(r"[a-z0-9_]{2,}", field_vals))

        # Goal-action and goal-history lexical alignment features
        vec[0] = float(min(1.0, max(0.0, remaining_budget_ratio)))
        vec[1] = float(np.tanh(prior_action_logprob / 5.0))
        vec[2] = float(np.tanh(prior_success_logit / 4.0))
        vec[3] = len(goal_tokens & act_tokens) / max(1.0, float(len(goal_tokens)))
        vec[4] = len(goal_tokens & field_tokens) / max(1.0, float(len(goal_tokens)))
        vec[5] = len(act_tokens & field_tokens) / max(1.0, float(len(act_tokens)))

        # Concrete action semantics
        act_l = action_text.strip().lower()
        vec[6] = 1.0 if act_l.startswith(("look", "inventory", "examine", "search[")) else 0.0
        vec[7] = 1.0 if act_l.startswith(("go to", "open", "click[")) else 0.0
        vec[8] = 1.0 if act_l.startswith(("take", "move", "put", "clean", "heat", "cool", "use")) else 0.0
        vec[9] = 1.0 if ("buy now" in act_l or "claim_done" in act_l or "submit" in act_l) else 0.0

        # Structured public field indicators (never guessing unobserved fields)
        vec[10] = 1.0 if public_fields.get("target_object_visible") == "yes" else (
            -1.0 if public_fields.get("target_object_visible") == "no" else 0.0
        )
        vec[11] = 1.0 if public_fields.get("inventory_object", "unknown") not in ("unknown", "none") else 0.0
        vec[12] = 1.0 if public_fields.get("object_treatment_state", "unknown") not in ("unknown", "raw") else 0.0
        vec[13] = 1.0 if public_fields.get("open_receptacle_state") == "open_with_items" else 0.0
        vec[14] = 1.0 if public_fields.get("current_asin", "unknown") not in ("unknown", "none") else 0.0
        vec[15] = 1.0 if public_fields.get("selected_options", "none") not in ("unknown", "none") else 0.0
        vec[16] = 1.0 if public_fields.get("inspected_attributes", "none") not in ("unknown", "none") else 0.0
        vec[17] = 1.0 if public_fields.get("price_rating_match") == "high" else 0.0
        vec[18] = 1.0 if public_fields.get("bound_identifiers", "count_0") not in ("unknown", "count_0") else 0.0
        vec[19] = 1.0 if public_fields.get("prerequisite_status") == "ready" else (
            -1.0 if public_fields.get("prerequisite_status") == "error" else 0.0
        )
        vec[20] = 1.0 if public_fields.get("workspace_artifact_state") == "verified_write" else 0.0

        if program is not None:
            vec[21] = min(1.0, len(program.step_templates) / 3.0)
            vec[22] = min(1.0, len(program.readout_fields) / 4.0)
            vec[23] = 1.0 if program.case_condition_field else 0.0

        # Deterministic projections of (goal, public_fields, action_text) for fine-grained discrimination
        for offset, payload in (
            (24, goal_text),
            (36, _canonical_json(dict(public_fields))),
            (48, action_text),
        ):
            digest = hashlib.sha256(payload.encode("utf-8")).digest()
            span = 12 if offset < 48 else 16
            for idx in range(span):
                vec[offset + idx] = (digest[idx] / 127.5) - 1.0

        return torch.tensor(vec, device=device, dtype=torch.float32)

    @classmethod
    def encode_program_features(
        cls,
        program: ExecutableProgram,
        public_fields: Mapping[str, str],
        device: torch.device,
        *,
        goal_text: str = "",
        action_text: str = "",
        remaining_budget_ratio: float = 1.0,
        prior_action_logprob: float = 0.0,
        prior_success_logit: float = 0.0,
    ) -> torch.Tensor:
        """Encode program + concrete action + public state into the 64-dim context tensor."""
        act_str = action_text or (program.step_templates[0] if program.step_templates else "")
        return cls.encode_context_vector(
            goal_text=goal_text,
            public_fields=public_fields,
            action_text=act_str,
            remaining_budget_ratio=remaining_budget_ratio,
            prior_action_logprob=prior_action_logprob,
            prior_success_logit=prior_success_logit,
            program=program,
            device=device,
        )

    def _coerce_context(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.numel() == self.context_dim:
            return feat.view(self.context_dim)
        padded = torch.zeros(self.context_dim, device=feat.device, dtype=feat.dtype)
        n = min(self.context_dim, feat.numel())
        padded[:n] = feat.reshape(-1)[:n]
        return padded

    def operator_matrices(
        self,
        context_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute normalized transition T_m(s'|s,c) and context-dependent emission G_m(o|s',c)."""
        c_vec = self._coerce_context(context_feat)
        h_ctx = self.context_encoder(c_vec)  # (C,)
        t_res = self.transition_residual_proj(h_ctx).view(
            self.ensemble_size, self.state_dim, self.state_dim
        )
        e_res = self.emission_residual_proj(h_ctx).view(
            self.ensemble_size, self.state_dim, self.outcome_bins
        )
        T = F.softmax(self.transition_logits + t_res, dim=1)  # sum_{s'} T[m, s', s] == 1
        G = F.softmax(self.emission_logits + e_res, dim=-1)  # sum_o G[m, s', o] == 1 (varies over o!)
        return T, G, h_ctx

    def observable_operator_step(
        self,
        belief: torch.Tensor,
        context_feat: torch.Tensor,
        observed_outcome_bin: int | None = None,
        *,
        prior_outcome_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply observable operator M_{a,o}^{(m)} to intra-task state s_m(h) and normalize."""
        T, G, _ = self.operator_matrices(context_feat)
        if prior_outcome_logits is not None and prior_outcome_logits.numel() == self.outcome_bins:
            prior_l = prior_outcome_logits.to(device=G.device, dtype=G.dtype).view(1, 1, self.outcome_bins)
            G = F.softmax(torch.log(G.clamp(min=1e-8)) + prior_l, dim=-1)
        next_state_prior = torch.einsum("mij,mj->mi", T, belief)  # (M, S)
        joint_next = next_state_prior[:, :, None] * G  # (M, S, O)
        outcome_probs = joint_next.sum(dim=1).clamp(min=1e-8)  # (M, O)
        outcome_probs = outcome_probs / outcome_probs.sum(dim=-1, keepdim=True)
        if observed_outcome_bin is None:
            return outcome_probs, next_state_prior
        o_idx = int(observed_outcome_bin) % self.outcome_bins
        unnorm_post = joint_next[:, :, o_idx]
        denom = unnorm_post.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        next_belief = unnorm_post / denom
        return outcome_probs, next_belief

    def observe(
        self,
        previous_state: torch.Tensor,
        *,
        goal_text: str,
        pre_public_fields: Mapping[str, str],
        actual_action: str,
        post_public_fields: Mapping[str, str],
        remaining_budget_ratio: float = 1.0,
        had_error: bool = False,
        terminal_done: bool = False,
        terminal_success: bool = False,
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        """Unified intra-task state update `next_state = observe(previous_state, actual_action, actual_public_response)`."""
        device = self.transition_logits.device
        outcome_bin = PublicBenchmarkBinder.encode_response_outcome_bin(
            pre_public_fields,
            post_public_fields,
            had_error=had_error,
            terminal_done=terminal_done,
            terminal_success=terminal_success,
        )
        ctx = self.encode_context_vector(
            goal_text=goal_text,
            public_fields=pre_public_fields,
            action_text=actual_action,
            remaining_budget_ratio=remaining_budget_ratio,
            device=device,
        )
        outcome_probs, next_state = self.observable_operator_step(
            previous_state, ctx, observed_outcome_bin=outcome_bin
        )
        return next_state, outcome_bin, outcome_probs

    def predict_transition_heads(
        self,
        belief: torch.Tensor,
        context_feat: torch.Tensor,
        *,
        prior_outcome_logits: torch.Tensor | None = None,
        prior_termination_logits: torch.Tensor | None = None,
        prior_success_logit: float = 0.0,
        remaining_budget_ratio: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Predict normalized distributions for response category, cost, termination, and terminal Y."""
        _, _, h_ctx = self.operator_matrices(context_feat)
        outcome_probs, next_state_prior = self.observable_operator_step(
            belief, context_feat, observed_outcome_bin=None, prior_outcome_logits=prior_outcome_logits
        )
        mean_state = next_state_prior.mean(dim=0)  # (S,)
        joint_in = torch.cat([mean_state, h_ctx], dim=-1)  # (S + C,)

        term_res = self.termination_head(joint_in).view(self.ensemble_size, len(TERMINATION_CATEGORIES))
        if prior_termination_logits is not None and prior_termination_logits.numel() == len(TERMINATION_CATEGORIES):
            base_term = prior_termination_logits.to(device=term_res.device, dtype=term_res.dtype).view(1, -1)
        else:
            # Base prior: mostly live unless budget is near zero
            base_term = torch.tensor(
                [[1.5, -1.0, -2.0 if remaining_budget_ratio > 0.1 else 1.0]],
                device=term_res.device,
                dtype=term_res.dtype,
            )
        termination_probs = F.softmax(base_term + term_res, dim=-1)  # (M, 3)

        cost_res = self.cost_head(joint_in)  # (M,)
        base_cost_logit = -2.5  # ~0.075 fraction of budget per step by default
        cost_ratio = torch.sigmoid(base_cost_logit + cost_res)  # (M,)

        out_in = torch.cat([mean_state, h_ctx, outcome_probs.mean(dim=0)], dim=-1)
        y_res = self.terminal_outcome_head(out_in)  # (M,)
        y_prob = torch.sigmoid(float(prior_success_logit) + y_res)  # (M,)

        return {
            "outcome_probs": outcome_probs,          # (M, O)
            "next_state_prior": next_state_prior,    # (M, S)
            "termination_probs": termination_probs,  # (M, 3)
            "cost_ratio": cost_ratio,                # (M,)
            "terminal_success_prob": y_prob,         # (M,)
        }

    @torch.no_grad()
    def evaluate_candidate_q(
        self,
        *,
        goal_text: str,
        public_fields: Mapping[str, str],
        belief: torch.Tensor,
        candidate_action: str,
        program: ExecutableProgram | None = None,
        remaining_budget_ratio: float = 1.0,
        prior_action_logprob: float = 0.0,
        prior_success_logit: float = 0.0,
        member_index: int | None = None,
        use_initial_params: bool = False,
    ) -> dict[str, Any]:
        """Compute unified long-horizon value Q_m^rho(g, h, beta) = E_{K_{phi,m}, rho -> pi_0}[U_B(Y, C)].

        All ensemble members evaluate the same public-history-dependent candidate rho;
        the deployment score is the ensemble mean 1/M sum_m Q_m^rho.
        """
        saved_state: dict[str, torch.Tensor] | None = None
        if use_initial_params and self._initial_state_tensors:
            saved_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
            self.restore_initial_parameters()

        try:
            device = self.transition_logits.device
            cur_belief = belief.clone().to(device)
            steps_to_roll = (
                list(program.step_templates)
                if (program is not None and program.step_templates)
                else [candidate_action]
            )
            steps_to_roll[0] = candidate_action

            cum_cost = torch.zeros(self.ensemble_size, device=device, dtype=torch.float32)
            alive_prob = torch.ones(self.ensemble_size, device=device, dtype=torch.float32)
            expected_u = torch.zeros(self.ensemble_size, device=device, dtype=torch.float32)
            rem_ratio = float(max(0.0, min(1.0, remaining_budget_ratio)))

            for step_idx, act_str in enumerate(steps_to_roll[:DEFAULT_MAX_PROBE_STEPS]):
                ctx = self.encode_context_vector(
                    goal_text=goal_text,
                    public_fields=public_fields,
                    action_text=act_str,
                    remaining_budget_ratio=max(0.0, rem_ratio - float(cum_cost.mean().item())),
                    prior_action_logprob=prior_action_logprob if step_idx == 0 else 0.0,
                    prior_success_logit=prior_success_logit,
                    program=program,
                    device=device,
                )
                preds = self.predict_transition_heads(
                    cur_belief,
                    ctx,
                    prior_success_logit=prior_success_logit,
                    remaining_budget_ratio=max(0.0, rem_ratio - float(cum_cost.mean().item())),
                )
                step_cost = preds["cost_ratio"] * max(0.02, 1.0 / max(1.0, 15.0))
                cum_cost = (cum_cost + step_cost).clamp(max=1.0)
                p_term = preds["termination_probs"][:, 1]  # native_terminal
                p_budget_term = preds["termination_probs"][:, 2]
                y_prob = preds["terminal_success_prob"]
                rem_factor = (rem_ratio - cum_cost).clamp(min=0.0, max=1.0)

                expected_u = expected_u + alive_prob * p_term * y_prob * rem_factor
                alive_prob = alive_prob * (1.0 - (p_term + p_budget_term).clamp(max=0.95))
                cur_belief = preds["next_state_prior"]

            # Tail continuation pi_0 from the end of rho to native/task-budget termination
            tail_ctx = self.encode_context_vector(
                goal_text=goal_text,
                public_fields=public_fields,
                action_text=f"tail:{TAIL_POLICY_VERSION}:{candidate_action}",
                remaining_budget_ratio=max(0.0, rem_ratio - float(cum_cost.mean().item())),
                prior_action_logprob=prior_action_logprob,
                prior_success_logit=prior_success_logit,
                program=program,
                device=device,
            )
            tail_preds = self.predict_transition_heads(
                cur_belief,
                tail_ctx,
                prior_success_logit=prior_success_logit,
                remaining_budget_ratio=max(0.0, rem_ratio - float(cum_cost.mean().item())),
            )
            tail_cost = (cum_cost + tail_preds["cost_ratio"] * 0.25).clamp(max=1.0)
            tail_rem_factor = (rem_ratio - tail_cost).clamp(min=0.0, max=1.0)
            tail_u = alive_prob * tail_preds["terminal_success_prob"] * tail_rem_factor
            q_members = (expected_u + tail_u).clamp(min=0.0, max=1.0)

            if member_index is not None:
                m_idx = int(member_index) % self.ensemble_size
                q_scalar = float(q_members[m_idx].item())
            else:
                q_scalar = float(q_members.mean().item())

            return {
                "q_mean": float(q_members.mean().item()),
                "q_selected": q_scalar,
                "q_members": [float(x) for x in q_members.detach().cpu().tolist()],
                "predicted_terminal_success": float(tail_preds["terminal_success_prob"].mean().item()),
                "predicted_cost_ratio": float(tail_cost.mean().item()),
            }
        finally:
            if saved_state is not None:
                self.load_state_dict(saved_state, strict=True)

    @torch.no_grad()
    def score_program_evsi(
        self,
        program: ExecutableProgram,
        public_fields: Mapping[str, str],
        belief: torch.Tensor,
        *,
        use_Temporary_Initial_Params: bool = False,
        goal_text: str = "",
    ) -> dict[str, float]:
        """Exact closed-form EVSI reference without artificial polarity modulation (fixes I13).

        Uses identical utility before and after observing outcome o:
          Delta(xi) = sum_o max_j sum_m w_m L_m(o; xi) V_{m,j} - max_j sum_m w_m V_{m,j}.
        When M=1 or L_m(o_0)=1 or V_{m,1} == V_{m,2}, Delta(xi) == 0.0 exactly.
        Also computes unified terminal value Q^rho for candidate comparison.
        """
        saved_state: dict[str, torch.Tensor] | None = None
        if use_Temporary_Initial_Params and self._initial_state_tensors:
            saved_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
            self.restore_initial_parameters()

        try:
            device = self.transition_logits.device
            feat = self.encode_program_features(
                program, public_fields, device, goal_text=goal_text
            )
            L_m, next_prior = self.observable_operator_step(belief, feat, observed_outcome_bin=None)  # (M, O)
            w = self.mechanism_weights / self.mechanism_weights.sum().clamp(min=1e-8)

            # Evaluate two canonical continuation branches under the SAME utility function (no 1.15/0.85 multiplier)
            u_act = program.case_true_template or (program.step_templates[0] if program.step_templates else "cont_u")
            v_act = program.case_false_template or (program.step_templates[-1] if len(program.step_templates) > 1 else "cont_v")
            q_u = self.evaluate_candidate_q(
                goal_text=goal_text,
                public_fields=public_fields,
                belief=next_prior,
                candidate_action=u_act,
                program=program,
            )["q_members"]
            q_v = self.evaluate_candidate_q(
                goal_text=goal_text,
                public_fields=public_fields,
                belief=next_prior,
                candidate_action=v_act,
                program=program,
            )["q_members"]
            V_mj = torch.tensor(
                [[q_u[m], q_v[m]] for m in range(self.ensemble_size)],
                device=device,
                dtype=torch.float32,
            )  # (M, 2) in [0, 1]

            prior_values = torch.einsum("m,mj->j", w, V_mj)  # (2,)
            before_val = float(prior_values.max().item())

            # Posterior value with identical V_mj (zero polarity_mod!)
            weighted_lv = w[:, None, None] * L_m[:, :, None] * V_mj[:, None, :]  # (M, O, 2)
            after_per_outcome = weighted_lv.sum(dim=0).max(dim=-1).values  # (O,)
            after_val = float(after_per_outcome.sum().item())

            evsi_delta = max(0.0, after_val - before_val)
            cost = max(1.0, float(program.expected_cost))
            return {
                "evsi_delta": evsi_delta,
                "evsi_per_cost": evsi_delta / cost,
                "dual_control_j": after_val,
                "feedback_use_value_i": evsi_delta,
                "expected_cost": cost,
                "unified_q": float(prior_values.max().item()),
            }
        finally:
            if saved_state is not None:
                self.load_state_dict(saved_state, strict=True)

    def transition_negative_log_likelihood(
        self,
        transitions: Sequence[PublicTransition],
        *,
        bootstrap_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Cluster-level bootstrap negative log-likelihood over unique physical transitions.

        L_m(phi) = -sum_{(h_t, a_t, x_{t+1})} b_{mc} log K_{phi,m}(x_{t+1} | g, h_t, a_t, beta_t).
        Unobserved native_outcome (live or job-budget truncated) remains masked out.
        """
        device = self.transition_logits.device
        belief = self.reset_intra_task_belief(device=device)
        if bootstrap_weights is None:
            b_weights = torch.ones(self.ensemble_size, device=device, dtype=torch.float32)
        else:
            b_weights = bootstrap_weights.to(device=device, dtype=torch.float32)
            if b_weights.sum() <= 0:
                b_weights = torch.ones_like(b_weights)
            b_weights = b_weights / b_weights.mean().clamp(min=1e-8)

        total_member_nll = torch.zeros(self.ensemble_size, device=device, dtype=torch.float32)
        observed_outcomes = 0

        for tr in transitions:
            rem_ratio = float(tr.remaining_budget.get("ratio", 1.0))
            ctx = self.encode_context_vector(
                goal_text=tr.goal_text,
                public_fields=tr.pre_public_fields,
                action_text=tr.action_text,
                remaining_budget_ratio=rem_ratio,
                device=device,
            )
            had_err = tr.typed_response.get("prerequisite_status") == "error"
            term_done = tr.termination in ("native_terminal", "task_budget_terminal")
            term_succ = bool(tr.native_outcome is True)
            obs_bin = PublicBenchmarkBinder.encode_response_outcome_bin(
                tr.pre_public_fields,
                tr.typed_response,
                had_error=had_err,
                terminal_done=term_done,
                terminal_success=term_succ,
            )
            preds = self.predict_transition_heads(
                belief, ctx, remaining_budget_ratio=rem_ratio
            )
            _, belief = self.observable_operator_step(
                belief, ctx, observed_outcome_bin=obs_bin
            )

            # 1. Public response category NLL
            resp_nll = -torch.log(preds["outcome_probs"][:, obs_bin].clamp(min=1e-8))

            # 2. Termination category NLL
            term_idx = (
                1 if tr.termination == "native_terminal"
                else 2 if tr.termination == "task_budget_terminal"
                else 0
            )
            term_nll = -torch.log(preds["termination_probs"][:, term_idx].clamp(min=1e-8))

            # 3. Cost ratio NLL (smooth L1 / Gaussian log-lik on normalized step cost)
            target_cost = float(min(1.0, max(0.0, tr.resource_delta.get("cost_ratio", 0.05))))
            cost_nll = 0.5 * (preds["cost_ratio"] - target_cost).square()

            # 4. Official native terminal outcome NLL (ONLY when observed; masked when None)
            if tr.native_outcome is not None:
                observed_outcomes += 1
                y_val = 1.0 if tr.native_outcome else 0.0
                p_y = preds["terminal_success_prob"].clamp(min=1e-6, max=1.0 - 1e-6)
                y_nll = -(y_val * torch.log(p_y) + (1.0 - y_val) * torch.log(1.0 - p_y))
            else:
                y_nll = torch.zeros_like(resp_nll)

            total_member_nll = total_member_nll + (resp_nll + 0.5 * term_nll + cost_nll + y_nll)

        n_steps = max(1, len(transitions))
        weighted_loss = (b_weights * total_member_nll).mean() / float(n_steps)
        return weighted_loss, belief.detach(), {
            "num_transitions": float(len(transitions)),
            "observed_terminal_outcomes": float(observed_outcomes),
            "mean_member_nll": float((total_member_nll.mean() / float(n_steps)).detach().item()),
        }

    def block_negative_log_likelihood(
        self,
        program: ExecutableProgram,
        public_fields: Mapping[str, str],
        belief: torch.Tensor,
        outcome_bin: int,
        u_score: float,
        v_score: float,
        *,
        goal_text: str = "",
        action_text: str = "",
        u_native_outcome: bool | None = None,
        v_native_outcome: bool | None = None,
        bootstrap_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalized response & terminal outcome NLL without unnormalized coupling energy (fixes I05, I14)."""
        device = self.transition_logits.device
        act_str = action_text or (program.step_templates[0] if program.step_templates else "step")
        ctx = self.encode_program_features(
            program, public_fields, device, goal_text=goal_text, action_text=act_str
        )
        preds = self.predict_transition_heads(belief, ctx)
        outcome_probs, next_belief = self.observable_operator_step(
            belief, ctx, observed_outcome_bin=outcome_bin
        )
        o_idx = int(outcome_bin) % self.outcome_bins
        resp_nll = -torch.log(outcome_probs[:, o_idx].clamp(min=1e-8))

        # Supervise terminal outcome head ONLY with observed official outcomes (not unnormalized coupling)
        y_nll = torch.zeros_like(resp_nll)
        for nat_out, util_val in ((u_native_outcome, u_score), (v_native_outcome, v_score)):
            if nat_out is not None:
                target = float(max(0.0, min(1.0, util_val if nat_out else 0.0)))
                p_y = preds["terminal_success_prob"].clamp(min=1e-6, max=1.0 - 1e-6)
                y_nll = y_nll - (target * torch.log(p_y) + (1.0 - target) * torch.log(1.0 - p_y))

        if bootstrap_weights is None:
            b_w = torch.ones(self.ensemble_size, device=device, dtype=torch.float32)
        else:
            b_w = bootstrap_weights.to(device=device, dtype=torch.float32)
            b_w = b_w / b_w.mean().clamp(min=1e-8)

        loss = (b_w * (resp_nll + 0.5 * y_nll)).mean()
        return loss, next_belief.detach()


class ProgramLibrary:
    """Persistable library Xi of typed executable programs."""

    def __init__(self, benchmark: str, max_programs: int = 32) -> None:
        self.benchmark = benchmark.split(":", 1)[0]
        self.max_programs = int(max_programs)
        self._programs: dict[str, ExecutableProgram] = {}
        self._starter_ids: list[str] = []
        for prog in PublicBenchmarkBinder.starter_programs(self.benchmark):
            self._programs[prog.program_id] = prog
            self._starter_ids.append(prog.program_id)

    def active_programs(self, *, starter_only: bool = False) -> list[ExecutableProgram]:
        if starter_only:
            return [self._programs[pid] for pid in self._starter_ids if pid in self._programs]
        return list(self._programs.values())

    def add_program(self, program: ExecutableProgram) -> bool:
        if program.program_id in self._programs:
            return False
        if len(self._programs) >= self.max_programs:
            return False
        self._programs[program.program_id] = program
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "max_programs": self.max_programs,
            "starter_ids": list(self._starter_ids),
            "programs": [prog.to_dict() for prog in self.active_programs()],
        }

    def load_dict(self, data: Mapping[str, Any]) -> None:
        self._programs.clear()
        self._starter_ids = [str(x) for x in data.get("starter_ids", ())]
        for item in data.get("programs", ()):
            prog = ExecutableProgram.from_dict(item)
            self._programs[prog.program_id] = prog
        if not self._programs:
            for prog in PublicBenchmarkBinder.starter_programs(self.benchmark):
                self._programs[prog.program_id] = prog
                self._starter_ids.append(prog.program_id)


class ActionContrastProposer:
    """Witness-guided executable program proposer with def-use Delete, scope-unified Concat,
    strict single-label-view isolation, and full candidate construction/check budget accounting.
    """

    def __init__(
        self,
        benchmark: str,
        *,
        max_probe_steps: int = DEFAULT_MAX_PROBE_STEPS,
        proposal_budget: int = DEFAULT_PROPOSAL_BUDGET,
        max_candidate_checks: int = DEFAULT_MAX_CANDIDATE_CHECKS,
        shuffle_witness_labels: bool = False,
        use_model_diff_variant: bool = False,
        seed: int = 42,
    ) -> None:
        self.benchmark = benchmark.split(":", 1)[0]
        self.max_probe_steps = int(max_probe_steps)
        self.proposal_budget = int(proposal_budget)
        self.max_candidate_checks = int(max_candidate_checks)
        self.shuffle_witness_labels = bool(shuffle_witness_labels)
        self.use_model_diff_variant = bool(use_model_diff_variant)
        self.rng = random.Random(int(seed))
        self.records: list[MeasurementRecord] = []
        self.unique_witness_pair_ids: set[tuple[str, str]] = set()
        self.cumulative_witness_visits: int = 0
        self.call_index: int = 0
        self.proposal_runs: list[ProposalRun] = []
        self.proposed_counts: dict[str, int] = {
            "witness_full": 0,
            "witness_delete": 0,
            "witness_concat": 0,
            "prior_fallback": 0,
        }

    @property
    def witness_pairs_found(self) -> int:
        """Unique witness pairs discovered (kept distinct from cumulative_witness_visits, fixes E02)."""
        return len(self.unique_witness_pair_ids)

    def record_measurement(self, record: MeasurementRecord) -> None:
        self.records.append(record)

    @staticmethod
    def _def_use_delete(program: ExecutableProgram, target_field: str) -> ExecutableProgram | None:
        """Remove `target_field` and ONLY the call step dedicated exclusively to it (fixes I09)."""
        if len(program.readout_fields) <= 1 or target_field not in program.readout_fields:
            return None
        remaining_fields = tuple(f for f in program.readout_fields if f != target_field)
        producer_for_target = {
            e.producer_step for e in program.def_use_edges if e.field_path == target_field
        }
        producers_for_remaining = {
            e.producer_step for e in program.def_use_edges if e.field_path in remaining_fields
        }
        removable_steps = producer_for_target - producers_for_remaining
        if removable_steps and len(program.step_templates) > 1:
            kept_steps = tuple(
                s for idx, s in enumerate(program.step_templates) if idx not in removable_steps
            )
            if not kept_steps:
                kept_steps = program.step_templates[:-1]
        else:
            kept_steps = program.step_templates
        return ExecutableProgram(
            program_id=_short_hash((program.benchmark, kept_steps, remaining_fields, "delete")),
            benchmark=program.benchmark,
            step_templates=kept_steps,
            readout_fields=remaining_fields,
            origin="witness_delete",
            parent_ids=(program.program_id,),
            expected_cost=float(len(kept_steps)),
            role_bindings=dict(program.role_bindings),
        )

    @staticmethod
    def _scope_unified_concat(
        p1: ExecutableProgram,
        p2: ExecutableProgram,
        differing_fields: Sequence[str],
        max_steps: int,
    ) -> ExecutableProgram | None:
        """Unify role bindings and variable scope across p1 and p2 before concatenating."""
        merged_bindings = {**p2.role_bindings, **p1.role_bindings}
        combined_steps: list[str] = []
        for step in (*p1.step_templates, *p2.step_templates):
            if step not in combined_steps and len(combined_steps) < max_steps:
                combined_steps.append(step)
        combined_fields: list[str] = []
        for fld in (*differing_fields, *p1.readout_fields, *p2.readout_fields):
            if fld not in combined_fields and len(combined_fields) < 4:
                combined_fields.append(fld)
        if not combined_steps or not combined_fields:
            return None
        cond_field = differing_fields[0] if differing_fields else ""
        return ExecutableProgram(
            program_id=_short_hash((p1.benchmark, tuple(combined_steps), tuple(combined_fields), cond_field)),
            benchmark=p1.benchmark,
            step_templates=tuple(combined_steps),
            readout_fields=tuple(combined_fields),
            origin="witness_concat",
            parent_ids=(p1.program_id, p2.program_id),
            expected_cost=float(len(combined_steps)),
            role_bindings=merged_bindings,
            case_condition_field=cond_field,
        )

    def _build_label_view(self) -> dict[str, float]:
        """Construct a single isolated action-difference label view (fixes E03).

        When `shuffle_witness_labels=True`, permutes labels within comparable
        (goal_category, continuation_pair_key, tail_policy_version) groups and returns
        the permuted dictionary. All downstream grouping, magnitude calculation, and
        candidate ranking read ONLY from this dictionary.
        """
        raw_map: dict[str, float] = {}
        for rec in self.records:
            if self.use_model_diff_variant and rec.d_model is not None:
                val = float(rec.d_model)
            elif rec.d_real is not None:
                val = float(rec.d_real)
            elif rec.u_native_outcome is not None and rec.v_native_outcome is not None:
                val = float(rec.action_diff_d)
            else:
                # Before first terminal contrast or when unobserved, real terminal difference is 0.0
                val = float(rec.action_diff_d) if (rec.u_done or rec.v_done) else 0.0
            raw_map[rec.record_id] = val

        if not self.shuffle_witness_labels:
            return raw_map

        grouped: dict[tuple[str, str, str], list[str]] = {}
        for rec in self.records:
            key = (rec.goal_category, rec.continuation_pair_key, rec.tail_policy_version)
            grouped.setdefault(key, []).append(rec.record_id)

        shuffled_map: dict[str, float] = {}
        # If all groups are singletons, permute within goal_category so permutation is non-trivial
        if all(len(ids) <= 1 for ids in grouped.values()) and len(self.records) >= 2:
            cat_groups: dict[str, list[str]] = {}
            for rec in self.records:
                cat_groups.setdefault(rec.goal_category, []).append(rec.record_id)
            if all(len(ids) <= 1 for ids in cat_groups.values()):
                cat_groups = {"all": [rec.record_id for rec in self.records]}
            for _, r_ids in sorted(cat_groups.items()):
                vals = [raw_map[rid] for rid in r_ids]
                self.rng.shuffle(vals)
                for rid, v in zip(r_ids, vals):
                    shuffled_map[rid] = v
            return shuffled_map

        for _, r_ids in sorted(grouped.items()):
            vals = [raw_map[rid] for rid in r_ids]
            self.rng.shuffle(vals)
            for rid, v in zip(r_ids, vals):
                shuffled_map[rid] = v
        return shuffled_map

    def grow_library(
        self,
        library: ProgramLibrary,
        *,
        dataset_version: str = "online",
        model_version: str = "current",
    ) -> list[ExecutableProgram]:
        self.call_index += 1
        if len(self.records) < 2:
            return []

        label_view = self._build_label_view()
        pos_records = [rec for rec in self.records if label_view.get(rec.record_id, 0.0) > 0.01]
        neg_records = [rec for rec in self.records if label_view.get(rec.record_id, 0.0) < -0.01]

        candidates_constructed = 0
        candidates_checked = 0
        candidates_scored = 0
        candidates: list[tuple[float, ExecutableProgram]] = []

        if pos_records and neg_records:
            for r1 in pos_records[-6:]:
                if candidates_checked >= self.max_candidate_checks:
                    break
                for r2 in neg_records[-6:]:
                    if candidates_checked >= self.max_candidate_checks:
                        break
                    if r1.goal_category != r2.goal_category and len(self.records) > 6:
                        continue
                    self.cumulative_witness_visits += 1
                    pair_key = tuple(sorted((r1.record_id, r2.record_id)))
                    self.unique_witness_pair_ids.add((pair_key[0], pair_key[1]))

                    # Strictly read from `label_view` (never `r1.action_diff_d`), preventing label leakage
                    d1_view = float(label_view[r1.record_id])
                    d2_view = float(label_view[r2.record_id])
                    contrast_mag = abs(d1_view - d2_view)

                    # Only inspect fields actually observed (not "unknown") in both anchor traces
                    differing_fields = [
                        k for k in r1.public_readout_z
                        if r1.public_readout_z.get(k, "unknown") != "unknown"
                        and r2.public_readout_z.get(k, "unknown") != "unknown"
                        and r1.public_readout_z.get(k) != r2.public_readout_z.get(k)
                    ]
                    ordered_fields = tuple(
                        differing_fields
                        + [k for k in r1.readout_fields if k not in differing_fields]
                    )[:4] or r1.readout_fields

                    # 1. Full original trace program candidate
                    full_steps = r1.rebound_trace_templates[: self.max_probe_steps] or r1.step_templates
                    full_fields = r1.rebound_trace_fields[:4] or ordered_fields
                    full_prog = ExecutableProgram(
                        program_id=_short_hash((self.benchmark, full_steps, full_fields)),
                        benchmark=self.benchmark,
                        step_templates=full_steps,
                        readout_fields=full_fields,
                        origin="witness_full",
                        parent_ids=(r1.program_id, r2.program_id),
                        expected_cost=float(len(full_steps)),
                        role_bindings=dict(r1.role_bindings),
                        case_condition_field=differing_fields[0] if differing_fields else "",
                        case_condition_value=r1.public_readout_z.get(differing_fields[0], "") if differing_fields else "",
                    )
                    candidates_constructed += 1
                    candidates_checked += 1
                    candidates_scored += 1
                    candidates.append((contrast_mag + 0.1, full_prog))
                    if candidates_checked >= self.max_candidate_checks:
                        break

                    # 2. Def-use Delete transformation
                    target_del_field = full_fields[-1] if len(full_fields) > 1 else ""
                    if target_del_field:
                        del_prog = self._def_use_delete(full_prog, target_del_field)
                        candidates_constructed += 1
                        candidates_checked += 1
                        if del_prog is not None:
                            candidates_scored += 1
                            candidates.append((contrast_mag + 0.2 * len(differing_fields), del_prog))
                    if candidates_checked >= self.max_candidate_checks:
                        break

                    # 3. Scope-unified Concat transformation
                    p2_prog = ExecutableProgram(
                        program_id=r2.program_id,
                        benchmark=self.benchmark,
                        step_templates=r2.rebound_trace_templates[: self.max_probe_steps] or r2.step_templates,
                        readout_fields=r2.rebound_trace_fields[:4] or r2.readout_fields,
                        role_bindings=dict(r2.role_bindings),
                    )
                    concat_prog = self._scope_unified_concat(
                        full_prog, p2_prog, differing_fields, self.max_probe_steps
                    )
                    candidates_constructed += 1
                    candidates_checked += 1
                    if concat_prog is not None:
                        candidates_scored += 1
                        candidates.append((contrast_mag + 0.3 * len(differing_fields), concat_prog))

        candidates.sort(key=lambda item: item[0], reverse=True)
        added: list[ExecutableProgram] = []
        for _, prog in candidates:
            if len(added) >= self.proposal_budget:
                break
            if library.add_program(prog):
                self.proposed_counts[prog.origin] = self.proposed_counts.get(prog.origin, 0) + 1
                added.append(prog)

        self.proposal_runs.append(
            ProposalRun(
                run_id=_short_hash((self.benchmark, dataset_version, self.call_index, self.shuffle_witness_labels)),
                benchmark=self.benchmark,
                dataset_version=dataset_version,
                model_version=model_version,
                grammar_version=GRAMMAR_VERSION,
                record_order=tuple(r.record_id for r in self.records),
                call_index=self.call_index,
                candidates_constructed=candidates_constructed,
                candidates_checked=candidates_checked,
                candidates_scored=candidates_scored,
                label_view_mode="label_shuffled" if self.shuffle_witness_labels else (
                    "d_model" if self.use_model_diff_variant else "true_d_real"
                ),
                label_view_values=tuple(float(label_view[r.record_id]) for r in self.records),
                unique_witness_pairs=len(self.unique_witness_pair_ids),
                cumulative_witness_visits=self.cumulative_witness_visits,
                added_program_ids=tuple(p.program_id for p in added),
            )
        )
        return added
