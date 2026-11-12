"""PRISM DCC v2 (Decision-Contrast Closure) unified mathematical and execution kernel.

Implements the complete revision specification (`docs/prism-dcc-v2-design.md`):
1. Pure symbolic AST (`ExecutableProgram`: `Call`, `Read`, `Seq`, `Case`, `Stop`) separated
   from task-local `ExecutionFrame` (`ProgramExecutionState`), with stack-based AST evaluation
   (eliminating `Case` loops), AST-derived consumer def-use graph for `Delete`, local-variable
   alpha-renaming for `Concat`, and explicit `Action | Complete | BindingFailure` outcomes.
2. Full LOCA `native_tools` (`function.name` / `function.parameters`) schema adaptation and
   unbiased candidate support across all feasible operations without I-before-E truncation.
3. Explicit `ResourceState(before, after)` in wall-clock seconds, environment steps, and model
   tokens aligned with official `NativeTrajectory.utility`: U_B = Y * max(0, 1 - C_seconds / B_seconds).
4. Real frozen backbone semantic context & normalized response prior interface (zero SHA256
   hashes, zero candidate-origin pseudo-logprobs), typed field-slot response predictions
   supporting `Case` lookahead and parameter binding, strict ensemble-member independence,
   absorbing termination, and persistent immutable `ModelView` (phi_0 vs phi_T).
5. Fork-tree marginal NLL computation where shared prefix nodes are forwarded once and sibling
   branches (u, v) both start from the differentiable post-q `fork_belief`, plus amortized
   fixed-tail response distribution H_{phi,m}^{pi_0}(Y, C_tail | Y=1) supervised strictly by
   genuine frozen pi_0 suffix intervals (`TailExample`).
6. Canonical ordered `Comparison` key shared by witness pairing and `Label-Shuffled` within-group
   permutation without singleton group expansion.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import itertools
import json
import math
import random
import re
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DCC_CONTRACT_VERSION = "prism-dcc-v2-unified-r5"
TAIL_POLICY_VERSION = "frozen-prior-tail-v2"
GRAMMAR_VERSION = "typed-ast-v5"
RESPONSE_SCHEMA_VERSION = "typed-field-slots-v5"
PRIOR_ADVANTAGE_WEIGHT = 0.25
ACTION_REPEAT_PENALTY = 0.05
DEFAULT_ENSEMBLE_SIZE = 8
DEFAULT_MAX_PROBE_STEPS = 3
DEFAULT_CONTINUATION_STEPS = 50
DEFAULT_PROPOSAL_BUDGET = 3
DEFAULT_MAX_CANDIDATE_CHECKS = 24
STATE_DIM = 16
FROZEN_HISTORY_DIM = 32
FROZEN_ACTION_DIM = 16
STRUCTURED_FIELD_DIM = 16
CONTEXT_DIM = FROZEN_HISTORY_DIM + FROZEN_ACTION_DIM + STRUCTURED_FIELD_DIM  # 64
OUTCOME_BINS = 8
NUM_FIELD_SLOTS = 4
NUM_SLOT_VALUES = 4
TAIL_COST_BINS = 8
TERMINATION_CATEGORIES = ("live", "native_terminal", "task_budget_terminal")

# Single canonical typed-slot vocabulary shared across raw adapter boundary, AST Case predicates, and K_phi
SLOT_VALUE_NAMES: tuple[tuple[str, str, str, str], ...] = (
    ("unknown", "no_or_low", "yes_or_high", "error"),
    ("unknown", "none", "bound_single", "bound_multiple"),
    ("unknown", "closed_or_pending", "open_empty_or_subpage", "open_with_items_or_ready"),
    ("unknown", "raw_or_unchanged", "inspected_or_selected", "treated_or_verified_write"),
)

# Multi-scale wall-clock cost ratio bins in [0, 1] resolving both fast short-horizon tails (0.005..0.10)
# and long-horizon tails (0.10..1.0) while preserving exact Bellman neutrality at phi_0.
TAIL_COST_BIN_EDGES: tuple[float, ...] = (0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.40, 0.70, 1.0)
TAIL_COST_BIN_CENTERS: tuple[float, ...] = tuple(
    0.5 * (TAIL_COST_BIN_EDGES[i] + TAIL_COST_BIN_EDGES[i + 1]) for i in range(TAIL_COST_BINS)
)

_JOINT_SLOT_GRID_NP = np.asarray(
    list(itertools.product(range(NUM_SLOT_VALUES), repeat=NUM_FIELD_SLOTS)), dtype=np.int64
)  # (256, 4)


def _tail_cost_bin_index(cost_ratio: float) -> int:
    c = float(min(0.999999, max(0.0, cost_ratio)))
    for idx in range(TAIL_COST_BINS - 1):
        if c < TAIL_COST_BIN_EDGES[idx + 1]:
            return idx
    return TAIL_COST_BINS - 1


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _short_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:12]


def _deterministic_seed(*parts: Any) -> int:
    digest = hashlib.sha256(_canonical_json(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


@dataclass(frozen=True)
class ResourceState:
    """Explicit pre/post resource ledger aligned with official wall-clock utility U_B."""

    elapsed_seconds: float = 0.0
    budget_seconds: float = 1800.0
    env_steps: int = 0
    max_env_steps: int = 100
    model_tokens: int = 0
    max_model_tokens: int = 262144
    side_forward_ops: int = 0

    @property
    def spent_seconds_ratio(self) -> float:
        if self.budget_seconds <= 0:
            return 1.0
        return float(min(1.0, max(0.0, self.elapsed_seconds / self.budget_seconds)))

    @property
    def remaining_seconds_ratio(self) -> float:
        return float(max(0.0, 1.0 - self.spent_seconds_ratio))

    @property
    def remaining_steps_ratio(self) -> float:
        if self.max_env_steps <= 0:
            return 0.0
        return float(max(0.0, 1.0 - (self.env_steps / float(self.max_env_steps))))

    def to_dict(self) -> dict[str, float]:
        return {
            "elapsed_seconds": float(self.elapsed_seconds),
            "budget_seconds": float(self.budget_seconds),
            "spent_seconds_ratio": self.spent_seconds_ratio,
            "remaining_seconds_ratio": self.remaining_seconds_ratio,
            "env_steps": float(self.env_steps),
            "max_env_steps": float(self.max_env_steps),
            "remaining_steps_ratio": self.remaining_steps_ratio,
            "model_tokens": float(self.model_tokens),
            "max_model_tokens": float(self.max_model_tokens),
            "side_forward_ops": float(self.side_forward_ops),
            "ratio": self.remaining_seconds_ratio,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResourceState":
        b_sec = float(data.get("budget_seconds", 1800.0))
        if "elapsed_seconds" in data:
            e_sec = float(data["elapsed_seconds"])
        elif "spent_seconds_ratio" in data:
            e_sec = float(data["spent_seconds_ratio"]) * b_sec
        elif "ratio" in data:
            e_sec = (1.0 - float(data["ratio"])) * b_sec
        else:
            e_sec = 0.0
        return cls(
            elapsed_seconds=e_sec,
            budget_seconds=b_sec,
            env_steps=int(float(data.get("env_steps", 0))),
            max_env_steps=int(float(data.get("max_env_steps", 100))),
            model_tokens=int(float(data.get("model_tokens", 0))),
            max_model_tokens=int(float(data.get("max_model_tokens", 262144))),
            side_forward_ops=int(float(data.get("side_forward_ops", 0))),
        )


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
    """Data-dependency edge derived from actual AST producer-consumer references."""

    producer_step: int
    variable_name: str
    field_path: str
    consumer_step: int | None  # None means consumed by final program readout or Case


def _derive_ast_and_def_use(
    step_templates: Sequence[str],
    readout_fields: Sequence[str],
    *,
    case_condition_field: str = "",
    case_condition_value: str = "",
    case_true_template: str = "",
    case_false_template: str = "",
    existing_ast: Sequence[ASTNode] = (),
) -> tuple[tuple[ASTNode, ...], tuple[str, ...], tuple[DefUseEdge, ...]]:
    """Build normalized AST and derive true producer->consumer def-use edges from AST nodes."""
    if existing_ast:
        nodes = list(existing_ast)
    else:
        nodes = []
        for idx, tmpl in enumerate(step_templates):
            var_name = f"ret_{idx}"
            slots = re.findall(r"\{([a-zA-Z0-9_]+)\}", tmpl)
            arg_map: dict[str, str] = {}
            for slot in slots:
                if slot.startswith("ret_") and "." in slot:
                    arg_map[slot] = f"${slot}"
                elif idx > 0 and slot in ("bound_id", "bound_path", "bound_token", "bound_key", "bound_record_id"):
                    arg_map[slot] = f"$ret_{idx - 1}.{slot}"
                else:
                    arg_map[slot] = f"${slot}"
            nodes.append(
                ASTNode(
                    op="Call",
                    tool_or_action=tmpl,
                    typed_arguments=arg_map,
                    bind_return_to=var_name,
                )
            )
            if idx < len(readout_fields):
                fld = readout_fields[idx]
                nodes.append(ASTNode(op="Read", variable=var_name, public_field_path=fld))
        for extra_idx in range(len(step_templates), len(readout_fields)):
            fld = readout_fields[extra_idx]
            src_step = max(0, len(step_templates) - 1)
            var_name = f"ret_{src_step}"
            nodes.append(ASTNode(op="Read", variable=var_name, public_field_path=fld))
        if case_condition_field and (case_true_template or case_false_template):
            nodes.append(
                ASTNode(
                    op="Case",
                    condition_field=case_condition_field,
                    condition_equals=case_condition_value,
                    children=(
                        ASTNode(
                            op="Call",
                            tool_or_action=case_true_template,
                            typed_arguments={
                                s: f"${s}" for s in re.findall(r"\{([a-zA-Z0-9_]+)\}", case_true_template)
                            },
                            bind_return_to="ret_case_true",
                        )
                        if case_true_template
                        else ASTNode(op="Stop"),
                        ASTNode(
                            op="Call",
                            tool_or_action=case_false_template,
                            typed_arguments={
                                s: f"${s}" for s in re.findall(r"\{([a-zA-Z0-9_]+)\}", case_false_template)
                            },
                            bind_return_to="ret_case_false",
                        )
                        if case_false_template
                        else ASTNode(op="Stop"),
                    ),
                )
            )
        if not nodes or nodes[-1].op != "Stop":
            nodes.append(ASTNode(op="Stop"))

    # Traverse AST to compute real producer -> consumer def-use edges
    call_Step_by_var: dict[str, int] = {}
    field_producer_step: dict[str, int] = {}
    scope: list[str] = []
    edges: list[DefUseEdge] = []
    call_idx = 0

    def _scan_nodes(seq: Sequence[ASTNode]) -> None:
        nonlocal call_idx
        for node in seq:
            if node.op == "Seq":
                _scan_nodes(node.children)
            elif node.op == "Call":
                cur_step = call_idx
                call_idx += 1
                if node.bind_return_to:
                    call_Step_by_var[node.bind_return_to] = cur_step
                    if node.bind_return_to not in scope:
                        scope.append(node.bind_return_to)
                for _, arg_ref in node.typed_arguments.items():
                    ref_clean = arg_ref.lstrip("$")
                    base_var = ref_clean.split(".", 1)[0]
                    fld_part = ref_clean.split(".", 1)[1] if "." in ref_clean else "return_value"
                    if base_var in call_Step_by_var:
                        edges.append(
                            DefUseEdge(
                                producer_step=call_Step_by_var[base_var],
                                variable_name=base_var,
                                field_path=fld_part,
                                consumer_step=cur_step,
                            )
                        )
            elif node.op == "Read":
                prod_step = call_Step_by_var.get(node.variable, max(0, call_idx - 1))
                field_producer_step[node.public_field_path] = prod_step
                edges.append(
                    DefUseEdge(
                        producer_step=prod_step,
                        variable_name=node.variable or f"ret_{prod_step}",
                        field_path=node.public_field_path,
                        consumer_step=None,
                    )
                )
            elif node.op == "Case":
                if node.condition_field in field_producer_step:
                    edges.append(
                        DefUseEdge(
                            producer_step=field_producer_step[node.condition_field],
                            variable_name=f"ret_{field_producer_step[node.condition_field]}",
                            field_path=node.condition_field,
                            consumer_step=call_idx,
                        )
                    )
                _scan_nodes(node.children)

    _scan_nodes(nodes)
    return tuple(nodes), tuple(scope), tuple(edges)


def _alpha_rename_ast_nodes(nodes: Sequence[ASTNode], prefix: str) -> tuple[ASTNode, ...]:
    """Perform local-variable alpha-renaming on AST nodes so concatenated scopes never collide."""
    renamed: list[ASTNode] = []
    for node in nodes:
        if node.op == "Stop":
            continue
        new_bind = f"{prefix}_{node.bind_return_to}" if node.bind_return_to.startswith("ret_") else node.bind_return_to
        new_var = f"{prefix}_{node.variable}" if node.variable.startswith("ret_") else node.variable
        new_args: dict[str, str] = {}
        for k, v in node.typed_arguments.items():
            if v.startswith("$ret_"):
                new_args[k] = f"${prefix}_{v[1:]}"
            else:
                new_args[k] = v
        new_children = _alpha_rename_ast_nodes(node.children, prefix) if node.children else ()
        renamed.append(
            ASTNode(
                op=node.op,
                tool_or_action=node.tool_or_action,
                typed_arguments=new_args,
                bind_return_to=new_bind,
                variable=new_var,
                public_field_path=node.public_field_path,
                children=new_children,
                condition_field=node.condition_field,
                condition_equals=node.condition_equals,
            )
        )
    return tuple(renamed)


@dataclass(frozen=True)
class ExecutableProgram:
    """Pure symbolic executable program e = (rho_e, chi_e).

    Stores ONLY structural AST, formal parameter slots, local variable scope, and AST-derived
    def-use edges. Task-specific concrete object IDs are never stored on the program (fixes F23).
    """

    program_id: str
    benchmark: str
    step_templates: tuple[str, ...]
    readout_fields: tuple[str, ...]
    origin: str = "starter"
    parent_ids: tuple[str, ...] = ()
    expected_cost: float = 1.0
    ast_nodes: tuple[ASTNode, ...] = ()
    variable_scope: tuple[str, ...] = ()
    formal_params: tuple[str, ...] = ()
    role_bindings: dict[str, str] = field(default_factory=dict)
    def_use_edges: tuple[DefUseEdge, ...] = ()
    case_condition_field: str = ""
    case_condition_value: str = ""
    case_true_template: str = ""
    case_false_template: str = ""

    def __post_init__(self) -> None:
        # Strip any concrete task instance bindings from Program structure (F23):
        # only symbolic formal parameter names may be retained.
        if self.role_bindings:
            object.__setattr__(self, "role_bindings", {})
        nodes, scope, edges = _derive_ast_and_def_use(
            self.step_templates,
            self.readout_fields,
            case_condition_field=self.case_condition_field,
            case_condition_value=self.case_condition_value,
            case_true_template=self.case_true_template,
            case_false_template=self.case_false_template,
            existing_ast=self.ast_nodes,
        )
        object.__setattr__(self, "ast_nodes", nodes)
        # Derive display `step_templates` directly from top-level AST Call nodes when AST is authoritative
        if self.ast_nodes:
            derived_steps: list[str] = []

            def _collect_top_calls(seq: Sequence[ASTNode]) -> None:
                for n in seq:
                    if n.op == "Seq":
                        _collect_top_calls(n.children)
                    elif n.op == "Call" and n.tool_or_action:
                        derived_steps.append(n.tool_or_action)

            _collect_top_calls(nodes)
            if derived_steps:
                object.__setattr__(self, "step_templates", tuple(derived_steps))
        object.__setattr__(self, "variable_scope", scope)
        object.__setattr__(self, "def_use_edges", edges)
        max_depth = self.max_call_depth()
        object.__setattr__(self, "expected_cost", float(max(1, max_depth)))
        if not self.formal_params:
            slots: list[str] = []
            for tmpl in (*self.step_templates, self.case_true_template, self.case_false_template):
                for s in re.findall(r"\{([a-zA-Z0-9_.]+)\}", tmpl):
                    if s not in slots:
                        slots.append(s)
            object.__setattr__(self, "formal_params", tuple(slots))

    def max_call_depth(self) -> int:
        """Compute maximum Call count along any root-to-Stop execution path in `ast_nodes`."""
        def _depth(seq: Sequence[ASTNode]) -> int:
            if not seq:
                return 0
            head, rest = seq[0], seq[1:]
            if head.op == "Stop":
                return 0
            if head.op == "Seq":
                return _depth(tuple(head.children) + tuple(rest))
            if head.op == "Call":
                return 1 + _depth(rest)
            if head.op == "Case":
                if not head.children:
                    return _depth(rest)
                return max(_depth((child,) + tuple(rest)) for child in head.children)
            return _depth(rest)

        return _depth(self.ast_nodes)

    def ast_signature(self) -> str:
        return _canonical_json([n.to_dict() for n in self.ast_nodes])

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
            "formal_params": list(self.formal_params),
            "role_bindings": {},
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
            formal_params=tuple(str(p) for p in data.get("formal_params", ())),
            role_bindings={},
            def_use_edges=edges,
            case_condition_field=str(data.get("case_condition_field", "")),
            case_condition_value=str(data.get("case_condition_value", "")),
            case_true_template=str(data.get("case_true_template", "")),
            case_false_template=str(data.get("case_false_template", "")),
        )


BoundedObservationProgram = ExecutableProgram


@dataclass
class ExecutionFrame:
    """Private task-instance execution stack frame consuming `ExecutableProgram.ast_nodes`.

    Fixes F05, F06, F23, F24:
      - Maintains an explicit `node_stack` of `ASTNode` objects.
      - When `Case` is evaluated, pushes ONLY the chosen branch child onto `node_stack`;
        once that branch child executes, the stack advances to `Stop` and terminates (never loops).
      - Stores concrete entity bindings (`bound_variables`) and `Call` return values (`local_env`)
        strictly in this frame, never mutating `ExecutableProgram`.
    """

    program: ExecutableProgram
    program_counter: int = 0
    bound_variables: dict[str, str] = field(default_factory=dict)
    local_env: dict[str, dict[str, str]] = field(default_factory=dict)
    observed_readouts: dict[str, str] = field(default_factory=dict)
    executed_actions: list[str] = field(default_factory=list)
    node_stack: list[ASTNode] = field(default_factory=list)
    pending_return_var: str = ""
    binding_failed: bool = False
    completed: bool = False
    last_status: str = "Ready"  # "Ready" | "Action" | "Complete" | "BindingFailure"

    def __post_init__(self) -> None:
        if not self.node_stack and not self.completed and not self.binding_failed:
            self.node_stack = list(self.program.ast_nodes)
            if self.program_counter > 0:
                # Advance stack past `program_counter` Call nodes for legacy state restoration
                calls_skipped = 0
                while self.node_stack and calls_skipped < self.program_counter:
                    head = self.node_stack.pop(0)
                    if head.op == "Call":
                        calls_skipped += 1
                    elif head.op == "Seq":
                        self.node_stack = list(head.children) + self.node_stack

    def record_call_return(self, raw_observation: str, post_fields: Mapping[str, str]) -> None:
        """Bind the return value of the most recently executed Call into `local_env`, then consume ready Read/Stop nodes."""
        self.observed_readouts.update({str(k): str(v) for k, v in post_fields.items()})
        var_name = self.pending_return_var or f"ret_{max(0, self.program_counter - 1)}"
        extracted: dict[str, str] = dict(post_fields)
        for key_name, val in re.findall(
            r'"([a-zA-Z0-9_]*(?:id|path|file|token|key|name|dir))"\s*:\s*"([^"]{1,64})"',
            raw_observation,
        ):
            k_low = key_name.lower()
            extracted[f"bound_{k_low}"] = val
            self.bound_variables[f"bound_{k_low}"] = val
            if "id" in k_low:
                extracted.setdefault("bound_id", val)
                self.bound_variables.setdefault("bound_id", val)
            if "path" in k_low or "file" in k_low:
                extracted.setdefault("bound_path", val)
                self.bound_variables.setdefault("bound_path", val)
        self.local_env[var_name] = extracted
        self.pending_return_var = ""

        # Consume Read nodes now that the real Call return is bound in `local_env`
        while self.node_stack:
            top = self.node_stack[0]
            if top.op == "Seq":
                self.node_stack.pop(0)
                self.node_stack = list(top.children) + self.node_stack
                continue
            if top.op == "Read":
                self.node_stack.pop(0)
                src_env = self.local_env.get(top.variable, {}) if top.variable else {}
                val = src_env.get(top.public_field_path, post_fields.get(top.public_field_path))
                if val is not None:
                    self.observed_readouts[top.public_field_path] = val
                continue
            if top.op == "Stop":
                self.node_stack.pop(0)
                self.completed = True
                self.last_status = "Complete"
                break
            break
        if not self.node_stack:
            self.completed = True
            self.last_status = "Complete"

    def record_symbolic_call_return(
        self,
        tool_or_action: str,
        step_index: int,
        predicted_slots: tuple[int, int, int, int] | None = None,
    ) -> None:
        """Bind typed symbolic return handles during model lookahead and consume ready Read/Stop nodes.

        Preserves exact variable origin, entity role identity, and equality relations (`<sym:ret_k:id>`)
        without guessing unseen concrete identifiers, matching `normalize_action_for_semantic_encoding`.
        """
        var_name = self.pending_return_var or f"ret_{max(0, self.program_counter - 1)}"
        sym_id = f"<sym:{var_name}:id>"
        sym_path = f"<sym:{var_name}:path>"
        sym_fields: dict[str, str] = dict(self.observed_readouts)
        if predicted_slots is not None:
            sym_fields = PublicBenchmarkBinder.slots_to_canonical_fields(sym_fields, predicted_slots)
            self.observed_readouts.update(sym_fields)
        extracted: dict[str, str] = dict(sym_fields)
        extracted["bound_id"] = sym_id
        extracted["bound_path"] = sym_path
        extracted.setdefault("bound_identifiers", sym_fields.get("bound_identifiers", "bound_single"))
        self.bound_variables.setdefault("bound_id", sym_id)
        self.bound_variables.setdefault("bound_path", sym_path)
        self.local_env[var_name] = extracted
        self.pending_return_var = ""

        while self.node_stack:
            top = self.node_stack[0]
            if top.op == "Seq":
                self.node_stack.pop(0)
                self.node_stack = list(top.children) + self.node_stack
                continue
            if top.op == "Read":
                self.node_stack.pop(0)
                src_env = self.local_env.get(top.variable, {}) if top.variable else {}
                val = src_env.get(top.public_field_path, sym_fields.get(top.public_field_path))
                if val is not None:
                    self.observed_readouts[top.public_field_path] = val
                continue
            if top.op == "Stop":
                self.node_stack.pop(0)
                self.completed = True
                self.last_status = "Complete"
                break
            break
        if not self.node_stack:
            self.completed = True
            self.last_status = "Complete"

    def remaining_call_nodes(self) -> list[ASTNode]:
        if self.completed or self.binding_failed:
            return []
        calls: list[ASTNode] = []
        stack_copy = list(self.node_stack)
        while stack_copy:
            node = stack_copy.pop(0)
            if node.op == "Stop":
                break
            if node.op == "Seq":
                stack_copy = list(node.children) + stack_copy
            elif node.op == "Call":
                calls.append(node)
            elif node.op == "Case" and node.children:
                cond_val = self.observed_readouts.get(node.condition_field, "unknown")
                if cond_val != "unknown":
                    lhs = PublicBenchmarkBinder.canonical_slot_predicate_value(node.condition_field, cond_val)
                    rhs = PublicBenchmarkBinder.canonical_slot_predicate_value(node.condition_field, node.condition_equals)
                    branch_idx = 0 if lhs == rhs else (1 if len(node.children) > 1 else 0)
                else:
                    branch_idx = 0
                if branch_idx < len(node.children):
                    stack_copy = [node.children[branch_idx]] + stack_copy
        return calls

    def remaining_templates(self) -> tuple[str, ...]:
        return tuple(n.tool_or_action for n in self.remaining_call_nodes() if n.tool_or_action)

    def residual_program(self) -> ExecutableProgram | None:
        if self.completed or self.binding_failed:
            return None
        rem_calls = self.remaining_call_nodes()
        if not rem_calls:
            return None
        rem_ast = tuple(self.node_stack)
        rem_tmpls = tuple(n.tool_or_action for n in rem_calls if n.tool_or_action)
        return ExecutableProgram(
            program_id=f"{self.program.program_id}:pc{self.program_counter}",
            benchmark=self.program.benchmark,
            step_templates=rem_tmpls,
            readout_fields=self.program.readout_fields,
            origin=self.program.origin,
            parent_ids=(self.program.program_id,),
            expected_cost=float(len(rem_tmpls)),
            ast_nodes=rem_ast,
        )

    def clone(self) -> "ExecutionFrame":
        return deepcopy(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program.program_id,
            "program_counter": self.program_counter,
            "bound_variables": dict(self.bound_variables),
            "local_env": {k: dict(v) for k, v in self.local_env.items()},
            "observed_readouts": dict(self.observed_readouts),
            "executed_actions": list(self.executed_actions),
            "node_stack": [n.to_dict() for n in self.node_stack],
            "pending_return_var": self.pending_return_var,
            "binding_failed": self.binding_failed,
            "completed": self.completed,
            "last_status": self.last_status,
            "remaining_templates": list(self.remaining_templates()),
        }


ProgramExecutionState = ExecutionFrame


@dataclass(frozen=True)
class InterpreterStepResult:
    """Explicit outcome of one AST interpreter step: Action, Complete, or BindingFailure."""

    status: str  # "Action" | "Complete" | "BindingFailure"
    concrete_action: str | None
    frame: ExecutionFrame
    program_attributed: bool


@dataclass(frozen=True)
class PublicTransition:
    """Fork-tree transition node (h_t, a_t, x_{t+1}) with explicit parent_transition_id and ResourceState."""

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
    native_outcome: bool | None
    parent_transition_id: str | None = None
    resource_before: dict[str, float] = field(default_factory=dict)
    resource_after: dict[str, float] = field(default_factory=dict)
    frozen_context_vec: tuple[float, ...] = ()
    frozen_action_vec: tuple[float, ...] = ()
    prior_action_logprob: float = 0.0
    controller_id: str = "dcc"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


TransitionNode = PublicTransition


@dataclass(frozen=True)
class TailExample:
    """Supervision example for H_{phi,m}^{pi_0}(Y, C_tail | h, g, beta) generated strictly by frozen pi_0 suffixes."""

    example_id: str
    cluster_id: str
    task_id: str
    benchmark: str
    step_index: int
    tail_policy_version: str
    goal_text: str
    pre_public_fields: dict[str, str]
    resource_before: dict[str, float]
    frozen_context_vec: tuple[float, ...] = ()
    frozen_tail_action_vec: tuple[float, ...] = ()
    step_budget_fraction: float = 0.05
    observed_y: bool | None = None
    tail_seconds_cost_ratio: float | None = None
    total_seconds_cost_ratio: float | None = None
    censored: bool = False
    prefix_node_id: str | None = None
    episode_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecisionContext:
    """Single read-only decision state shared identically across q-selection, post-q d_model, observe, and deployment."""

    benchmark: str
    goal_text: str
    public_fields: dict[str, str]
    typed_slots: tuple[int, int, int, int]
    role_bindings: dict[str, str]
    action_schema: Mapping[str, Any]
    remaining_budget_ratio: float
    spent_seconds_ratio: float
    remaining_steps_ratio: float
    step_budget_fraction: float
    frozen_context_vec: tuple[float, ...]
    frozen_tail_action_vec: tuple[float, ...]
    member_index: int | None = None
    controller_mode: str = "expected_utility"
    action_templates: dict[str, str] = field(default_factory=dict)

    def template_for_action(self, action_text: str) -> str:
        act_s = str(action_text).strip()
        if act_s in self.action_templates:
            return self.action_templates[act_s]
        return PublicBenchmarkBinder.abstract_action_to_template(
            self.benchmark, act_s, self.role_bindings
        )


@dataclass(frozen=True)
class Comparison:
    """Canonical ordered continuation pair (u, v) fixed before observing branch outcomes (fixes F15)."""

    comparison_key: str
    goal_category: str
    u_policy_id: str
    v_policy_id: str
    tail_policy_version: str
    polarity_sign: float = 1.0

    @classmethod
    def canonicalize(
        cls,
        *,
        goal_category: str,
        u_policy_id: str,
        v_policy_id: str,
        tail_policy_version: str = TAIL_POLICY_VERSION,
    ) -> "Comparison":
        u_str = str(u_policy_id).strip()
        v_str = str(v_policy_id).strip()
        if u_str <= v_str:
            c_u, c_v, sign = u_str, v_str, 1.0
        else:
            c_u, c_v, sign = v_str, u_str, -1.0
        key = f"{goal_category}::{c_u}__vs__{c_v}::{tail_policy_version}"
        return cls(
            comparison_key=key,
            goal_category=goal_category,
            u_policy_id=c_u,
            v_policy_id=c_v,
            tail_policy_version=tail_policy_version,
            polarity_sign=sign,
        )


@dataclass
class MeasurementRecord:
    """Paired intervention record with canonical Comparison key and separated d_real / d_model."""

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
    comparison_key: str = ""
    program_executed_cleanly: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.public_readout_z, Mapping):
            self.public_readout_z = {
                str(k): str(v) for k, v in self.public_readout_z.items() if not str(k).startswith("_")
            }
        if not self.comparison_key:
            comp = Comparison.canonicalize(
                goal_category=self.goal_category,
                u_policy_id=self.u_policy_id,
                v_policy_id=self.v_policy_id,
                tail_policy_version=self.tail_policy_version,
            )
            self.comparison_key = comp.comparison_key
            self.continuation_pair_key = f"{comp.u_policy_id}__vs__{comp.v_policy_id}"
            if comp.polarity_sign < 0:
                self.u_policy_id, self.v_policy_id = comp.u_policy_id, comp.v_policy_id
                self.u_score, self.v_score = self.v_score, self.u_score
                self.u_cost, self.v_cost = self.v_cost, self.u_cost
                self.u_done, self.v_done = self.v_done, self.u_done
                self.u_native_outcome, self.v_native_outcome = self.v_native_outcome, self.u_native_outcome
                self.action_diff_d = -float(self.action_diff_d)
                if self.d_real is not None:
                    self.d_real = -float(self.d_real)
                if self.d_model is not None:
                    self.d_model = -float(self.d_model)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    permutable_groups: int = 0
    permutable_records: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PublicBenchmarkBinder:
    """Public field extractor, LOCA nested schema adapter, and stack-based AST interpreter."""

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
        """Extract the public task instruction g from message history, skipping system prompts."""
        if not messages:
            return fallback
        for msg in messages:
            role = str(msg.get("role", "")).lower()
            content = str(msg.get("content", "")).strip()
            if role == "user" and content:
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, Mapping) and parsed.get("instruction"):
                        return str(parsed["instruction"]).strip()
                except Exception:
                    pass
                return content
        for msg in reversed(messages):
            if str(msg.get("role", "")).lower() != "system":
                content = str(msg.get("content", "")).strip()
                if content:
                    return content
        return fallback or str(messages[0].get("content", ""))

    @classmethod
    def _extract_loca_tools(cls, action_schema: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Extract normalized LOCA tool definitions from `native_tools` or `program_context` (fixes F07)."""
        raw_tools: list[Any] = []
        if isinstance(action_schema.get("native_tools"), Sequence):
            raw_tools.extend(action_schema["native_tools"])
        prog_ctx = action_schema.get("program_context", {})
        if isinstance(prog_ctx, Mapping) and isinstance(prog_ctx.get("tools"), Sequence):
            raw_tools.extend(prog_ctx["tools"])
        if isinstance(action_schema.get("tools"), Sequence):
            raw_tools.extend(action_schema["tools"])

        open_modes = action_schema.get("open_text_modes", {})
        mode_by_name: dict[str, str] = {}
        if isinstance(open_modes, Mapping):
            for m_key in ("I", "E"):
                for t_name in open_modes.get(m_key, ()):
                    mode_by_name[str(t_name)] = m_key

        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in raw_tools:
            if isinstance(item, Mapping):
                fn = item.get("function") if isinstance(item.get("function"), Mapping) else item
                name = str(fn.get("name", "")).strip()
                params = fn.get("parameters", {"type": "object"})
                if not isinstance(params, Mapping):
                    params = {"type": "object"}
            else:
                name = str(item).strip()
                params = {"type": "object"}
            if not name or name in seen:
                continue
            seen.add(name)
            mode = mode_by_name.get(name)
            if mode is None:
                is_read = any(
                    k in name.lower()
                    for k in ("get", "list", "read", "search", "inspect", "find", "query", "status", "view", "check", "show")
                )
                mode = "I" if is_read else "E"
            normalized.append({"name": name, "parameters": dict(params), "mode": mode})

        # Fallback if only `open_text_modes` names were supplied
        if not normalized and isinstance(open_modes, Mapping):
            for m_key in ("I", "E"):
                for t_name in open_modes.get(m_key, ()):
                    name = str(t_name).strip()
                    if name and name not in seen:
                        seen.add(name)
                        normalized.append({"name": name, "parameters": {"type": "object"}, "mode": m_key})
        return normalized

    @classmethod
    def _build_loca_tool_arguments(
        cls,
        tool_spec: Mapping[str, Any],
        bindings: Mapping[str, str],
        task_instruction: str,
    ) -> dict[str, Any] | None:
        """Construct schema-compliant JSON arguments for a LOCA tool using bound values and task literals."""
        params = tool_spec.get("parameters", {})
        if not isinstance(params, Mapping):
            return {}
        props = params.get("properties", {})
        if not isinstance(props, Mapping):
            props = {}
        required = list(params.get("required", ()))

        args: dict[str, Any] = {}
        for req_key in required:
            prop_schema = props.get(req_key, {}) if isinstance(props.get(req_key), Mapping) else {}
            if "const" in prop_schema:
                args[req_key] = prop_schema["const"]
                continue
            if "enum" in prop_schema and isinstance(prop_schema["enum"], Sequence) and prop_schema["enum"]:
                args[req_key] = prop_schema["enum"][0]
                continue
            if "default" in prop_schema:
                args[req_key] = prop_schema["default"]
                continue

            # Check live bindings from previous tool returns or task entities
            bound_val = (
                bindings.get(f"bound_{req_key}")
                or bindings.get(req_key)
                or (bindings.get("bound_id") if "id" in req_key.lower() else None)
                or (bindings.get("bound_path") if ("path" in req_key.lower() or "file" in req_key.lower() or "dir" in req_key.lower()) else None)
                or (bindings.get("bound_query") if ("query" in req_key.lower() or "keyword" in req_key.lower() or "search" in req_key.lower()) else None)
            )
            p_type = str(prop_schema.get("type", "string")).lower()
            if bound_val is not None:
                if p_type == "integer":
                    try:
                        args[req_key] = int(bound_val)
                    except ValueError:
                        args[req_key] = 1
                elif p_type == "number":
                    try:
                        args[req_key] = float(bound_val)
                    except ValueError:
                        args[req_key] = 1.0
                elif p_type == "boolean":
                    args[req_key] = bound_val.lower() in ("true", "1", "yes")
                else:
                    args[req_key] = str(bound_val)
            else:
                if p_type == "integer":
                    args[req_key] = 1
                elif p_type == "number":
                    args[req_key] = 1.0
                elif p_type == "boolean":
                    args[req_key] = True
                elif p_type == "array":
                    args[req_key] = []
                elif p_type == "object":
                    args[req_key] = {}
                elif "path" in req_key.lower() or "dir" in req_key.lower():
                    args[req_key] = "."
                elif "query" in req_key.lower():
                    words = [w for w in re.findall(r"[a-zA-Z0-9_-]{3,}", task_instruction)[:3]]
                    args[req_key] = " ".join(words) if words else "status"
                else:
                    # Cannot safely synthesize an unbound opaque required string ID
                    if "id" in req_key.lower() or "token" in req_key.lower():
                        return None
                    args[req_key] = "."
        return args

    @classmethod
    def extract_role_bindings(
        cls,
        benchmark: str,
        task_instruction: str,
        events: Sequence[Any],
        action_schema: Mapping[str, Any],
    ) -> dict[str, str]:
        """Construct a task-local role-binding table mapping symbolic slots to concrete entities."""
        name = benchmark.split(":", 1)[0]
        obs_texts = [str(getattr(ev, "observation", "")) for ev in events]
        joined_obs = "\n".join(obs_texts[-6:]) if obs_texts else ""
        bindings: dict[str, str] = {}

        if name == "alfworld":
            goal_lower = task_instruction.lower()
            goal_stopwords = {
                "put", "the", "and", "in", "on", "to", "at", "by", "of", "an", "as", "or",
                "from", "with", "look", "find", "clean", "heat", "cool", "examine", "turn",
                "under", "some", "two", "there", "is", "are", "you",
            }
            goal_nouns = [
                tok for tok in re.findall(r"\b([a-z]{2,})\b", goal_lower)
                if tok not in goal_stopwords
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
            for idx, rec in enumerate(receptacles[:6]):
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
            ws_stop = {
                "instruction", "find", "me", "buy", "with", "and", "the", "for",
                "price", "lower", "than", "dollars", "dollar", "item", "need", "want",
            }
            words = [
                w for w in re.findall(r"[a-z0-9]+", task_instruction.lower())
                if len(w) >= 2 and w not in ws_stop and not re.fullmatch(r"\d+(?:00)?", w)
            ][:16]
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
                for idx, asin in enumerate(asins[:4]):
                    bindings[f"candidate_asin_{idx + 1}"] = asin
            return bindings

        # LOCA / structured tool environments (fixes F07)
        loca_tools = cls._extract_loca_tools(action_schema)
        read_tools = [t["name"] for t in loca_tools if t["mode"] == "I"]
        write_tools = [t["name"] for t in loca_tools if t["mode"] == "E"]
        if read_tools:
            bindings["primary_read_tool"] = read_tools[0]
            bindings["secondary_read_tool"] = read_tools[min(1, len(read_tools) - 1)]
        elif loca_tools:
            bindings["primary_read_tool"] = loca_tools[0]["name"]
            bindings["secondary_read_tool"] = loca_tools[0]["name"]
        if write_tools:
            bindings["primary_write_tool"] = write_tools[0]

        path_matches = re.findall(r"([a-zA-Z0-9_./-]+\.[a-zA-Z0-9]{1,6})", task_instruction)
        if path_matches:
            bindings["bound_path"] = path_matches[0]
        for ev in events[-6:]:
            obs_str = str(getattr(ev, "observation", ""))
            for key_name, val in re.findall(
                r'"([a-zA-Z0-9_]*(?:id|path|file|token|key|name|dir))"\s*:\s*"([^"]{1,64})"',
                obs_str,
            ):
                k_low = key_name.lower()
                bindings[f"bound_{k_low}"] = val
                if "id" in k_low and "bound_id" not in bindings:
                    bindings["bound_id"] = val
                if ("path" in k_low or "file" in k_low) and "bound_path" not in bindings:
                    bindings["bound_path"] = val
        return bindings

    @classmethod
    def extract_public_fields(
        cls,
        benchmark: str,
        task_instruction: str,
        events: Sequence[Any],
        action_schema: Mapping[str, Any],
    ) -> dict[str, str]:
        """Extract typed public fields from observed history without guessing unobserved facts."""
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
            goal_stopwords = {
                "put", "the", "and", "in", "on", "to", "at", "by", "of", "an", "as", "or",
                "from", "with", "look", "find", "some", "two", "clean", "heat", "cool",
                "examine", "turn", "under", "there", "is", "are", "you",
            }
            goal_tokens = [
                tok for tok in re.findall(r"\b[a-z]{2,}\b", goal_lower)
                if tok not in goal_stopwords
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

            held_item = "none"
            treatment = "raw"
            last_act_norm = "none"
            tried_in_regime: list[str] = []
            for ev in events:
                obs_l = str(getattr(ev, "observation", "")).lower()
                raw_act = str(
                    getattr(getattr(ev, "generation", None), "text", "")
                    or (getattr(ev, "action", {}) or {}).get("text", "")
                    or getattr(ev, "response", "")
                ).strip()
                act_l = raw_act.lower()
                prev_regime = (held_item, treatment)
                picked = re.findall(r"(?:you pick up|you are carrying:)\s+([^\.\n]+)", obs_l)
                if picked and "nothing" not in picked[-1]:
                    held_item = picked[-1].strip()
                elif act_l.startswith("take ") and ("you pick up" in obs_l or "you take" in obs_l):
                    held_item = act_l.replace("take ", "").split(" from ")[0].strip()
                elif (act_l.startswith("put ") or act_l.startswith("move ")) and ("you put" in obs_l or "you move" in obs_l):
                    held_item = "none"
                for tag, phrase in (
                    ("clean", "you clean"),
                    ("hot", "you heat"),
                    ("cool", "you cool"),
                    ("examined", "you turn on"),
                ):
                    if phrase in obs_l:
                        treatment = tag
                if (held_item, treatment) != prev_regime or "you open" in obs_l:
                    tried_in_regime = []
                elif act_l and act_l not in tried_in_regime:
                    tried_in_regime.append(act_l)
                if act_l:
                    last_act_norm = act_l

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
                "_last_action": last_act_norm,
                "_regime_tried": "|".join(tried_in_regime[-16:]),
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
            chosen_opts: list[str] = []
            inspected_set: list[str] = []
            last_act_norm = "none"
            tried_in_regime: list[str] = []
            cur_query = ""
            cur_item = "none"
            for ev in events:
                raw_act = str(
                    (getattr(ev, "action", {}) or {}).get("text", "")
                    or getattr(getattr(ev, "generation", None), "text", "")
                    or getattr(ev, "response", "")
                ).strip()
                act_l = raw_act.lower()
                prev_regime = (cur_query, cur_item, tuple(chosen_opts))
                if act_l.startswith("search["):
                    cur_query = act_l
                    cur_item = "none"
                    chosen_opts = []
                    inspected_set = []
                elif act_l == "click[back to search]":
                    cur_item = "none"
                    chosen_opts = []
                    inspected_set = []
                else:
                    m_asin = re.match(r"^click\[(b0[0-9a-z]{8})\]$", act_l)
                    if m_asin:
                        cur_item = m_asin.group(1)
                        chosen_opts = []
                        inspected_set = []
                    elif act_l in ("click[description]", "click[features]", "click[attributes]", "click[reviews]"):
                        sub_name = act_l[6:-1]
                        if sub_name not in inspected_set:
                            inspected_set.append(sub_name)
                    elif (
                        act_l.startswith("click[")
                        and "buy now" not in act_l
                        and act_l not in ("click[< prev]", "click[next >]", "click[search]")
                    ):
                        if raw_act not in chosen_opts:
                            chosen_opts.append(raw_act)
                if (cur_query, cur_item, tuple(chosen_opts)) != prev_regime:
                    tried_in_regime = []
                elif act_l and act_l not in tried_in_regime:
                    tried_in_regime.append(act_l)
                if act_l:
                    last_act_norm = act_l

            goal_words = set(re.findall(r"[a-z0-9]+", goal_lower))
            obs_words = set(re.findall(r"[a-z0-9]+", joined_obs.lower()))
            overlap = len(goal_words & obs_words) / max(1, len(goal_words))
            return {
                "page_type": page_type,
                "current_asin": current_asin,
                "selected_options": ",".join(chosen_opts[-3:]) or "none",
                "inspected_attributes": ",".join(inspected_set) or "none",
                "price_rating_match": "high" if overlap >= 0.45 else "low",
                "_last_action": last_act_norm,
                "_regime_tried": "|".join(tried_in_regime[-16:]),
            }

        tool_families = []
        bound_ids: list[str] = []
        had_error = False
        artifact_state = "unchanged"
        last_act_norm = "none"
        tried_in_regime = []
        for ev in events:
            raw_act = str(
                getattr(getattr(ev, "generation", None), "text", "")
                or getattr(ev, "response", "")
            ).strip()
            obs_str = str(getattr(ev, "observation", ""))
            prev_regime = (len(set(bound_ids)), artifact_state)
            for m in re.findall(r'"(?:id|path|file|token|key|record_id)"\s*:\s*"([^"]{1,32})"', obs_str):
                bound_ids.append(m)
            step_err = "error" in obs_str.lower() or "exception" in obs_str.lower() or "invalid" in obs_str.lower()
            if not step_err and ("success" in obs_str.lower() or "written" in obs_str.lower() or "updated" in obs_str.lower()):
                artifact_state = "verified_write"
            norm_a = cls.normalize_action_for_semantic_encoding(raw_act).lower()
            if (len(set(bound_ids)), artifact_state) != prev_regime:
                tried_in_regime = []
            elif norm_a and norm_a not in tried_in_regime:
                tried_in_regime.append(norm_a)
            if norm_a:
                last_act_norm = norm_a
        for ev in events[-4:]:
            act_text = str(getattr(ev, "generation", None).text if getattr(ev, "generation", None) else "")
            obs_str = str(getattr(ev, "observation", ""))
            tool_families.append(act_text[:32] or "call")
            if "error" in obs_str.lower() or "exception" in obs_str.lower() or "invalid" in obs_str.lower():
                had_error = True
        return {
            "last_tool_family": tool_families[-1] if tool_families else "none",
            "bound_identifiers": f"count_{min(4, len(set(bound_ids)))}",
            "prerequisite_status": "error" if had_error else ("ready" if bound_ids else "pending"),
            "workspace_artifact_state": artifact_state,
            "_last_action": last_act_norm,
            "_regime_tried": "|".join(tried_in_regime[-16:]),
        }

    @classmethod
    def encode_typed_field_slots(cls, public_fields: Mapping[str, str]) -> tuple[int, int, int, int]:
        """Idempotently map raw environment fields OR canonical `SLOT_VALUE_NAMES` into `{0, 1, 2, 3}^4`.

        Slot 0 (visibility / match):
          0=unknown, 1=no_or_low, 2=yes_or_high, 3=error
        Slot 1 (entity / inventory / identifier binding):
          0=unknown, 1=none, 2=bound_single, 3=bound_multiple
        Slot 2 (container / page / prerequisite state):
          0=unknown, 1=closed_or_pending, 2=open_empty_or_subpage, 3=open_with_items_or_ready
        Slot 3 (treatment / attribute / workspace artifact state):
          0=unknown, 1=raw_or_unchanged, 2=inspected_or_selected, 3=treated_or_verified_write
        """
        # Slot 0
        vis = public_fields.get("target_object_visible", public_fields.get("price_rating_match", "unknown"))
        if public_fields.get("prerequisite_status") == "error" or vis == "error":
            s0 = 3
        elif vis in ("yes", "high", "yes_or_high"):
            s0 = 2
        elif vis in ("no", "low", "no_or_low"):
            s0 = 1
        else:
            s0 = 0

        # Slot 1
        inv = public_fields.get(
            "inventory_object",
            public_fields.get("current_asin", public_fields.get("bound_identifiers", "unknown")),
        )
        if inv == "unknown":
            s1 = 0
        elif inv in ("none", "count_0"):
            s1 = 1
        elif inv in ("bound_multiple", "count_2", "count_3", "count_4") or "," in inv:
            s1 = 3
        else:
            s1 = 2

        # Slot 2
        st = public_fields.get(
            "open_receptacle_state",
            public_fields.get("page_type", public_fields.get("prerequisite_status", "unknown")),
        )
        if st == "unknown":
            s2 = 0
        elif st in ("closed", "search_home", "pending", "error", "closed_or_pending"):
            s2 = 1
        elif st in ("open_empty", "search_results", "item_subpage", "open_empty_or_subpage"):
            s2 = 2
        else:
            s2 = 3

        # Slot 3
        tr = public_fields.get(
            "object_treatment_state",
            public_fields.get("workspace_artifact_state", public_fields.get("selected_options", "unknown")),
        )
        insp = public_fields.get("inspected_attributes", "none")
        if tr == "unknown":
            s3 = 0
        elif tr in ("clean", "hot", "cool", "examined", "verified_write", "treated_or_verified_write"):
            s3 = 3
        elif tr in ("inspected", "selected", "inspected_or_selected") or (
            tr not in ("raw", "none", "unchanged", "raw_or_unchanged")
            or insp not in ("none", "unknown", "raw_or_unchanged")
        ):
            s3 = 2
        else:
            s3 = 1

        return (int(s0), int(s1), int(s2), int(s3))

    @classmethod
    def slots_to_canonical_fields(
        cls,
        base_fields: Mapping[str, str],
        slots: tuple[int, int, int, int],
    ) -> dict[str, str]:
        """Project a typed 4-slot assignment `(s0, s1, s2, s3)` onto canonical field keys with strict round-trip identity."""
        out = dict(base_fields)
        s0, s1, s2, s3 = (int(slots[0]) % 4, int(slots[1]) % 4, int(slots[2]) % 4, int(slots[3]) % 4)
        k0 = "price_rating_match" if "price_rating_match" in out else "target_object_visible"
        k1 = (
            "current_asin" if "current_asin" in out
            else "bound_identifiers" if "bound_identifiers" in out
            else "inventory_object"
        )
        k2 = (
            "page_type" if "page_type" in out
            else "prerequisite_status" if ("prerequisite_status" in out and "open_receptacle_state" not in out)
            else "open_receptacle_state"
        )
        k3 = (
            "workspace_artifact_state" if "workspace_artifact_state" in out
            else "selected_options" if "selected_options" in out
            else "object_treatment_state"
        )
        out[k0] = SLOT_VALUE_NAMES[0][s0]
        out[k1] = SLOT_VALUE_NAMES[1][s1]
        out[k2] = SLOT_VALUE_NAMES[2][s2]
        out[k3] = SLOT_VALUE_NAMES[3][s3]
        if s0 == 3:
            out["prerequisite_status"] = "error"
        elif out.get("prerequisite_status") == "error" and k2 != "prerequisite_status":
            out["prerequisite_status"] = SLOT_VALUE_NAMES[2][s2]
        if s3 != 2 and "inspected_attributes" in out:
            out["inspected_attributes"] = "none"
        return out

    @classmethod
    def canonical_slot_predicate_value(cls, condition_field: str, raw_value: str) -> str:
        """Normalize a field value into the exact categorical predicate event supported by both AST execution and K_phi."""
        if raw_value == "unknown":
            return "unknown"
        slot_idx, val_idx = cls.condition_slot_and_value(condition_field, raw_value)
        return SLOT_VALUE_NAMES[slot_idx][val_idx]

    @classmethod
    def condition_slot_and_value(cls, condition_field: str, condition_equals: str) -> tuple[int, int]:
        """Map a Case(condition_field, condition_equals) predicate to its corresponding (slot_idx, value_idx)."""
        if condition_field in ("target_object_visible", "price_rating_match"):
            s_idx = 0
        elif condition_field in ("inventory_object", "current_asin", "bound_identifiers"):
            s_idx = 1
        elif condition_field in ("open_receptacle_state", "page_type", "prerequisite_status"):
            s_idx = 2
        else:
            s_idx = 3
        for v_i, canon_name in enumerate(SLOT_VALUE_NAMES[s_idx]):
            if condition_equals == canon_name:
                return s_idx, v_i
        dummy = {condition_field: condition_equals}
        slots = cls.encode_typed_field_slots(dummy)
        return s_idx, slots[s_idx]

    @classmethod
    def normalize_action_for_semantic_encoding(
        cls,
        action_text: str,
        role_bindings: Mapping[str, str] | None = None,
    ) -> str:
        """Canonicalize opaque session/return identifiers so training NLL and symbolic AST lookahead share identical action features."""
        act = str(action_text or "").strip()
        if not act or act == TAIL_POLICY_VERSION:
            return act
        # Normalize symbolic return handles `<sym:ret_k:id>` / `<sym:ret_k:path>` to role-canonical tokens
        act = re.sub(r"<sym:[^:>]+:id>", "<sym:bound_id>", act)
        act = re.sub(r"<sym:[^:>]+:path>", "<sym:bound_path>", act)
        if role_bindings:
            # Map concrete bound ASINs to their positional role (`<sym:candidate_asin_k>`) before generic ASIN scrub
            for rk in sorted(role_bindings.keys()):
                rv = str(role_bindings.get(rk, "")).strip()
                if not rv or rv in ("unknown", "none"):
                    continue
                if rk.startswith("candidate_asin_") and len(rv) >= 8:
                    act = re.sub(re.escape(rv), f"<sym:{rk}>", act, flags=re.IGNORECASE)
                elif rk == "bound_id" and len(rv) >= 4 and not rv.startswith("<sym:"):
                    act = act.replace(f'"{rv}"', '"<sym:bound_id>"')
                elif rk == "bound_path" and len(rv) >= 4 and not rv.startswith("<sym:"):
                    act = act.replace(f'"{rv}"', '"<sym:bound_path>"')
        act = re.sub(r"\bb0[0-9a-z]{8}\b", "<sym:candidate_asin_1>", act, flags=re.IGNORECASE)
        return act

    @staticmethod
    def native_terminal_utility(
        native_outcome: bool | None,
        cost_used: float,
        cost_budget: float,
    ) -> float | None:
        """Compute official native terminal utility U_B(Y, C_seconds) = Y * max(0, 1 - C_seconds / B_seconds)."""
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
        """Return official terminal utility only; mid-episode hand-crafted progress is zero."""
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
        """Encode 8-bin coarse summary while checking explicit positive binding values (fixes F09 'no' bug)."""
        changed_fields = sum(
            1 for k, v in post_fields.items()
            if not str(k).startswith("_") and pre_fields.get(k, "unknown") != v
        )
        info_bit = 1 if changed_fields > 0 else 0
        entity_bound = (
            post_fields.get("inventory_object", "unknown") not in ("unknown", "none")
            or post_fields.get("current_asin", "unknown") not in ("unknown", "none")
            or post_fields.get("bound_identifiers", "unknown") not in ("unknown", "none", "count_0")
            or post_fields.get("target_object_visible") == "yes"
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
        """Rebind concrete entity IDs to distinguishable rebindable role slots (preserving distinct u/v roles)."""
        name = benchmark.split(":", 1)[0]
        text = action_text.strip()
        bindings = dict(role_bindings or {})

        if name == "alfworld":
            lowered = text.lower()
            # Priority role order so distinct receptacles/objects keep distinguishable role identities
            priority_roles = (
                "target_object",
                "target_receptacle",
                "source_receptacle",
                "receptacle_1",
                "receptacle_2",
                "receptacle_3",
                "receptacle_4",
                "receptacle_5",
                "receptacle_6",
            )
            for slot_name in priority_roles:
                entity_val = str(bindings.get(slot_name, "")).strip().lower()
                if entity_val and len(entity_val) >= 3 and entity_val in lowered:
                    lowered = lowered.replace(entity_val, f"{{{slot_name}}}")
            # Preserve ordinal index on any remaining unbound entity so `drawer 1` vs `drawer 2` stay distinct
            lowered = re.sub(r"\b([a-z]+)\s+(\d+)\b", r"{\1_\2}", lowered)
            return lowered[:64]

        if name == "webshop":
            if text.lower().startswith("search["):
                return "search[{goal_query}]"
            m_asin = re.search(r"click\[(b0[0-9a-z]{8})\]", text.lower())
            if m_asin:
                asin_val = m_asin.group(1)
                for idx in range(1, 5):
                    slot_k = f"candidate_asin_{idx}"
                    if str(bindings.get(slot_k, "")).lower() == asin_val:
                        return f"click[{{{slot_k}}}]"
                if str(bindings.get("candidate_asin", "")).lower() == asin_val:
                    return "click[{candidate_asin_1}]"
                return f"click[{{candidate_asin_{asin_val[-2:]}}}]"
            return text[:64]

        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "name" in obj:
                tool_name = str(obj["name"])
                args = obj.get("arguments", {})
                if isinstance(args, dict) and args:
                    inv_bound = {
                        str(v): k for k, v in bindings.items()
                        if k.startswith("bound_") and v is not None and str(v)
                    }
                    arg_parts: list[str] = []
                    for ak in sorted(args.keys())[:3]:
                        av = args[ak]
                        av_str = str(av)
                        if av_str in inv_bound:
                            arg_parts.append(f"{ak}=${inv_bound[av_str]}")
                        elif isinstance(av, (int, float, bool)):
                            arg_parts.append(f"{ak}={av}")
                        else:
                            arg_parts.append(f"{ak}={av_str[:12]}")
                    return f"{tool_name}({','.join(arg_parts)})"[:80]
                return f"{tool_name}()"
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
        """Return unbiased admissible actions across both I and E without truncating E behind I (fixes F07, F08)."""
        name = benchmark.split(":", 1)[0]
        bindings = dict(role_bindings or {})
        if not bindings:
            bindings = cls.extract_role_bindings(benchmark, task_instruction, (), action_schema)

        if name == "loca" or action_schema.get("native_tools") or action_schema.get("native_action_schemas_by_mode"):
            candidates: list[str] = []
            # 1. Include ready zero-arg probe actions from environment schema
            for p_act in action_schema.get("probe_actions", ()):
                s = str(p_act).strip()
                if s and s not in candidates:
                    candidates.append(s)
            # 2. Construct valid JSON calls across both I and E tools from native_tools
            loca_tools = cls._extract_loca_tools(action_schema)
            i_tools = [t for t in loca_tools if t["mode"] == "I"]
            e_tools = [t for t in loca_tools if t["mode"] == "E"]
            # Interleave I and E tools so neither mode is starved by any downstream budget cap
            interleaved: list[dict[str, Any]] = []
            for idx in range(max(len(i_tools), len(e_tools))):
                if idx < len(i_tools):
                    interleaved.append(i_tools[idx])
                if idx < len(e_tools):
                    interleaved.append(e_tools[idx])
            for t_spec in interleaved:
                args_obj = cls._build_loca_tool_arguments(t_spec, bindings, task_instruction)
                if args_obj is None:
                    continue
                call_str = json.dumps(
                    {"name": t_spec["name"], "arguments": args_obj},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if call_str not in candidates:
                    candidates.append(call_str)
            return candidates

        mode_actions = action_schema.get("mode_actions", {})
        i_list = [str(a).strip() for a in mode_actions.get("I", ()) if str(a).strip()]
        e_list = [str(a).strip() for a in mode_actions.get("E", ()) if str(a).strip()]
        # Interleave E and I actions so every feasible E action (e.g., take/put/clean/Buy Now) is always represented
        candidates = []
        for idx in range(max(len(i_list), len(e_list))):
            if idx < len(e_list) and e_list[idx] not in candidates:
                candidates.append(e_list[idx])
            if idx < len(i_list) and i_list[idx] not in candidates:
                candidates.append(i_list[idx])
        for act in action_schema.get("available_actions", ()):
            s = str(act).strip()
            if s and s not in candidates:
                candidates.append(s)
        if name == "webshop" and action_schema.get("search_available"):
            query = bindings.get("goal_query", "")
            if not query and task_instruction:
                words = [w for w in re.findall(r"[a-z0-9]+", task_instruction.lower()) if len(w) >= 2][:16]
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
        local_env: Mapping[str, Mapping[str, str]] | None = None,
        typed_arguments: Mapping[str, str] | None = None,
        *,
        allow_symbolic: bool = False,
    ) -> str | None:
        """Bind a symbolic step template or AST Call node (with `typed_arguments`) to a concrete or symbolic action."""
        name = benchmark.split(":", 1)[0]
        bindings = dict(role_bindings or {})
        if not bindings:
            bindings = cls.extract_role_bindings(benchmark, task_instruction, (), action_schema)
        if local_env:
            for var_name, field_map in local_env.items():
                for fk, fv in field_map.items():
                    bindings[f"{var_name}.{fk}"] = fv
                    if fk.startswith("bound_"):
                        bindings.setdefault(fk, fv)

        # Resolve `Call.typed_arguments` ($ret_k.field or $slot) directly into active parameter bindings
        if typed_arguments:
            for param_slot, ref_expr in typed_arguments.items():
                clean_ref = str(ref_expr).lstrip("$")
                resolved_val: str | None = None
                if "." in clean_ref and local_env:
                    var_part, fld_part = clean_ref.split(".", 1)
                    if var_part in local_env and fld_part in local_env[var_part]:
                        resolved_val = local_env[var_part][fld_part]
                if resolved_val is None and clean_ref in bindings:
                    resolved_val = bindings[clean_ref]
                if resolved_val is None and allow_symbolic and "." in clean_ref:
                    var_part, fld_part = clean_ref.split(".", 1)
                    sym_kind = "path" if ("path" in fld_part or "file" in fld_part) else "id"
                    resolved_val = f"<sym:{var_part}:{sym_kind}>"
                if resolved_val is not None and resolved_val != "unknown":
                    bindings[param_slot] = resolved_val
                    bindings[f"bound_{param_slot}"] = resolved_val

        all_candidates = cls.enumerate_admissible_actions(benchmark, action_schema, task_instruction, bindings)
        if template in all_candidates:
            return template

        resolved = template
        for slot in re.findall(r"\{([a-zA-Z0-9_.]+)\}", template):
            if slot in bindings and bindings[slot]:
                resolved = resolved.replace(f"{{{slot}}}", bindings[slot])
            elif re.match(r"^[a-z]+_\d+$", slot) and not slot.startswith("candidate_asin"):
                literal_entity = slot.replace("_", " ")
                resolved = resolved.replace(f"{{{slot}}}", literal_entity)
            elif allow_symbolic:
                resolved = resolved.replace(f"{{{slot}}}", f"<sym:{slot}>")
        if resolved in all_candidates:
            return resolved
        for cand in all_candidates:
            if cand.lower() == resolved.lower():
                return cand
        if allow_symbolic and "<sym:" in resolved and not name == "loca":
            return resolved

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
                    slot_type = re.sub(r"_\d+$", "", slot_matches[0]).replace("_id", "").replace("target_", "").replace("source_", "")
                    if slot_type and slot_type not in ("receptacle", "object") and slot_type not in cand_l:
                        continue
                matching.append(cand)
            if matching:
                return matching[0]
            return resolved if (allow_symbolic and "{" not in resolved) else None

        if name == "webshop":
            if template.startswith("search[") and (action_schema.get("search_available") or allow_symbolic):
                query = bindings.get("goal_query", "")
                if not query:
                    words = [w for w in re.findall(r"[a-z0-9]+", task_instruction.lower()) if len(w) >= 2][:16]
                    query = " ".join(words)
                return f"search[{query}]" if query else None
            m_asin_slot = re.match(r"^click\[\{(candidate_asin(?:_\d+)?)\}\]$", template.strip())
            if m_asin_slot:
                slot_name = m_asin_slot.group(1)
                target_asin = bindings.get(slot_name) or bindings.get("candidate_asin", "")
                for cand in all_candidates:
                    if target_asin and f"click[{target_asin}]" == cand.lower():
                        return cand
                asin_clicks = [c for c in all_candidates if re.search(r"click\[b0[0-9a-z]{8}\]", c.lower())]
                if slot_name.startswith("candidate_asin_") and slot_name[-1].isdigit():
                    idx_pos = max(0, int(slot_name.rsplit("_", 1)[-1]) - 1)
                    if idx_pos < len(asin_clicks):
                        return asin_clicks[idx_pos]
                if asin_clicks:
                    return asin_clicks[0]
                if allow_symbolic:
                    norm_slot = slot_name if slot_name != "candidate_asin" else "candidate_asin_1"
                    return f"click[<sym:{norm_slot}>]"
                return None
            for cand in all_candidates:
                if cand.lower() == template.lower():
                    return cand
            return resolved if (allow_symbolic and "{" not in resolved) else None

        # LOCA / structured tool environment (fixes F07 + typed_arguments resolution)
        loca_tools = cls._extract_loca_tools(action_schema)
        tool_by_name = {t["name"]: t for t in loca_tools}
        for slot_key in ("primary_read_tool", "secondary_read_tool", "primary_write_tool"):
            if f"{{{slot_key}}}" in resolved and bindings.get(slot_key):
                resolved = resolved.replace(f"{{{slot_key}}}", bindings[slot_key])
        tool_head = resolved.split(":", 1)[-1].split("(", 1)[0].strip()
        if tool_head in tool_by_name:
            args_payload = cls._build_loca_tool_arguments(tool_by_name[tool_head], bindings, task_instruction)
            if args_payload is None:
                return None
            return json.dumps(
                {"name": tool_head, "arguments": args_payload},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        return None

    @classmethod
    def step_ast_frame(
        cls,
        frame: ExecutionFrame,
        action_schema: Mapping[str, Any],
        task_instruction: str,
        events: Sequence[Any],
    ) -> InterpreterStepResult:
        """Consume AST nodes from `frame.node_stack` until the next `Call`, `Stop`, or `BindingFailure`.

        Resolves `Call.typed_arguments` against `local_env` and `bound_variables`, and suspends
        trailing `Read` nodes until `record_call_return` binds the actual environment response.
        """
        next_frame = frame.clone()
        if next_frame.completed:
            next_frame.last_status = "Complete"
            return InterpreterStepResult("Complete", None, next_frame, False)
        if next_frame.binding_failed:
            next_frame.last_status = "BindingFailure"
            return InterpreterStepResult("BindingFailure", None, next_frame, False)

        live_bindings = cls.extract_role_bindings(
            next_frame.program.benchmark, task_instruction, events, action_schema
        )
        # Merge live_bindings without overwriting return-bound local variables
        merged_vars = dict(live_bindings)
        for k, v in next_frame.bound_variables.items():
            if k.startswith("bound_") and v:
                merged_vars[k] = v
        next_frame.bound_variables = merged_vars

        current_fields = cls.extract_public_fields(
            next_frame.program.benchmark, task_instruction, events, action_schema
        )
        for fk, fv in current_fields.items():
            if fk not in next_frame.observed_readouts or next_frame.observed_readouts[fk] == "unknown":
                next_frame.observed_readouts[fk] = fv

        if events and next_frame.pending_return_var:
            last_obs = str(getattr(events[-1], "observation", ""))
            next_frame.record_call_return(last_obs, current_fields)
            if next_frame.completed:
                return InterpreterStepResult("Complete", None, next_frame, False)

        steps_guard = 0
        while next_frame.node_stack and steps_guard < 32:
            steps_guard += 1
            node = next_frame.node_stack.pop(0)
            if node.op == "Stop":
                next_frame.completed = True
                next_frame.last_status = "Complete"
                return InterpreterStepResult("Complete", None, next_frame, False)
            if node.op == "Seq":
                next_frame.node_stack = list(node.children) + next_frame.node_stack
                continue
            if node.op == "Read":
                if node.variable and node.variable in next_frame.local_env:
                    val = next_frame.local_env[node.variable].get(node.public_field_path)
                    if val is not None:
                        next_frame.observed_readouts[node.public_field_path] = val
                elif node.public_field_path in current_fields:
                    next_frame.observed_readouts[node.public_field_path] = current_fields[node.public_field_path]
                continue
            if node.op == "Case":
                cond_val = next_frame.observed_readouts.get(node.condition_field, "unknown")
                if cond_val == "unknown":
                    next_frame.binding_failed = True
                    next_frame.last_status = "BindingFailure"
                    return InterpreterStepResult("BindingFailure", None, next_frame, False)
                lhs = cls.canonical_slot_predicate_value(node.condition_field, cond_val)
                rhs = cls.canonical_slot_predicate_value(node.condition_field, node.condition_equals)
                take_true = lhs == rhs
                chosen_child = (
                    node.children[0]
                    if (take_true and len(node.children) >= 1)
                    else (node.children[1] if (not take_true and len(node.children) >= 2) else ASTNode(op="Stop"))
                )
                next_frame.node_stack = [chosen_child] + next_frame.node_stack
                continue
            if node.op == "Call":
                concrete_action = cls.instantiate_template_action(
                    next_frame.program.benchmark,
                    node.tool_or_action,
                    action_schema,
                    task_instruction,
                    next_frame.bound_variables,
                    next_frame.local_env,
                    node.typed_arguments,
                )
                if concrete_action is None:
                    next_frame.binding_failed = True
                    next_frame.last_status = "BindingFailure"
                    return InterpreterStepResult("BindingFailure", None, next_frame, False)
                next_frame.program_counter += 1
                next_frame.pending_return_var = node.bind_return_to or f"ret_{next_frame.program_counter - 1}"
                next_frame.executed_actions.append(concrete_action)
                # Only mark completed immediately if no Read, Case, or Call remains on the stack
                if not next_frame.node_stack or (
                    len(next_frame.node_stack) == 1 and next_frame.node_stack[0].op == "Stop"
                ):
                    next_frame.node_stack = []
                    next_frame.completed = True
                next_frame.last_status = "Action"
                return InterpreterStepResult("Action", concrete_action, next_frame, True)

        next_frame.completed = True
        next_frame.last_status = "Complete"
        return InterpreterStepResult("Complete", None, next_frame, False)

    @classmethod
    def step_program_interpreter(
        cls,
        state: ExecutionFrame,
        action_schema: Mapping[str, Any],
        task_instruction: str,
        events: Sequence[Any],
    ) -> tuple[str | None, ExecutionFrame]:
        """Step the stack-based AST interpreter and return `(concrete_action_or_None, updated_frame)`."""
        res = cls.step_ast_frame(state, action_schema, task_instruction, events)
        return res.concrete_action, res.frame


class ControlledResponseEnsemble(nn.Module):
    """Normalized context-conditioned residual response ensemble K_{phi,m} and tail head H_{phi,m}^{pi_0}.

    Fixes F01, F02, F04, F09, F10, F11, F12, F13, F14, F22:
      - Zero-initialized base and residual logits so K_{phi_0} == p_0 strictly holds.
      - Consumes real frozen backbone semantic vectors (`frozen_context_vec`, `frozen_action_vec`),
        zero SHA256 hashes, and zero candidate-origin pseudo-logprobs.
      - Predicts typed field-slot value distributions `(M, NUM_FIELD_SLOTS, NUM_SLOT_VALUES)`
        alongside outcome bins so `Case` conditions and downstream bindings have genuine forward lookahead.
      - Strictly independent ensemble members `m = 0..M-1`: each member maintains its own private
        state `belief[m, :]`, field distributions, termination probabilities, and elapsed seconds ratio;
        no `.mean(dim=0)` across members occurs inside transition or tail heads.
      - Absorbing termination: `alive_prob` multiplies directly by `termination_probs[:, 0]` (`live`).
      - Dedicated amortized fixed-tail head `H_{phi,m}^{pi_0}(Y, C_tail | Y=1)` predicting both
        terminal success probability and an 8-bin conditional wall-clock time-cost distribution given Y=1.
      - Fork-tree marginal NLL: shared prefix transitions are forwarded once, caching differentiable
        `fork_belief` so both u and v suffixes start from `b_{post-q}`.
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

        # Shared semantic context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(self.context_dim, self.context_dim),
            nn.LayerNorm(self.context_dim),
            nn.SiLU(),
            nn.Linear(self.context_dim, self.context_dim),
        )
        # Per-member transition and emission residual projections
        self.transition_residual_proj = nn.Linear(
            self.context_dim, self.ensemble_size * self.state_dim * self.state_dim
        )
        self.emission_residual_proj = nn.Linear(
            self.context_dim, self.ensemble_size * self.state_dim * self.outcome_bins
        )
        # Per-member typed field-slot distribution head: (M, S, NUM_FIELD_SLOTS, NUM_SLOT_VALUES)
        self.field_slot_proj = nn.Linear(
            self.context_dim,
            self.ensemble_size * self.state_dim * NUM_FIELD_SLOTS * NUM_SLOT_VALUES,
        )
        # Break intra-member latent state symmetry on T and initial belief while keeping G and all
        # observable readout heads at zero so the initial observable distribution strictly equals p_0 (fixes B).
        init_gen = torch.Generator(device="cpu").manual_seed(20261110)
        asym_t = 0.15 * torch.randn(
            (self.ensemble_size, self.state_dim, self.state_dim), generator=init_gen
        )
        self.transition_logits = nn.Parameter(asym_t)
        self.emission_logits = nn.Parameter(
            torch.zeros(self.ensemble_size, self.state_dim, self.outcome_bins)
        )

        # Per-member private single-step heads: input is per-member state (S) + context (C), no cross-member mean (fixes F13)
        member_in_dim = self.state_dim + self.context_dim
        self.member_termination_weight = nn.Parameter(
            torch.zeros(self.ensemble_size, member_in_dim, len(TERMINATION_CATEGORIES))
        )
        self.member_termination_bias = nn.Parameter(
            torch.zeros(self.ensemble_size, len(TERMINATION_CATEGORIES))
        )
        self.member_step_cost_weight = nn.Parameter(
            torch.zeros(self.ensemble_size, member_in_dim)
        )
        self.member_step_cost_bias = nn.Parameter(torch.zeros(self.ensemble_size))
        self.member_step_outcome_weight = nn.Parameter(
            torch.zeros(self.ensemble_size, member_in_dim + self.outcome_bins)
        )
        self.member_step_outcome_bias = nn.Parameter(torch.zeros(self.ensemble_size))

        # Dedicated amortized fixed-tail head H_{phi,m}^{pi_0}(Y, C_tail | Y=1, s_m, field_slots, c_tail)
        tail_in_dim = member_in_dim + (NUM_FIELD_SLOTS * NUM_SLOT_VALUES)
        self.tail_pi0_success_weight = nn.Parameter(
            torch.zeros(self.ensemble_size, tail_in_dim)
        )
        self.tail_pi0_success_bias = nn.Parameter(torch.zeros(self.ensemble_size))
        self.tail_pi0_cost_bins_weight = nn.Parameter(
            torch.zeros(self.ensemble_size, tail_in_dim, TAIL_COST_BINS)
        )
        self.tail_pi0_cost_bins_bias = nn.Parameter(
            torch.zeros(self.ensemble_size, TAIL_COST_BINS)
        )

        with torch.no_grad():
            self.transition_residual_proj.weight.copy_(
                0.05 * torch.randn(self.transition_residual_proj.weight.shape, generator=init_gen)
            )
        nn.init.zeros_(self.transition_residual_proj.bias)
        nn.init.zeros_(self.emission_residual_proj.weight)
        nn.init.zeros_(self.emission_residual_proj.bias)
        nn.init.zeros_(self.field_slot_proj.weight)
        nn.init.zeros_(self.field_slot_proj.bias)

        # Asymmetric initial belief logits per member/state so all 16 states receive distinct gradients from step 1
        state_ramp = torch.linspace(-0.6, 0.6, self.state_dim, dtype=torch.float32)
        init_b_logits = torch.stack(
            [torch.roll(state_ramp, shifts=m_i) for m_i in range(self.ensemble_size)], dim=0
        )
        self.register_buffer("initial_belief_logits", init_b_logits)
        self.register_buffer(
            "mechanism_weights",
            torch.full((self.ensemble_size,), 1.0 / self.ensemble_size, dtype=torch.float32),
        )
        self._initial_state_tensors: dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in self.state_dict().items()
        }

    def snapshot_initial_parameters(self) -> None:
        self._initial_state_tensors = {
            k: v.detach().clone() for k, v in self.state_dict().items()
        }
        if hasattr(self, "_cached_initial_view"):
            delattr(self, "_cached_initial_view")

    def initial_state_dict_snapshot(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self._initial_state_tensors.items()}

    def active_model_view(self, use_initial_params: bool = False) -> "ControlledResponseEnsemble":
        """Return an immutable phi_0 view when `use_initial_params=True` without mutating or cloning per call."""
        if not use_initial_params or not self._initial_state_tensors:
            return self
        view = getattr(self, "_cached_initial_view", None)
        if view is None:
            view = ControlledResponseEnsemble(
                ensemble_size=self.ensemble_size,
                state_dim=self.state_dim,
                outcome_bins=self.outcome_bins,
                context_dim=self.context_dim,
            ).to(device=self.transition_logits.device, dtype=self.transition_logits.dtype)
            restored = {
                k: v.to(device=self.transition_logits.device, dtype=self.transition_logits.dtype).detach().clone()
                for k, v in self._initial_state_tensors.items()
            }
            view.load_state_dict(restored, strict=True)
            for p in view.parameters():
                p.requires_grad_(False)
            view.eval()
            object.__setattr__(self, "_cached_initial_view", view)
        return view

    def load_initial_state_dict_snapshot(self, state_map: Mapping[str, torch.Tensor]) -> None:
        dev = self.transition_logits.device
        self._initial_state_tensors = {
            k: v.to(dev).detach().clone() for k, v in state_map.items() if k in self.state_dict()
        }
        if hasattr(self, "_cached_initial_view"):
            delattr(self, "_cached_initial_view")

    def restore_initial_parameters(self) -> None:
        if self._initial_state_tensors:
            dev = self.transition_logits.device
            restored = {k: v.to(dev).clone() for k, v in self._initial_state_tensors.items()}
            self.load_state_dict(restored, strict=True)

    def reset_intra_task_belief(self, device: torch.device | None = None) -> torch.Tensor:
        dev = device or self.transition_logits.device
        return F.softmax(self.initial_belief_logits.to(device=dev, dtype=torch.float32), dim=-1)

    @staticmethod
    def base_tail_pi0_logits(
        remaining_steps_ratio: float,
        step_budget_fraction: float,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[float, torch.Tensor]:
        """Pure state-resource finite-horizon Bellman-neutral base logits for H_{phi_0}^{pi_0}(Y, C_tail | Y=1, K).

        Depends strictly on remaining step horizon K (never on any prior root candidate action label!),
        guaranteeing identical baseline semantics across TailExample supervision, fork-head supervision,
        and deployment leaf evaluation, plus exact multi-step Bellman neutrality Q_{phi_0}^{(1)} == Q_{phi_0}^{(2)} == Q_{phi_0}^{(3)}.
        """
        sf = max(1e-4, float(step_budget_fraction))
        k_steps = max(0, int(round(max(0.0, float(remaining_steps_ratio)) / sf)))
        if k_steps <= 0:
            return -20.0, torch.zeros(TAIL_COST_BINS, device=device, dtype=dtype)

        p_live_0 = 0.96
        p_succ_0 = 0.0048
        c_0 = 1.0 / (1.0 + math.exp(5.0))

        centers = TAIL_COST_BIN_CENTERS
        bin_mass = [0.0] * TAIL_COST_BINS
        total_y = 0.0
        surv = 1.0
        for j in range(1, k_steps + 1):
            q_j = surv * p_succ_0
            total_y += q_j
            tau = min(centers[-1], max(centers[0], j * c_0))
            # Two-point barycentric projection onto adjacent bin centers
            placed = False
            for r in range(TAIL_COST_BINS - 1):
                if tau <= centers[r + 1] + 1e-12:
                    span = max(1e-12, centers[r + 1] - centers[r])
                    lam = min(1.0, max(0.0, (tau - centers[r]) / span))
                    bin_mass[r] += q_j * (1.0 - lam)
                    bin_mass[r + 1] += q_j * lam
                    placed = True
                    break
            if not placed:
                bin_mass[-1] += q_j
            surv *= p_live_0

        p_y_clamped = min(1.0 - 1e-6, max(1e-6, total_y))
        base_y_logit = math.log(p_y_clamped / (1.0 - p_y_clamped))
        norm_bins = [max(1e-8, m / max(1e-12, total_y)) for m in bin_mass]
        s_bins = sum(norm_bins)
        base_bin_logits = torch.tensor(
            [math.log(m / s_bins) for m in norm_bins], device=device, dtype=dtype
        )
        return float(base_y_logit), base_bin_logits

    def encode_tail_state_context(
        self,
        *,
        goal_text: str,
        public_fields: Mapping[str, str],
        remaining_budget_ratio: float = 1.0,
        spent_seconds_ratio: float | None = None,
        remaining_steps_ratio: float = 1.0,
        frozen_context_vec: Sequence[float] = (),
        frozen_tail_action_vec: Sequence[float] = (),
        typed_slots: tuple[int, int, int, int] | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Canonical tail-state encoder shared identically by TailExample training and Q leaf evaluation."""
        dev = device or self.transition_logits.device
        c_vec = self.encode_context_vector(
            goal_text=goal_text,
            public_fields=public_fields,
            action_text=TAIL_POLICY_VERSION,
            remaining_budget_ratio=remaining_budget_ratio,
            spent_seconds_ratio=spent_seconds_ratio,
            remaining_steps_ratio=remaining_steps_ratio,
            frozen_context_vec=frozen_context_vec,
            frozen_action_vec=frozen_tail_action_vec,
            typed_slots=typed_slots,
            device=dev,
        )
        return self.context_encoder(self._coerce_context(c_vec))

    @classmethod
    def deterministic_semantic_projection(cls, text: str, dim: int) -> tuple[float, ...]:
        """Fallback continuous bag-of-n-grams projection ONLY when no backbone model is resident."""
        vec = np.zeros(dim, dtype=np.float32)
        tokens = re.findall(r"[a-z0-9_]+", text.lower())
        if not tokens:
            return tuple(float(x) for x in vec)
        for tok in tokens:
            h = int.from_bytes(hashlib.blake2b(tok.encode("utf-8"), digest_size=4).digest(), "big")
            idx = h % dim
            sign = 1.0 if ((h >> 16) & 1) == 0 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 1e-6:
            vec = vec / norm
        return tuple(float(x) for x in vec)

    @classmethod
    def encode_context_vector(
        cls,
        *,
        goal_text: str,
        public_fields: Mapping[str, str],
        action_text: str,
        remaining_budget_ratio: float = 1.0,
        spent_seconds_ratio: float | None = None,
        remaining_steps_ratio: float = 1.0,
        frozen_context_vec: Sequence[float] = (),
        frozen_action_vec: Sequence[float] = (),
        role_bindings: Mapping[str, str] | None = None,
        typed_slots: tuple[int, int, int, int] | None = None,
        prior_action_logprob: float = 0.0,
        prior_success_logit: float = 0.0,
        program: ExecutableProgram | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Encode c = (frozen_context, frozen_action, typed_slots, resource_before) into a 64-dim vector.

        Uses canonicalized action text (`normalize_action_for_semantic_encoding`) and unified integer
        `typed_slots` so NLL training, online `observe` filtering, and multi-step AST lookahead share
        bit-identical feature semantics.
        """
        vec = np.zeros(CONTEXT_DIM, dtype=np.float32)
        norm_act_text = PublicBenchmarkBinder.normalize_action_for_semantic_encoding(action_text, role_bindings)
        slot_0, slot_1, slot_2, slot_3 = (
            typed_slots if typed_slots is not None else PublicBenchmarkBinder.encode_typed_field_slots(public_fields)
        )
        all_canon_status = {w for row in SLOT_VALUE_NAMES for w in row} | {
            "yes", "no", "high", "low", "closed", "open_empty", "open_with_items",
            "search_home", "search_results", "item_subpage", "item_detail", "pending",
            "ready", "raw", "unchanged", "inspected", "selected", "clean", "hot", "cool",
            "examined", "verified_write", "count_0", "count_1", "count_2", "count_3", "count_4",
        }
        entity_pairs = {
            k: v for k, v in sorted(public_fields.items())
            if not str(k).startswith("_") and str(v).lower() not in all_canon_status
        }

        # 0..31: Frozen history representation (invariant to raw vs canonical status spelling)
        hist_seq = (
            list(frozen_context_vec)[:FROZEN_HISTORY_DIM]
            if frozen_context_vec
            else list(cls.deterministic_semantic_projection(f"{goal_text}\n{_canonical_json(entity_pairs)}", FROZEN_HISTORY_DIM))
        )
        vec[: len(hist_seq)] = np.asarray(hist_seq, dtype=np.float32)

        # 32..47: Frozen action representation
        act_seq = (
            list(frozen_action_vec)[:FROZEN_ACTION_DIM]
            if frozen_action_vec
            else list(cls.deterministic_semantic_projection(norm_act_text, FROZEN_ACTION_DIM))
        )
        vec[FROZEN_HISTORY_DIM : FROZEN_HISTORY_DIM + len(act_seq)] = np.asarray(act_seq, dtype=np.float32)

        # 48..63: Pre-action resource state, public regime-idempotence indicators & typed field slots
        rem_sec = float(min(1.0, max(0.0, remaining_budget_ratio)))
        spent_sec = float(1.0 - rem_sec if spent_seconds_ratio is None else min(1.0, max(0.0, spent_seconds_ratio)))
        rem_steps = float(min(1.0, max(0.0, remaining_steps_ratio)))
        goal_tokens = set(re.findall(r"[a-z0-9]{2,}", goal_text.lower()))
        act_tokens = set(re.findall(r"[a-z0-9]{2,}", norm_act_text.lower()))
        entity_vals = " ".join(f"{k}:{v}" for k, v in sorted(entity_pairs.items())).lower()
        if role_bindings:
            entity_vals = entity_vals + " " + " ".join(str(v).lower() for v in role_bindings.values())
        field_tokens = set(re.findall(r"[a-z0-9_]{2,}", entity_vals))

        raw_act_l = str(action_text or "").strip().lower()
        norm_act_l = norm_act_text.strip().lower()
        last_act_l = str(public_fields.get("_last_action", "none")).strip().lower()
        tried_set = {
            x.strip().lower()
            for x in str(public_fields.get("_regime_tried", "")).split("|")
            if x.strip()
        }
        is_imm_repeat = 1.0 if (raw_act_l and (raw_act_l == last_act_l or norm_act_l == last_act_l)) else 0.0
        is_tried_regime = 1.0 if (raw_act_l and (raw_act_l in tried_set or norm_act_l in tried_set)) else 0.0

        base_idx = FROZEN_HISTORY_DIM + FROZEN_ACTION_DIM  # 48
        vec[base_idx + 0] = rem_sec
        vec[base_idx + 1] = spent_sec
        vec[base_idx + 2] = rem_steps
        vec[base_idx + 3] = len(goal_tokens & act_tokens) / max(1.0, float(len(goal_tokens)))
        vec[base_idx + 4] = is_imm_repeat
        vec[base_idx + 5] = is_tried_regime

        act_l = norm_act_text.strip().lower()
        vec[base_idx + 6] = 1.0 if act_l.startswith(("look", "inventory", "examine", "search[")) else 0.0
        vec[base_idx + 7] = 1.0 if act_l.startswith(("go to", "open", "click[")) else 0.0
        vec[base_idx + 8] = 1.0 if act_l.startswith(("take", "move", "put", "clean", "heat", "cool", "use")) else 0.0
        vec[base_idx + 9] = 1.0 if ("buy now" in act_l or "claim_done" in act_l or "submit" in act_l) else 0.0

        vec[base_idx + 10] = (slot_0 - 1.5) / 1.5
        vec[base_idx + 11] = (slot_1 - 1.5) / 1.5
        vec[base_idx + 12] = (slot_2 - 1.5) / 1.5
        vec[base_idx + 13] = (slot_3 - 1.5) / 1.5
        vec[base_idx + 14] = 1.0 if slot_0 == 3 else 0.0
        vec[base_idx + 15] = 1.0 if slot_3 == 3 else 0.0

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
        frozen_context_vec: Sequence[float] = (),
        frozen_action_vec: Sequence[float] = (),
        prior_action_logprob: float = 0.0,
        prior_success_logit: float = 0.0,
    ) -> torch.Tensor:
        act_str = action_text or (program.step_templates[0] if program.step_templates else "")
        return cls.encode_context_vector(
            goal_text=goal_text,
            public_fields=public_fields,
            action_text=act_str,
            remaining_budget_ratio=remaining_budget_ratio,
            frozen_context_vec=frozen_context_vec,
            frozen_action_vec=frozen_action_vec,
            device=device,
        )

    @staticmethod
    def compute_centered_prior_advantages(raw_logprobs: Mapping[str, float]) -> dict[str, float]:
        """Normalize full-sequence+EOS frozen action log-probs log pi_0(a | g, h) across admissible candidates
        and center relative to candidate-prior entropy so E_{a ~ pi_0}[alpha_0(a)] == 0 identically.
        """
        if not raw_logprobs:
            return {}
        keys = list(raw_logprobs.keys())
        vals = np.asarray([float(raw_logprobs[k]) for k in keys], dtype=np.float64)
        finite_mask = np.isfinite(vals)
        if not finite_mask.any():
            return {k: 0.0 for k in keys}
        max_v = float(vals[finite_mask].max())
        shifted = np.where(finite_mask, vals - max_v, -50.0)
        log_z = max_v + float(np.log(np.maximum(1e-18, np.exp(shifted).sum())))
        log_tilde = np.where(finite_mask, vals - log_z, -50.0)
        probs = np.exp(log_tilde)
        probs = probs / max(1e-18, float(probs.sum()))
        entropy_baseline = float((probs * log_tilde).sum())
        centered = np.clip(log_tilde - entropy_baseline, -8.0, 4.0)
        return {k: float(centered[i]) for i, k in enumerate(keys)}

    @staticmethod
    def apply_prior_advantage_to_q(
        q_model: torch.Tensor,
        p_y_model: torch.Tensor,
        prior_advantage: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Combine pure response-model continuation (Q_model, P_y_model) with centered full-EOS action prior
        alpha_0(a | g, h) in log-odds space (shared identically by fork_head_nll and evaluate_candidate_q).
        """
        adv = float(max(-8.0, min(4.0, prior_advantage)))
        p_clamped = p_y_model.clamp(min=1e-6, max=1.0 - 1e-6)
        if abs(adv) <= 1e-12:
            return q_model.clamp(min=0.0, max=1.0), p_clamped
        cond_time_factor = (q_model / p_clamped).clamp(min=0.0, max=1.0)
        p_post = torch.sigmoid(torch.logit(p_clamped) + PRIOR_ADVANTAGE_WEIGHT * adv).clamp(min=1e-6, max=1.0 - 1e-6)
        q_post = (p_post * cond_time_factor).clamp(min=0.0, max=1.0)
        return q_post, p_post

    def _coerce_context(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() == 2 and feat.shape[0] == self.ensemble_size and feat.shape[1] == self.context_dim:
            return feat
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
        """Compute normalized transition T_m(s'|s,c_m) and emission G_m(o|s',c_m) supporting both shared `(C,)`
        and strictly private per-member `(M, C)` contexts without cross-member averaging.
        """
        c_vec = self._coerce_context(context_feat)
        h_ctx = self.context_encoder(c_vec)  # (C,) or (M, C)
        if h_ctx.dim() == 2:
            t_w = self.transition_residual_proj.weight.view(
                self.ensemble_size, self.state_dim, self.state_dim, self.context_dim
            )
            t_b = self.transition_residual_proj.bias.view(
                self.ensemble_size, self.state_dim, self.state_dim
            )
            t_res = torch.einsum("mc,mijc->mij", h_ctx, t_w) + t_b
            e_w = self.emission_residual_proj.weight.view(
                self.ensemble_size, self.state_dim, self.outcome_bins, self.context_dim
            )
            e_b = self.emission_residual_proj.bias.view(
                self.ensemble_size, self.state_dim, self.outcome_bins
            )
            e_res = torch.einsum("mc,msoc->mso", h_ctx, e_w) + e_b
        else:
            t_res = self.transition_residual_proj(h_ctx).view(
                self.ensemble_size, self.state_dim, self.state_dim
            )
            e_res = self.emission_residual_proj(h_ctx).view(
                self.ensemble_size, self.state_dim, self.outcome_bins
            )
        T = F.softmax(self.transition_logits + t_res, dim=1)
        G = F.softmax(self.emission_logits + e_res, dim=-1)
        return T, G, h_ctx

    def step_and_predict(
        self,
        belief: torch.Tensor,
        context_feat: torch.Tensor,
        observed_outcome_bin: int | None = None,
        *,
        prior_outcome_logits: torch.Tensor | None = None,
        prior_termination_logits: torch.Tensor | None = None,
        prior_action_logprob: float = 0.0,
        prior_success_logit: float = 0.0,
        remaining_budget_ratio: float | torch.Tensor = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Single-pass per-member operator update, typed field-slot prediction, and pure physical transition heads."""
        T, G, h_ctx = self.operator_matrices(context_feat)
        if prior_outcome_logits is not None and prior_outcome_logits.numel() == self.outcome_bins:
            prior_l = prior_outcome_logits.to(device=G.device, dtype=G.dtype).view(1, 1, self.outcome_bins)
            G = F.softmax(torch.log(G.clamp(min=1e-8)) + prior_l, dim=-1)

        next_state_prior = torch.einsum("mij,mj->mi", T, belief)  # (M, S)
        joint_next = next_state_prior[:, :, None] * G  # (M, S, O)
        outcome_probs = joint_next.sum(dim=1).clamp(min=1e-8)  # (M, O)
        outcome_probs = outcome_probs / outcome_probs.sum(dim=-1, keepdim=True)

        if observed_outcome_bin is None:
            post_belief = next_state_prior
        else:
            o_idx = int(observed_outcome_bin) % self.outcome_bins
            unnorm_post = joint_next[:, :, o_idx]
            post_belief = unnorm_post / unnorm_post.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Typed field-slot distributions per member: (M, NUM_FIELD_SLOTS, NUM_SLOT_VALUES)
        if h_ctx.dim() == 2:
            f_w = self.field_slot_proj.weight.view(
                self.ensemble_size, self.state_dim, NUM_FIELD_SLOTS, NUM_SLOT_VALUES, self.context_dim
            )
            f_b = self.field_slot_proj.bias.view(
                self.ensemble_size, self.state_dim, NUM_FIELD_SLOTS, NUM_SLOT_VALUES
            )
            slot_logits_s = torch.einsum("mc,msjkc->msjk", h_ctx, f_w) + f_b
            ctx_expanded = h_ctx
        else:
            slot_logits_s = self.field_slot_proj(h_ctx).view(
                self.ensemble_size, self.state_dim, NUM_FIELD_SLOTS, NUM_SLOT_VALUES
            )
            ctx_expanded = h_ctx.unsqueeze(0).expand(self.ensemble_size, -1)  # (M, C)
        slot_probs_s = F.softmax(slot_logits_s, dim=-1)
        field_slot_probs = torch.einsum("ms,mskv->mkv", next_state_prior, slot_probs_s).clamp(min=1e-8)
        field_slot_probs = field_slot_probs / field_slot_probs.sum(dim=-1, keepdim=True)

        # Strictly private per-member inputs: (M, S + C)
        member_in = torch.cat([next_state_prior, ctx_expanded], dim=-1)  # (M, S + C)

        term_res = torch.einsum("md,mdc->mc", member_in, self.member_termination_weight) + self.member_termination_bias
        if isinstance(remaining_budget_ratio, torch.Tensor):
            rem_vec = remaining_budget_ratio.to(device=term_res.device, dtype=term_res.dtype).view(self.ensemble_size)
        else:
            rem_vec = torch.full(
                (self.ensemble_size,), float(remaining_budget_ratio), device=term_res.device, dtype=term_res.dtype
            )
        if prior_termination_logits is not None and prior_termination_logits.numel() == len(TERMINATION_CATEGORIES):
            base_term = prior_termination_logits.to(device=term_res.device, dtype=term_res.dtype).view(1, -1).expand(self.ensemble_size, -1)
        else:
            # Pure physical coupled 3-way step outcome law: (0=live, 1=native_terminal_success, 2=terminal_failure)
            l_live = math.log(0.96)
            l_succ = math.log(0.0048)
            l_fail = torch.where(
                rem_vec <= 0.05,
                torch.full_like(rem_vec, 1.0),
                torch.full_like(rem_vec, math.log(0.0352)),
            )
            base_term = torch.stack(
                [torch.full_like(rem_vec, l_live), torch.full_like(rem_vec, l_succ), l_fail],
                dim=-1,
            )
        termination_probs = F.softmax(base_term + term_res, dim=-1)  # (M, 3)

        cost_res = (member_in * self.member_step_cost_weight).sum(dim=-1) + self.member_step_cost_bias  # (M,)
        cost_ratio = torch.sigmoid(-5.0 + cost_res)  # (M,) calibrated initial wall-clock step cost fraction (~0.0067)

        # Conditional success given termination = P(term_succ) / (P(term_succ) + P(term_fail))
        term_mass = (termination_probs[:, 1] + termination_probs[:, 2]).clamp(min=1e-8)
        y_prob = (termination_probs[:, 1] / term_mass).clamp(min=1e-6, max=1.0 - 1e-6)

        return {
            "h_ctx": h_ctx,
            "outcome_probs": outcome_probs,          # (M, O)
            "field_slot_probs": field_slot_probs,    # (M, 4, 4)
            "slot_probs_per_state": slot_probs_s,    # (M, S, 4, 4)
            "next_state_prior": next_state_prior,    # (M, S)
            "post_belief": post_belief,              # (M, S)
            "termination_probs": termination_probs,  # (M, 3)
            "cost_ratio": cost_ratio,                # (M,)
            "terminal_success_prob": y_prob,         # (M,)
        }

    @staticmethod
    def _one_hot_field_slots(
        public_fields: Mapping[str, str],
        ensemble_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        slots = PublicBenchmarkBinder.encode_typed_field_slots(public_fields)
        oh = torch.zeros((1, NUM_FIELD_SLOTS, NUM_SLOT_VALUES), device=device, dtype=dtype)
        for s_i, v_i in enumerate(slots):
            oh[0, s_i, int(v_i) % NUM_SLOT_VALUES] = 1.0
        return oh.expand(ensemble_size, -1, -1).clone()

    def predict_tail_pi0(
        self,
        belief: torch.Tensor,
        h_ctx: torch.Tensor,
        spent_seconds_ratio: torch.Tensor,
        *,
        field_slot_probs: torch.Tensor | None = None,
        public_fields: Mapping[str, str] | None = None,
        remaining_steps_ratio: float = 1.0,
        step_budget_fraction: float = 0.05,
    ) -> dict[str, torch.Tensor]:
        """Evaluate pure state-conditioned fixed-tail law H_{phi,m}^{pi_0}(Y, C_tail | Y=1, b_m, z_one_hot, c_tail).

        Identical baseline across TailExample direct supervision, fork-head marginal supervision, and
        deployment leaf evaluation (never conditioned on root candidate action labels).
        """
        base_y_logit, base_bin_logits = self.base_tail_pi0_logits(
            remaining_steps_ratio=remaining_steps_ratio,
            step_budget_fraction=step_budget_fraction,
            device=belief.device,
            dtype=belief.dtype,
        )
        centers = torch.tensor(TAIL_COST_BIN_CENTERS, device=belief.device, dtype=belief.dtype)

        if belief.dim() == 3:
            # Batched joint-assignment evaluation: belief=(M, G, S), h_ctx=(M, G, C), field_slot_probs=(M, G, 16)
            m_sz, g_sz, _ = belief.shape
            if field_slot_probs is None:
                raise ValueError("Batched predict_tail_pi0 requires explicit one-hot field_slot_probs (M, G, 16).")
            slot_feat = field_slot_probs.to(device=belief.device, dtype=belief.dtype).reshape(
                m_sz, g_sz, NUM_FIELD_SLOTS * NUM_SLOT_VALUES
            )
            member_in = torch.cat([belief, h_ctx, slot_feat], dim=-1)  # (M, G, S + C + 16)
            y_res = (
                (member_in * self.tail_pi0_success_weight.unsqueeze(1)).sum(dim=-1)
                + self.tail_pi0_success_bias.unsqueeze(1)
            )
            y_prob = torch.sigmoid(base_y_logit + y_res)  # (M, G)
            bin_res = (
                torch.einsum("mgd,mdk->mgk", member_in, self.tail_pi0_cost_bins_weight)
                + self.tail_pi0_cost_bins_bias.unsqueeze(1)
            )
            bin_probs = F.softmax(base_bin_logits.view(1, 1, TAIL_COST_BINS) + bin_res, dim=-1)  # (M, G, K)
            spent = spent_seconds_ratio.view(m_sz, 1, 1).clamp(min=0.0, max=1.0)
            rem_after_tail = (1.0 - spent - centers.view(1, 1, TAIL_COST_BINS)).clamp(min=0.0, max=1.0)
            cond_utility_given_y1 = (bin_probs * rem_after_tail).sum(dim=-1)  # (M, G)
            expected_tail_utility = y_prob * cond_utility_given_y1  # (M, G)
            expected_cond_tail_cost = (bin_probs * centers.view(1, 1, TAIL_COST_BINS)).sum(dim=-1)  # (M, G)
            return {
                "tail_y_prob": y_prob,
                "tail_cost_bin_probs": bin_probs,
                "expected_cond_tail_cost": expected_cond_tail_cost,
                "expected_tail_utility": expected_tail_utility,
            }

        ctx_expanded = h_ctx.unsqueeze(0).expand(self.ensemble_size, -1) if h_ctx.dim() == 1 else h_ctx
        if field_slot_probs is None:
            slot_feat = self._one_hot_field_slots(
                public_fields or {}, self.ensemble_size, belief.device, belief.dtype
            ).reshape(self.ensemble_size, -1)
        else:
            slot_feat = field_slot_probs.to(device=belief.device, dtype=belief.dtype).reshape(
                self.ensemble_size, NUM_FIELD_SLOTS * NUM_SLOT_VALUES
            )
        member_in = torch.cat([belief, ctx_expanded, slot_feat], dim=-1)  # (M, S + C + 16)

        y_logit = (member_in * self.tail_pi0_success_weight).sum(dim=-1) + self.tail_pi0_success_bias
        y_prob = torch.sigmoid(base_y_logit + y_logit)  # (M,)

        bin_logits = (
            base_bin_logits.view(1, TAIL_COST_BINS)
            + torch.einsum("md,mdk->mk", member_in, self.tail_pi0_cost_bins_weight)
            + self.tail_pi0_cost_bins_bias
        )
        bin_probs = F.softmax(bin_logits, dim=-1)  # (M, TAIL_COST_BINS)

        spent = spent_seconds_ratio.view(self.ensemble_size, 1).clamp(min=0.0, max=1.0)
        rem_after_tail = (1.0 - spent - centers.view(1, TAIL_COST_BINS)).clamp(min=0.0, max=1.0)  # (M, K)
        cond_utility_given_y1 = (bin_probs * rem_after_tail).sum(dim=-1)  # (M,)
        expected_tail_utility = y_prob * cond_utility_given_y1  # (M,)
        expected_cond_tail_cost = (bin_probs * centers.view(1, TAIL_COST_BINS)).sum(dim=-1)  # (M,)

        return {
            "tail_y_prob": y_prob,
            "tail_cost_bin_probs": bin_probs,
            "expected_cond_tail_cost": expected_cond_tail_cost,
            "expected_tail_utility": expected_tail_utility,
        }

    def observable_operator_step(
        self,
        belief: torch.Tensor,
        context_feat: torch.Tensor,
        observed_outcome_bin: int | None = None,
        *,
        prior_outcome_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        res = self.step_and_predict(
            belief,
            context_feat,
            observed_outcome_bin=observed_outcome_bin,
            prior_outcome_logits=prior_outcome_logits,
        )
        return res["outcome_probs"], res["post_belief"]

    def observe(
        self,
        previous_state: torch.Tensor,
        *,
        goal_text: str,
        pre_public_fields: Mapping[str, str],
        actual_action: str,
        post_public_fields: Mapping[str, str],
        remaining_budget_ratio: float = 1.0,
        spent_seconds_ratio: float | None = None,
        remaining_steps_ratio: float = 1.0,
        role_bindings: Mapping[str, str] | None = None,
        frozen_context_vec: Sequence[float] = (),
        frozen_action_vec: Sequence[float] = (),
        had_error: bool = False,
        terminal_done: bool = False,
        terminal_success: bool = False,
        decision_ctx: DecisionContext | None = None,
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        """Unified intra-task state update `next_state = observe(previous_state, actual_action, actual_public_response)`."""
        device = self.transition_logits.device
        if decision_ctx is not None:
            goal_text = decision_ctx.goal_text
            pre_public_fields = decision_ctx.public_fields
            remaining_budget_ratio = decision_ctx.remaining_budget_ratio
            spent_seconds_ratio = decision_ctx.spent_seconds_ratio
            remaining_steps_ratio = decision_ctx.remaining_steps_ratio
            role_bindings = decision_ctx.role_bindings
            frozen_context_vec = decision_ctx.frozen_context_vec
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
            spent_seconds_ratio=spent_seconds_ratio,
            remaining_steps_ratio=remaining_steps_ratio,
            role_bindings=role_bindings,
            frozen_context_vec=frozen_context_vec,
            frozen_action_vec=frozen_action_vec,
            device=device,
        )
        res = self.step_and_predict(
            previous_state,
            ctx,
            observed_outcome_bin=outcome_bin,
            remaining_budget_ratio=remaining_budget_ratio,
        )
        return res["post_belief"], outcome_bin, res["outcome_probs"]

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
        return self.step_and_predict(
            belief,
            context_feat,
            observed_outcome_bin=None,
            prior_outcome_logits=prior_outcome_logits,
            prior_termination_logits=prior_termination_logits,
            prior_success_logit=prior_success_logit,
            remaining_budget_ratio=remaining_budget_ratio,
        )

    @torch.no_grad()
    def evaluate_candidate_q(
        self,
        *,
        goal_text: str = "",
        public_fields: Mapping[str, str] | None = None,
        belief: torch.Tensor,
        candidate_action: str,
        program: ExecutableProgram | None = None,
        initial_frame: ExecutionFrame | None = None,
        decision_ctx: DecisionContext | None = None,
        benchmark: str = "alfworld",
        action_schema: Mapping[str, Any] | None = None,
        role_bindings: Mapping[str, str] | None = None,
        remaining_budget_ratio: float = 1.0,
        remaining_steps_ratio: float = 1.0,
        step_budget_fraction: float = 0.05,
        frozen_context_vec: Sequence[float] = (),
        frozen_action_vec: Sequence[float] = (),
        frozen_tail_action_vec: Sequence[float] = (),
        action_embedder: Any | None = None,
        prior_action_logprob: float = 0.0,
        prior_success_logit: float = 0.0,
        member_index: int | None = None,
        controller_mode: str = "expected_utility",
        use_initial_params: bool = False,
    ) -> dict[str, Any]:
        """Compute unified long-horizon value Q_m^rho(g, h, beta) = E_{K_{phi,m}, rho -> pi_0}[U_B(Y, C_seconds)].

        First-principles semantics:
          1. Primary `expected_utility` controller advances alive mass strictly via normalized `live`
             probability `termination_probs[:, 0]` (zero artificial no-change contraction) and selects
             by ensemble mean `q_mean` (zero ad-hoc LCB penalty).
          2. Leaves evaluate the exact state-conditioned joint field-slot expectation
             E_Z[H_{phi,m}^{pi_0}(b_m(. | Z), one_hot(Z), spent_m)] in one vectorized batch pass,
             preserving per-member resource states and avoiding Jensen's inequality bias.
          3. Multi-step AST lookahead executes a cloned `ExecutionFrame` (`Call`, `Read`, `Case`, `Seq`, `Stop`),
             binding symbolic return handles (`record_symbolic_call_return`), instantiating subsequent `Call`s,
             and updating `Case` false branches with the exact 3-value complement conditional distribution.
        """
        if decision_ctx is not None:
            benchmark = decision_ctx.benchmark
            goal_text = decision_ctx.goal_text
            public_fields = decision_ctx.public_fields
            role_bindings = decision_ctx.role_bindings
            action_schema = decision_ctx.action_schema
            remaining_budget_ratio = decision_ctx.remaining_budget_ratio
            remaining_steps_ratio = decision_ctx.remaining_steps_ratio
            step_budget_fraction = decision_ctx.step_budget_fraction
            frozen_context_vec = decision_ctx.frozen_context_vec
            frozen_tail_action_vec = decision_ctx.frozen_tail_action_vec
            member_index = decision_ctx.member_index if member_index is None else member_index
            controller_mode = decision_ctx.controller_mode

        if use_initial_params and self._initial_state_tensors:
            return self.active_model_view(use_initial_params=True).evaluate_candidate_q(
                goal_text=goal_text,
                public_fields=public_fields,
                belief=belief,
                candidate_action=candidate_action,
                program=program,
                initial_frame=initial_frame,
                benchmark=benchmark,
                action_schema=action_schema,
                role_bindings=role_bindings,
                remaining_budget_ratio=remaining_budget_ratio,
                remaining_steps_ratio=remaining_steps_ratio,
                step_budget_fraction=step_budget_fraction,
                frozen_context_vec=frozen_context_vec,
                frozen_action_vec=frozen_action_vec,
                frozen_tail_action_vec=frozen_tail_action_vec,
                action_embedder=action_embedder,
                prior_action_logprob=prior_action_logprob,
                prior_success_logit=prior_success_logit,
                member_index=member_index,
                controller_mode=controller_mode,
                use_initial_params=False,
            )

        device = self.transition_logits.device
        init_belief = belief.clone().to(device)
        rem_ratio = float(max(0.0, min(1.0, remaining_budget_ratio)))
        init_rem_steps = float(max(0.0, min(1.0, remaining_steps_ratio)))
        step_frac = float(max(1e-4, min(1.0, step_budget_fraction)))
        init_spent = torch.full(
            (self.ensemble_size,), 1.0 - rem_ratio, device=device, dtype=torch.float32
        )
        init_alive = torch.ones(self.ensemble_size, device=device, dtype=torch.float32)
        base_fields = dict(public_fields or {})
        base_bindings = dict(role_bindings or {})
        base_schema = action_schema or {}
        eff_benchmark = (program.benchmark if program is not None else benchmark) or "alfworld"

        root_slots = PublicBenchmarkBinder.encode_typed_field_slots(base_fields)
        init_slot_probs_s = torch.zeros(
            (self.ensemble_size, self.state_dim, NUM_FIELD_SLOTS, NUM_SLOT_VALUES),
            device=device,
            dtype=torch.float32,
        )
        for s_i, v_i in enumerate(root_slots):
            init_slot_probs_s[:, :, s_i, int(v_i) % NUM_SLOT_VALUES] = 1.0

        grid_idx = torch.tensor(_JOINT_SLOT_GRID_NP, device=device, dtype=torch.long)  # (256, 4)
        base_tail_cvec = self.encode_context_vector(
            goal_text=goal_text,
            public_fields=base_fields,
            action_text=TAIL_POLICY_VERSION,
            remaining_budget_ratio=rem_ratio,
            spent_seconds_ratio=1.0 - rem_ratio,
            remaining_steps_ratio=init_rem_steps,
            frozen_context_vec=frozen_context_vec,
            frozen_action_vec=frozen_tail_action_vec,
            typed_slots=root_slots,
            device=device,
        )

        def _leaf_tail_value(
            cur_b: torch.Tensor,
            cur_spent: torch.Tensor,
            cur_alive: torch.Tensor,
            cur_rem_steps: float,
            cur_slot_probs_s: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if cur_rem_steps <= 1e-7:
                zeros = torch.zeros(self.ensemble_size, device=device, dtype=torch.float32)
                return zeros, cur_spent.clamp(max=1.0), zeros

            if controller_mode == "legacy_lcb_v3":
                rem_mean = float((1.0 - cur_spent).clamp(min=0.0, max=1.0).mean().item())
                spent_mean = float(cur_spent.clamp(min=0.0, max=1.0).mean().item())
                tail_h_ctx = self.encode_tail_state_context(
                    goal_text=goal_text,
                    public_fields=base_fields,
                    remaining_budget_ratio=rem_mean,
                    spent_seconds_ratio=spent_mean,
                    remaining_steps_ratio=cur_rem_steps,
                    frozen_context_vec=frozen_context_vec,
                    frozen_tail_action_vec=frozen_tail_action_vec,
                    device=device,
                )
                leaf_field_probs = torch.einsum("ms,mskv->mkv", cur_b, cur_slot_probs_s).clamp(min=1e-8)
                leaf_field_probs = leaf_field_probs / leaf_field_probs.sum(dim=-1, keepdim=True)
                tail_out = self.predict_tail_pi0(
                    cur_b,
                    tail_h_ctx,
                    cur_spent,
                    field_slot_probs=leaf_field_probs,
                    public_fields=base_fields,
                    remaining_steps_ratio=cur_rem_steps,
                    step_budget_fraction=step_frac,
                )
                return (
                    cur_alive * tail_out["expected_tail_utility"],
                    (cur_spent + cur_alive * tail_out["expected_cond_tail_cost"]).clamp(max=1.0),
                    cur_alive * tail_out["tail_y_prob"],
                )

            # Exact state-conditioned joint field-slot distribution P_m(z) = sum_s b_m(s) prod_j G_{m,s,j}(z_j)
            p0 = cur_slot_probs_s[:, :, 0, grid_idx[:, 0]]
            p1 = cur_slot_probs_s[:, :, 1, grid_idx[:, 1]]
            p2 = cur_slot_probs_s[:, :, 2, grid_idx[:, 2]]
            p3 = cur_slot_probs_s[:, :, 3, grid_idx[:, 3]]
            lik_z = p0 * p1 * p2 * p3  # (M, S, 256)
            unnorm_post = cur_b.unsqueeze(-1) * lik_z  # (M, S, 256)
            p_z = unnorm_post.sum(dim=1)  # (M, 256)
            p_z = p_z / p_z.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            active_mask = p_z.max(dim=0).values > 1e-6
            active_cols = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
            if active_cols.numel() == 0:
                active_cols = p_z.mean(dim=0).argmax().view(1)

            g_act = int(active_cols.numel())
            p_z_act = p_z[:, active_cols]  # (M, G_act)
            p_z_act = p_z_act / p_z_act.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            post_b_act = unnorm_post[:, :, active_cols].permute(0, 2, 1)  # (M, G_act, S)
            post_b_act = post_b_act / post_b_act.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            act_slots = grid_idx[active_cols]  # (G_act, 4)

            c_batch = base_tail_cvec.view(1, 1, self.context_dim).expand(
                self.ensemble_size, g_act, self.context_dim
            ).clone()
            spent_clamped = cur_spent.clamp(min=0.0, max=1.0)
            rem_clamped = (1.0 - spent_clamped).clamp(min=0.0, max=1.0)
            b_idx = FROZEN_HISTORY_DIM + FROZEN_ACTION_DIM
            c_batch[:, :, b_idx + 0] = rem_clamped.unsqueeze(1)
            c_batch[:, :, b_idx + 1] = spent_clamped.unsqueeze(1)
            c_batch[:, :, b_idx + 2] = float(min(1.0, max(0.0, cur_rem_steps)))
            c_batch[:, :, b_idx + 10 : b_idx + 14] = (
                (act_slots.to(dtype=c_batch.dtype) - 1.5) / 1.5
            ).unsqueeze(0)
            c_batch[:, :, b_idx + 14] = (act_slots[:, 0] == 3).to(dtype=c_batch.dtype).unsqueeze(0)
            c_batch[:, :, b_idx + 15] = (act_slots[:, 3] == 3).to(dtype=c_batch.dtype).unsqueeze(0)

            h_ctx_batch = self.context_encoder(c_batch.reshape(-1, self.context_dim)).view(
                self.ensemble_size, g_act, self.context_dim
            )
            slot_oh_act = (
                F.one_hot(act_slots, num_classes=NUM_SLOT_VALUES)
                .to(dtype=cur_b.dtype)
                .reshape(1, g_act, NUM_FIELD_SLOTS * NUM_SLOT_VALUES)
                .expand(self.ensemble_size, g_act, -1)
            )
            tail_out = self.predict_tail_pi0(
                post_b_act,
                h_ctx_batch,
                spent_clamped,
                field_slot_probs=slot_oh_act,
                remaining_steps_ratio=cur_rem_steps,
                step_budget_fraction=step_frac,
            )
            exp_tail_u = (p_z_act * tail_out["expected_tail_utility"]).sum(dim=-1)
            exp_tail_c = (p_z_act * tail_out["expected_cond_tail_cost"]).sum(dim=-1)
            exp_tail_y = (p_z_act * tail_out["tail_y_prob"]).sum(dim=-1)
            return (
                cur_alive * exp_tail_u,
                (spent_clamped + cur_alive * exp_tail_c).clamp(max=1.0),
                cur_alive * exp_tail_y,
            )

        def _eval_call_step(
            cur_b: torch.Tensor,
            cur_spent: torch.Tensor,
            cur_alive: torch.Tensor,
            cur_rem_steps: float,
            cur_slot_probs_s: torch.Tensor,
            cur_fields: Mapping[str, str],
            cur_bindings: Mapping[str, str],
            act_str: str,
            is_first: bool,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int, int, int]]:
            spent_clamped = cur_spent.clamp(min=0.0, max=1.0)
            rem_vec = (1.0 - spent_clamped).clamp(min=0.0, max=1.0)
            norm_act = PublicBenchmarkBinder.normalize_action_for_semantic_encoding(act_str, cur_bindings)
            if is_first and frozen_action_vec:
                act_vec: Sequence[float] = frozen_action_vec
            elif action_embedder is not None:
                act_vec = action_embedder(norm_act)
            else:
                act_vec = ()
            # Per-member private context `(M, 64)`: each member retains its own spent/remaining budget and
            # its own state-marginalized slot expectations without cross-member `mean(dim=0)` collapse!
            marg_m = torch.einsum("ms,mskv->mkv", cur_b, cur_slot_probs_s).clamp(min=1e-8)  # (M, 4, 4)
            marg_m = marg_m / marg_m.sum(dim=-1, keepdim=True)
            modal_slots = tuple(int(x) for x in marg_m.mean(dim=0).argmax(dim=-1).tolist())
            ctx_base = self.encode_context_vector(
                goal_text=goal_text,
                public_fields=cur_fields,
                action_text=act_str,
                remaining_budget_ratio=float(rem_vec.mean().item()),
                spent_seconds_ratio=float(spent_clamped.mean().item()),
                remaining_steps_ratio=cur_rem_steps,
                role_bindings=cur_bindings,
                typed_slots=modal_slots,  # type: ignore[arg-type]
                frozen_context_vec=frozen_context_vec,
                frozen_action_vec=act_vec,
                device=device,
            )
            ctx_m = ctx_base.unsqueeze(0).expand(self.ensemble_size, -1).clone()
            b_idx = FROZEN_HISTORY_DIM + FROZEN_ACTION_DIM
            val_grid = torch.arange(NUM_SLOT_VALUES, device=device, dtype=ctx_m.dtype).view(1, 1, NUM_SLOT_VALUES)
            exp_slot_m = (marg_m * val_grid).sum(dim=-1)  # (M, 4)
            ctx_m[:, b_idx + 0] = rem_vec
            ctx_m[:, b_idx + 1] = spent_clamped
            ctx_m[:, b_idx + 2] = float(min(1.0, max(0.0, cur_rem_steps)))
            ctx_m[:, b_idx + 10 : b_idx + 14] = (exp_slot_m - 1.5) / 1.5
            ctx_m[:, b_idx + 14] = marg_m[:, 0, 3]
            ctx_m[:, b_idx + 15] = marg_m[:, 3, 3]

            preds = self.step_and_predict(
                cur_b,
                ctx_m,
                observed_outcome_bin=None,
                remaining_budget_ratio=rem_vec,
            )
            next_spent = (spent_clamped + preds["cost_ratio"]).clamp(min=0.0, max=1.0)
            next_rem_steps = max(0.0, cur_rem_steps - step_frac)
            p_live = preds["termination_probs"][:, 0]
            p_term_succ = preds["termination_probs"][:, 1]
            rem_after = (1.0 - next_spent).clamp(min=0.0, max=1.0)
            step_y = cur_alive * p_term_succ
            step_u = step_y * rem_after
            horizon_alive = 1.0 if next_rem_steps > 1e-7 else 0.0
            if controller_mode == "legacy_lcb_v3":
                p_no_change = preds["outcome_probs"][:, :4].sum(dim=-1).clamp(min=0.0, max=1.0)
                contraction = (1.0 - p_no_change * max(step_frac, 0.05)).clamp(min=0.0, max=1.0)
                next_alive = cur_alive * p_live * contraction * horizon_alive
            else:
                next_alive = cur_alive * p_live * horizon_alive
            next_marg = torch.einsum("ms,mskv->mkv", preds["next_state_prior"], preds["slot_probs_per_state"]).mean(dim=0)
            next_modal_slots = tuple(int(x) for x in next_marg.argmax(dim=-1).tolist())
            return (
                preds["next_state_prior"],
                next_spent,
                next_alive,
                next_rem_steps,
                step_u,
                step_y,
                preds["slot_probs_per_state"],
                next_modal_slots,  # type: ignore[return-value]
            )

        def _eval_ast_frame(
            sim_frame: ExecutionFrame,
            cur_b: torch.Tensor,
            cur_spent: torch.Tensor,
            cur_alive: torch.Tensor,
            cur_rem_steps: float,
            cur_slot_probs_s: torch.Tensor,
            calls_used: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if (
                sim_frame.completed
                or sim_frame.binding_failed
                or not sim_frame.node_stack
                or calls_used >= DEFAULT_MAX_PROBE_STEPS + 1
                or cur_rem_steps <= 1e-7
            ):
                return _leaf_tail_value(cur_b, cur_spent, cur_alive, cur_rem_steps, cur_slot_probs_s)

            head = sim_frame.node_stack[0]
            rest = sim_frame.node_stack[1:]
            if head.op == "Stop":
                return _leaf_tail_value(cur_b, cur_spent, cur_alive, cur_rem_steps, cur_slot_probs_s)

            if head.op == "Seq":
                next_f = sim_frame.clone()
                next_f.node_stack = list(head.children) + list(rest)
                return _eval_ast_frame(next_f, cur_b, cur_spent, cur_alive, cur_rem_steps, cur_slot_probs_s, calls_used)

            if head.op == "Read":
                next_f = sim_frame.clone()
                next_f.node_stack = list(rest)
                if head.variable and head.variable in next_f.local_env:
                    val = next_f.local_env[head.variable].get(head.public_field_path)
                    if val is not None:
                        next_f.observed_readouts[head.public_field_path] = val
                return _eval_ast_frame(next_f, cur_b, cur_spent, cur_alive, cur_rem_steps, cur_slot_probs_s, calls_used)

            if head.op == "Call":
                next_f = sim_frame.clone()
                next_f.node_stack = list(rest)
                if calls_used == 0:
                    act_str = candidate_action
                else:
                    inst_act = PublicBenchmarkBinder.instantiate_template_action(
                        eff_benchmark,
                        head.tool_or_action or candidate_action,
                        base_schema,
                        goal_text,
                        next_f.bound_variables,
                        next_f.local_env,
                        head.typed_arguments,
                        allow_symbolic=True,
                    )
                    act_str = inst_act or head.tool_or_action or candidate_action
                next_f.program_counter += 1
                next_f.pending_return_var = head.bind_return_to or f"ret_{next_f.program_counter - 1}"
                next_f.executed_actions.append(act_str)

                nb, n_spent, n_alive, n_rem_steps, step_u, step_y, n_slots_s, next_modal_slots = _eval_call_step(
                    cur_b,
                    cur_spent,
                    cur_alive,
                    cur_rem_steps,
                    cur_slot_probs_s,
                    next_f.observed_readouts or base_fields,
                    next_f.bound_variables,
                    act_str,
                    is_first=(calls_used == 0),
                )
                next_f.record_symbolic_call_return(act_str, calls_used, next_modal_slots)
                next_f.observed_readouts["_last_action"] = act_str.strip().lower()
                prev_tried = [
                    x for x in str(next_f.observed_readouts.get("_regime_tried", "")).split("|") if x.strip()
                ]
                if act_str.strip().lower() not in prev_tried:
                    prev_tried.append(act_str.strip().lower())
                next_f.observed_readouts["_regime_tried"] = "|".join(prev_tried[-16:])
                rem_u, end_cost, tail_y = _eval_ast_frame(
                    next_f, nb, n_spent, n_alive, n_rem_steps, n_slots_s, calls_used + 1
                )
                return step_u + rem_u, end_cost, step_y + tail_y

            if head.op == "Case":
                slot_idx, val_idx = PublicBenchmarkBinder.condition_slot_and_value(
                    head.condition_field, head.condition_equals
                )
                lik_true = cur_slot_probs_s[:, :, slot_idx, val_idx].clamp(min=0.0, max=1.0)
                unnorm_b_true = cur_b * lik_true.clamp(min=1e-8)
                p_true = (cur_b * lik_true).sum(dim=-1).clamp(min=0.0, max=1.0)
                b_true = unnorm_b_true / unnorm_b_true.sum(dim=-1, keepdim=True).clamp(min=1e-8)

                slots_s_true = cur_slot_probs_s.clone()
                slots_s_true[:, :, slot_idx, :] = 0.0
                slots_s_true[:, :, slot_idx, val_idx] = 1.0

                # Exact 3-value complement distribution given Z_{slot_idx} != val_idx
                slots_s_false_raw = cur_slot_probs_s.clone()
                slots_s_false_raw[:, :, slot_idx, val_idx] = 0.0
                lik_false = slots_s_false_raw[:, :, slot_idx, :].sum(dim=-1).clamp(min=0.0, max=1.0)
                unnorm_b_false = cur_b * lik_false.clamp(min=1e-8)
                p_false = (1.0 - p_true).clamp(min=0.0, max=1.0)
                b_false = unnorm_b_false / unnorm_b_false.sum(dim=-1, keepdim=True).clamp(min=1e-8)

                comp_mask = torch.ones(NUM_SLOT_VALUES, device=device, dtype=cur_b.dtype)
                comp_mask[val_idx] = 0.0
                comp_denom = slots_s_false_raw[:, :, slot_idx, :].sum(dim=-1, keepdim=True)
                slots_s_false = slots_s_false_raw.clone()
                slots_s_false[:, :, slot_idx, :] = torch.where(
                    comp_denom > 1e-8,
                    slots_s_false_raw[:, :, slot_idx, :] / comp_denom.clamp(min=1e-8),
                    comp_mask.view(1, 1, NUM_SLOT_VALUES) / float(NUM_SLOT_VALUES - 1),
                )

                true_child = [head.children[0]] if len(head.children) >= 1 else [ASTNode(op="Stop")]
                false_child = [head.children[1]] if len(head.children) >= 2 else [ASTNode(op="Stop")]

                frame_true = sim_frame.clone()
                frame_true.observed_readouts[head.condition_field] = SLOT_VALUE_NAMES[slot_idx][val_idx]
                frame_true.node_stack = true_child + list(rest)

                comp_marg = torch.einsum("ms,msv->mv", b_false, slots_s_false[:, :, slot_idx, :]).mean(dim=0)
                best_comp_v = int(comp_marg.argmax().item())
                frame_false = sim_frame.clone()
                frame_false.observed_readouts[head.condition_field] = SLOT_VALUE_NAMES[slot_idx][best_comp_v]
                frame_false.node_stack = false_child + list(rest)

                u_t, c_t, y_t = _eval_ast_frame(
                    frame_true, b_true, cur_spent, cur_alive * p_true, cur_rem_steps, slots_s_true, calls_used
                )
                u_f, c_f, y_f = _eval_ast_frame(
                    frame_false, b_false, cur_spent, cur_alive * p_false, cur_rem_steps, slots_s_false, calls_used
                )
                return u_t + u_f, p_true * c_t + p_false * c_f, y_t + y_f

            next_f = sim_frame.clone()
            next_f.node_stack = list(rest)
            return _eval_ast_frame(next_f, cur_b, cur_spent, cur_alive, cur_rem_steps, cur_slot_probs_s, calls_used)

        if initial_frame is not None:
            # `initial_frame` is the post-Call-0 frame produced by `step_ast_frame`: execute Call-0 (`candidate_action`)
            # then continue `initial_frame` after binding its symbolic return handle!
            nb0, n_spent0, n_alive0, n_rem0, step_u0, step_y0, n_slots0, modal_slots0 = _eval_call_step(
                init_belief,
                init_spent,
                init_alive,
                init_rem_steps,
                init_slot_probs_s,
                initial_frame.observed_readouts or base_fields,
                initial_frame.bound_variables or base_bindings,
                candidate_action,
                is_first=True,
            )
            cont_frame = initial_frame.clone()
            cont_frame.record_symbolic_call_return(candidate_action, 0, modal_slots0)
            cont_frame.observed_readouts["_last_action"] = candidate_action.strip().lower()
            prev_tried0 = [
                x for x in str(cont_frame.observed_readouts.get("_regime_tried", "")).split("|") if x.strip()
            ]
            if candidate_action.strip().lower() not in prev_tried0:
                prev_tried0.append(candidate_action.strip().lower())
            cont_frame.observed_readouts["_regime_tried"] = "|".join(prev_tried0[-16:])
            rem_u0, end_spent, tail_y_rem = _eval_ast_frame(
                cont_frame, nb0, n_spent0, n_alive0, n_rem0, n_slots0, 1
            )
            q_model_members = step_u0 + rem_u0
            total_y_model_members = step_y0 + tail_y_rem
        else:
            ast_seq = (
                program.ast_nodes
                if (program is not None and program.ast_nodes)
                else (ASTNode(op="Call", tool_or_action=candidate_action), ASTNode(op="Stop"))
            )
            wrapper_prog = program or ExecutableProgram(
                program_id=f"atomic:{_short_hash(candidate_action)}",
                benchmark=eff_benchmark,
                step_templates=(candidate_action,),
                readout_fields=(),
                ast_nodes=ast_seq,
            )
            root_frame = ExecutionFrame(
                program=wrapper_prog,
                bound_variables=dict(base_bindings),
                observed_readouts=dict(base_fields),
                node_stack=list(ast_seq),
            )
            q_model_members, end_spent, total_y_model_members = _eval_ast_frame(
                root_frame, init_belief, init_spent, init_alive, init_rem_steps, init_slot_probs_s, 0
            )

        q_model_members = q_model_members.clamp(min=0.0, max=1.0)
        q_members, total_y_members = self.apply_prior_advantage_to_q(
            q_model_members, total_y_model_members, float(prior_action_logprob) + float(prior_success_logit)
        )
        q_mean_t = q_members.mean()
        q_std_t = q_members.std(unbiased=False) if self.ensemble_size > 1 else torch.zeros_like(q_mean_t)
        q_lcb_val = float((q_mean_t - 0.5 * q_std_t).clamp(min=0.0).item())

        if member_index is not None:
            m_idx = int(member_index) % self.ensemble_size
            q_scalar = float(q_members[m_idx].item())
        elif controller_mode == "prior_only_reranker":
            q_scalar = float(prior_action_logprob)
        elif controller_mode == "legacy_lcb_v3":
            q_scalar = q_lcb_val
        else:
            q_scalar = float(q_mean_t.item())

        return {
            "q_mean": float(q_mean_t.item()),
            "q_model_only_mean": float(q_model_members.mean().item()),
            "prior_advantage": float(prior_action_logprob),
            "q_std": float(q_std_t.item()),
            "q_lcb": q_lcb_val,
            "q_selected": q_scalar,
            "q_members": [float(x) for x in q_members.detach().cpu().tolist()],
            "predicted_terminal_success": float(total_y_members.mean().item()),
            "predicted_cost_ratio": float(end_spent.mean().item()),
        }

    @torch.no_grad()
    def evaluate_decision_candidates(
        self,
        decision_ctx: DecisionContext,
        belief: torch.Tensor,
        candidates: Sequence[tuple[str, str, ExecutableProgram | None, ExecutionFrame | None]],
        *,
        action_embedder: Any | None = None,
        precomputed_action_vecs: Mapping[str, Sequence[float]] | None = None,
        precomputed_prior_logprobs: Mapping[str, float] | None = None,
        use_initial_params: bool = False,
    ) -> list[dict[str, Any]]:
        """Single unified candidate-scoring entrypoint shared by training q-selection, post-q d_model, R_h(u,v), and act()."""
        results: list[dict[str, Any]] = []
        vec_map = precomputed_action_vecs or {}
        lp_map = precomputed_prior_logprobs or {}
        last_act = str(decision_ctx.public_fields.get("_last_action", "")).strip().lower()
        is_loca = decision_ctx.benchmark.split(":", 1)[0] == "loca"
        apply_repeat_penalty = (
            bool(last_act)
            and last_act != "none"
            and decision_ctx.controller_mode != "prior_only_reranker"
        )
        for cid, c_act, c_prog, c_frame in candidates:
            norm_act = PublicBenchmarkBinder.normalize_action_for_semantic_encoding(
                c_act, decision_ctx.role_bindings
            )
            act_vec = vec_map.get(norm_act) or vec_map.get(c_act) or (
                action_embedder(norm_act) if action_embedder is not None else ()
            )
            cand_lp = float(lp_map.get(c_act, lp_map.get(c_act.strip(), 0.0)))
            q_eval = self.evaluate_candidate_q(
                belief=belief,
                candidate_action=c_act,
                program=c_prog,
                initial_frame=c_frame,
                decision_ctx=decision_ctx,
                frozen_action_vec=act_vec,
                action_embedder=action_embedder,
                prior_action_logprob=cand_lp,
                use_initial_params=use_initial_params,
            )
            cand_cmp = (
                PublicBenchmarkBinder.normalize_action_for_semantic_encoding(c_act).strip().lower()
                if is_loca
                else str(c_act).strip().lower()
            )
            if apply_repeat_penalty and cand_cmp == last_act:
                q_eval["q_selected"] = float(q_eval["q_selected"]) - ACTION_REPEAT_PENALTY
                q_eval["repetition_penalty_applied"] = -ACTION_REPEAT_PENALTY
            else:
                q_eval["repetition_penalty_applied"] = 0.0
            results.append(
                {
                    "candidate_id": cid,
                    "candidate_action": c_act,
                    "program": c_prog,
                    "frame": c_frame,
                    "prior_action_logprob": cand_lp,
                    **q_eval,
                }
            )
        return results

    @staticmethod
    def select_fork_continuation_pair(
        scored_candidates: Sequence[Mapping[str, Any]],
        *,
        benchmark: str,
        goal_category: str,
        role_bindings: Mapping[str, str] | None = None,
        action_templates: Mapping[str, str] | None = None,
        prior_action_text: str = "",
        existing_comparison_keys: Mapping[str, set[str]] | None = None,
        current_readout_sig: str = "",
        rng: random.Random | None = None,
    ) -> dict[str, Any] | None:
        """Select continuation pair (u, v) by ensemble decision-disagreement R_h(u, v) per paired execution cost,
        using the exact `role_bindings` / `action_templates` from `DecisionContext` so acquisition keys match
        stored `MeasurementRecord` and `ActionContrastProposer` keys identically.
        """
        tmpl_cache = dict(action_templates or {})

        def _tmpl_of(act_str: str) -> str:
            s_a = str(act_str).strip()
            if s_a not in tmpl_cache:
                tmpl_cache[s_a] = PublicBenchmarkBinder.abstract_action_to_template(
                    benchmark, s_a, role_bindings
                )
            return tmpl_cache[s_a]

        distinct: list[Mapping[str, Any]] = []
        seen_tmpls: set[str] = set()
        seen_acts: set[str] = set()
        # Preserve prior proposal first so tie-breaking favors a_{pi_0}
        ordered_in = sorted(
            scored_candidates,
            key=lambda item: (0 if str(item.get("candidate_action", "")).strip() == prior_action_text.strip() else 1),
        )
        for item in ordered_in:
            act = str(item.get("candidate_action", "")).strip()
            if not act or act in seen_acts:
                continue
            tmpl = _tmpl_of(act)
            if tmpl in seen_tmpls and len(scored_candidates) > 2:
                continue
            seen_acts.add(act)
            seen_tmpls.add(tmpl)
            distinct.append(item)

        if len(distinct) < 2:
            distinct = []
            seen_acts.clear()
            for item in ordered_in:
                act = str(item.get("candidate_action", "")).strip()
                if act and act not in seen_acts:
                    seen_acts.add(act)
                    distinct.append(item)
        if len(distinct) < 2:
            return None

        # Identify mean-optimal candidate (prior-anchored on exact ties)
        best_u_idx = max(
            range(len(distinct)),
            key=lambda i: (
                round(float(distinct[i]["q_mean"]), 7),
                1 if str(distinct[i]["candidate_action"]).strip() == prior_action_text.strip() else 0,
            ),
        )

        best_pair: tuple[int, int] | None = None
        best_r_per_cost = -1.0
        best_r_raw = 0.0
        max_unthresholded_r = 0.0
        existing_map = existing_comparison_keys or {}

        # Evaluate pairs involving competitive candidates (top-8 by q_mean + prior + previously measured keys)
        ranked_indices = sorted(
            range(len(distinct)),
            key=lambda i: float(distinct[i]["q_mean"]),
            reverse=True,
        )
        anchor_indices = list(dict.fromkeys([0, best_u_idx, *ranked_indices[:8]]))

        for idx_pos, i in enumerate(anchor_indices):
            qi = np.asarray(distinct[i]["q_members"], dtype=np.float64)
            q_bar_i = float(qi.mean())
            cost_i = max(0.01, float(distinct[i].get("predicted_cost_ratio", 0.05)))
            tmpl_i = _tmpl_of(str(distinct[i]["candidate_action"]))
            for j in range(len(distinct)):
                if i == j:
                    continue
                qj = np.asarray(distinct[j]["q_members"], dtype=np.float64)
                q_bar_j = float(qj.mean())
                r_val = float(np.maximum(qi, qj).mean() - max(q_bar_i, q_bar_j))
                if r_val > max_unthresholded_r:
                    max_unthresholded_r = r_val
                cost_j = max(0.01, float(distinct[j].get("predicted_cost_ratio", 0.05)))
                pair_cost = max(0.05, cost_i + cost_j)
                tmpl_j = _tmpl_of(str(distinct[j]["candidate_action"]))
                comp = Comparison.canonicalize(
                    goal_category=goal_category,
                    u_policy_id=tmpl_i,
                    v_policy_id=tmpl_j,
                )
                seen_sigs = existing_map.get(comp.comparison_key, set())
                cross_anchor_bonus = 1.05 if (seen_sigs and current_readout_sig not in seen_sigs) else 1.0
                score = (r_val / pair_cost) * cross_anchor_bonus
                if score > best_r_per_cost + 1e-12:
                    best_r_per_cost = score
                    best_r_raw = r_val
                    best_pair = (i, j) if q_bar_i >= q_bar_j else (j, i)

        selection_mode = "decision_disagreement_r_h"
        if best_pair is None or best_r_raw <= 5e-4:
            # Below meaningful disagreement threshold: anchor u on frozen LM prior proposal (index 0 if valid)
            u_idx = 0 if (prior_action_text and str(distinct[0]["candidate_action"]).strip() == prior_action_text.strip()) else best_u_idx
            u_act = str(distinct[u_idx]["candidate_action"])
            u_tmpl = _tmpl_of(u_act)
            u_verb = u_act.split()[0].split("[")[0].lower() if u_act else ""

            # 1) Reuse a canonical comparison_key already measured at a distinct public readout z' != z
            reusable_v: list[int] = []
            diverse_v: list[int] = []
            all_other_v: list[int] = []
            for j in range(len(distinct)):
                if j == u_idx:
                    continue
                all_other_v.append(j)
                v_act = str(distinct[j]["candidate_action"])
                v_tmpl = _tmpl_of(v_act)
                comp = Comparison.canonicalize(goal_category=goal_category, u_policy_id=u_tmpl, v_policy_id=v_tmpl)
                seen_sigs = existing_map.get(comp.comparison_key, set())
                if seen_sigs and current_readout_sig not in seen_sigs:
                    reusable_v.append(j)
                v_verb = v_act.split()[0].split("[")[0].lower() if v_act else ""
                if v_verb != u_verb:
                    diverse_v.append(j)

            chooser = rng or random.Random(42)
            if reusable_v:
                v_idx = chooser.choice(reusable_v)
                selection_mode = "prior_cross_anchor_key_reuse"
            elif diverse_v:
                v_idx = chooser.choice(diverse_v)
                selection_mode = "prior_role_diverse_sample"
            else:
                v_idx = chooser.choice(all_other_v)
                selection_mode = "prior_uniform_support_sample"
            best_pair = (u_idx, v_idx)
            best_r_raw = 0.0

        idx_u, idx_v = best_pair
        chosen_u_act = str(distinct[idx_u]["candidate_action"])
        chosen_v_act = str(distinct[idx_v]["candidate_action"])
        chosen_comp = Comparison.canonicalize(
            goal_category=goal_category,
            u_policy_id=_tmpl_of(chosen_u_act),
            v_policy_id=_tmpl_of(chosen_v_act),
        )
        return {
            "u_entry": distinct[idx_u],
            "v_entry": distinct[idx_v],
            "comparison": chosen_comp,
            "disagreement_r": float(best_r_raw),
            "max_raw_disagreement_r": float(max_unthresholded_r),
            "selection_mode": selection_mode,
            "num_distinct_candidates": len(distinct),
        }

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
        """Exact closed-form EVSI reference without artificial polarity modulation."""
        if use_Temporary_Initial_Params and self._initial_state_tensors:
            return self.active_model_view(use_initial_params=True).score_program_evsi(
                program,
                public_fields,
                belief,
                use_Temporary_Initial_Params=False,
                goal_text=goal_text,
            )

        if True:
            device = self.transition_logits.device
            feat = self.encode_program_features(
                program, public_fields, device, goal_text=goal_text
            )
            L_m, next_prior = self.observable_operator_step(belief, feat, observed_outcome_bin=None)
            w = self.mechanism_weights / self.mechanism_weights.sum().clamp(min=1e-8)

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
            )

            prior_values = torch.einsum("m,mj->j", w, V_mj)
            before_val = float(prior_values.max().item())

            weighted_lv = w[:, None, None] * L_m[:, :, None] * V_mj[:, None, :]
            after_per_outcome = weighted_lv.sum(dim=0).max(dim=-1).values
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

    def transition_negative_log_likelihood(
        self,
        transitions: Sequence[PublicTransition],
        *,
        tail_examples: Sequence[TailExample] = (),
        bootstrap_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Fork-tree marginal negative log-likelihood + fixed-tail H_{phi,m}^{pi_0} supervision.

        Fixes F02, F04, F09, F11, F13, F22:
          - Every transition node is evaluated starting from `belief_by_node[tr.parent_transition_id]`
            so shared prefix nodes are forwarded once, and counterfactual suffix v starts from
            `fork_belief` (b_{post-q}) rather than u's terminal belief!
          - Supervises typed field-slot distributions, outcome bins, wall-clock step cost ratio,
            and termination category.
          - Supervises `predict_tail_pi0` strictly on genuine `TailExample` records from frozen pi_0 suffixes.
        """
        device = self.transition_logits.device
        root_belief = self.reset_intra_task_belief(device=device)
        if bootstrap_weights is None:
            b_weights = torch.ones(self.ensemble_size, device=device, dtype=torch.float32)
        else:
            b_weights = bootstrap_weights.to(device=device, dtype=torch.float32)
            if b_weights.sum() <= 0:
                b_weights = torch.ones_like(b_weights)
            b_weights = b_weights / b_weights.mean().clamp(min=1e-8)

        total_member_nll = torch.zeros(self.ensemble_size, device=device, dtype=torch.float32)
        observed_outcomes = 0
        tail_supervised_count = 0

        # Pre-index episode-level pi_0 branch outcomes so fork-head steps `(u_0, v_0)` receive exact
        # marginal continuation likelihood supervision and multi-step branches are normalized 1:1 per episode.
        ep_tail_counts: dict[str, int] = {}
        ep_outcome_info: dict[str, tuple[bool, tuple[float, ...], float]] = {}
        for tex in tail_examples:
            if not tex.censored and tex.observed_y is not None:
                ep_key = str(tex.episode_id or "default_ep")
                ep_tail_counts[ep_key] = ep_tail_counts.get(ep_key, 0) + 1
                res_tb = ResourceState.from_dict(tex.resource_before)
                sf_tb = (
                    float(tex.step_budget_fraction)
                    if tex.step_budget_fraction > 0
                    else 1.0 / max(1.0, float(res_tb.max_env_steps))
                )
                ep_outcome_info[ep_key] = (bool(tex.observed_y), tex.frozen_tail_action_vec, sf_tb)

        ep_pi0_step_counts: dict[str, int] = {}
        for tr_c in transitions:
            ep_c = str(tr_c.episode_id or "")
            if tr_c.controller_id == TAIL_POLICY_VERSION and ep_c:
                ep_pi0_step_counts[ep_c] = ep_pi0_step_counts.get(ep_c, 0) + 1

        seen_branch_heads: set[str] = set()
        fork_head_nll = torch.zeros(self.ensemble_size, device=device, dtype=torch.float32)
        fork_heads_supervised = 0
        grid_idx = torch.tensor(_JOINT_SLOT_GRID_NP, device=device, dtype=torch.long)

        # Map transition_id -> post-step belief tensor (retaining computation graph for shared prefix gradients!)
        belief_by_node: dict[str, torch.Tensor] = {}
        last_belief = root_belief

        for tr in transitions:
            if tr.parent_transition_id and tr.parent_transition_id in belief_by_node:
                in_belief = belief_by_node[tr.parent_transition_id]
            elif tr.step_index == 0 or tr.parent_transition_id is None:
                in_belief = root_belief
            else:
                in_belief = last_belief

            res_before = ResourceState.from_dict(tr.resource_before or tr.remaining_budget)
            rem_sec_ratio = res_before.remaining_seconds_ratio
            ctx = self.encode_context_vector(
                goal_text=tr.goal_text,
                public_fields=tr.pre_public_fields,
                action_text=tr.action_text,
                remaining_budget_ratio=rem_sec_ratio,
                spent_seconds_ratio=res_before.spent_seconds_ratio,
                remaining_steps_ratio=res_before.remaining_steps_ratio,
                role_bindings=tr.role_bindings,
                frozen_context_vec=tr.frozen_context_vec,
                frozen_action_vec=tr.frozen_action_vec,
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
            preds = self.step_and_predict(
                in_belief,
                ctx,
                observed_outcome_bin=obs_bin,
                prior_action_logprob=float(tr.prior_action_logprob),
                remaining_budget_ratio=rem_sec_ratio,
            )
            belief_by_node[tr.transition_id] = preds["post_belief"]
            last_belief = preds["post_belief"]

            # 1. Outcome bin NLL + typed field-slot NLL
            resp_nll = -torch.log(preds["outcome_probs"][:, obs_bin].clamp(min=1e-8))
            target_slots = PublicBenchmarkBinder.encode_typed_field_slots(tr.typed_response)
            slot_nll = torch.zeros_like(resp_nll)
            for s_idx, v_idx in enumerate(target_slots):
                slot_nll = slot_nll - 0.25 * torch.log(
                    preds["field_slot_probs"][:, s_idx, v_idx].clamp(min=1e-8)
                )

            # 2. Coupled 3-way step outcome NLL: 0=live, 1=native_terminal_success, 2=terminal_failure
            if term_done:
                observed_outcomes += 1
                term_idx = 1 if term_succ else 2
            else:
                term_idx = 0
            term_nll = -torch.log(preds["termination_probs"][:, term_idx].clamp(min=1e-8))

            # 3. Wall-clock step cost ratio loss
            target_cost = float(
                min(1.0, max(0.0, tr.resource_delta.get("seconds_ratio", tr.resource_delta.get("cost_ratio", 0.01))))
            )
            cost_nll = 0.5 * (preds["cost_ratio"] - target_cost).square()

            total_member_nll = total_member_nll + (resp_nll + slot_nll + term_nll + cost_nll)

            # 4. Exact 1-step marginal continuation likelihood on every pi_0-followed transition `(a_t -> pi_0)`:
            #    P_m(Y_branch | b_t, a_t, do(pi_0)) = p_term_succ(a_t) + p_live(a_t) * E_Z[P_m^{pi_0}(Y=1 | b_m(.|Z), Z)]
            #    Paired fork-head interventions `(u, v)` carry unit weight 1.0; subsequent on-policy tail steps
            #    carry episode-normalized weight `1 / max(1, N_ep - 1)` so every stagnant or successful tail step
            #    trains action-conditioned continuation without inflating episode sample count.
            ep_id_str = str(tr.episode_id or "")
            if (
                tr.controller_id == TAIL_POLICY_VERSION
                and (ep_id_str in ep_outcome_info or (term_done and tr.native_outcome is not None))
            ):
                is_fork_head = ep_id_str not in seen_branch_heads
                seen_branch_heads.add(ep_id_str)
                n_ep_steps = ep_pi0_step_counts.get(ep_id_str, 1)
                cont_weight = 1.0 if is_fork_head else (1.0 / float(max(1, n_ep_steps - 1)))
                if ep_id_str in ep_outcome_info:
                    y_branch_bool, t_act_vec, sf_val = ep_outcome_info[ep_id_str]
                else:
                    y_branch_bool = bool(tr.native_outcome)
                    t_act_vec = ()
                    sf_val = 1.0 / max(1.0, float(res_before.max_env_steps))
                next_rem_s = max(0.0, res_before.remaining_steps_ratio - sf_val)
                p_live_h = preds["termination_probs"][:, 0]
                p_succ_h = preds["termination_probs"][:, 1]
                if next_rem_s > 1e-7:
                    sp_s = preds["slot_probs_per_state"]
                    nb_h = preds["next_state_prior"]
                    lik_g = (
                        sp_s[:, :, 0, grid_idx[:, 0]]
                        * sp_s[:, :, 1, grid_idx[:, 1]]
                        * sp_s[:, :, 2, grid_idx[:, 2]]
                        * sp_s[:, :, 3, grid_idx[:, 3]]
                    )
                    un_g = nb_h.unsqueeze(-1) * lik_g
                    pz_g = un_g.sum(dim=1)
                    pz_g = pz_g / pz_g.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                    pb_g = (un_g.permute(0, 2, 1)) / un_g.sum(dim=1).unsqueeze(-1).clamp(min=1e-8)

                    base_c_tail = self.encode_context_vector(
                        goal_text=tr.goal_text,
                        public_fields=tr.pre_public_fields,
                        action_text=TAIL_POLICY_VERSION,
                        remaining_budget_ratio=rem_sec_ratio,
                        spent_seconds_ratio=res_before.spent_seconds_ratio,
                        remaining_steps_ratio=next_rem_s,
                        frozen_context_vec=tr.frozen_context_vec,
                        frozen_action_vec=t_act_vec,
                        device=device,
                    )
                    c_b = base_c_tail.view(1, 1, self.context_dim).expand(
                        self.ensemble_size, 256, self.context_dim
                    ).clone()
                    sp_after = (res_before.spent_seconds_ratio + preds["cost_ratio"]).clamp(0.0, 1.0)
                    rm_after = (1.0 - sp_after).clamp(0.0, 1.0)
                    bi = FROZEN_HISTORY_DIM + FROZEN_ACTION_DIM
                    c_b[:, :, bi + 0] = rm_after.unsqueeze(1)
                    c_b[:, :, bi + 1] = sp_after.unsqueeze(1)
                    c_b[:, :, bi + 2] = next_rem_s
                    c_b[:, :, bi + 10 : bi + 14] = ((grid_idx.to(dtype=c_b.dtype) - 1.5) / 1.5).unsqueeze(0)
                    c_b[:, :, bi + 14] = (grid_idx[:, 0] == 3).to(dtype=c_b.dtype).unsqueeze(0)
                    c_b[:, :, bi + 15] = (grid_idx[:, 3] == 3).to(dtype=c_b.dtype).unsqueeze(0)
                    h_b = self.context_encoder(c_b.reshape(-1, self.context_dim)).view(
                        self.ensemble_size, 256, self.context_dim
                    )
                    oh_b = (
                        F.one_hot(grid_idx, num_classes=NUM_SLOT_VALUES)
                        .to(dtype=nb_h.dtype)
                        .reshape(1, 256, NUM_FIELD_SLOTS * NUM_SLOT_VALUES)
                        .expand(self.ensemble_size, 256, -1)
                    )
                    t_out_h = self.predict_tail_pi0(
                        pb_g,
                        h_b,
                        sp_after,
                        field_slot_probs=oh_b,
                        remaining_steps_ratio=next_rem_s,
                        step_budget_fraction=sf_val,
                    )
                    marg_tail_y = (pz_g * t_out_h["tail_y_prob"]).sum(dim=-1)
                    p_branch_y_model = (p_succ_h + p_live_h * marg_tail_y).clamp(min=1e-6, max=1.0 - 1e-6)
                else:
                    p_branch_y_model = p_succ_h.clamp(min=1e-6, max=1.0 - 1e-6)

                _, p_branch_y = self.apply_prior_advantage_to_q(
                    p_branch_y_model, p_branch_y_model, float(tr.prior_action_logprob)
                )

                y_br_t = 1.0 if y_branch_bool else 0.0
                fork_head_nll = fork_head_nll - cont_weight * (
                    y_br_t * torch.log(p_branch_y) + (1.0 - y_br_t) * torch.log(1.0 - p_branch_y)
                )
                if is_fork_head:
                    fork_heads_supervised += 1

        # 5. Supervise fixed-tail head H_{phi,m}^{pi_0}(Y, C_tail | Y=1) normalized 1:1 per episode
        tail_member_nll = torch.zeros(self.ensemble_size, device=device, dtype=torch.float32)
        num_tail_episodes = max(1, len(ep_tail_counts))
        for tex in tail_examples:
            if tex.censored or tex.observed_y is None:
                continue
            tail_supervised_count += 1
            ep_w = 1.0 / float(max(1, ep_tail_counts.get(str(tex.episode_id or "default_ep"), 1)))
            res_b = ResourceState.from_dict(tex.resource_before)
            eff_step_frac = (
                float(tex.step_budget_fraction)
                if tex.step_budget_fraction > 0
                else 1.0 / max(1.0, float(res_b.max_env_steps))
            )
            h_ctx_t = self.encode_tail_state_context(
                goal_text=tex.goal_text,
                public_fields=tex.pre_public_fields,
                remaining_budget_ratio=res_b.remaining_seconds_ratio,
                spent_seconds_ratio=res_b.spent_seconds_ratio,
                remaining_steps_ratio=res_b.remaining_steps_ratio,
                frozen_context_vec=tex.frozen_context_vec,
                frozen_tail_action_vec=tex.frozen_tail_action_vec,
                device=device,
            )
            spent_vec = torch.full(
                (self.ensemble_size,), res_b.spent_seconds_ratio, device=device, dtype=torch.float32
            )
            if tex.prefix_node_id in (None, "", "__root__"):
                tex_belief = root_belief
            else:
                tex_belief = belief_by_node.get(str(tex.prefix_node_id), root_belief)
            obs_slot_oh = self._one_hot_field_slots(
                tex.pre_public_fields, self.ensemble_size, device, tex_belief.dtype
            )
            tail_preds = self.predict_tail_pi0(
                tex_belief,
                h_ctx_t,
                spent_vec,
                field_slot_probs=obs_slot_oh,
                public_fields=tex.pre_public_fields,
                remaining_steps_ratio=res_b.remaining_steps_ratio,
                step_budget_fraction=eff_step_frac,
            )
            p_y = tail_preds["tail_y_prob"].clamp(min=1e-6, max=1.0 - 1e-6)
            y_target = 1.0 if tex.observed_y else 0.0
            t_loss = -(y_target * torch.log(p_y) + (1.0 - y_target) * torch.log(1.0 - p_y))
            if tex.observed_y and tex.tail_seconds_cost_ratio is not None:
                bin_idx = _tail_cost_bin_index(float(tex.tail_seconds_cost_ratio))
                p_bin = tail_preds["tail_cost_bin_probs"][:, bin_idx].clamp(min=1e-8)
                t_loss = t_loss - torch.log(p_bin)
            tail_member_nll = tail_member_nll + ep_w * t_loss

        n_steps = max(1, len(transitions))
        combined_member_loss = (
            (total_member_nll / float(n_steps))
            + (tail_member_nll / float(num_tail_episodes) if tail_supervised_count > 0 else 0.0)
            + (fork_head_nll / float(max(1, fork_heads_supervised)) if fork_heads_supervised > 0 else 0.0)
        )
        weighted_loss = (b_weights * combined_member_loss).mean()
        return weighted_loss, last_belief.detach(), {
            "num_transitions": float(len(transitions)),
            "observed_terminal_outcomes": float(observed_outcomes),
            "tail_examples_supervised": float(tail_supervised_count),
            "mean_member_nll": float(combined_member_loss.mean().detach().item()),
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
        device = self.transition_logits.device
        act_str = action_text or (program.step_templates[0] if program.step_templates else "step")
        ctx = self.encode_program_features(
            program, public_fields, device, goal_text=goal_text, action_text=act_str
        )
        preds = self.step_and_predict(belief, ctx, observed_outcome_bin=outcome_bin)
        o_idx = int(outcome_bin) % self.outcome_bins
        resp_nll = -torch.log(preds["outcome_probs"][:, o_idx].clamp(min=1e-8))

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
        return loss, preds["post_belief"].detach()


class ProgramLibrary:
    """Persistable library Xi of pure symbolic typed executable programs."""

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
    """Witness-guided executable program proposer with canonical Comparison matching,
    AST consumer-graph Delete, alpha-renaming Concat, and strict non-expanding Label-Shuffled control.
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
        proposer_mode: str = "witness_hard",
        seed: int = 42,
    ) -> None:
        self.benchmark = benchmark.split(":", 1)[0]
        self.max_probe_steps = int(max_probe_steps)
        self.proposal_budget = int(proposal_budget)
        self.max_candidate_checks = int(max_candidate_checks)
        self.shuffle_witness_labels = bool(shuffle_witness_labels)
        self.use_model_diff_variant = bool(use_model_diff_variant)
        self.proposer_mode = str(proposer_mode or "witness_hard")
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
        return len(self.unique_witness_pair_ids)

    def record_measurement(self, record: MeasurementRecord) -> None:
        self.records.append(record)

    @staticmethod
    def _def_use_delete(program: ExecutableProgram, target_field: str) -> ExecutableProgram | None:
        """Transform `program.ast_nodes` directly, removing `target_field` and ONLY unreferenced producer Calls (fixes E)."""
        if len(program.readout_fields) <= 1 or target_field not in program.readout_fields:
            return None
        remaining_fields = tuple(f for f in program.readout_fields if f != target_field)
        producer_for_target = {
            e.producer_step
            for e in program.def_use_edges
            if e.field_path == target_field and e.consumer_step is None
        }
        protected_producers = {
            e.producer_step
            for e in program.def_use_edges
            if e.consumer_step is not None or e.field_path in remaining_fields
        }
        removable_steps = producer_for_target - protected_producers

        new_ast: list[ASTNode] = []
        removed_vars: set[str] = set()
        call_counter = 0
        total_top_calls = sum(1 for n in program.ast_nodes if n.op == "Call")
        for node in program.ast_nodes:
            if node.op == "Call":
                cur_idx = call_counter
                call_counter += 1
                if cur_idx in removable_steps and total_top_calls > 1:
                    if node.bind_return_to:
                        removed_vars.add(node.bind_return_to)
                    total_top_calls -= 1
                    continue
                new_ast.append(node)
            elif node.op == "Read":
                if node.public_field_path == target_field or (node.variable and node.variable in removed_vars):
                    continue
                new_ast.append(node)
            else:
                new_ast.append(node)

        kept_steps = tuple(n.tool_or_action for n in new_ast if n.op == "Call" and n.tool_or_action)
        if not kept_steps:
            return None
        return ExecutableProgram(
            program_id=_short_hash((program.benchmark, tuple(n.to_dict() for n in new_ast), remaining_fields, "delete")),
            benchmark=program.benchmark,
            step_templates=kept_steps,
            readout_fields=remaining_fields,
            origin="witness_delete",
            parent_ids=(program.program_id,),
            ast_nodes=tuple(new_ast),
            case_condition_field=program.case_condition_field,
            case_condition_value=program.case_condition_value,
            case_true_template=program.case_true_template,
            case_false_template=program.case_false_template,
        )

    @staticmethod
    def _scope_unified_concat(
        p1: ExecutableProgram,
        p2: ExecutableProgram,
        differing_fields: Sequence[str],
        max_steps: int,
        *,
        case_condition_value: str = "",
        case_true_template: str = "",
        case_false_template: str = "",
    ) -> ExecutableProgram | None:
        """Concatenate two programs directly on AST with alpha-renaming and consistent AST/budget Call count (fixes E)."""
        combined_fields: list[str] = []
        for fld in (*differing_fields, *p1.readout_fields, *p2.readout_fields):
            if fld not in combined_fields and len(combined_fields) < 4:
                combined_fields.append(fld)
        if not combined_fields:
            return None

        cond_field = differing_fields[0] if differing_fields else p1.case_condition_field
        raw_cond_val = case_condition_value or p1.case_condition_value
        cond_val = (
            PublicBenchmarkBinder.canonical_slot_predicate_value(cond_field, raw_cond_val)
            if (cond_field and raw_cond_val)
            else ""
        )
        t_tmpl = case_true_template or p1.case_true_template
        f_tmpl = case_false_template or p1.case_false_template
        has_valid_case = bool(cond_field and cond_val and t_tmpl and f_tmpl and t_tmpl != f_tmpl)
        body_call_cap = max(1, max_steps - (1 if has_valid_case else 0))

        p1_body = [n for n in p1.ast_nodes if n.op not in ("Stop", "Case")]
        p2_body = list(_alpha_rename_ast_nodes([n for n in p2.ast_nodes if n.op not in ("Stop", "Case")], "p2"))
        merged_ast: list[ASTNode] = []
        kept_vars: set[str] = set()
        last_call_tmpl: str | None = None
        last_kept_var: str = ""
        remapped_vars: dict[str, str] = {}
        calls_count = 0

        for node in (*p1_body, *p2_body):
            if node.op == "Call":
                if last_call_tmpl is not None and node.tool_or_action == last_call_tmpl:
                    # Collapse immediately adjacent identical idempotent Call in AST itself (matching display templates!)
                    if node.bind_return_to and last_kept_var:
                        remapped_vars[node.bind_return_to] = last_kept_var
                    continue
                if calls_count >= body_call_cap:
                    continue
                calls_count += 1
                last_call_tmpl = node.tool_or_action
                if node.bind_return_to:
                    kept_vars.add(node.bind_return_to)
                    last_kept_var = node.bind_return_to
                merged_ast.append(node)
            elif node.op == "Read":
                target_var = remapped_vars.get(node.variable, node.variable)
                if target_var and target_var not in kept_vars:
                    continue
                merged_ast.append(
                    ASTNode(
                        op="Read",
                        variable=target_var,
                        public_field_path=node.public_field_path,
                    )
                )

        kept_steps = tuple(n.tool_or_action for n in merged_ast if n.op == "Call" and n.tool_or_action)
        if not kept_steps:
            return None

        if has_valid_case:
            merged_ast.append(
                ASTNode(
                    op="Case",
                    condition_field=cond_field,
                    condition_equals=cond_val,
                    children=(
                        ASTNode(
                            op="Call",
                            tool_or_action=t_tmpl,
                            typed_arguments={s: f"${s}" for s in re.findall(r"\{([a-zA-Z0-9_.]+)\}", t_tmpl)},
                            bind_return_to="ret_case_true",
                        ),
                        ASTNode(
                            op="Call",
                            tool_or_action=f_tmpl,
                            typed_arguments={s: f"${s}" for s in re.findall(r"\{([a-zA-Z0-9_.]+)\}", f_tmpl)},
                            bind_return_to="ret_case_false",
                        ),
                    ),
                )
            )
        merged_ast.append(ASTNode(op="Stop"))

        return ExecutableProgram(
            program_id=_short_hash(
                (p1.benchmark, tuple(n.to_dict() for n in merged_ast), tuple(combined_fields))
            ),
            benchmark=p1.benchmark,
            step_templates=kept_steps,
            readout_fields=tuple(combined_fields),
            origin="witness_concat",
            parent_ids=(p1.program_id, p2.program_id),
            ast_nodes=tuple(merged_ast),
            case_condition_field=cond_field if has_valid_case else "",
            case_condition_value=cond_val if has_valid_case else "",
            case_true_template=t_tmpl if has_valid_case else "",
            case_false_template=f_tmpl if has_valid_case else "",
        )

    def _build_label_view(self) -> tuple[dict[str, float], int, int]:
        """Construct a single isolated action-difference label view grouped strictly by `comparison_key`.

        Fixes F15:
          - Permutes ONLY within matching `comparison_key` groups (`(goal_category, canonical_u_v, tail_policy_version)`).
          - Singleton groups remain unchanged (never expanded to `goal_category` or `all`!).
          - Returns `(label_map, permutable_groups_count, permutable_records_count)`.
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
                val = 0.0
            raw_map[rec.record_id] = val

        grouped: dict[str, list[str]] = {}
        for rec in self.records:
            grouped.setdefault(rec.comparison_key, []).append(rec.record_id)

        permutable_groups = sum(1 for ids in grouped.values() if len(ids) >= 2)
        permutable_records = sum(len(ids) for ids in grouped.values() if len(ids) >= 2)

        if not self.shuffle_witness_labels:
            return raw_map, permutable_groups, permutable_records

        shuffled_map: dict[str, float] = {}
        for _, r_ids in sorted(grouped.items()):
            vals = [raw_map[rid] for rid in r_ids]
            if len(vals) >= 2:
                self.rng.shuffle(vals)
            for rid, v in zip(r_ids, vals):
                shuffled_map[rid] = v
        return shuffled_map, permutable_groups, permutable_records

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

        label_view, permutable_groups, permutable_records = self._build_label_view()

        # Group records strictly by canonical `comparison_key` so witnesses only pair identical ordered (u, v) (fixes F15)
        by_comparison: dict[str, list[MeasurementRecord]] = {}
        for rec in self.records:
            if rec.program_executed_cleanly:
                by_comparison.setdefault(rec.comparison_key, []).append(rec)

        candidates_constructed = 0
        candidates_checked = 0
        candidates_scored = 0
        candidates: list[tuple[float, ExecutableProgram]] = []

        for _, comp_records in sorted(by_comparison.items()):
            if candidates_checked >= self.max_candidate_checks:
                break
            if self.proposer_mode == "conditional_gain":
                # Exact binary Case(Z_f == z*: true_branch, false_branch) selection gain Gamma_binary(f, z*; u, v)
                # evaluated on the identical observed population D_{u,v,f} = {r in D_{u,v} : Z_f(r) != "unknown"}.
                if len(comp_records) < 2:
                    continue
                all_f_keys = list(dict.fromkeys(k for r in comp_records for k in r.public_readout_z.keys()))
                best_f = ""
                best_gamma = 0.0
                best_pos_z = ""
                best_true_is_u = True
                for fk in all_f_keys:
                    obs_pairs: list[tuple[str, float]] = []
                    for r in comp_records:
                        raw_z = r.public_readout_z.get(fk, "unknown")
                        if raw_z == "unknown":
                            continue
                        cz = PublicBenchmarkBinder.canonical_slot_predicate_value(fk, raw_z)
                        obs_pairs.append((cz, float(label_view.get(r.record_id, 0.0))))
                    if len(obs_pairs) < 2:
                        continue
                    n_obs = float(len(obs_pairs))
                    base_gain_f = max(0.0, sum(dv for _, dv in obs_pairs) / n_obs)
                    unique_zs = list(dict.fromkeys(cz for cz, _ in obs_pairs))
                    if len(unique_zs) < 2:
                        continue
                    for z_cand in unique_zs:
                        eq_vals = [dv for cz, dv in obs_pairs if cz == z_cand]
                        neq_vals = [dv for cz, dv in obs_pairs if cz != z_cand]
                        if not eq_vals or not neq_vals:
                            continue
                        mean_eq = sum(eq_vals) / float(len(eq_vals))
                        mean_neq = sum(neq_vals) / float(len(neq_vals))
                        p_eq = float(len(eq_vals)) / n_obs
                        p_neq = float(len(neq_vals)) / n_obs
                        bin_gain = p_eq * max(0.0, mean_eq) + p_neq * max(0.0, mean_neq)
                        gamma_bin = bin_gain - base_gain_f
                        if gamma_bin > best_gamma + 1e-6:
                            best_gamma = gamma_bin
                            best_f = fk
                            best_pos_z = z_cand
                            best_true_is_u = bool(mean_eq > 0.0)
                if best_gamma <= 1e-6 or not best_f:
                    continue
                pos_records = [r for r in comp_records if float(label_view.get(r.record_id, 0.0)) > 0.0]
                neg_records = [r for r in comp_records if float(label_view.get(r.record_id, 0.0)) < 0.0]
                if not pos_records or not neg_records:
                    continue
            else:
                best_f = ""
                best_gamma = 0.0
                best_pos_z = ""
                best_true_is_u = True
                pos_records = [rec for rec in comp_records if label_view.get(rec.record_id, 0.0) > 0.01]
                neg_records = [rec for rec in comp_records if label_view.get(rec.record_id, 0.0) < -0.01]
                if not pos_records or not neg_records:
                    continue

            for r1 in pos_records[-6:]:
                if candidates_checked >= self.max_candidate_checks:
                    break
                for r2 in neg_records[-6:]:
                    if candidates_checked >= self.max_candidate_checks:
                        break
                    self.cumulative_witness_visits += 1
                    pair_key = tuple(sorted((r1.record_id, r2.record_id)))
                    self.unique_witness_pair_ids.add((pair_key[0], pair_key[1]))

                    d1_view = float(label_view[r1.record_id])
                    d2_view = float(label_view[r2.record_id])
                    contrast_mag = (
                        float(best_gamma) + 0.5 * abs(d1_view - d2_view)
                        if self.proposer_mode == "conditional_gain"
                        else abs(d1_view - d2_view)
                    )

                    differing_fields = [
                        k for k in r1.public_readout_z
                        if r1.public_readout_z.get(k, "unknown") != "unknown"
                        and r2.public_readout_z.get(k, "unknown") != "unknown"
                        and r1.public_readout_z.get(k) != r2.public_readout_z.get(k)
                    ]
                    if self.proposer_mode == "conditional_gain" and best_f and best_f not in differing_fields:
                        differing_fields = [best_f, *differing_fields]
                    ordered_fields = tuple(
                        differing_fields
                        + [k for k in r1.readout_fields if k not in differing_fields]
                    )[:4] or r1.readout_fields

                    cond_f = best_f if (self.proposer_mode == "conditional_gain" and best_f) else (
                        differing_fields[0] if differing_fields else ""
                    )
                    raw_cond_v = r1.public_readout_z.get(cond_f, "") if cond_f else ""
                    cond_v = best_pos_z if (self.proposer_mode == "conditional_gain" and best_pos_z) else (
                        PublicBenchmarkBinder.canonical_slot_predicate_value(cond_f, raw_cond_v)
                        if (cond_f and raw_cond_v)
                        else ""
                    )
                    if self.proposer_mode == "conditional_gain":
                        true_branch_tmpl = r1.u_policy_id if best_true_is_u else r1.v_policy_id
                        false_branch_tmpl = r1.v_policy_id if best_true_is_u else r1.u_policy_id
                    else:
                        true_branch_tmpl = r1.u_policy_id if d1_view > 0 else r1.v_policy_id
                        false_branch_tmpl = r1.v_policy_id if d1_view > 0 else r1.u_policy_id
                    use_case = bool(cond_f and cond_v and true_branch_tmpl != false_branch_tmpl)

                    # 1. Full original trace program candidate with canonical Case(z: u, v) branches
                    body_cap = max(1, self.max_probe_steps - (1 if use_case else 0))
                    full_steps = (r1.rebound_trace_templates or r1.step_templates)[:body_cap]
                    full_fields = r1.rebound_trace_fields[:4] or ordered_fields
                    full_prog = ExecutableProgram(
                        program_id=_short_hash(
                            (self.benchmark, full_steps, full_fields, cond_f if use_case else "", cond_v if use_case else "", true_branch_tmpl, false_branch_tmpl)
                        ),
                        benchmark=self.benchmark,
                        step_templates=full_steps,
                        readout_fields=full_fields,
                        origin="witness_full",
                        parent_ids=(r1.program_id, r2.program_id),
                        case_condition_field=cond_f if use_case else "",
                        case_condition_value=cond_v if use_case else "",
                        case_true_template=true_branch_tmpl if use_case else "",
                        case_false_template=false_branch_tmpl if use_case else "",
                    )
                    candidates_constructed += 1
                    candidates_checked += 1
                    candidates_scored += 1
                    candidates.append((contrast_mag + 0.1, full_prog))
                    if candidates_checked >= self.max_candidate_checks:
                        break

                    # 2. AST consumer-graph Delete transformation
                    target_del_field = full_fields[-1] if len(full_fields) > 1 else ""
                    if target_del_field and target_del_field != cond_f:
                        del_prog = self._def_use_delete(full_prog, target_del_field)
                        candidates_constructed += 1
                        candidates_checked += 1
                        if del_prog is not None:
                            candidates_scored += 1
                            candidates.append((contrast_mag + 0.2 * len(differing_fields), del_prog))
                    if candidates_checked >= self.max_candidate_checks:
                        break

                    # 3. Alpha-renaming Concat transformation
                    p2_prog = ExecutableProgram(
                        program_id=r2.program_id,
                        benchmark=self.benchmark,
                        step_templates=r2.rebound_trace_templates[: self.max_probe_steps] or r2.step_templates,
                        readout_fields=r2.rebound_trace_fields[:4] or r2.readout_fields,
                    )
                    concat_prog = self._scope_unified_concat(
                        full_prog,
                        p2_prog,
                        differing_fields,
                        self.max_probe_steps,
                        case_condition_value=cond_v,
                        case_true_template=true_branch_tmpl if cond_f else "",
                        case_false_template=false_branch_tmpl if cond_f else "",
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
                permutable_groups=permutable_groups,
                permutable_records=permutable_records,
            )
        )
        return added
