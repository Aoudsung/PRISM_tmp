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
import json
import math
import random
import re
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DCC_CONTRACT_VERSION = "prism-dcc-v2-unified-r3"
TAIL_POLICY_VERSION = "frozen-prior-tail-v2"
GRAMMAR_VERSION = "typed-ast-v3"
RESPONSE_SCHEMA_VERSION = "typed-field-slots-v3"
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

# Canonical midpoints of 8 uniform wall-clock cost ratio bins in [0, 1]
TAIL_COST_BIN_CENTERS = tuple((i + 0.5) / TAIL_COST_BINS for i in range(TAIL_COST_BINS))


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
    observed_y: bool | None = None
    tail_seconds_cost_ratio: float | None = None
    total_seconds_cost_ratio: float | None = None
    censored: bool = False
    prefix_node_id: str | None = None
    episode_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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

    @classmethod
    def encode_typed_field_slots(cls, public_fields: Mapping[str, str]) -> tuple[int, int, int, int]:
        """Encode concrete program-read public fields into 4 typed slot indices in {0, 1, 2, 3} (fixes F09).

        Slot 0 (visibility / match):
          0=unknown, 1=no/low, 2=yes/high, 3=error
        Slot 1 (entity / inventory / identifier binding):
          0=unknown, 1=none/count_0, 2=bound_single, 3=bound_multiple
        Slot 2 (container / page / prerequisite state):
          0=unknown, 1=closed/search_home/pending, 2=open_empty/search_results/item_subpage, 3=open_with_items/item_detail/ready
        Slot 3 (treatment / attribute / workspace artifact state):
          0=unknown, 1=raw/none/unchanged, 2=inspected/selected, 3=clean_hot_cool_examined/verified_write
        """
        # Slot 0
        vis = public_fields.get("target_object_visible", public_fields.get("price_rating_match", "unknown"))
        if public_fields.get("prerequisite_status") == "error":
            s0 = 3
        elif vis in ("yes", "high"):
            s0 = 2
        elif vis in ("no", "low"):
            s0 = 1
        else:
            s0 = 0

        # Slot 1
        inv = public_fields.get("inventory_object", public_fields.get("current_asin", public_fields.get("bound_identifiers", "unknown")))
        if inv == "unknown":
            s1 = 0
        elif inv in ("none", "count_0"):
            s1 = 1
        elif inv in ("count_2", "count_3", "count_4") or "," in inv:
            s1 = 3
        else:
            s1 = 2

        # Slot 2
        st = public_fields.get("open_receptacle_state", public_fields.get("page_type", public_fields.get("prerequisite_status", "unknown")))
        if st == "unknown":
            s2 = 0
        elif st in ("closed", "search_home", "pending", "error"):
            s2 = 1
        elif st in ("open_empty", "search_results", "item_subpage"):
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
        elif tr in ("clean", "hot", "cool", "examined", "verified_write"):
            s3 = 3
        elif tr not in ("raw", "none", "unchanged") or insp not in ("none", "unknown"):
            s3 = 2
        else:
            s3 = 1

        return (int(s0), int(s1), int(s2), int(s3))

    @classmethod
    def canonical_slot_predicate_value(cls, condition_field: str, raw_value: str) -> str:
        """Normalize a field value into the exact categorical predicate event supported by both AST execution and K_phi."""
        if raw_value == "unknown":
            return "unknown"
        slot_idx, val_idx = cls.condition_slot_and_value(condition_field, raw_value)
        slot_names = (
            ("unknown", "no_or_low", "yes_or_high", "error"),
            ("unknown", "none", "bound_single", "bound_multiple"),
            ("unknown", "closed_or_pending", "open_empty_or_subpage", "open_with_items_or_ready"),
            ("unknown", "raw_or_unchanged", "inspected_or_selected", "treated_or_verified_write"),
        )
        return slot_names[slot_idx][val_idx]

    @classmethod
    def condition_slot_and_value(cls, condition_field: str, condition_equals: str) -> tuple[int, int]:
        """Map a Case(condition_field, condition_equals) predicate to its corresponding (slot_idx, value_idx)."""
        canonical_map = {
            (0, "unknown"): 0, (0, "no_or_low"): 1, (0, "yes_or_high"): 2, (0, "error"): 3,
            (1, "unknown"): 0, (1, "none"): 1, (1, "bound_single"): 2, (1, "bound_multiple"): 3,
            (2, "unknown"): 0, (2, "closed_or_pending"): 1, (2, "open_empty_or_subpage"): 2, (2, "open_with_items_or_ready"): 3,
            (3, "unknown"): 0, (3, "raw_or_unchanged"): 1, (3, "inspected_or_selected"): 2, (3, "treated_or_verified_write"): 3,
        }
        if condition_field in ("target_object_visible", "price_rating_match"):
            s_idx = 0
        elif condition_field in ("inventory_object", "current_asin", "bound_identifiers"):
            s_idx = 1
        elif condition_field in ("open_receptacle_state", "page_type", "prerequisite_status"):
            s_idx = 2
        else:
            s_idx = 3
        if (s_idx, condition_equals) in canonical_map:
            return s_idx, canonical_map[(s_idx, condition_equals)]
        dummy = {condition_field: condition_equals}
        slots = cls.encode_typed_field_slots(dummy)
        return s_idx, slots[s_idx]

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
        changed_fields = sum(1 for k, v in post_fields.items() if pre_fields.get(k, "unknown") != v)
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
        local_env: Mapping[str, Mapping[str, str]] | None = None,
        typed_arguments: Mapping[str, str] | None = None,
    ) -> str | None:
        """Bind a symbolic step template or AST Call node (with `typed_arguments`) to a concrete admissible action."""
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
            elif re.match(r"^[a-z]+_\d+$", slot):
                literal_entity = slot.replace("_", " ")
                resolved = resolved.replace(f"{{{slot}}}", literal_entity)
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
                    slot_type = re.sub(r"_\d+$", "", slot_matches[0]).replace("_id", "").replace("target_", "").replace("source_", "")
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
                return asin_clicks[0] if asin_clicks else None
            for cand in all_candidates:
                if cand.lower() == template.lower():
                    return cand
            return None

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

        # Dedicated amortized fixed-tail head H_{phi,m}^{pi_0}(Y, C_tail | Y=1, h, g, beta) (fixes F02, F11)
        self.tail_pi0_success_weight = nn.Parameter(
            torch.zeros(self.ensemble_size, member_in_dim)
        )
        self.tail_pi0_success_bias = nn.Parameter(torch.zeros(self.ensemble_size))
        self.tail_pi0_cost_bins_weight = nn.Parameter(
            torch.zeros(self.ensemble_size, member_in_dim, TAIL_COST_BINS)
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
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Canonical tail-state encoder shared identically by TailExample training and Q leaf evaluation (fixes D)."""
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
        prior_action_logprob: float = 0.0,
        prior_success_logit: float = 0.0,
        program: ExecutableProgram | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Encode c = (frozen_context, frozen_action, public_fields, resource_before) into a 64-dim vector.

        Fixes F01 and F10:
          - First 32 dims: frozen backbone semantic representation of (goal, history).
          - Next 16 dims: frozen backbone semantic embedding of `action_text`.
          - Final 16 dims: pre-action ResourceState and typed public field slot indicators
            shared identically by NLL training and Q planning (no fake candidate-origin constants).
        """
        vec = np.zeros(CONTEXT_DIM, dtype=np.float32)

        # 0..31: Frozen history representation
        hist_seq = (
            list(frozen_context_vec)[:FROZEN_HISTORY_DIM]
            if frozen_context_vec
            else list(cls.deterministic_semantic_projection(f"{goal_text}\n{_canonical_json(dict(public_fields))}", FROZEN_HISTORY_DIM))
        )
        vec[: len(hist_seq)] = np.asarray(hist_seq, dtype=np.float32)

        # 32..47: Frozen action representation
        act_seq = (
            list(frozen_action_vec)[:FROZEN_ACTION_DIM]
            if frozen_action_vec
            else list(cls.deterministic_semantic_projection(action_text, FROZEN_ACTION_DIM))
        )
        vec[FROZEN_HISTORY_DIM : FROZEN_HISTORY_DIM + len(act_seq)] = np.asarray(act_seq, dtype=np.float32)

        # 48..63: Pre-action resource state & typed public field slots (identical in training and planning)
        rem_sec = float(min(1.0, max(0.0, remaining_budget_ratio)))
        spent_sec = float(1.0 - rem_sec if spent_seconds_ratio is None else min(1.0, max(0.0, spent_seconds_ratio)))
        rem_steps = float(min(1.0, max(0.0, remaining_steps_ratio)))
        goal_tokens = set(re.findall(r"[a-z0-9]{3,}", goal_text.lower()))
        act_tokens = set(re.findall(r"[a-z0-9]{2,}", action_text.lower()))
        field_vals = " ".join(f"{k}:{v}" for k, v in sorted(public_fields.items())).lower()
        field_tokens = set(re.findall(r"[a-z0-9_]{2,}", field_vals))

        base_idx = FROZEN_HISTORY_DIM + FROZEN_ACTION_DIM  # 48
        vec[base_idx + 0] = rem_sec
        vec[base_idx + 1] = spent_sec
        vec[base_idx + 2] = rem_steps
        vec[base_idx + 3] = len(goal_tokens & act_tokens) / max(1.0, float(len(goal_tokens)))
        vec[base_idx + 4] = len(goal_tokens & field_tokens) / max(1.0, float(len(goal_tokens)))
        vec[base_idx + 5] = len(act_tokens & field_tokens) / max(1.0, float(len(act_tokens)))

        act_l = action_text.strip().lower()
        vec[base_idx + 6] = 1.0 if act_l.startswith(("look", "inventory", "examine", "search[")) else 0.0
        vec[base_idx + 7] = 1.0 if act_l.startswith(("go to", "open", "click[")) else 0.0
        vec[base_idx + 8] = 1.0 if act_l.startswith(("take", "move", "put", "clean", "heat", "cool", "use")) else 0.0
        vec[base_idx + 9] = 1.0 if ("buy now" in act_l or "claim_done" in act_l or "submit" in act_l) else 0.0

        slot_0, slot_1, slot_2, slot_3 = PublicBenchmarkBinder.encode_typed_field_slots(public_fields)
        vec[base_idx + 10] = (slot_0 - 1.5) / 1.5
        vec[base_idx + 11] = (slot_1 - 1.5) / 1.5
        vec[base_idx + 12] = (slot_2 - 1.5) / 1.5
        vec[base_idx + 13] = (slot_3 - 1.5) / 1.5
        vec[base_idx + 14] = 1.0 if public_fields.get("prerequisite_status") == "error" else 0.0
        vec[base_idx + 15] = 1.0 if public_fields.get("workspace_artifact_state") == "verified_write" else 0.0

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
        """Compute normalized transition T_m(s'|s,c) and emission G_m(o|s',c) in a single pass."""
        c_vec = self._coerce_context(context_feat)
        h_ctx = self.context_encoder(c_vec)  # (C,)
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
        prior_success_logit: float = 0.0,
        remaining_budget_ratio: float | torch.Tensor = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Single-pass per-member operator update, typed field-slot prediction, and transition heads (fixes F13, F22)."""
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
        slot_logits_s = self.field_slot_proj(h_ctx).view(
            self.ensemble_size, self.state_dim, NUM_FIELD_SLOTS, NUM_SLOT_VALUES
        )
        slot_probs_s = F.softmax(slot_logits_s, dim=-1)
        field_slot_probs = torch.einsum("ms,mskv->mkv", next_state_prior, slot_probs_s).clamp(min=1e-8)
        field_slot_probs = field_slot_probs / field_slot_probs.sum(dim=-1, keepdim=True)

        # Strictly private per-member inputs: (M, S + C), never averaging belief across members (fixes F13)
        ctx_expanded = h_ctx.unsqueeze(0).expand(self.ensemble_size, -1)  # (M, C)
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
            budget_low = torch.where(rem_vec <= 0.05, torch.ones_like(rem_vec), torch.full_like(rem_vec, -2.0))
            base_term = torch.stack(
                [torch.full_like(rem_vec, 1.5), torch.full_like(rem_vec, -1.0), budget_low],
                dim=-1,
            )
        termination_probs = F.softmax(base_term + term_res, dim=-1)  # (M, 3)

        cost_res = (member_in * self.member_step_cost_weight).sum(dim=-1) + self.member_step_cost_bias  # (M,)
        cost_ratio = torch.sigmoid(-3.0 + cost_res)  # (M,) wall-clock step cost fraction in (0, 1)

        out_in = torch.cat([member_in, outcome_probs], dim=-1)  # (M, S + C + O)
        y_res = (out_in * self.member_step_outcome_weight).sum(dim=-1) + self.member_step_outcome_bias  # (M,)
        y_prob = torch.sigmoid(float(prior_success_logit) + y_res)  # (M,)

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

    def predict_tail_pi0(
        self,
        belief: torch.Tensor,
        h_ctx: torch.Tensor,
        spent_seconds_ratio: torch.Tensor,
        *,
        prior_success_logit: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Evaluate fixed-tail response distribution H_{phi,m}^{pi_0}(Y, C_tail | Y=1, h, g, beta) per member.

        Fixes F02, F11, F13:
          - Predicts per-member P_m^{pi_0}(Y = 1 | h, g, beta) and conditional 8-bin tail time-cost
            distribution P_m^{pi_0}(C_tail in bin_k | Y = 1, h, g, beta).
          - Computes exact truncated expected utility:
            E_m[U_B] = P_m(Y=1) * sum_k P_m(bin_k | Y=1) * max(0, 1 - spent_m - center_k),
            preserving Cov(Y, C_tail) and respecting the budget ceiling B_seconds.
        """
        ctx_expanded = h_ctx.unsqueeze(0).expand(self.ensemble_size, -1)
        member_in = torch.cat([belief, ctx_expanded], dim=-1)  # (M, S + C)

        y_logit = (member_in * self.tail_pi0_success_weight).sum(dim=-1) + self.tail_pi0_success_bias
        y_prob = torch.sigmoid(float(prior_success_logit) + y_logit)  # (M,)

        bin_logits = (
            torch.einsum("md,mdk->mk", member_in, self.tail_pi0_cost_bins_weight)
            + self.tail_pi0_cost_bins_bias
        )
        bin_probs = F.softmax(bin_logits, dim=-1)  # (M, TAIL_COST_BINS)

        centers = torch.tensor(
            TAIL_COST_BIN_CENTERS, device=belief.device, dtype=belief.dtype
        ).view(1, TAIL_COST_BINS)  # (1, K)
        spent = spent_seconds_ratio.view(self.ensemble_size, 1).clamp(min=0.0, max=1.0)
        rem_after_tail = (1.0 - spent - centers).clamp(min=0.0, max=1.0)  # (M, K)
        cond_utility_given_y1 = (bin_probs * rem_after_tail).sum(dim=-1)  # (M,)
        expected_tail_utility = y_prob * cond_utility_given_y1  # (M,)
        expected_cond_tail_cost = (bin_probs * centers).sum(dim=-1)  # (M,)

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
        frozen_context_vec: Sequence[float] = (),
        frozen_action_vec: Sequence[float] = (),
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
            frozen_context_vec=frozen_context_vec,
            frozen_action_vec=frozen_action_vec,
            device=device,
        )
        res = self.step_and_predict(previous_state, ctx, observed_outcome_bin=outcome_bin)
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
        goal_text: str,
        public_fields: Mapping[str, str],
        belief: torch.Tensor,
        candidate_action: str,
        program: ExecutableProgram | None = None,
        remaining_budget_ratio: float = 1.0,
        frozen_context_vec: Sequence[float] = (),
        frozen_action_vec: Sequence[float] = (),
        frozen_tail_action_vec: Sequence[float] = (),
        action_embedder: Any | None = None,
        prior_action_logprob: float = 0.0,
        prior_success_logit: float = 0.0,
        member_index: int | None = None,
        use_initial_params: bool = False,
    ) -> dict[str, Any]:
        """Compute unified long-horizon value Q_m^rho(g, h, beta) = E_{K_{phi,m}, rho -> pi_0}[U_B(Y, C_seconds)].

        Fixes D and E:
          - Unrolls `program.ast_nodes` with Bayesian state conditioning inside `Case`:
            b_m(s' | z) proportional to b_m(s') * P_m(z | s', c), updating both member beliefs and
            branch public field states before evaluating true/false subtrees.
          - Evaluates leaves via the canonical `encode_tail_state_context` shared with `TailExample` training.
        """
        if use_initial_params and self._initial_state_tensors:
            return self.active_model_view(use_initial_params=True).evaluate_candidate_q(
                goal_text=goal_text,
                public_fields=public_fields,
                belief=belief,
                candidate_action=candidate_action,
                program=program,
                remaining_budget_ratio=remaining_budget_ratio,
                frozen_context_vec=frozen_context_vec,
                frozen_action_vec=frozen_action_vec,
                frozen_tail_action_vec=frozen_tail_action_vec,
                action_embedder=action_embedder,
                prior_action_logprob=prior_action_logprob,
                prior_success_logit=prior_success_logit,
                member_index=member_index,
                use_initial_params=False,
            )

        device = self.transition_logits.device
        init_belief = belief.clone().to(device)
        rem_ratio = float(max(0.0, min(1.0, remaining_budget_ratio)))
        init_spent = torch.full(
            (self.ensemble_size,), 1.0 - rem_ratio, device=device, dtype=torch.float32
        )
        init_alive = torch.ones(self.ensemble_size, device=device, dtype=torch.float32)
        base_fields = dict(public_fields)

        def _leaf_tail_value(
            cur_b: torch.Tensor,
            cur_spent: torch.Tensor,
            cur_alive: torch.Tensor,
            cur_fields: Mapping[str, str],
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            rem_mean = float((1.0 - cur_spent).clamp(min=0.0, max=1.0).mean().item())
            spent_mean = float(cur_spent.clamp(min=0.0, max=1.0).mean().item())
            tail_h_ctx = self.encode_tail_state_context(
                goal_text=goal_text,
                public_fields=cur_fields,
                remaining_budget_ratio=rem_mean,
                spent_seconds_ratio=spent_mean,
                frozen_context_vec=frozen_context_vec,
                frozen_tail_action_vec=frozen_tail_action_vec,
                device=device,
            )
            tail_out = self.predict_tail_pi0(
                cur_b, tail_h_ctx, cur_spent, prior_success_logit=prior_success_logit
            )
            leaf_u = cur_alive * tail_out["expected_tail_utility"]
            leaf_cost = (cur_spent + cur_alive * tail_out["expected_cond_tail_cost"]).clamp(max=1.0)
            return leaf_u, leaf_cost, tail_out["tail_y_prob"]

        def _eval_call_step(
            cur_b: torch.Tensor,
            cur_spent: torch.Tensor,
            cur_alive: torch.Tensor,
            cur_fields: Mapping[str, str],
            act_str: str,
            is_first: bool,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            rem_vec = (1.0 - cur_spent).clamp(min=0.0, max=1.0)
            if is_first and frozen_action_vec:
                act_vec: Sequence[float] = frozen_action_vec
            elif action_embedder is not None:
                act_vec = action_embedder(act_str)
            else:
                act_vec = ()
            ctx = self.encode_context_vector(
                goal_text=goal_text,
                public_fields=cur_fields,
                action_text=act_str,
                remaining_budget_ratio=float(rem_vec.mean().item()),
                spent_seconds_ratio=float(cur_spent.mean().item()),
                frozen_context_vec=frozen_context_vec,
                frozen_action_vec=act_vec,
                prior_success_logit=prior_success_logit,
                device=device,
            )
            preds = self.step_and_predict(
                cur_b,
                ctx,
                observed_outcome_bin=None,
                prior_success_logit=prior_success_logit,
                remaining_budget_ratio=rem_vec,
            )
            next_spent = (cur_spent + preds["cost_ratio"]).clamp(min=0.0, max=1.0)
            p_live = preds["termination_probs"][:, 0]
            p_nat_term = preds["termination_probs"][:, 1]
            y_prob = preds["terminal_success_prob"]
            rem_after = (1.0 - next_spent).clamp(min=0.0, max=1.0)
            step_u = cur_alive * p_nat_term * y_prob * rem_after
            next_alive = cur_alive * p_live
            return (
                preds["next_state_prior"],
                next_spent,
                next_alive,
                step_u,
                preds["slot_probs_per_state"],
            )

        def _eval_ast_sequence(
            nodes: Sequence[ASTNode],
            cur_b: torch.Tensor,
            cur_spent: torch.Tensor,
            cur_alive: torch.Tensor,
            cur_slot_probs_s: torch.Tensor | None,
            cur_fields: dict[str, str],
            calls_used: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if not nodes or calls_used >= DEFAULT_MAX_PROBE_STEPS + 1:
                return _leaf_tail_value(cur_b, cur_spent, cur_alive, cur_fields)

            head = nodes[0]
            rest = nodes[1:]
            if head.op == "Stop":
                return _leaf_tail_value(cur_b, cur_spent, cur_alive, cur_fields)

            if head.op == "Seq":
                return _eval_ast_sequence(
                    tuple(head.children) + tuple(rest),
                    cur_b,
                    cur_spent,
                    cur_alive,
                    cur_slot_probs_s,
                    cur_fields,
                    calls_used,
                )

            if head.op == "Read":
                return _eval_ast_sequence(
                    rest, cur_b, cur_spent, cur_alive, cur_slot_probs_s, cur_fields, calls_used
                )

            if head.op == "Call":
                act_str = candidate_action if calls_used == 0 else (head.tool_or_action or candidate_action)
                nb, n_spent, n_alive, step_u, n_slots_s = _eval_call_step(
                    cur_b, cur_spent, cur_alive, cur_fields, act_str, is_first=(calls_used == 0)
                )
                rem_u, end_cost, tail_y = _eval_ast_sequence(
                    rest, nb, n_spent, n_alive, n_slots_s, cur_fields, calls_used + 1
                )
                return step_u + rem_u, end_cost, tail_y

            if head.op == "Case":
                slot_idx, val_idx = PublicBenchmarkBinder.condition_slot_and_value(
                    head.condition_field, head.condition_equals
                )
                if cur_slot_probs_s is not None:
                    # Bayesian belief conditioning: b_m(s' | z) proportional to b_m(s') * P_m(z | s', c)
                    lik_true = cur_slot_probs_s[:, :, slot_idx, val_idx].clamp(min=1e-6, max=1.0 - 1e-6)  # (M, S)
                    lik_false = (1.0 - lik_true).clamp(min=1e-6, max=1.0 - 1e-6)
                    unnorm_b_true = cur_b * lik_true
                    p_true = unnorm_b_true.sum(dim=-1).clamp(min=0.0, max=1.0)  # (M,)
                    b_true = unnorm_b_true / unnorm_b_true.sum(dim=-1, keepdim=True).clamp(min=1e-8)

                    unnorm_b_false = cur_b * lik_false
                    p_false = (1.0 - p_true).clamp(min=0.0, max=1.0)
                    b_false = unnorm_b_false / unnorm_b_false.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                else:
                    obs_val = cur_fields.get(head.condition_field, "unknown")
                    if obs_val != "unknown":
                        lhs = PublicBenchmarkBinder.canonical_slot_predicate_value(head.condition_field, obs_val)
                        rhs = PublicBenchmarkBinder.canonical_slot_predicate_value(head.condition_field, head.condition_equals)
                        p_t_val = 1.0 if lhs == rhs else 0.0
                    else:
                        p_t_val = 0.5
                    p_true = torch.full((self.ensemble_size,), p_t_val, device=device, dtype=torch.float32)
                    p_false = 1.0 - p_true
                    b_true = cur_b
                    b_false = cur_b

                fields_true = {**cur_fields, head.condition_field: head.condition_equals}
                alt_val = "no_or_low" if head.condition_equals == "yes_or_high" else "none"
                fields_false = {**cur_fields, head.condition_field: alt_val}

                true_child = (head.children[0],) if len(head.children) >= 1 else (ASTNode(op="Stop"),)
                false_child = (head.children[1],) if len(head.children) >= 2 else (ASTNode(op="Stop"),)
                u_t, c_t, y_t = _eval_ast_sequence(
                    true_child + tuple(rest),
                    b_true,
                    cur_spent,
                    cur_alive * p_true,
                    cur_slot_probs_s,
                    fields_true,
                    calls_used,
                )
                u_f, c_f, y_f = _eval_ast_sequence(
                    false_child + tuple(rest),
                    b_false,
                    cur_spent,
                    cur_alive * p_false,
                    cur_slot_probs_s,
                    fields_false,
                    calls_used,
                )
                return u_t + u_f, p_true * c_t + p_false * c_f, p_true * y_t + p_false * y_f

            return _eval_ast_sequence(
                rest, cur_b, cur_spent, cur_alive, cur_slot_probs_s, cur_fields, calls_used
            )

        ast_seq = (
            program.ast_nodes
            if (program is not None and program.ast_nodes)
            else (ASTNode(op="Call", tool_or_action=candidate_action), ASTNode(op="Stop"))
        )
        q_members, end_spent, tail_y_members = _eval_ast_sequence(
            ast_seq, init_belief, init_spent, init_alive, None, base_fields, 0
        )
        q_members = q_members.clamp(min=0.0, max=1.0)

        if member_index is not None:
            m_idx = int(member_index) % self.ensemble_size
            q_scalar = float(q_members[m_idx].item())
        else:
            q_scalar = float(q_members.mean().item())

        return {
            "q_mean": float(q_members.mean().item()),
            "q_selected": q_scalar,
            "q_members": [float(x) for x in q_members.detach().cpu().tolist()],
            "predicted_terminal_success": float(tail_y_members.mean().item()),
            "predicted_cost_ratio": float(end_spent.mean().item()),
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
                remaining_budget_ratio=rem_sec_ratio,
            )
            belief_by_node[tr.transition_id] = preds["post_belief"]
            last_belief = preds["post_belief"]

            # 1. Outcome bin NLL + typed field-slot NLL (fixes F09)
            resp_nll = -torch.log(preds["outcome_probs"][:, obs_bin].clamp(min=1e-8))
            target_slots = PublicBenchmarkBinder.encode_typed_field_slots(tr.typed_response)
            slot_nll = torch.zeros_like(resp_nll)
            for s_idx, v_idx in enumerate(target_slots):
                slot_nll = slot_nll - 0.25 * torch.log(
                    preds["field_slot_probs"][:, s_idx, v_idx].clamp(min=1e-8)
                )

            # 2. Termination category NLL
            term_idx = (
                1 if tr.termination == "native_terminal"
                else 2 if tr.termination == "task_budget_terminal"
                else 0
            )
            term_nll = -torch.log(preds["termination_probs"][:, term_idx].clamp(min=1e-8))

            # 3. Wall-clock step cost ratio loss (aligned with seconds/budget_seconds, fixes F11)
            target_cost = float(
                min(1.0, max(0.0, tr.resource_delta.get("seconds_ratio", tr.resource_delta.get("cost_ratio", 0.01))))
            )
            cost_nll = 0.5 * (preds["cost_ratio"] - target_cost).square()

            # 4. Terminal outcome NLL on terminal transitions (masked when None)
            if tr.native_outcome is not None and term_done:
                observed_outcomes += 1
                y_val = 1.0 if tr.native_outcome else 0.0
                p_y = preds["terminal_success_prob"].clamp(min=1e-6, max=1.0 - 1e-6)
                y_nll = -(y_val * torch.log(p_y) + (1.0 - y_val) * torch.log(1.0 - p_y))
            else:
                y_nll = torch.zeros_like(resp_nll)

            total_member_nll = total_member_nll + (resp_nll + slot_nll + 0.5 * term_nll + cost_nll + y_nll)

        # 5. Supervise fixed-tail head H_{phi,m}^{pi_0}(Y, C_tail | Y=1) using exact fork-tree `prefix_node_id`
        # and the canonical `encode_tail_state_context` shared with Q leaf evaluation (fixes C & D).
        tail_member_nll = torch.zeros(self.ensemble_size, device=device, dtype=torch.float32)
        for tex in tail_examples:
            if tex.censored or tex.observed_y is None:
                continue
            tail_supervised_count += 1
            res_b = ResourceState.from_dict(tex.resource_before)
            h_ctx_t = self.encode_tail_state_context(
                goal_text=tex.goal_text,
                public_fields=tex.pre_public_fields,
                remaining_budget_ratio=res_b.remaining_seconds_ratio,
                spent_seconds_ratio=res_b.spent_seconds_ratio,
                remaining_steps_ratio=res_b.remaining_steps_ratio,
                frozen_context_vec=tex.frozen_context_vec,
                device=device,
            )
            spent_vec = torch.full(
                (self.ensemble_size,), res_b.spent_seconds_ratio, device=device, dtype=torch.float32
            )
            if tex.prefix_node_id in (None, "", "__root__"):
                tex_belief = root_belief
            else:
                tex_belief = belief_by_node.get(str(tex.prefix_node_id), root_belief)
            tail_preds = self.predict_tail_pi0(tex_belief, h_ctx_t, spent_vec)
            p_y = tail_preds["tail_y_prob"].clamp(min=1e-6, max=1.0 - 1e-6)
            y_target = 1.0 if tex.observed_y else 0.0
            t_loss = -(y_target * torch.log(p_y) + (1.0 - y_target) * torch.log(1.0 - p_y))
            if tex.observed_y and tex.tail_seconds_cost_ratio is not None:
                c_ratio = float(min(0.999, max(0.0, tex.tail_seconds_cost_ratio)))
                bin_idx = min(TAIL_COST_BINS - 1, int(c_ratio * TAIL_COST_BINS))
                p_bin = tail_preds["tail_cost_bin_probs"][:, bin_idx].clamp(min=1e-8)
                t_loss = t_loss - torch.log(p_bin)
            tail_member_nll = tail_member_nll + t_loss

        n_steps = max(1, len(transitions))
        combined_member_loss = (total_member_nll / float(n_steps)) + (
            tail_member_nll / float(max(1, tail_supervised_count)) if tail_supervised_count > 0 else 0.0
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
                    contrast_mag = abs(d1_view - d2_view)

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

                    cond_f = differing_fields[0] if differing_fields else ""
                    raw_cond_v = r1.public_readout_z.get(cond_f, "") if cond_f else ""
                    cond_v = (
                        PublicBenchmarkBinder.canonical_slot_predicate_value(cond_f, raw_cond_v)
                        if (cond_f and raw_cond_v)
                        else ""
                    )
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
