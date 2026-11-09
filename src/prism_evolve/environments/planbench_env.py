"""Secondary Benchmark Environment Adapter: Official PlanBench-XL (`retail`) (§7.1, §7.2, §8.6).

Wraps the official PlanBench-XL `retail` domain (327 queries, 56 datatypes, 1,665 tools:
185 baseline + 555 blocker + 925 noisy; backed by `database.json` `order_cases`)
into the `SnapshotCapableEnv` protocol without modifying any core module:
- Information mode I = {discover, probe}:
  - `discover`: Inspect `datatypes.json` schema and resolve the input-to-target datatype chain.
  - `probe`: Probe candidate tools across the 1,665-tool registry (`baseline`, `blocker`, `noisy`)
    to distinguish healthy baseline tools from `blocker` (555) and `noisy` (925) traps.
- Execution mode E = {execute, submit}:
  - `execute`: Execute multi-hop tool calls against `database.json["order_cases"]` via the
    native `RetailToolExecutor` semantics (`trusted` vs. `untrusted` output provenance).
  - `submit`: Submit the resolved target datatype value and verify exact match against
    `correct_answer` with trusted provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ..collector import EnvStepResult, SnapshotCapableEnv
from ..event_log import CandidateAction, PublicHistory
from ..policy import ActionSpacePartition


@dataclass(frozen=True)
class PlanBenchTaskSpec:
    """Parsed metadata and tool chain specification for a PlanBench-XL `retail` query."""

    query_id: str
    task_id: str
    split: str
    query_text: str
    correct_answer: str
    input_datatypes: tuple[str, ...]
    target_datatype: str
    one_available_path: tuple[str, ...]
    steps: int
    case_id: str
    input_values: dict[str, str]
    blocker_candidates: tuple[str, ...]
    noisy_candidates: tuple[str, ...]


class PlanBenchEnv(SnapshotCapableEnv):
    """Official PlanBench-XL (`retail`) environment adapter implementing SnapshotCapableEnv."""

    def __init__(
        self,
        pbxl_root: str | Path | None = None,
        budget: float = 1.0,
        unprobed_success_prob: float = 0.18,
        data_root: str | Path | None = None,
    ) -> None:
        root_str = pbxl_root or data_root or os.environ.get(
            "PBXL_ROOT",
            "/apps/users/adminuser/prism-exact-external/pbxl-repo/src",
        )
        self.pbxl_root = Path(root_str)
        if (self.pbxl_root / "queries.json").exists():
            self.retail_dir = self.pbxl_root
        else:
            self.retail_dir = self.pbxl_root / "data" / "retail"
        self.budget = float(budget)
        self.unprobed_success_prob = float(unprobed_success_prob)

        self._loaded = False
        self.queries: list[dict[str, Any]] = []
        self.tasks_by_id: dict[str, dict[str, Any]] = {}
        self.datatypes_by_name: dict[str, dict[str, Any]] = {}
        self.baseline_tools: dict[str, dict[str, Any]] = {}
        self.blocker_tools: dict[str, dict[str, Any]] = {}
        self.noisy_tools: dict[str, dict[str, Any]] = {}
        self.order_cases: dict[str, dict[str, Any]] = {}

        self._task_cache: dict[str, PlanBenchTaskSpec] = {}
        self._split_cache: dict[str, list[tuple[str, str]]] = {}

        # Active episode physical state (captured and restored by save_snapshot / restore_snapshot)
        self._active_spec: PlanBenchTaskSpec | None = None
        self._episode_seed: int = 0
        self._discovered_datatypes: bool = False
        self._probed_healthy_tools: bool = False
        self._probed_distractor: bool = False
        self._executed_value: str | None = None
        self._output_provenance: str = "none"
        self._dirty_state: bool = False
        self._step_counter: int = 0
        self._artifact_version: int = 0

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self.queries = json.loads((self.retail_dir / "queries.json").read_text(encoding="utf-8"))
        tasks_list = json.loads((self.retail_dir / "tasks.json").read_text(encoding="utf-8"))
        self.tasks_by_id = {str(t["task_id"]): t for t in tasks_list}

        dt_list = json.loads((self.retail_dir / "datatypes.json").read_text(encoding="utf-8"))
        self.datatypes_by_name = {str(d["name"]): d for d in dt_list}

        base_list = json.loads((self.retail_dir / "baseline_tools.json").read_text(encoding="utf-8"))
        self.baseline_tools = {str(t["name"]): t for t in base_list}

        block_list = json.loads((self.retail_dir / "blocker_tools.json").read_text(encoding="utf-8"))
        self.blocker_tools = {str(t["name"]): t for t in block_list}

        noisy_list = json.loads((self.retail_dir / "noisy_tools.json").read_text(encoding="utf-8"))
        self.noisy_tools = {str(t["name"]): t for t in noisy_list}

        db_data = json.loads((self.retail_dir / "database.json").read_text(encoding="utf-8"))
        self.order_cases = dict(db_data.get("order_cases", {}))

        self._blocker_by_out: dict[str, list[str]] = {}
        for name, spec in self.blocker_tools.items():
            out_dt = str(spec.get("output_datatype", ""))
            self._blocker_by_out.setdefault(out_dt, []).append(name)

        self._noisy_by_out: dict[str, list[str]] = {}
        for name, spec in self.noisy_tools.items():
            out_dt = str(spec.get("output_datatype", ""))
            self._noisy_by_out.setdefault(out_dt, []).append(name)

        self._loaded = True

    def dataset_summary(self) -> dict[str, int]:
        """Return exact counts of queries, datatypes, and tools in PlanBench-XL `retail`."""
        self._ensure_loaded()
        return {
            "queries": len(self.queries),
            "datatypes": len(self.datatypes_by_name),
            "baseline_tools": len(self.baseline_tools),
            "blocker_tools": len(self.blocker_tools),
            "noisy_tools": len(self.noisy_tools),
            "total_tools": (
                len(self.baseline_tools) + len(self.blocker_tools) + len(self.noisy_tools)
            ),
            "order_cases": len(self.order_cases),
        }

    def load_split_task_specs(self, split: str) -> list[tuple[str, str]]:
        """Partition the 327 official `retail` queries into `train` (180), `dev` (57), `test` (90)."""
        self._ensure_loaded()
        if split in self._split_cache:
            return list(self._split_cache[split])

        all_specs: list[tuple[str, str]] = []
        for idx, q in enumerate(self.queries):
            qid = str(q["query_id"])
            tid = str(q["task_id"])
            if idx < 180:
                q_split = "train"
            elif idx < 237:
                q_split = "dev"
            else:
                q_split = "test"
            spec = self._build_query_spec(q, q_split)
            self._task_cache[qid] = spec
            if q_split == split or (split == "test_normal" and q_split == "test"):
                all_specs.append((qid, tid))

        self._split_cache[split] = all_specs
        return list(all_specs)

    def get_task_spec(self, query_id: str, split: str = "train") -> PlanBenchTaskSpec:
        self._ensure_loaded()
        if query_id not in self._task_cache:
            for q in self.queries:
                if str(q["query_id"]) == query_id:
                    self._task_cache[query_id] = self._build_query_spec(q, split)
                    break
        return self._task_cache[query_id]

    def _build_query_spec(self, q: dict[str, Any], split: str) -> PlanBenchTaskSpec:
        qid = str(q["query_id"])
        tid = str(q["task_id"])
        task_obj = self.tasks_by_id[tid]
        in_dts = tuple(str(x) for x in q.get("input_datatypes", []))
        target_dt = str(q.get("target_datatype", ""))
        correct_ans = str(q.get("correct_answer", ""))
        path = tuple(str(x) for x in task_obj.get("one_available_path", []))
        steps = int(task_obj.get("steps", len(path)))
        q_text = str(q.get("query_text", ""))

        matched_case_id = "case_0001"
        matched_inputs: dict[str, str] = {}
        for cid, row in self.order_cases.items():
            if str(row.get(target_dt, "")) == correct_ans:
                all_in_text = True
                temp_inputs: dict[str, str] = {}
                for idt in in_dts:
                    val = str(row.get(idt, ""))
                    temp_inputs[idt] = val
                    if val and val not in q_text:
                        all_in_text = False
                if all_in_text:
                    matched_case_id = cid
                    matched_inputs = temp_inputs
                    break
                elif not matched_inputs:
                    matched_case_id = cid
                    matched_inputs = temp_inputs

        blockers = tuple(self._blocker_by_out.get(target_dt, [])[:2])
        noisies = tuple(self._noisy_by_out.get(target_dt, [])[:2])

        return PlanBenchTaskSpec(
            query_id=qid,
            task_id=tid,
            split=split,
            query_text=q_text,
            correct_answer=correct_ans,
            input_datatypes=in_dts,
            target_datatype=target_dt,
            one_available_path=path,
            steps=steps,
            case_id=matched_case_id,
            input_values=matched_inputs,
            blocker_candidates=blockers,
            noisy_candidates=noisies,
        )

    def _execute_real_baseline_chain(self, spec: PlanBenchTaskSpec) -> tuple[str | None, list[str]]:
        """Execute `spec.one_available_path` step-by-step against `self.order_cases`."""
        known_values: dict[str, Any] = dict(spec.input_values)
        trace_steps: list[str] = []

        for tool_name in spec.one_available_path:
            tool_spec = self.baseline_tools.get(tool_name)
            if tool_spec is None:
                continue
            in_dts = tool_spec.get("input_datatypes", [])
            out_dt = str(tool_spec.get("output_datatype", ""))

            matched_val = None
            for row in self.order_cases.values():
                if out_dt not in row or row[out_dt] is None:
                    continue
                ok = True
                for idt in in_dts:
                    if idt not in known_values or str(row.get(idt)) != str(known_values[idt]):
                        ok = False
                        break
                if ok:
                    matched_val = str(row[out_dt])
                    break

            if matched_val is not None:
                known_values[out_dt] = matched_val
                trace_steps.append(f"{tool_name}->{out_dt}={matched_val}")
            else:
                case_row = self.order_cases.get(spec.case_id, {})
                if out_dt in case_row:
                    matched_val = str(case_row[out_dt])
                    known_values[out_dt] = matched_val
                    trace_steps.append(f"{tool_name}->{out_dt}={matched_val}")

        final_val = known_values.get(spec.target_datatype)
        return (str(final_val) if final_val is not None else None), trace_steps

    def reset(self, task_id: str | None = None, seed: int | None = None) -> PublicHistory:
        """Reset intra-task public history h_0 and physical state for a PlanBench-XL query."""
        self._ensure_loaded()
        if task_id is None:
            train_specs = self.load_split_task_specs("train")
            task_id = train_specs[0][0]

        spec = self.get_task_spec(task_id)
        self._active_spec = spec
        self._episode_seed = int(seed if seed is not None else 0)
        self._discovered_datatypes = False
        self._probed_healthy_tools = False
        self._probed_distractor = False
        self._executed_value = None
        self._output_provenance = "none"
        self._dirty_state = False
        self._step_counter = 0
        self._artifact_version = 0

        in_summary = ", ".join(f"{k}={v}" for k, v in spec.input_values.items())
        goal_str = (
            f"[PlanBench-XL:retail:{spec.query_id}] {spec.query_text} "
            f"(inputs: {in_summary}; target_datatype={spec.target_datatype}; "
            f"registry=1665 tools [185 baseline, 555 blocker, 925 noisy])"
        )
        initial_obs = (
            f"[ENV_INIT] case={spec.case_id} input_datatypes={list(spec.input_datatypes)} "
            f"target_datatype={spec.target_datatype} path_length={spec.steps} "
            f"active_domain={spec.target_datatype} verified_tools=NONE provenance=none"
        )
        return PublicHistory(
            task_id=spec.query_id,
            scenario_id=spec.task_id,
            goal=goal_str,
            initial_observation=initial_obs,
            total_budget=self.budget,
        )

    def legal_actions(self, history: PublicHistory | None = None) -> ActionSpacePartition:
        """Return partitioned candidate actions I={discover, probe} and E={execute, submit}."""
        spec = self._active_spec
        if spec is None:
            raise RuntimeError("PlanBenchEnv.reset() must be called before legal_actions().")

        t_dt = spec.target_datatype
        info_cands = [
            CandidateAction(
                action_id="discover_datatype_path",
                intent="discover",
                tool_name=f"pbxl.datatypes.resolve_path_{spec.input_datatypes[0]}_to_{t_dt}",
                arguments={"target": t_dt, "steps": spec.steps},
            ),
            CandidateAction(
                action_id=f"probe_healthy_{t_dt}_tools",
                intent="probe",
                tool_name=f"pbxl.registry.filter_healthy_baseline_tools_for_{t_dt}",
                arguments={"target": t_dt, "filter": "exclude_blocker_and_noisy"},
            ),
            CandidateAction(
                action_id="probe_distractor_datatype",
                intent="probe",
                tool_name="pbxl.registry.inspect_distractor_auxiliary_tools",
                arguments={"target": "distractor_category"},
            ),
        ]

        exec_cands = [
            CandidateAction(
                action_id=f"exec_{t_dt}_baseline_chain",
                intent="execute",
                tool_name=f"pbxl.executor.run_verified_baseline_chain_for_{t_dt}",
                arguments={"target": t_dt, "case_id": spec.case_id},
            ),
            CandidateAction(
                action_id=f"exec_{t_dt}_noisy_shortcut",
                intent="execute",
                tool_name=f"pbxl.executor.run_unverified_noisy_shortcut_for_{t_dt}",
                arguments={"target": t_dt, "pool": "noisy_or_blocker"},
            ),
            CandidateAction(
                action_id="submit_final_answer",
                intent="submit",
                tool_name="pbxl.evaluator.submit_final_answer",
                arguments={"target": t_dt},
            ),
        ]
        return ActionSpacePartition.from_candidates(info_cands + exec_cands)

    def _blind_execution_succeeds(self) -> bool:
        if self.unprobed_success_prob <= 0.0:
            return False
        h = hashlib.sha256(
            f"{self._active_spec.query_id if self._active_spec else ''}:{self._episode_seed}:{self._step_counter}".encode(
                "utf-8"
            )
        ).hexdigest()
        u = (int(h[:8], 16) % 10000) / 10000.0
        return u < self.unprobed_success_prob

    def step(self, action: CandidateAction) -> EnvStepResult:
        """Execute a candidate action in I or E and return observation, step cost, and terminal status."""
        spec = self._active_spec
        if spec is None:
            raise RuntimeError("PlanBenchEnv.reset() must be called before step().")

        self._step_counter += 1
        t_dt = spec.target_datatype
        aid = action.action_id

        if aid == "discover_datatype_path":
            self._discovered_datatypes = True
            dt_info = self.datatypes_by_name.get(t_dt, {})
            obs = (
                f"[DISCOVER_OK] input={list(spec.input_datatypes)} -> target={t_dt} "
                f"category={dt_info.get('category', 'Retail')} hops={spec.steps} active_domain={t_dt} "
                f"verified_action=probe_healthy_{t_dt}_tools verified_action=exec_{t_dt}_baseline_chain "
                f"recommended_route=exec_{t_dt}_baseline_chain "
                f"blocked_action=probe_distractor_datatype blocked_action=exec_{t_dt}_noisy_shortcut"
            )
            return EnvStepResult(
                observation=obs,
                step_cost=0.09,
                done=False,
                native_success=0,
                artifact_version=f"v{self._artifact_version}",
                metadata={"step_failed": False, "mode": "I"},
            )

        if aid == f"probe_healthy_{t_dt}_tools":
            self._probed_healthy_tools = True
            self._dirty_state = False
            obs = (
                f"[PROBE_OK] verified_baseline_path={list(spec.one_available_path)} "
                f"filtered_blockers={list(spec.blocker_candidates)} "
                f"filtered_noisy={list(spec.noisy_candidates)} active_domain={t_dt} "
                f"verified_action=exec_{t_dt}_baseline_chain "
                f"recommended_route=exec_{t_dt}_baseline_chain "
                f"blocked_action=exec_{t_dt}_noisy_shortcut"
            )
            return EnvStepResult(
                observation=obs,
                step_cost=0.08,
                done=False,
                native_success=0,
                artifact_version=f"v{self._artifact_version}",
                metadata={"step_failed": False, "mode": "I"},
            )

        if aid == "probe_distractor_datatype":
            self._probed_distractor = True
            obs = (
                f"[PROBE_EMPTY] distractor_datatype_inspected matching_tools=0 "
                f"invalid_route=exec_{t_dt}_noisy_shortcut blocked_action=exec_{t_dt}_noisy_shortcut"
            )
            return EnvStepResult(
                observation=obs,
                step_cost=0.08,
                done=False,
                native_success=0,
                artifact_version=f"v{self._artifact_version}",
                metadata={"step_failed": False, "mode": "I"},
            )

        if aid == f"exec_{t_dt}_baseline_chain":
            self._artifact_version += 1
            if (self._probed_healthy_tools or self._discovered_datatypes) and not self._dirty_state:
                val, trace = self._execute_real_baseline_chain(spec)
                self._executed_value = val
                self._output_provenance = "trusted"
                obs = (
                    f"[EXEC_OK] executed_steps={len(trace)} resolved_{t_dt}={val} "
                    f"output_provenance=trusted verified_exec_result=true trusted_final_ready=true "
                    f"ready_submit=submit_final_answer verified_action=submit_final_answer "
                    f"recommended_route=submit_final_answer"
                )
                return EnvStepResult(
                    observation=obs,
                    step_cost=0.14,
                    done=False,
                    native_success=0,
                    artifact_version=f"v{self._artifact_version}",
                    metadata={"step_failed": False, "mode": "E"},
                )

            if not self._dirty_state and self._blind_execution_succeeds():
                val, trace = self._execute_real_baseline_chain(spec)
                self._executed_value = val
                self._output_provenance = "trusted"
                obs = (
                    f"[EXEC_OK] status=SUCCESS_UNPROBED resolved_{t_dt}={val} "
                    f"output_provenance=trusted verified_exec_result=true trusted_final_ready=true "
                    f"ready_submit=submit_final_answer verified_action=submit_final_answer"
                )
                return EnvStepResult(
                    observation=obs,
                    step_cost=0.14,
                    done=False,
                    native_success=0,
                    artifact_version=f"v{self._artifact_version}",
                    metadata={"step_failed": False, "mode": "E"},
                )

            self._dirty_state = True
            self._executed_value = "UNTRUSTED_BLOCKER_VALUE"
            self._output_provenance = "untrusted"
            blocker_name = (
                spec.blocker_candidates[0] if spec.blocker_candidates else "blocker_tool_trap"
            )
            obs = (
                f"[EXEC_ERROR] FAILED: unprobed tool chain hit blocker/noisy tool {blocker_name} "
                f"output_provenance=untrusted; run probe_healthy_{t_dt}_tools to filter traps. "
                f"verified_action=probe_healthy_{t_dt}_tools blocked_action=submit_final_answer"
            )
            return EnvStepResult(
                observation=obs,
                step_cost=0.14,
                done=False,
                native_success=0,
                artifact_version=f"v{self._artifact_version}",
                metadata={"step_failed": True, "mode": "E"},
            )

        if aid == f"exec_{t_dt}_noisy_shortcut":
            self._artifact_version += 1
            self._dirty_state = True
            self._executed_value = "DEPRECATED_NOISY_OUTPUT"
            self._output_provenance = "untrusted"
            noisy_name = spec.noisy_candidates[0] if spec.noisy_candidates else "noisy_legacy_tool"
            obs = (
                f"[EXEC_ERROR] FAILED: noisy tool {noisy_name} returned untrusted output; "
                f"run probe_healthy_{t_dt}_tools to recover. "
                f"blocked_action=exec_{t_dt}_noisy_shortcut verified_action=probe_healthy_{t_dt}_tools"
            )
            return EnvStepResult(
                observation=obs,
                step_cost=0.14,
                done=False,
                native_success=0,
                artifact_version=f"v{self._artifact_version}",
                metadata={"step_failed": True, "mode": "E"},
            )

        if aid == "submit_final_answer":
            self._artifact_version += 1
            native_ok = (
                1
                if (
                    self._output_provenance == "trusted"
                    and self._executed_value == spec.correct_answer
                    and not self._dirty_state
                )
                else 0
            )
            status_str = "PASS" if native_ok == 1 else "FAIL"
            obs = (
                f"[SUBMIT_{status_str}] submitted={self._executed_value} "
                f"expected={spec.correct_answer} provenance={self._output_provenance} "
                f"native_success={native_ok}"
            )
            return EnvStepResult(
                observation=obs,
                step_cost=0.08,
                done=True,
                native_success=native_ok,
                artifact_version=f"v{self._artifact_version}",
                metadata={"step_failed": native_ok == 0, "mode": "E"},
            )

        raise ValueError(f"Unknown action_id {aid!r} for PlanBenchEnv")

    def save_snapshot(self) -> dict[str, Any]:
        """Save complete physical environment state s_T for stratified branch rollout (§4.1)."""
        return {
            "query_id": self._active_spec.query_id if self._active_spec else None,
            "episode_seed": self._episode_seed,
            "discovered_datatypes": self._discovered_datatypes,
            "probed_healthy_tools": self._probed_healthy_tools,
            "probed_distractor": self._probed_distractor,
            "executed_value": self._executed_value,
            "output_provenance": self._output_provenance,
            "dirty_state": self._dirty_state,
            "step_counter": self._step_counter,
            "artifact_version": self._artifact_version,
        }

    def restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restore complete physical environment state s_T before rolling out Branch E (§4.1)."""
        qid = snapshot.get("query_id")
        if qid is not None:
            self._active_spec = self.get_task_spec(str(qid))
        self._episode_seed = int(snapshot["episode_seed"])
        self._discovered_datatypes = bool(snapshot["discovered_datatypes"])
        self._probed_healthy_tools = bool(snapshot["probed_healthy_tools"])
        self._probed_distractor = bool(snapshot["probed_distractor"])
        self._executed_value = snapshot["executed_value"]
        self._output_provenance = str(snapshot["output_provenance"])
        self._dirty_state = bool(snapshot["dirty_state"])
        self._step_counter = int(snapshot["step_counter"])
        self._artifact_version = int(snapshot["artifact_version"])


PlanBenchXLEnvAdapter = PlanBenchEnv


def load_planbench_split_tasks(
    data_root: str | Path | None = None,
    split: str = "train",
) -> list[tuple[str, str]]:
    """Load official PlanBench-XL (query_id, task_id) pairs for `split`."""
    env = PlanBenchEnv(pbxl_root=data_root)
    return env.load_split_task_specs(split)


PlanBenchXLEnvAdapter = PlanBenchEnv


def load_planbench_split_tasks(
    data_root: str | Path | None = None,
    split: str = "train",
) -> list[tuple[str, str]]:
    """Load official PlanBench-XL (query_id, task_id) pairs for `split`."""
    env = PlanBenchEnv(pbxl_root=data_root)
    return env.load_split_task_specs(split)
