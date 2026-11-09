"""PoS's native LOCA 75-case/8k protocol and isolated, original MCP tools.

The official test instances are preserved. Train/dev change only generator
seeds and measure same-template instance generalization. 8k is the native
environment-size setting, not an actor context-window limit.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from .native_core import BudgetMeter, NativeStep, monotonic_seconds
from .native_loca_capabilities import classify_loca_tool
from .native_loca_evaluation_guard import (
    PausedMCPProcesses, materialize_evaluation_workspace,
    validate_public_workspace,
)
from .native_loca_sandbox import LOCASandbox
from .native_loca_vm import LOCAVMSession, _pack_dir_b64, _unpack_dir_b64
from .native_text_envs import public_text_action_schema


POS_COMMIT = "6818cfa6434dbc91fcf67f0b3eb12a91db47433f"
TEST_SEEDS = (42, 123, 456, 789, 2024)
_SNAPSHOT_GAP = (
    "LOCA full-process snapshot and restore are not implemented. Original services retain "
    "login sessions, PDF search sessions/cache, live SQLite connection state "
    "and arbitrary agent-created processes; these interfaces do not provide "
    "seed replay or filesystem-only substitute snapshots."
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _information_signature(observation: str) -> str:
    # PoS serializes tool messages with a fresh call ID on every turn. IDs are
    # transport bookkeeping, so repeated public content must retain its identity.
    try:
        messages, end = json.JSONDecoder().raw_decode(observation)
    except ValueError:
        return _json(observation)
    if isinstance(messages, list) and all(isinstance(item, dict) and item.get("role") == "tool"
                                         for item in messages):
        return _json({"content": [item.get("content") for item in messages],
                      "suffix": observation[end:]})
    return _json(observation)


def _load_cases(loca_root: Path, seeds: Mapping[str, Sequence[int]]) -> tuple[dict, dict, dict]:
    source = loca_root / "task-configs/final_8k_set_config.json"
    official = json.loads(source.read_text(encoding="utf-8"))["configurations"]
    if len(official) != 75:
        raise ValueError("Pinned PoS 8k must provide exactly 75 official test cases.")
    templates: dict[str, dict] = {}
    official_ids: set[tuple[str, int]] = set()
    for case in official:
        name, seed = str(case["name"]), int(case["env_params"]["seed"])
        if (name, seed) in official_ids:
            raise ValueError("Duplicate official LOCA test instance.")
        official_ids.add((name, seed))
        template = deepcopy(case)
        template["env_params"].pop("seed")
        if name in templates and template != templates[name]:
            raise ValueError(f"LOCA {name} differs across official seeds beyond the seed parameter.")
        templates[name] = template
    if len(templates) != 15 or official_ids != {(n, s) for n in templates for s in TEST_SEEDS}:
        raise ValueError("Pinned PoS LOCA test template/seed manifest changed.")
    normalized = {split: tuple(int(seed) for seed in values) for split, values in seeds.items()}
    if normalized["test"] != TEST_SEEDS:
        raise ValueError("LOCA formal test must retain the five official seeds in their original order.")
    for split, values in normalized.items():
        if not values or len(values) != len(set(values)):
            raise ValueError(f"LOCA {split} seeds must be nonempty and unique.")
    if any(set(normalized[a]) & set(normalized[b])
           for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise ValueError("LOCA train/dev/test generation seeds must be disjoint.")
    cases, rows, splits = {}, {}, {}
    for split, values in normalized.items():
        splits[split] = []
        for name, template in templates.items():
            for seed in values:
                case = deepcopy(template)
                case["env_params"]["seed"] = seed
                task_id = f"loca/8k/{name}/seed_{seed}"
                row = {
                    "task_id": task_id, "scenario_id": task_id, "split": split,
                    "task_category": name, "task_type": name, "template_id": name,
                    "environment_length": "8k", "instance_seed": seed,
                    "partition_source": "PoS.official" if split == "test" else "project.same_template_new_seed",
                }
                cases[task_id], rows[task_id] = case, row
                splits[split].append(row)
    return cases, rows, splits


class NativeLOCAEnv:
    benchmark = "loca"

    def __init__(self, pos_repo_root: str | Path, *, output_dir: str | Path,
                 environment_length: str = "8k", train_seeds: Sequence[int] = tuple(range(10001, 10021)),
                 dev_seeds: Sequence[int] = tuple(range(20001, 20006)),
                 test_seeds: Sequence[int] = TEST_SEEDS, max_steps: int = 100,
                 runtime_paths: Sequence[str] = (), runtime_roots: Sequence[str] = (),
                 bwrap_executable: str = "bwrap", vm_config: Mapping[str, Any] | None = None,
                 meter: BudgetMeter | None = None) -> None:
        if environment_length != "8k" or int(max_steps) != 100:
            raise ValueError("Registered PoS LOCA protocol requires 8k and 100 policy/tool turns.")
        self.pos_repo_root = Path(pos_repo_root).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loca_root = self.pos_repo_root / "benchmarks/LOCA-Bench"
        self._cases, self._rows, self._splits = _load_cases(self.loca_root, {
            "train": train_seeds, "dev": dev_seeds, "test": test_seeds,
        })
        self.split_manifest = {
            "benchmark": self.benchmark, "source_commit": POS_COMMIT,
            "environment_length": "8k", "train_dev_protocol": "same_template_new_seed",
            "splits": {split: {"count": len(rows), "tasks": deepcopy(rows)}
                       for split, rows in self._splits.items()},
        }
        self._sandbox = LOCASandbox(self.loca_root, self.output_dir / ".loca-private",
                                    runtime_paths=runtime_paths, runtime_roots=runtime_roots,
                                    bwrap_executable=bwrap_executable)
        self._vm_config = self._resolve_vm_config(vm_config)
        self._vm_session: LOCAVMSession | None = None
        if str(self.pos_repo_root) not in sys.path:
            sys.path.insert(0, str(self.pos_repo_root))
        official_module = importlib.import_module("benchmarks.adapter")
        self._official_class = official_module._LOCABench
        self.meter, self.max_steps = meter, 100
        self._adapter: Any = None
        self._task_id = ""
        self._rollout_seed = 0
        self._attempts = self._tool_uses = 0
        self._done = self._submitted = False
        self._score = 0.0
        self._state: dict[str, Any] = {}
        self._seen_information: set[str] = set()
        self._tool_modes: dict[str, str] = {}
        self._last_observation = ""
        self.last_native_result: dict[str, Any] | None = None
        self.replayed_env_steps = 0
        self.snapshot_restore_seconds = 0.0
        self.action_schema: dict[str, Any] = {
            "benchmark": self.benchmark, "classic_text": True,
            "native_interface": "PoS LOCA original MCP tools; one JSON tool call per turn.",
            "native_format": '{"name": "public tool name", "arguments": {"native arguments": "..."}}',
            "native_system_prompt": "Complete the task with the available native tools. Issue exactly one JSON "
                                    "object with name and arguments per turn. Call claim_done when finished.",
            "information_semantics": "information_intent_with_native_cache_and_cursor_effects",
            "mode_actions": {"I": [], "E": []}, "open_text_modes": {"I": [], "E": []},
            "native_action_schemas_by_mode": {}, "native_tools": [], "probe_actions": [],
            "probe_tool_names": [], "feasible_modes": [], "search_available": False,
            "exact_snapshot_available": bool(self._vm_config.get("enabled", False)),
        }

    def _resolve_vm_config(self, vm_config: Mapping[str, Any] | None) -> dict[str, Any]:
        cfg = dict(vm_config or {})
        if cfg.get("enabled") is False:
            return {"enabled": False}
        ext_dir = self.pos_repo_root.parent
        qemu_dir = Path(cfg.get("qemu_dir", ext_dir / "qemu")).expanduser().resolve()
        vm_dir = Path(cfg.get("vm_dir", ext_dir / "loca-vm")).expanduser().resolve()
        kernels = sorted((qemu_dir / "root/boot").glob("vmlinuz-*")) if (qemu_dir / "root/boot").is_dir() else []
        default_kernel = kernels[-1] if kernels else vm_dir / "vmlinuz"
        resolved = {
            "enabled": cfg.get("enabled", True),
            "qemu_system_executable": str(cfg.get("qemu_system_executable", qemu_dir / "bin/qemu-system-x86_64")),
            "qemu_img_executable": str(cfg.get("qemu_img_executable", qemu_dir / "bin/qemu-img")),
            "kernel_path": str(cfg.get("kernel_path", default_kernel)),
            "initrd_path": str(cfg.get("initrd_path", vm_dir / "initramfs.cpio.gz")),
            "runtime_sqsh_path": str(cfg.get("runtime_sqsh_path", vm_dir / "runtime.sqsh")),
            "base_qcow2_path": str(cfg.get("base_qcow2_path", vm_dir / "loca-base.qcow2")),
            "boot_snapshot_tag": str(cfg.get("boot_snapshot_tag", "boot_ready")),
            "memory_mb": int(cfg.get("memory_mb", 2048)),
            "smp": int(cfg.get("smp", 2)),
        }
        if not all(Path(resolved[k]).is_file() for k in (
            "qemu_system_executable", "qemu_img_executable", "kernel_path",
            "initrd_path", "runtime_sqsh_path", "base_qcow2_path",
        )):
            if cfg.get("enabled") is True:
                raise FileNotFoundError(f"Configured LOCA QEMU VM artifacts missing: {resolved}")
            return {"enabled": False}
        return resolved

    def list_tasks(self, split: str) -> list[dict[str, Any]]:
        if split not in self._splits:
            raise ValueError(f"Unknown LOCA split: {split}")
        return deepcopy(self._splits[split])

    def schema_for_history(self, messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return public_text_action_schema(self.action_schema, messages)

    def _new_adapter(self) -> Any:
        sandbox = self._sandbox
        official = self._official_class
        if self._vm_config.get("enabled", False):
            self._vm_session = LOCAVMSession(self._vm_config, self.output_dir / ".loca-vm-session")
        vm_session = self._vm_session

        class IsolatedLOCABench(official):
            def _mcp_config(inner, server_configs: dict, task_workspace: Path,
                            agent_workspace: Path, npm_cache: Path | None = None) -> dict:
                server_configs = deepcopy(server_configs)
                for server in server_configs.values():
                    if server.get("type") == "terminal":
                        params = server.setdefault("params", {})
                        # Upstream's allowed_dir='.' otherwise overrides agent_workspace.
                        if params.get("allowed_dir") in (None, "."):
                            params["allowed_dir"] = str(agent_workspace)
                native = official._mcp_config(server_configs, task_workspace, agent_workspace, npm_cache)
                types = {name: value["type"] for name, value in server_configs.items()
                         if value.get("enabled", True)}
                artifact_dir = task_workspace / ".pos_artifacts/tool_results"
                if vm_session is not None:
                    specs, databases, code_mounts = sandbox.vm_service_specs(
                        native["mcpServers"], server_types=types,
                        task_workspace=task_workspace, agent_workspace=agent_workspace,
                        artifact_dir=artifact_dir,
                    )
                    return vm_session.prepare_episode(
                        specs, agent_workspace=agent_workspace,
                        artifact_dir=artifact_dir, databases=databases,
                        code_mounts=code_mounts,
                    )
                return sandbox.wrap(native["mcpServers"], server_types=types,
                                    task_workspace=task_workspace, agent_workspace=agent_workspace,
                                    artifact_dir=artifact_dir)

            def _start_environment(inner, case: dict, agent_workspace: Path) -> None:
                super()._start_environment(case, agent_workspace)
                native_reset = inner.environment.env.reset
                native_step = inner.environment.env.step
                execute_tool = inner.tool.execute_tool
                inner._inspection_pause = None

                def reset_with_artifact_directory(*args: Any, **kwargs: Any) -> Any:
                    value = native_reset(*args, **kwargs)
                    # Original reset recreates task_dir. Create the official public
                    # artifact mount before MCP discovery starts its subprocesses.
                    inner.artifact_dir.mkdir(parents=True, exist_ok=True)
                    return value

                inner.environment.env.reset = reset_with_artifact_directory

                def execute_with_quiescent_inspection(*args: Any, **kwargs: Any) -> Any:
                    result = execute_tool(*args, **kwargs)
                    if vm_session is not None:
                        vm_session.sync_workspace_to_host()
                    pause = PausedMCPProcesses()
                    pause.freeze()
                    inner._inspection_pause = pause
                    validate_public_workspace(agent_workspace)
                    return result

                def evaluate_materialized_outputs(*args: Any, **kwargs: Any) -> Any:
                    if inner._inspection_pause is None:
                        raise RuntimeError("LOCA evaluator requires quiescent native tool processes.")
                    # The workspace was synced before freeze, which also stops QEMU.
                    materialize_evaluation_workspace(agent_workspace)
                    return native_step(*args, **kwargs)

                inner.tool.execute_tool = execute_with_quiescent_inspection
                inner.environment.env.step = evaluate_materialized_outputs

            def step(inner, action: dict) -> dict:
                try:
                    return super().step(action)
                finally:
                    if inner._inspection_pause is not None:
                        inner._inspection_pause.resume()
                        inner._inspection_pause = None

        return IsolatedLOCABench({
            "method": "prism_native", "benchmark": "LOCA-Bench",
            "loca_environment_length": "8k", "start_case": 0, "end_case": 75,
            "max_steps": 100, "loca_max_tool_uses": 100, "loca_tool_timeout_seconds": 120,
            "loca_workspace_retention": "all", "loca_npm_cache": str(self.output_dir / ".npm-unused"),
            "loca_observation_artifacts": {"enabled": True, "inline_max_chars": 8000,
                                           "preview_items": 3, "structured_only": True},
        })

    def reset(self, task_id: str, seed: int) -> dict[str, Any]:
        if task_id not in self._cases:
            raise ValueError("LOCA task must belong to the registered public manifest.")
        self.close()
        self._task_id = task_id
        # The rollout seed never overwrites the immutable instance-generation seed.
        self._rollout_seed = int(seed)
        self._adapter = self._new_adapter()
        self._adapter.cases, self._adapter.total, self._adapter.cursor = [deepcopy(self._cases[task_id])], 1, 0
        try:
            self._state = self._adapter.reset()
            self._attempts = self._tool_uses = 0
            self._done = self._submitted = False
            self._score = 0.0
            self._seen_information = set()
            self.last_native_result = None
            observation = self._publish(str(self._state["observation"]))
        except BaseException:
            self.close()
            raise
        return {"benchmark": self.benchmark, "task_id": task_id, "scenario_id": task_id,
                "instruction": str(self._state["task"]), "observation": observation,
                "system_prompt": self.action_schema["native_system_prompt"],
                "action_schema": deepcopy(self.action_schema),
                "initial_information_signature": _json({"observation": self._state["observation"]})}

    def _publish(self, observation: str) -> str:
        types = {name: config["type"] for name, config in self._cases[self._task_id]["mcp_servers"].items()
                 if config.get("enabled", True)}
        tools = deepcopy(self._state.get("actions", [])) if not self._done else []
        by_mode: dict[str, list[dict]] = {"I": [], "E": []}
        names: dict[str, list[str]] = {"I": [], "E": []}
        probes: list[str] = []
        self._tool_modes = {}
        for tool in tools:
            function = tool["function"]
            name, parameters = str(function["name"]), deepcopy(function.get("parameters", {"type": "object"}))
            mode = classify_loca_tool(name, types)
            self._tool_modes[name] = mode
            names[mode].append(name)
            by_mode[mode].append({"type": "object", "properties": {
                "name": {"const": name}, "arguments": parameters},
                "required": ["name", "arguments"], "additionalProperties": False})
            if mode == "I" and not parameters.get("required"):
                probes.append(_json({"name": name, "arguments": {}}))
        state = {
            "benchmark": self.benchmark, "mode_actions": {"I": [], "E": []},
            "open_text_modes": names, "native_tools": tools,
            "native_action_schemas_by_mode": {mode: {"anyOf": variants} for mode, variants in by_mode.items() if variants},
            "probe_tool_names": names["I"], "probe_actions": probes,
            "available_actions": [], "feasible_modes": [mode for mode in ("I", "E") if names[mode]],
            "search_available": False,
        }
        state["public_state_id"] = _digest({"observation": observation, **state})
        self.action_schema.update(deepcopy(state))
        self.action_schema["exact_snapshot_available"] = self._vm_session is not None
        self._last_observation = _json({"feedback_type": "native_text_transition",
                                       "native_observation": observation,
                                       "classic_text_action_state": state})
        return self._last_observation

    def step(self, action: Mapping[str, Any]) -> NativeStep:
        return self.step_native(action)

    def step_native(self, action: Mapping[str, Any], *, probe: bool = False) -> NativeStep:
        if self._adapter is None or self._done:
            raise RuntimeError("LOCA step requires a live nonterminal task.")
        if self.meter is not None:
            self.meter.ensure_env_steps(1)
        started = monotonic_seconds()
        try:
            result = self._step_unmetered(action, probe=probe)
            return NativeStep(result.observation, result.done, result.success,
                              {**result.metadata, "environment_seconds": monotonic_seconds() - started})
        finally:
            if self.meter is not None:
                self.meter.record_env_step(monotonic_seconds() - started)

    def _step_unmetered(self, action: Mapping[str, Any], *, probe: bool) -> NativeStep:
        self._attempts += 1
        mode, native, executed, called, hard_bad = None, None, False, False, False
        try:
            if action.get("kind") != "native_text" or not isinstance(action.get("text"), str):
                raise ValueError("LOCA requires a native_text action containing one JSON tool call.")
            text = action["text"]
            from jsonschema import Draft202012Validator
            native = json.loads(text)
            modes = [candidate for candidate, schema in self.action_schema["native_action_schemas_by_mode"].items()
                     if Draft202012Validator(schema).is_valid(native)]
            mode = modes[0] if len(modes) == 1 else None
            if mode is None:
                raise ValueError("Tool call does not match the current public native tool schema.")
            if action.get("mode") not in (None, mode):
                raise ValueError("Tool call does not belong to the selected I/E mode.")
            if probe and mode != "I":
                raise ValueError("LOCA probes require an information-intent tool.")
        except (TypeError, ValueError) as error:
            hard_bad, observation = True, f"Invalid action: {error}"
        else:
            before = int(self._adapter.last_info.get("tool_use_counter", 0))
            self._state = self._adapter.step(native)
            after = int(self._adapter.last_info.get("tool_use_counter", 0))
            if after - before != 1:
                raise RuntimeError("Official LOCA did not account for exactly one native MCP call.")
            called = True
            self._tool_uses = after
            # The original OpenAI wrapper advances tool_success_counter only
            # for claim_done. PoS's adapter supplies the corrected native
            # execution record for every call without changing evaluator reward.
            executed = bool(self._adapter.last_execution.get("success"))
            if self.meter is not None:
                self.meter.record_api_calls(1, successful=int(executed))
            hard_bad = not executed
            observation = str(self._state["observation"])
            self._done = bool(self._state["done"])
            self._submitted = self._done and (native["name"] == "claim_done" or native["name"].endswith("_claim_done"))
            self._score = float(self._state["score"])
            if not math.isfinite(self._score):
                raise RuntimeError("Official LOCA evaluator returned a non-finite score.")
        if self._attempts >= 100 or self._tool_uses >= 100:
            self._done = True
        success = bool(self._submitted and self._score >= 1.0)
        signature = _information_signature(observation) if called and mode == "I" else None
        new_information = not hard_bad and signature is not None and signature not in self._seen_information
        if signature is not None:
            self._seen_information.add(signature)
        published = self._publish(observation)
        return NativeStep(published, self._done, success, {
            "classic_text": True, "native_action": native, "action_mode": mode,
            "native_executed": called, "native_execution_succeeded": executed,
            "step_failed": hard_bad, "submitted": self._submitted,
            "information_action_executed": called and mode == "I",
            "native_success": success if self._done else None,
            "native_transition": {
                "state_changed": None, "task_progress": success,
                "new_information_candidate": bool(new_information),
                "public_information_evidence": observation if signature is not None else None,
                "public_information_signature": signature, "hard_bad": hard_bad, "final_success": success,
            },
            "termination": "submitted" if self._submitted else "environment_step_limit" if self._done else "running",
        })

    def finalize_evaluation(self) -> dict[str, Any]:
        if not self._task_id:
            raise RuntimeError("LOCA final scoring requires a selected task.")
        self.last_native_result = {
            **deepcopy(self._rows[self._task_id]), "benchmark": self.benchmark,
            "success": bool(self._submitted and self._score >= 1.0), "score": self._score,
            "submitted": self._submitted, "steps": self._attempts, "tool_use_counter": self._tool_uses,
            "native_evaluator": "PoS._LOCABench.original_claim_done",
            "upstream_source_reference": POS_COMMIT,
        }
        return deepcopy(self.last_native_result)

    def training_reference(self, task_id: str | None = None) -> dict[str, Any]:
        selected = self._task_id if task_id is None else str(task_id)
        if selected not in self._rows or self._rows[selected]["split"] != "train":
            raise ValueError("LOCA reference supervision is restricted to registered training cases.")
        if selected != self._task_id or self._adapter is None:
            raise ValueError("LOCA reference requires the currently generated training instance.")
        from .native_loca_reference import read_training_reference
        reference = read_training_reference(Path(self._adapter.workspace), self._rows[selected]["template_id"])
        return {"benchmark": self.benchmark, "task_id": selected,
                "reference_id": selected, "source": "official_generated_groundtruth; training_supervision_only",
                "reference": reference}

    def save_snapshot(self) -> dict[str, Any]:
        if self._vm_session is None:
            raise NotImplementedError(_SNAPSHOT_GAP)
        if self._adapter is None or not self._task_id or self._done:
            raise RuntimeError("LOCA snapshots require a live nonterminal episode.")
        started = monotonic_seconds()
        vm_tag = self._vm_session.save_vm_snapshot()
        artifact_dir = Path(self._adapter.artifact_dir)
        wrapper_chain = []
        cur = getattr(self._adapter, "environment", None)
        while cur is not None:
            wrapper_chain.append({
                k: deepcopy(getattr(cur, k))
                for k in ("tool_use_counter", "tool_success_counter", "step_count", "steps", "current_step", "num_steps")
                if hasattr(cur, k) and isinstance(getattr(cur, k), (int, float, bool, str))
            })
            cur = getattr(cur, "env", None)
        adapter_scalars = {
            k: deepcopy(getattr(self._adapter, k))
            for k in ("steps", "step_count", "done")
            if hasattr(self._adapter, k) and isinstance(getattr(self._adapter, k), (int, float, bool, str))
        }
        snapshot = {
            "benchmark": self.benchmark,
            "task_id": self._task_id,
            "seed": self._rollout_seed,
            "vm_snapshot_tag": vm_tag,
            "attempts": self._attempts,
            "tool_uses": self._tool_uses,
            "done": self._done,
            "submitted": self._submitted,
            "score": self._score,
            "seen_information": sorted(self._seen_information),
            "state": deepcopy(self._state),
            "public_observation": self._last_observation,
            "adapter_last_info": deepcopy(getattr(self._adapter, "last_info", None) or {}),
            "adapter_last_execution": deepcopy(getattr(self._adapter, "last_execution", None) or {}),
            # PoS._LOCABench.step uses host-side artifact tracking as well as
            # guest process state. Isolate it across same-prefix branches.
            "adapter_branch_state": {
                name: deepcopy(getattr(self._adapter, name))
                for name in ("artifact_contracts", "_workspace_file_index",
                             "_last_tool_success_counter", "score")
            },
            "wrapper_chain": wrapper_chain,
            "adapter_scalars": adapter_scalars,
            "artifact_dir_b64": _pack_dir_b64(artifact_dir) if artifact_dir.exists() else "",
        }
        if self.meter is not None:
            self.meter.record_snapshot_save(monotonic_seconds() - started)
        return snapshot

    def restore_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        if self._vm_session is None:
            raise NotImplementedError(_SNAPSHOT_GAP)
        if snapshot.get("benchmark") != self.benchmark:
            raise ValueError("Snapshot belongs to another native benchmark.")
        if snapshot.get("task_id") != self._task_id or self._adapter is None:
            raise ValueError("LOCA QEMU snapshot restore requires the active episode's VM session.")
        vm_tag = str(snapshot.get("vm_snapshot_tag", ""))
        if not vm_tag:
            raise ValueError("LOCA snapshot is missing its QEMU savevm tag.")
        # Reject incomplete legacy snapshots before changing the live VM;
        # retaining current branch fields would silently contaminate replay.
        branch_fields = ("artifact_contracts", "_workspace_file_index",
                         "_last_tool_success_counter", "score")
        branch_state = snapshot.get("adapter_branch_state")
        if not isinstance(branch_state, Mapping) or any(name not in branch_state for name in branch_fields):
            raise ValueError("LOCA snapshot is missing host adapter branch state; capture a new snapshot.")
        restored_branch_state = {name: deepcopy(branch_state[name]) for name in branch_fields}
        started = monotonic_seconds()
        try:
            self._vm_session.restore_vm_snapshot(vm_tag)
            artifact_dir = Path(self._adapter.artifact_dir)
            _unpack_dir_b64(str(snapshot.get("artifact_dir_b64", "")), artifact_dir, clean=True)
            self._vm_session.sync_artifacts_to_guest()
            self._attempts = int(snapshot["attempts"])
            self._tool_uses = int(snapshot["tool_uses"])
            self._done = bool(snapshot["done"])
            self._submitted = bool(snapshot["submitted"])
            self._score = float(snapshot["score"])
            self._seen_information = set(snapshot.get("seen_information", ()))
            self._state = deepcopy(dict(snapshot["state"]))
            self._adapter.last_info = deepcopy(dict(snapshot.get("adapter_last_info") or {}))
            self._adapter.last_execution = deepcopy(dict(snapshot.get("adapter_last_execution") or {}))
            for name, value in restored_branch_state.items():
                setattr(self._adapter, name, value)
            cur = getattr(self._adapter, "environment", None)
            for saved_attrs in snapshot.get("wrapper_chain", ()):
                if cur is None:
                    break
                for k, v in saved_attrs.items():
                    setattr(cur, k, deepcopy(v))
                cur = getattr(cur, "env", None)
            for k, v in snapshot.get("adapter_scalars", {}).items():
                setattr(self._adapter, k, deepcopy(v))
            published = self._publish(str(self._state["observation"]))
            if published != snapshot.get("public_observation"):
                raise RuntimeError("LOCA QEMU snapshot restore diverged from recorded public observation.")
        finally:
            elapsed = monotonic_seconds() - started
            self.snapshot_restore_seconds += elapsed
            if self.meter is not None:
                self.meter.record_restore(elapsed, 0)

    def close(self) -> None:
        if self._adapter is not None:
            try:
                self._adapter.close()
            finally:
                self._adapter = None
        if self._vm_session is not None:
            try:
                self._vm_session.close()
            finally:
                self._vm_session = None
