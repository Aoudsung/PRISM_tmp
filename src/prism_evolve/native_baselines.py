"""Native transfers of complete upstream inference methods.

Each upstream checkout runs in its own Python process. The process owns method
state only; model calls and environment transitions are serviced here through
the shared, measured budget. No upstream API client is allowed to call a model
outside that path.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import re
from pathlib import Path
import selectors
import subprocess
import sys
from typing import Any, Mapping, MutableMapping

from .native_core import (
    BudgetExhausted, BudgetMeter, NativeEvent, NativeGeneration, NativeTrajectory,
    append_interaction, initial_messages, monotonic_seconds,
)


INFERENCE_METHODS = ("pos", "worldevolver", "envprobe_simple", "envprobe_judge")
_REPO_NAMES = {"pos": "pos-official", "worldevolver": "worldevolver-official",
               "envprobe_simple": "envprobe-official", "envprobe_judge": "envprobe-official"}
_TEXT_BENCHMARKS = {"alfworld", "webshop", "clindiag", "loca"}


def _task_artifact_path(output: Path, task_id: str, suffix: str) -> Path:
    """Keep existing flat IDs; isolate path-like IDs without path traversal or basename collisions."""
    if not task_id:
        raise ValueError("Task artifacts require a nonempty task ID.")
    if (task_id not in {".", ".."} and not any(char in task_id for char in "/\\")
            and all(ord(char) >= 32 and ord(char) != 127 for char in task_id)
            and len((task_id + suffix).encode("utf-8")) <= 255):
        return output / f"{task_id}{suffix}"
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    return output / "_task_ids" / f"{digest}{suffix}"


def _action_mapping(value: Any, *, benchmark: str, forced_mode: str | None = None) -> dict[str, Any]:
    """Transport an upstream action without replacing its arguments or route."""
    if value is None or not isinstance(value, (Mapping, str)) or isinstance(value, str) and not value.strip():
        return {"kind": "invalid", "raw_text": json.dumps(value, ensure_ascii=False),
                "validation_stage": "controller_output", "error": "The controller supplied no native action."}
    if isinstance(value, Mapping):
        action = dict(value)
        if set(action) == {"action"}:
            return _action_mapping(action["action"], benchmark=benchmark, forced_mode=forced_mode)
        if benchmark.lower() in _TEXT_BENCHMARKS and action.get("kind") not in {"native_text", "invalid"}:
            # ClinDiag's official tool object is the native payload, not a program AST.
            action = {"kind": "native_text", "text": json.dumps(action, ensure_ascii=False,
                                                                 separators=(",", ":"), allow_nan=False)}
    else:
        raw = str(value).strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, Mapping):
            return _action_mapping(parsed, benchmark=benchmark, forced_mode=forced_mode)
        if isinstance(parsed, str):
            return _action_mapping(parsed, benchmark=benchmark, forced_mode=forced_mode)
        if benchmark.lower() in _TEXT_BENCHMARKS:
            action = {"kind": "native_text", "text": raw}
        elif "planbench" in benchmark.lower():
            action = {"raw_text": raw}
        else:
            # Strip only a transport fence; keep the controller's program intact.
            fenced = re.fullmatch(r"```(?:python|py)?\s*\n(.*?)\n```", raw, flags=re.DOTALL)
            action = {"kind": "code", "code": fenced.group(1) if fenced else raw}
    if forced_mode is not None:
        if action.get("mode", forced_mode) != forced_mode:
            return {"kind": "invalid", "mode": forced_mode,
                    "raw_text": json.dumps(value, ensure_ascii=False) if isinstance(value, Mapping) else str(value),
                    "validation_stage": "mode",
                    "error": "Upstream action mode contradicts the controller's required mode."}
        action["mode"] = forced_mode
    return action


class OfficialBaselineSession:
    """A single method/benchmark/seed's upstream state and isolated imports.

    Keep this object for a prequential WorldEvolver test sequence. Close it when
    the sequence ends. Its public checkpoint is saved after every completed
    episode; method configuration changes require a new session.
    """

    def __init__(self, method: str, config: Mapping[str, Any], output_dir: str | Path) -> None:
        if method not in INFERENCE_METHODS:
            raise ValueError(f"Unsupported complete inference method: {method}")
        self.method = method
        self.config = dict(config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        repos = dict(config.get("official_repos", {}))
        if "external_root" not in config and method not in repos:
            raise ValueError("Configure the official checkout location on the authorized execution host.")
        root = Path(config.get("external_root", "."))
        self.repo = Path(repos.get(method, root / _REPO_NAMES[method])).resolve()
        if not self.repo.is_dir():
            raise FileNotFoundError(f"Official {method} checkout is missing: {self.repo}")
        python_config = config.get("official_python", {})
        executable = python_config.get(method, sys.executable) if isinstance(python_config, Mapping) else python_config
        self.stderr_path = self.output_dir / f"{method}.official.stderr.log"
        self._stderr = self.stderr_path.open("a", encoding="utf-8")
        worker = Path(__file__).with_name("native_official_worker.py")
        self.process = subprocess.Popen(
            [str(executable), "-u", str(worker), "--method", method, "--repo", str(self.repo)],
            cwd=str(self.repo), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._stderr, text=True, encoding="utf-8", bufsize=1,
        )
        self._selector = selectors.DefaultSelector()
        self._selector.register(self.process.stdout, selectors.EVENT_READ)
        self._closed = False
        self._run_identity: tuple[str, str, int] | None = None
        self.checkpoint: dict[str, Any] | None = None
        self.completed_tasks: list[str] = []
        self.provenance: dict[str, Any] = {}
        self._budget_cleanup_deadline = 0.0

    def _send(self, record: Mapping[str, Any]) -> None:
        if self.process.stdin is None:
            raise RuntimeError("Official worker input is closed.")
        self.process.stdin.write(json.dumps(dict(record), ensure_ascii=False, allow_nan=False) + "\n")
        self.process.stdin.flush()

    def _read(self, meter: BudgetMeter) -> dict[str, Any]:
        while True:
            if self._selector.select(timeout=0.25):
                line = self.process.stdout.readline()
                if not line:
                    raise RuntimeError(f"Official {self.method} worker exited; see {self.stderr_path}")
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise RuntimeError("Invalid official-worker protocol record.")
                return record
            if self.process.poll() is not None:
                raise RuntimeError(f"Official {self.method} worker exited with {self.process.returncode}; see {self.stderr_path}")
            if (meter.elapsed_seconds >= meter.limits.max_wall_seconds
                    and monotonic_seconds() >= self._budget_cleanup_deadline):
                raise BudgetExhausted("wall_seconds")

    def run(
        self, policy: Any, env: Any, task: Mapping[str, Any], seed: int,
        config: Mapping[str, Any], meter: BudgetMeter, output_dir: str | Path,
    ) -> NativeTrajectory:
        benchmark = str(config.get("benchmark", getattr(env, "benchmark", type(env).__name__)))
        native_benchmark = str(getattr(env, "benchmark", benchmark)).split(":", 1)[0]
        backbone = str(config.get("model_id", config.get("backbone", getattr(policy, "model_path", ""))))
        identity = (benchmark, backbone, int(seed))
        if self._run_identity is not None and identity != self._run_identity:
            raise ValueError("An official session cannot share memory across benchmark, model, or seed.")
        self._run_identity = identity
        task_id = str(task["task_id"])
        if task_id in self.completed_tasks:
            raise ValueError(f"Task {task_id} already completed in this method sequence.")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        trace_path = _task_artifact_path(output, task_id, ".official.jsonl")
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        if trace_path.exists():
            raise FileExistsError(f"Refusing to overwrite a previous task trace: {trace_path}")
        old_policy_meter = getattr(policy, "meter", None)
        old_env_meter = getattr(env, "meter", None)
        policy.meter = meter
        env.meter = meter
        try:
            context = env.reset(task_id, int(task.get("environment_seed", seed)))
        except Exception:
            policy.meter, env.meter = old_policy_meter, old_env_meter
            self.close()
            raise
        context["benchmark"] = native_benchmark
        context["action_schema"] = env.action_schema
        trajectory = NativeTrajectory(task_id, str(context.get("scenario_id", task.get("scenario_id", task_id))),
                                      budget_seconds=meter.limits.max_wall_seconds)
        trajectory.metadata["official_trace_path"] = str(trace_path)
        messages = initial_messages(context)
        actor_generation: NativeGeneration | None = None
        generation_by_id: dict[int, NativeGeneration] = {}
        native_done = False
        terminal_elapsed: float | None = None
        budget_reason: str | None = None
        start = monotonic_seconds()
        # Environment construction/reset and an upstream process's one-time
        # startup are setup costs, outside the common deployment clock.
        meter.started = start
        try:
            self._send({"kind": "run", "context": context, "config": dict(config),
                        "seed": seed, "checkpoint": self.checkpoint,
                        "completed_tasks": list(self.completed_tasks)})
            with trace_path.open("x", encoding="utf-8") as trace:
                while True:
                    request = self._read(meter)
                    kind = request.get("kind")
                    if kind == "result":
                        trace.write(json.dumps({"upstream_result": request.get("result", {})},
                                               ensure_ascii=False, allow_nan=False) + "\n")
                        trace.flush()
                        result = dict(request.get("result", {}))
                        self.provenance = dict(result.pop("provenance", self.provenance))
                        checkpoint = result.pop("checkpoint", None)
                        if checkpoint is not None:
                            self.checkpoint = checkpoint
                            checkpoint_path = _task_artifact_path(output, task_id, ".worldevolver-memory.json")
                            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                            with checkpoint_path.open("x", encoding="utf-8") as handle:
                                json.dump(checkpoint, handle, ensure_ascii=False, allow_nan=False)
                            trajectory.metadata["official_memory_path"] = str(checkpoint_path)
                        trajectory.metadata.update({"official_result": result,
                                                    "official_provenance": self.provenance})
                        if result.get("status") == "configuration_error":
                            raise RuntimeError(result.get("error", "Official method configuration failed."))
                        if result.get("status") == "method_error":
                            trajectory.metadata["method_error"] = result.get("error")
                        break
                    request_id = int(request["id"])
                    payload = dict(request.get("payload", {}))
                    trace.write(json.dumps({"request": request}, ensure_ascii=False, allow_nan=False) + "\n")
                    trace.flush()
                    try:
                        if budget_reason is not None and kind not in ("log", "token_count"):
                            raise BudgetExhausted(budget_reason)
                        if kind == "begin_episode":
                            start = monotonic_seconds()
                            meter.started = start
                            reply = {"started": True}
                        elif kind == "model_call":
                            role = str(payload.get("role", self.method))
                            model_messages = [dict(item) for item in payload["messages"]]
                            actor_call = role.endswith("actor") or role.endswith("probe_mapping")
                            greedy = bool(config.get("evaluation_greedy", False)) if actor_call else True
                            if actor_call:
                                budget = {"remaining_environment_steps": max(0, meter.limits.max_env_steps - meter.env_steps),
                                          "remaining_wall_seconds": max(0.0, meter.limits.max_wall_seconds - meter.elapsed_seconds),
                                          "remaining_model_tokens": max(0, meter.limits.max_model_tokens - meter.total_model_tokens)}
                                model_messages[-1]["content"] += "\nPublic remaining budget: " + json.dumps(budget, sort_keys=True)
                            with policy.frozen_reference():
                                generation = policy.complete(
                                    model_messages, role=role,
                                    max_new_tokens=int(payload.get("max_new_tokens", config.get("max_new_tokens", 512))),
                                    temperature=1.0, greedy=greedy,
                                )
                            generation_by_id[request_id] = generation
                            if role.endswith("actor") or role.endswith("probe_mapping"):
                                actor_generation = generation
                            reply = {"text": generation.text, "input_tokens": generation.input_tokens,
                                     "output_tokens": generation.output_tokens, "generation_id": request_id}
                            if payload.get("return_logprobs"):
                                import torch
                                with torch.no_grad(), policy.frozen_reference():
                                    token_logprobs = policy.score_tokens(generation, include_mode=False)
                                if token_logprobs.numel() == 0:
                                    raise ValueError("WorldEvolver confidence requires generated response tokens.")
                                reply["mean_logprob"] = float(token_logprobs.mean().detach().cpu())
                        elif kind == "token_count":
                            tokenizer = policy.tokenizer
                            if "messages" in payload:
                                ids = tokenizer.apply_chat_template(payload["messages"], tokenize=True,
                                                                    add_generation_prompt=True, return_dict=False)
                            else:
                                ids = tokenizer.encode(str(payload.get("text", "")), add_special_tokens=False)
                            reply = {"count": len(ids)}
                        elif kind == "env_step":
                            if native_done:
                                raise RuntimeError("Official controller attempted an action after native termination.")
                            action = _action_mapping(payload.get("action"), benchmark=native_benchmark)
                            t0 = monotonic_seconds()
                            step = env.step_native(action, probe=bool(payload.get("probe", False)))
                            elapsed = monotonic_seconds() - t0
                            generation_id = payload.get("generation_id")
                            chosen = (generation_by_id.get(generation_id) if generation_id is not None
                                      else actor_generation)
                            if chosen is None:
                                raise RuntimeError("Official controller executed an action before a model decision.")
                            event_generation = NativeGeneration(**asdict(chosen))
                            event_generation.action = action
                            trajectory.events.append(NativeEvent(
                                list(messages), event_generation, step.observation, elapsed,
                                {"official_decision": payload, "native_step": dict(step.metadata),
                                 "model_generation_id": generation_id},
                            ))
                            messages = append_interaction(messages, event_generation, step.observation)
                            native_done = bool(step.done)
                            if native_done:
                                trajectory.metadata["native_terminal_success"] = bool(step.success)
                                terminal_elapsed = monotonic_seconds() - start
                                if native_benchmark.lower() == "clindiag":
                                    judge_seconds = float(step.metadata.get("terminal_judge_seconds", 0.0))
                                    if not math.isfinite(judge_seconds) or not 0 <= judge_seconds <= terminal_elapsed:
                                        raise ValueError("ClinDiag returned invalid terminal judge timing.")
                                    trajectory.metadata["terminal_wall_seconds_including_judge"] = terminal_elapsed
                                    trajectory.metadata["terminal_judge_seconds"] = judge_seconds
                                    terminal_elapsed -= judge_seconds
                                    # Scoring and terminal bookkeeping retain real costs, but C ends at submission.
                                    self._budget_cleanup_deadline = monotonic_seconds() + 5.0
                                trajectory.success = bool(step.success) and terminal_elapsed <= meter.limits.max_wall_seconds
                                if step.success and not trajectory.success:
                                    trajectory.metadata["native_success_after_deadline"] = True
                            reply = {"observation": step.observation, "done": step.done,
                                     "success": step.success, "action_schema": env.action_schema}
                        elif kind == "log":
                            reply = {"ok": True}
                        else:
                            raise RuntimeError(f"Unknown official-worker operation: {kind}")
                        self._send({"id": request_id, "ok": True, "result": reply})
                        trace.write(json.dumps({"id": request_id, "response": reply}, ensure_ascii=False,
                                               allow_nan=False) + "\n")
                        trace.flush()
                    except BudgetExhausted as exc:
                        budget_reason = str(exc)
                        self._budget_cleanup_deadline = monotonic_seconds() + 5.0
                        self._send({"id": request_id, "ok": False, "error_kind": "budget_exhausted",
                                    "error": budget_reason})
                    except Exception as exc:
                        self._send({"id": request_id, "ok": False, "error_kind": "callback_error",
                                    "error": f"{type(exc).__name__}: {exc}"})
                        raise
        except BudgetExhausted as exc:
            budget_reason = str(exc)
            self.close()
        except Exception:
            self.close()
            raise
        finally:
            policy.meter = old_policy_meter
            env.meter = old_env_meter
        trajectory.elapsed_seconds = terminal_elapsed if terminal_elapsed is not None else monotonic_seconds() - start
        official_evaluation = dict(env.finalize_evaluation())
        trajectory.metadata["official_evaluation"] = official_evaluation
        trajectory.metadata.setdefault("native_terminal_success", False)
        # A cutoff-state score is independent of a submitted, budget-compliant
        # terminal success; it must never promote an unfinished controller run.
        trajectory.success = bool(native_done and trajectory.success and
                                  (budget_reason is None or native_benchmark.lower() == "clindiag"))
        trajectory.termination = ("method_error" if "method_error" in trajectory.metadata else
                                  ("success" if trajectory.success else "native_failure") if native_done else
                                  f"budget:{budget_reason}" if budget_reason else "step_limit")
        trajectory.metadata.update({"method": self.method, "meter": meter.snapshot(),
                                    "parameter_training": "not_applicable",
                                    "actor_contract": "official-controller-native-v1",
                                    "task_order_index": len(self.completed_tasks),
                                    "memory_protocol": "prequential" if self.method == "worldevolver" else "episode_reset"})
        if budget_reason:
            trajectory.metadata["budget_exhausted"] = budget_reason
        self.completed_tasks.append(task_id)
        return trajectory

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self._selector.close()
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()
        self._stderr.close()

    def __enter__(self) -> OfficialBaselineSession:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def run_inference_method(
    method: str, policy: Any, env: Any, task: Mapping[str, Any], seed: int,
    config: Mapping[str, Any], meter: BudgetMeter, output_dir: str | Path,
    memory_state: MutableMapping[str, Any] | OfficialBaselineSession | None = None,
) -> NativeTrajectory:
    """Run one native task; pass the same memory_state for a method sequence."""
    temporary = False
    if isinstance(memory_state, OfficialBaselineSession):
        session = memory_state
    elif isinstance(memory_state, MutableMapping):
        session = memory_state.get("session")
        if session is None:
            session = OfficialBaselineSession(method, config, output_dir)
            memory_state["session"] = session
            session.checkpoint = memory_state.get("checkpoint")
            session.completed_tasks = list(memory_state.get("completed_tasks", []))
        elif session._closed:
            raise RuntimeError("The official method session has ended; automatic reconstruction is disabled.")
    else:
        if method == "worldevolver":
            raise ValueError("WorldEvolver requires persistent memory_state for its prequential task sequence.")
        session = OfficialBaselineSession(method, config, output_dir)
        temporary = True
    if session.method != method:
        raise ValueError("A method cannot reuse another method's official worker.")
    try:
        result = session.run(policy, env, task, seed, config, meter, output_dir)
        if isinstance(memory_state, MutableMapping):
            memory_state["checkpoint"] = session.checkpoint
            memory_state["completed_tasks"] = list(session.completed_tasks)
        return result
    finally:
        if temporary:
            session.close()
