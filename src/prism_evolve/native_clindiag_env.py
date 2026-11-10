"""PoS ClinDiag tools with isolated labels and exact in-memory branch restoration.

The fixed PoS 604-case manifest remains the test set. The user-approved training
extension partitions the remaining usable public cases 80/20. Patient records
are read directly from the official archive; they are never copied into traces.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import re
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence
import zipfile

from .native_core import BudgetExhausted, BudgetMeter, NativeStep, monotonic_seconds


_OFFICIAL_MODEL = "qwen3.7-plus"
_SOURCE_SYMBOLS = {
    "SUBMIT_TOOL_NAME", "QUERY_TOOLS", "PROVIDER_SYSTEM_PROMPT",
    "JUDGE_SYSTEM_PROMPT", "JUDGE_USER_PROMPT", "ClinDiagCase",
    "provider_tool_schemas", "ask_provider", "validate_submission",
    "judge_submission", "_parse_judge_response", "answer_key_access_problem",
    "cap_observation",
}


def _official_runtime(repo_root: Path) -> Any:
    """Load the pinned upstream pure tools without its global API/client imports."""
    source = repo_root / "benchmarks" / "clindiag_env.py"
    contents = source.read_text(encoding="utf-8")
    nodes = []
    for node in ast.parse(contents, filename=str(source)).body:
        names: set[str] = set()
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        if names & _SOURCE_SYMBOLS:
            nodes.append(node)
    namespace: dict[str, Any] = {
        "__name__": "_prism_official_clindiag", "json": json, "re": re,
        "Path": Path, "Callable": Callable,
    }
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in _SOURCE_SYMBOLS})


class _PrivateJudgeLogger:
    def log(self, event: str, **fields: Any) -> None:
        # Upstream judge events contain the reference diagnosis and full record.
        pass


def _native_call_schema(tool: Mapping[str, Any]) -> dict[str, Any]:
    function = tool["function"]
    return {
        "type": "object", "properties": {
            "name": {"const": function["name"]},
            "arguments": deepcopy(function["parameters"]),
        },
        "required": ["name", "arguments"], "additionalProperties": False,
    }


class NativeClinDiagEnv:
    """One real clinical episode; only queried evidence reaches the policy."""

    benchmark = "clindiag"

    def __init__(
        self, pos_repo_root: str | Path, data_archive: str | Path, *,
        split_seed: int = 20261105, max_steps: int = 30,
        meter: BudgetMeter | None = None,
        provider_config: Mapping[str, Any] | None = None,
        judge_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.repo_root = Path(pos_repo_root).resolve()
        self.data_archive = Path(data_archive).resolve()
        if max_steps <= 0 or max_steps > 30:
            raise ValueError("ClinDiag uses a positive action budget of at most 30.")
        self.max_steps, self.meter = int(max_steps), meter
        self._runtime = _official_runtime(self.repo_root)
        self._archive = zipfile.ZipFile(self.data_archive)
        self._clients: dict[str, Any] = {}
        self._provider_config = self._model_config("provider", provider_config)
        self._judge_config = self._model_config("judge", judge_config)
        self._splits = self._build_splits(int(split_seed))
        self._task_rows = {row["task_id"]: row for rows in self._splits.values() for row in rows}
        self._tools = self._runtime.provider_tool_schemas()
        mode_tools = {
            mode: [tool for tool in self._tools if (
                tool["function"]["name"] == self._runtime.SUBMIT_TOOL_NAME) == (mode == "E")]
            for mode in ("I", "E")
        }
        instruction = (
            'Emit one native JSON tool call: {"name":"tool_name","arguments":{...}}. '
            "Use exactly one listed tool per action. Ask one specific question to obtain "
            "evidence, then submit_diagnosis once when ready. The final diagnosis is "
            "judged against private reference labels."
        )
        self.action_schema = {
            "benchmark": self.benchmark, "classic_text": True, "static_native_tools": True,
            "kind": "native-text-v1", "native_interface": instruction,
            "native_system_prompt": instruction, "native_format": "official_pos_json_tool_call",
            "mode_actions": {"I": [], "E": []}, "search_available": False,
            "open_text_modes": {mode: [t["function"]["name"] for t in rows]
                                for mode, rows in mode_tools.items()},
            "native_action_schemas_by_mode": {
                mode: {"anyOf": [_native_call_schema(tool) for tool in rows]}
                for mode, rows in mode_tools.items()
            },
            "native_tools": deepcopy(self._tools),
            "probe_actions": [], "probe_tool_names": list(self._runtime.QUERY_TOOLS),
            "I": "Request existing patient evidence using one of the five provider tools.",
            "E": "Submit the final diagnosis through submit_diagnosis.",
        }
        self._task_id = ""
        self._case: Any = None
        self._seed = 0
        self._attempts = 0
        self._tool_uses = 0
        self._done = False
        self._submitted = False
        self._pending_judgment = False
        self._verdict: dict[str, Any] | None = None
        self._history: list[dict[str, Any]] = []
        self._evidence: set[str] = set()
        self._judge_seconds = 0.0
        self._model_calls = {"provider": 0, "judge": 0}
        self.last_native_result: dict[str, Any] | None = None

    @staticmethod
    def _model_config(role: str, override: Mapping[str, Any] | None) -> dict[str, Any]:
        config = {
            "model": _OFFICIAL_MODEL,
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key_env": "DASHSCOPE_API_KEY", "temperature": 0,
            "max_tokens": 1024 if role == "provider" else 512,
            "extra_body": {"enable_thinking": False},
            "timeout": 120.0,
        }
        if role == "judge":
            config["max_json_retries"] = 2
            config["extra_body"]["response_format"] = {"type": "json_object"}
        if override:
            config.update(deepcopy(dict(override)))
        if (config["model"] != _OFFICIAL_MODEL or config["temperature"] != 0
                or config["max_tokens"] != (1024 if role == "provider" else 512)
                or config["extra_body"].get("enable_thinking") is not False
                or (role == "judge" and (config.get("max_json_retries") != 2
                    or config["extra_body"].get("response_format") != {"type": "json_object"}))):
            raise ValueError("ClinDiag provider/judge must retain the registered PoS model settings.")
        if not isinstance(config["api_key_env"], str) or not config["api_key_env"]:
            raise ValueError("ClinDiag requires a credential environment-variable name.")
        if not math.isfinite(float(config["timeout"])) or float(config["timeout"]) <= 0:
            raise ValueError("ClinDiag model timeout must be finite and positive.")
        return config

    def _read_json(self, case_id: str, filename: str) -> Any:
        return json.loads(self._archive.read(f"{case_id}/{filename}").decode("utf-8"))

    def _diagnosis_valid(self, case_id: str) -> bool:
        try:
            payload = self._read_json(case_id, "diagnosis.json")
        except (KeyError, ValueError):
            return False
        final = str(payload.get("diagnosis", {}).get("final_diagnosis", "")).strip()
        return bool(final) and final.lower() not in {"not specified", "none"}

    def _build_splits(self, seed: int) -> dict[str, list[dict[str, Any]]]:
        manifest_path = self.repo_root / "benchmarks" / "ClinDiag" / "manifest.txt"
        official = [line.strip() for line in manifest_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
        tops = sorted({name.split("/")[0] for name in self._archive.namelist()})
        valid = [case_id for case_id in tops if self._diagnosis_valid(case_id)]
        # Upstream includes legitimate report variants such as PMID_1. Match
        # its opaque case IDs while keeping archive paths inside their case.
        if any(case_id in {".", ".."} or "/" in case_id or "\\" in case_id
               or "\x00" in case_id for case_id in valid):
            raise ValueError("The ClinDiag archive contains an unsafe case identifier.")
        challenging = [case_id for case_id in valid if not case_id.startswith("rare")]
        test_set = set(official)
        remainder = [f"challenging/{case_id}" for case_id in challenging
                     if f"challenging/{case_id}" not in test_set]
        usable = []
        excluded = []
        for case_ref in remainder:
            upstream_id = case_ref.split("/", 1)[1]
            try:
                initial = self._read_json(upstream_id, "initial_information.json")
                ready = isinstance(initial, dict) and bool(str(initial.get("initial_information", "")).strip())
            except (KeyError, ValueError):
                ready = False
            (usable if ready else excluded).append(case_ref)
        random.Random(seed).shuffle(usable)
        boundary = len(usable) * 4 // 5
        if boundary == 0 or boundary == len(usable):
            raise ValueError("The public ClinDiag remainder cannot supply both train and dev.")
        assigned = {"train": sorted(usable[:boundary]), "dev": sorted(usable[boundary:]), "test": official}
        rows = {split: [{"task_id": task_id, "scenario_id": task_id, "split": split,
                         "subset": task_id.split("/", 1)[0]} for task_id in tasks]
                for split, tasks in assigned.items()}
        self.split_manifest = {
            "benchmark": self.benchmark, "scheme": "pos_fixed_604_public_remainder_80_20",
            "split_seed": seed, "train_fraction": 0.8, "rounding": "floor_train",
            "test_manifest": str(manifest_path), "data_archive": str(self.data_archive),
            "excluded_unusable_remainder_ids": excluded,
            "splits": {split: {"count": len(tasks), "tasks": deepcopy(tasks)}
                       for split, tasks in rows.items()},
        }
        return rows

    def list_tasks(self, split: str) -> list[dict[str, Any]]:
        if split not in self._splits:
            raise ValueError(f"Unknown ClinDiag split: {split}")
        return deepcopy(self._splits[split])

    def schema_for_history(self, messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        # These public tool schemas are static and contain no patient evidence.
        return deepcopy(self.action_schema)

    def _load_case(self, task_id: str) -> Any:
        upstream_id = task_id.split("/", 1)[1]
        case = self._runtime.ClinDiagCase.__new__(self._runtime.ClinDiagCase)
        case.case_dir = zipfile.Path(self._archive, at=f"{upstream_id}/")
        case.initial_information = str(self._read_json(upstream_id, "initial_information.json")["initial_information"])
        case._records = {}
        return case

    def reset(self, task_id: str, seed: int) -> dict[str, Any]:
        if task_id not in self._task_rows:
            raise ValueError("ClinDiag task must belong to the registered public manifest.")
        if self._case is not None:
            self._case.release()
        self._task_id, self._seed = str(task_id), int(seed)
        self._case = self._load_case(self._task_id)
        self._attempts = self._tool_uses = 0
        self._done = self._submitted = self._pending_judgment = False
        self._verdict, self.last_native_result = None, None
        self._history, self._evidence = [], set()
        self._judge_seconds = 0.0
        self._model_calls = {"provider": 0, "judge": 0}
        task = {"case_id": self._task_id, "subset": self._task_rows[self._task_id]["subset"],
                "initial_presentation": self._case.initial_information}
        observation = (
            "The patient is in front of you; only the initial presentation is known so far. "
            "The full record (history, physical examination, diagnostic tests) must be obtained "
            "through the tools, one specific question per turn. Investigate, then call "
            "submit_diagnosis exactly once with your conclusion."
        )
        return {"task_id": self._task_id, "scenario_id": self._task_id,
                "instruction": json.dumps(task, ensure_ascii=False),
                "observation": self._observation(observation),
                "system_prompt": self.action_schema["native_system_prompt"]}

    def _observation(self, text: str) -> str:
        return json.dumps({
            "feedback_type": "native_transition", "native_observation": text,
            "classic_text_action_state": {
                key: deepcopy(self.action_schema[key]) for key in (
                    "mode_actions", "search_available", "open_text_modes",
                    "native_action_schemas_by_mode", "native_tools")
            },
        }, ensure_ascii=False, allow_nan=False)

    def _ensure_request_budget(self, role: str, reservation: int, max_tokens: int) -> None:
        if self.meter is None:
            return
        if role == "judge":
            # The diagnosis was already submitted; posterior grading is outside C.
            if self.meter.remaining_model_tokens < reservation + max_tokens:
                raise BudgetExhausted("model_tokens")
            if self.meter.parent is not None:
                self.meter.parent.ensure_model_call(reservation, max_tokens, "clindiag_judge")
        else:
            self.meter.ensure_model_call(reservation, max_tokens, "clindiag_provider")

    def _call_model(self, role: str, messages: list[dict[str, Any]]) -> str:
        config = self._provider_config if role == "provider" else self._judge_config
        # The API owns its tokenizer. Reserve a conservative byte bound and record
        # only the service's measured usage; never report this bound as token usage.
        reservation = sum(len(str(row.get("content", "")).encode("utf-8")) for row in messages)
        reservation += 256 + 32 * len(messages)
        self._ensure_request_budget(role, reservation, int(config["max_tokens"]))
        if role not in self._clients:
            key = os.environ.get(config["api_key_env"])
            if not key:
                raise RuntimeError(f"ClinDiag requires the configured {config['api_key_env']} credential.")
            from openai import OpenAI
            self._clients[role] = OpenAI(api_key=key, base_url=config["base_url"], max_retries=0)
        started = monotonic_seconds()
        try:
            response = self._clients[role].chat.completions.create(
                model=config["model"], messages=messages, temperature=0,
                max_tokens=config["max_tokens"], extra_body=config["extra_body"],
                timeout=float(config["timeout"]),
            )
        except Exception as error:
            elapsed = monotonic_seconds() - started
            if self.meter is not None:
                self.meter._append({"kind": "environment_model_request_failed", "role": f"clindiag_{role}",
                                    "elapsed_seconds": elapsed, "usage_reported": False,
                                    "error_type": type(error).__name__})
            # The project propagates infrastructure failures: a missing service
            # usage receipt cannot be represented as a zero-cost clinical failure.
            raise RuntimeError(f"ClinDiag {role} request failed: {type(error).__name__}") from None
        elapsed = monotonic_seconds() - started
        usage = response.usage
        if (usage is None or not isinstance(usage.prompt_tokens, int)
                or not isinstance(usage.completion_tokens, int)
                or min(usage.prompt_tokens, usage.completion_tokens) < 0):
            raise RuntimeError(f"ClinDiag {role} response lacks measured token usage.")
        self._model_calls[role] += 1
        if self.meter is not None:
            self.meter.record_model_call(usage.prompt_tokens, usage.completion_tokens, elapsed,
                                         role=f"clindiag_{role}")
            budget = self.meter
            while budget is not None:
                if budget.total_model_tokens > budget.limits.max_model_tokens:
                    raise BudgetExhausted("model_tokens")
                budget = budget.parent
        if usage.prompt_tokens > reservation:
            raise RuntimeError("ClinDiag input tokens exceeded the registered request reservation.")
        content = response.choices[0].message.content if response.choices else None
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"ClinDiag {role} returned no textual response.")
        return content

    def _step_unmetered(self, action: Mapping[str, Any], probe: bool = False) -> NativeStep:
        if self._case is None or self._done:
            raise RuntimeError("ClinDiag stepping requires a live episode.")
        self._attempts += 1
        self.last_native_result = None
        mode: str | None = None
        hard_bad, information, judge_seconds = False, None, 0.0
        executed = False
        try:
            if action.get("kind") != "native_text" or not isinstance(action.get("text"), str):
                raise ValueError("Expected one native JSON tool call.")
            payload = json.loads(action["text"])
            if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
                raise ValueError('The action must be {"name":"tool name","arguments":{}}.')
            arguments = payload.get("arguments")
            if arguments is not None and not isinstance(arguments, dict):
                raise ValueError("Arguments must be a JSON object.")
            name, arguments = payload["name"], arguments or {}
            if name not in self._runtime.QUERY_TOOLS and name != self._runtime.SUBMIT_TOOL_NAME:
                raise ValueError("Unknown tool; choose a tool from the public tool list.")
            mode = "E" if name == self._runtime.SUBMIT_TOOL_NAME else "I"
            if action.get("mode") not in {None, mode}:
                raise ValueError("The native tool does not belong to the selected action mode.")
            if probe and mode != "I":
                raise ValueError("A ClinDiag probe must request patient evidence.")
        except (ValueError, TypeError) as error:
            hard_bad = True
            observation = f"Invalid action: {error}"
        else:
            self._tool_uses += 1
            if mode == "E":
                problems = self._runtime.validate_submission(arguments)
                if problems:
                    hard_bad = True
                    observation = ("Submission rejected: " + "; ".join(problems)
                                   + ". Fix the arguments and call submit_diagnosis again.")
                else:
                    self._submitted = self._pending_judgment = True
                    started = monotonic_seconds()
                    ground_truth = self._read_json(self._task_id.split("/", 1)[1], "diagnosis.json")["diagnosis"]
                    verdict = self._runtime.judge_submission(
                        lambda messages: self._call_model("judge", messages), arguments, ground_truth,
                        max_retries=int(self._judge_config["max_json_retries"]), logger=_PrivateJudgeLogger(),
                        case_id=self._task_id, step=self._attempts,
                    )
                    judge_seconds = monotonic_seconds() - started
                    # Drop the reference diagnosis and explanation at the private boundary.
                    self._verdict = {key: verdict[key] for key in
                                     ("success", "score", "diagnosis_correct", "differential_hit")}
                    self._pending_judgment = False
                    executed = True
                    self._done = True
                    self._judge_seconds += judge_seconds
                    observation = ("Diagnosis received: " + json.dumps(
                        {"final_diagnosis": arguments["final_diagnosis"]}, ensure_ascii=False)
                        + ". The case is finished.")
            else:
                problem = self._runtime.answer_key_access_problem(arguments)
                question = arguments.get("question")
                if problem is not None:
                    hard_bad, observation = True, f"Tool call rejected: {problem}"
                elif not isinstance(question, str) or not question.strip():
                    hard_bad, observation = True, f"Tool call error: {name} requires one non-empty 'question' string argument."
                else:
                    answer = self._runtime.ask_provider(
                        lambda messages: self._call_model("provider", messages), name, question, self._case)
                    observation = self._runtime.cap_observation(answer, 8000)
                    information = observation
                    executed = True
        if self._attempts >= self.max_steps:
            self._done = True
        success = bool(self._verdict and self._verdict["success"])
        signature = json.dumps(information, ensure_ascii=False, separators=(",", ":")) if information else None
        is_new = bool(signature is not None and signature not in self._evidence)
        if signature is not None:
            self._evidence.add(signature)
        public_observation = self._observation(observation)
        metadata = {
            "native_action": "clindiag_tool", "action_mode": mode, "classic_text": True,
            "native_executed": executed, "submitted": self._submitted,
            "step_failed": hard_bad, "native_execution_succeeded": executed,
            "native_success": success if self._done else None,
            "information_action_executed": mode == "I" and executed,
            "terminal_judge_seconds": judge_seconds,
            "native_transition": {
                "state_changed": False, "task_progress": success, "new_information_candidate": is_new,
                "public_information_evidence": information,
                "public_information_signature": signature,
                "hard_bad": hard_bad, "final_success": success,
            },
            "termination": "submitted" if self._submitted else "environment_step_limit" if self._done else "running",
        }
        self._history.append({"action": deepcopy(dict(action)), "observation": public_observation,
                              "mode": mode, "hard_bad": hard_bad})
        return NativeStep(public_observation, self._done, success, metadata)

    def step(self, action: Mapping[str, Any]) -> NativeStep:
        return self.step_native(action)

    def step_native(self, action: Mapping[str, Any], *, probe: bool = False) -> NativeStep:
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

    def finalize_evaluation(self) -> dict[str, Any]:
        if self._case is None:
            raise RuntimeError("ClinDiag final scoring requires a selected task.")
        if self._pending_judgment:
            raise RuntimeError("ClinDiag submission has no completed official judge verdict.")
        verdict = self._verdict or {"success": False, "diagnosis_correct": False,
                                    "differential_hit": False, "score": 0.0}
        self.last_native_result = {
            "benchmark": self.benchmark, "task_id": self._task_id, "scenario_id": self._task_id,
            "split": self._task_rows[self._task_id]["split"],
            "subset": self._task_rows[self._task_id]["subset"],
            "success": bool(verdict["success"]), "diagnosis_correct": bool(verdict["diagnosis_correct"]),
            "differential_hit": bool(verdict["differential_hit"]), "score": float(verdict["score"]),
            "submitted": self._submitted, "steps": self._attempts, "tool_use_counter": self._tool_uses,
            "native_evaluator": "PoS.ClinDiag.judge_submission",
            "provider_model": _OFFICIAL_MODEL, "judge_model": _OFFICIAL_MODEL,
            "terminal_judge_seconds": self._judge_seconds, "benchmark_model_calls": dict(self._model_calls),
        }
        return deepcopy(self.last_native_result)

    def training_reference(self, task_id: str | None = None) -> dict[str, Any]:
        selected = self._task_id if task_id is None else str(task_id)
        if selected not in self._task_rows or self._task_rows[selected]["split"] != "train":
            raise ValueError("ClinDiag reference supervision is restricted to registered training cases.")
        truth = self._read_json(selected.split("/", 1)[1], "diagnosis.json")["diagnosis"]
        return {"benchmark": self.benchmark, "task_id": selected,
                "source": "ClinDiag.public_archive.diagnosis.json; training_supervision_only",
                "reference_id": f"clindiag/train/{selected}",
                "reference": {"final_diagnosis": truth.get("final_diagnosis", ""),
                              "differential_diagnosis": truth.get("differential_diagnosis", ""),
                              "diagnostic_reasoning": truth.get("diagnostic_reasoning", "")}}

    def save_snapshot(self) -> dict[str, Any]:
        if self._case is None or self._done or self._pending_judgment:
            raise RuntimeError("ClinDiag snapshots require a live nonterminal case.")
        started = monotonic_seconds()
        result = {"benchmark": self.benchmark,
                  "task_id": self._task_id, "seed": self._seed, "attempts": self._attempts,
                  "tool_uses": self._tool_uses, "history": deepcopy(self._history),
                  "evidence": sorted(self._evidence), "model_calls": dict(self._model_calls)}
        if self.meter is not None:
            self.meter.record_snapshot_save(monotonic_seconds() - started)
        return result

    def restore_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        if snapshot.get("benchmark") != self.benchmark:
            raise ValueError("ClinDiag snapshot belongs to another benchmark.")
        history = deepcopy(list(snapshot["history"]))
        if len(history) != int(snapshot["attempts"]):
            raise ValueError("ClinDiag snapshot contains an incomplete public history.")
        started = monotonic_seconds()
        self.reset(str(snapshot["task_id"]), int(snapshot["seed"]))
        self._attempts, self._tool_uses = int(snapshot["attempts"]), int(snapshot["tool_uses"])
        self._history, self._evidence = history, set(snapshot["evidence"])
        self._model_calls = dict(snapshot["model_calls"])
        # Provider requests are stateless functions of immutable records and a
        # question. Restoring their already observed outputs makes no API call.
        if self.meter is not None:
            self.meter.record_restore(monotonic_seconds() - started, replay_steps=0)

    def close(self) -> None:
        if self._case is not None:
            self._case.release()
            self._case = None
        self._archive.close()
        for client in self._clients.values():
            client.close()
        self._clients.clear()
