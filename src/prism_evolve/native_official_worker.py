"""Isolated upstream-method controller for native_baselines.py.

Execute this file directly in the configured upstream Python environment. It
does not load a model or a native benchmark. Only public observations cross the
RPC boundary; all model and environment work is requested from the parent.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import importlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import traceback
from typing import Any


_CHANNEL = sys.stdout
_REQUEST_ID = 0


class RemoteBudgetExhausted(BaseException):
    """Bypass upstream Exception fallbacks when the common budget is spent."""


class InvalidNativeAction(BaseException):
    """Keep missing actions visible across upstream Exception recovery paths."""

    def __init__(self, raw_output: str, message: str = "The model returned no native action.") -> None:
        super().__init__(message)
        self.raw_output = raw_output


class UpstreamCallError(BaseException):
    """Preserve callback/structured-output failures across upstream recovery."""

    def __init__(self, message: str, raw_output: str | None = None) -> None:
        super().__init__(message)
        self.raw_output = raw_output


def _emit(record: dict[str, Any]) -> None:
    _CHANNEL.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    _CHANNEL.flush()


def _rpc(kind: str, **payload: Any) -> dict[str, Any]:
    global _REQUEST_ID
    _REQUEST_ID += 1
    request_id = _REQUEST_ID
    _emit({"kind": kind, "id": request_id, "payload": payload})
    line = sys.stdin.readline()
    if not line:
        raise EOFError("The native evaluation parent closed its RPC channel.")
    response = json.loads(line)
    if response.get("id") != request_id:
        raise RuntimeError("Out-of-order parent response.")
    if not response.get("ok"):
        if response.get("error_kind") == "budget_exhausted":
            raise RemoteBudgetExhausted(str(response.get("error", "budget")))
        raise UpstreamCallError(str(response.get("error", "Parent callback failed.")))
    return dict(response.get("result", {}))


def _json_object(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```json") and value.endswith("```"):
        value = value[7:-3].strip()
    elif value.startswith("```") and value.endswith("```"):
        value = value[3:-3].strip()
    result = json.loads(value)
    if not isinstance(result, dict):
        raise ValueError("An upstream structured response must be a JSON object.")
    return result


def _action_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, dict) else str(value)


class RPCLogger:
    def log(self, event: str, **fields: Any) -> None:
        _rpc("log", event=event, **fields)

    def close(self) -> None:
        pass


class RPCModel:
    def __init__(self, config: dict[str, Any], role: str) -> None:
        self.role = role
        self.max_tokens = int(config.get("max_new_tokens", 512))
        self.context_length = int(config.get("max_context_tokens", 32768))
        self.stop = None
        self.total_usage = {"n_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self.last_usage: dict[str, int] = {}
        self.last_generation_id: int | None = None
        self.last_text = ""

    def call(self, messages: list[dict[str, str]], *, role: str | None = None,
             max_tokens: int | None = None, logprobs: bool = False) -> dict[str, Any]:
        result = _rpc("model_call", messages=messages, role=role or self.role,
                      max_new_tokens=int(max_tokens if max_tokens is not None else self.max_tokens),
                      temperature=0.0, return_logprobs=logprobs)
        self.last_generation_id = result["generation_id"]
        self.last_text = result["text"]
        self.last_usage = {"prompt_tokens": int(result["input_tokens"]),
                           "completion_tokens": int(result["output_tokens"])}
        self.total_usage["n_calls"] += 1
        for key, value in self.last_usage.items():
            self.total_usage[key] += value
        return result

    def generate(self, system: str, user: str):
        result = self.call([{"role": "system", "content": system}, {"role": "user", "content": user}])
        return True, result["text"]

    def generate_raw(self, system: str, user: str):
        return self.generate(system, user)

    def generate_with_logprobs(self, system: str, user: str):
        result = self.call([{"role": "system", "content": system}, {"role": "user", "content": user}],
                           logprobs=True)
        return True, result["text"], result["mean_logprob"]

    def num_tokens_from_messages(self, messages: list[dict[str, str]]) -> int:
        return int(_rpc("token_count", messages=messages)["count"])

    @contextmanager
    def as_role(self, role: str):
        previous = self.role
        self.role = role
        try:
            yield
        finally:
            self.role = previous


def _schema_instruction(context: dict[str, Any]) -> str:
    public_schema = context.get("action_schema", {})
    classic = bool(public_schema.get("classic_text"))
    schema = (public_schema.get("native_interface", context.get("native_action_schema")) if classic else
              context.get("native_action_schema", public_schema.get("native_interface")))
    if schema is None:
        if classic:
            schema = {"action": "One native text command or one declared native tool JSON object."}
        elif context["benchmark"] == "appworld":
            schema = {"action": "One Python code block executed by AppWorld; variables persist.",
                      "apis": "Use the injected apis object and print results to observe them.",
                      "discovery": "Discover public API documentation using apis.api_docs."}
        else:
            schema = {"action": "One retrieve_tools, tool_call, or final_answer XML block.",
                      "retrieve_tools": "JSON with meta_tool by_output_info, by_inputs or by_io_info; "
                                        "write the corresponding output_information and/or input_information.",
                      "tool_call": "JSON with a discovered name or tool_id and arguments.",
                      "final_answer": "The final answer text, checked by the native environment."}
    current_actions = ("\nCurrent public actions:\n" + json.dumps(_public_actions(context), ensure_ascii=False)
                       if classic else "")
    payload_format = ("encode the exact native command or declared tool JSON inside that string. "
                      if classic else
                      "encode one native Python code block or PlanBench action block inside that string. ")
    return (
        "Instructions for the inner native-action payload:\n"
        + str(context.get("native_system_prompt", public_schema.get("native_system_prompt", "")))
        + "\nNative environment action interface:\n"
        + json.dumps(schema, ensure_ascii=False)
        + current_actions
        + "\nUse one executable native action. Preserve tool names and argument values exactly. "
          "Keep the upstream controller's outer response format. When it requires a string action, "
        + payload_format
        + "Do not generate a second I/E mode inside the action. Available action schemas describe syntax, not an answer or a "
          "recommended route. Discover missing information through real environment calls."
    )


def _public_actions(context: dict[str, Any]) -> list[Any]:
    """Expose the current native candidates, never a controller-generated route."""
    schema = context.get("action_schema", {})
    if not schema.get("classic_text"):
        return [_schema_instruction(context)]
    if schema.get("native_tools"):
        return deepcopy(list(schema["native_tools"]))
    actions = list(schema.get("available_actions", []))
    if not actions:
        for candidates in schema.get("mode_actions", {}).values():
            for action in candidates:
                if action not in actions:
                    actions.append(action)
    if schema.get("search_available") and "search[<your query>]" not in actions:
        actions.append("search[<your query>]")
    return actions


def _public_probe_actions(context: dict[str, Any]) -> list[Any]:
    schema = context.get("action_schema", {})
    if not schema.get("classic_text"):
        return [_schema_instruction(context) + "\nProbe actions read actual business state and must be read-only."]
    if "probe_actions" not in schema and "probe_tool_names" not in schema:
        raise ValueError("The native text environment must explicitly declare its public probe interface.")
    actions = list(schema.get("probe_actions", []))
    names = set(schema.get("probe_tool_names", []))
    for tool in schema.get("native_tools", []):
        definition = tool.get("function", tool)
        if definition.get("name") in names:
            actions.append(deepcopy(tool))
    return actions


def _update_public_schema(context: dict[str, Any], result: dict[str, Any]) -> None:
    schema = result.get("action_schema")
    if isinstance(schema, dict):
        context["action_schema"] = deepcopy(schema)


class PoSController:
    def __init__(self, repo: Path) -> None:
        sys.path.insert(0, str(repo))
        from baselines.react import ReAct
        from belief.manager import BeliefManager
        self.ReAct, self.BeliefManager = ReAct, BeliefManager
        self.repo = repo
        self.components = ["baselines/react.py", "belief/manager.py", "belief/beliefSentinel.py",
                           "belief/trappingDetection.py", "belief/recoveryPlanning.py"]

    def run(self, context: dict[str, Any], config: dict[str, Any], checkpoint=None) -> dict[str, Any]:
        import yaml
        benchmark = str(context["benchmark"]).lower()
        default_config = {"clindiag": "configs/clindiag/pos.yaml", "loca": "configs/loca/pos.yaml"}.get(
            benchmark, "configs/alfworld/pos.yaml")
        cfg_path = Path(config.get("pos", {}).get("config_path", self.repo / default_config))
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if benchmark == "clindiag":
            cfg["benchmark"] = "ClinDiag"
        elif benchmark == "loca":
            cfg["benchmark"] = "LOCA-Bench"
        elif benchmark != "alfworld":
            cfg["benchmark"] = str(context["benchmark"])
        cfg["max_steps"] = int(config.get("max_steps", 100))
        cfg["max_actions_per_turn"] = 1
        cfg["include_actions"] = True
        # Keep the fixed official configuration's bounded JSON repair budgets.
        cfg["context"]["sentinel"].update(enabled=True, consistency_enabled=True, progress_enabled=True)
        cfg["context"]["trapping"]["enabled"] = True
        cfg["context"]["recovery"]["enabled"] = True
        pos_cfg = dict(config.get("pos", {}))
        model = RPCModel(config, "pos_actor")
        logger = RPCLogger()

        def caller(role: str, token_limit: int):
            def call(messages):
                text = model.call(messages, role=role, max_tokens=token_limit)["text"]
                if role != "pos_consistency":
                    return text
                stripped = text.strip()
                if stripped.startswith("```json") and stripped.endswith("```"):
                    stripped = stripped[7:-3].strip()
                elif stripped.startswith("```") and stripped.endswith("```"):
                    stripped = stripped[3:-3].strip()
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    return text
                if isinstance(parsed, list):
                    normalized = json.dumps({"issues": parsed}, ensure_ascii=False)
                    logger.log("pos_consistency_container_normalized", raw_response=text,
                               normalized_response=normalized)
                    return normalized
                return text
            return call

        manager = self.BeliefManager(
            cfg, caller("pos_belief", int(pos_cfg.get("belief_max_tokens", 4096))), logger,
            call_sentinel=caller("pos_consistency", int(pos_cfg.get("sentinel_max_tokens", 512))),
            call_progress=caller("pos_progress", int(pos_cfg.get("progress_max_tokens", 512))),
        )
        ReAct = self.ReAct

        class NativeReAct(ReAct):
            def _build_messages(self, state, *, turn=0):
                messages = super()._build_messages(state, turn=turn)
                payload_instruction = ('Generate exactly one current native command or declared native tool object.'
                                       if context.get("action_schema", {}).get("classic_text") else
                                       'Generate concrete code and argument values using the public schema.')
                messages[0] = {**messages[0], "content": messages[0]["content"] + "\n" + _schema_instruction(context)
                               + '\nThe "action" may be the native JSON object or its JSON-encoded string. '
                               + payload_instruction}
                return messages

        class Adapter:
            def __init__(self):
                self.state = {"case_id": context["task_id"], "task": context["instruction"],
                              "observation": context["observation"], "actions": _public_actions(context),
                              "done": False, "success": False, "score": 0.0}

            def reset(self):
                return deepcopy(self.state)

            def step(self, action):
                result = _rpc("env_step", action=action, generation_id=model.last_generation_id)
                _update_public_schema(context, result)
                self.state.update(observation=result["observation"], done=result["done"],
                                  success=result["success"], score=float(result["success"]),
                                  actions=_public_actions(context))
                return deepcopy(self.state)

            def close(self):
                pass

        runtime = NativeReAct(cfg, caller("pos_actor", int(config.get("max_new_tokens", 512))),
                              logger, context_provider=manager)
        result = runtime.run(Adapter())
        return {"status": "completed", "official_result": result,
                "format_repair": "official_config_and_react_replacement_decision",
                "enabled_components": ["belief", "consistency", "progress", "trapping", "recovery"]}


class WorldEvolverController:
    def __init__(self, repo: Path) -> None:
        sys.path.insert(0, str(repo / "src"))
        from agents.wm_react_agent import WMReactAgent
        from world_model import build_wm
        spec = importlib.util.spec_from_file_location("native_worldevolver_common", repo / "src/tasks/new/common.py")
        common = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = common
        spec.loader.exec_module(common)
        self.agent_class, self.build_wm, self.mixin = WMReactAgent, build_wm, common.CLSWMTaskMixin
        self.model = None
        self.wm = None
        self.settings = None
        self.components = ["src/agents/wm_react_agent.py", "src/agents/react_agent.py",
                           "src/agents/foresight.py", "src/tasks/new/common.py",
                           "src/world_model/episodic_semantic.py", "src/world_model/modules/memory_mixins.py",
                           "src/world_model/modules/selective_foresight.py"]

    def run(self, context: dict[str, Any], config: dict[str, Any], checkpoint=None) -> dict[str, Any]:
        settings = dict(config.get("worldevolver", {}))
        sf_threshold = float(settings.get("sf_confidence_pct", 50.0))
        if not 0 < sf_threshold <= 100:
            raise ValueError("Complete WorldEvolver requires an enabled selective-foresight threshold.")
        if self.settings is not None and settings != self.settings:
            raise ValueError("WorldEvolver settings must stay fixed throughout the prequential sequence.")
        if self.wm is None:
            self.settings = deepcopy(settings)
            self.model = RPCModel(config, "worldevolver_actor")
            self.wm = self.build_wm(
                "wm-episodic-semantic", llm=self.model, env_name=str(context["benchmark"]),
                top_k=int(settings.get("top_k", 5)), batch_k=int(settings.get("batch_k", 1)),
                token_budget=int(settings.get("memory_token_budget", 2048)), filter_enabled=True,
                reset_scope="none", use_history=0, sf_confidence_pct=sf_threshold,
                wm_max_tokens=int(settings.get("wm_max_tokens", 1000)), episodic_retriever="jaccard_topk",
            )
            if checkpoint is not None:
                self.wm.load_checkpoint_state(checkpoint)
        self.wm.reset()

        def native_instruction():
            if context.get("action_schema", {}).get("classic_text"):
                output = (
                    "Return Thought: brief reasoning followed by Action: and exactly one native action "
                    "on a single line. Use the literal command for text actions, or the declared native "
                    "tool JSON object for a structured tool."
                )
            else:
                output = (
                    "Return Thought: brief reasoning followed by Action: and a JSON-encoded string "
                    "containing one native Python code block or PlanBench action block on a single line. "
                    "Encode newlines in that string as JSON escapes."
                )
            return "Solve the native interactive task from public observations. " + output + "\n" + _schema_instruction(context)

        instruction = native_instruction()
        agent = self.agent_class(self.model, memory_size=int(settings.get("memory_size", 100)),
                                 instruction=instruction, examples=[], need_goal=True,
                                 check_actions=None, check_inventory=None, use_parser=False,
                                 max_think_iters=int(settings.get("max_think_iters", 3)))
        agent.reset(str(context["instruction"]), str(context["observation"]))
        model = self.model
        mixin = self.mixin

        class NativePlanningLoop(mixin):
            def _stage3_wm_call(self, kind, call):
                with model.as_role(f"worldevolver_{kind}"):
                    return super()._stage3_wm_call(kind, call)

        loop = NativePlanningLoop()
        loop.agent, loop.wm = agent, self.wm
        loop.itp_max_k = 0
        loop.reset_stage3_planning_metrics()
        trajectory = loop.initial_trajectory(str(context["instruction"]), str(context["observation"]))
        observation = str(context["observation"])
        done = success = False
        for step in range(int(config.get("max_steps", 100))):
            if context.get("action_schema", {}).get("classic_text"):
                agent.instruction = native_instruction()
            loop.set_stage3_planning_context(0, 0, step)
            ok, action, prediction = loop.choose_action_with_current_foresight(
                run_agent=lambda: agent.run(), state=observation, goal=str(context["instruction"]),
                trajectory=trajectory, step_id=step, normalize_action=lambda x: str(x).strip(),
            )
            if not ok:
                raise RuntimeError("Official WorldEvolver actor failed to generate an action.")
            previous = observation
            result = _rpc("env_step", action=action)
            _update_public_schema(context, result)
            observation = str(result["observation"])
            done, success = bool(result["done"]), bool(result["success"])
            agent.update(action=action, state=observation)
            trajectory.extend([{"Action": action, "id": step}, {"Observation": observation, "id": step}])
            loop.update_wm_after_stage3_step(
                state=previous, action=action, prediction=prediction, gold_next_state=observation,
                info={"task": context["task_id"]},
            )
            loop.record_stage3_planning_step(0, 0, step)
            if done:
                break
        return {"status": "completed", "success": success, "done": done,
                "planning_metrics": loop.summarize_stage3_planning_metrics(),
                "checkpoint": self.wm.checkpoint_state(), "memory": self.wm.state_dict(),
                "sf_confidence_pct": sf_threshold, "trial_protocol": "success_at_1",
                "sf_threshold_definition": "100*exp(mean_prediction_token_logprob)>=sf_confidence_pct",
                "sf_threshold_kind": "absolute_probability_percent",
                "enabled_components": ["episodic_memory", "semantic_memory", "selective_foresight"]}


class EnvProbeController:
    def __init__(self, repo: Path, method: str) -> None:
        sys.path.insert(0, str(repo))
        self.agent_module = importlib.import_module("src.agents.llm_agent")
        self.methods = importlib.import_module("src.methods")
        self.base = importlib.import_module("src.methods.base")
        self.simple_module = importlib.import_module("src.methods.envprobe_simple")
        self.method_name = method
        self.components = ["src/agents/llm_agent.py", "src/agents/prompts.py", "src/methods/base.py",
                           "src/methods/envprobe_simple.py", "src/methods/envprobe_judge.py",
                           "src/methods/periodic_probe.py"]

    def run(self, context: dict[str, Any], config: dict[str, Any], checkpoint=None) -> dict[str, Any]:
        model = RPCModel(config, f"{self.method_name}_actor")
        schema = _schema_instruction(context)
        settings = dict(config.get("envprobe", {}))
        horizon = int(config.get("max_steps", 100))
        probe_budget = int(settings.get("probe_budget", max(1, horizon // 4)))
        if not 0 <= probe_budget <= horizon:
            raise ValueError("EnvProbe probe budget must lie within the shared episode horizon.")
        prefix = self.method_name

        decision_generations: dict[str, int] = {}
        actor_candidates: dict[str, Any] = {}

        class Client:
            def chat_json(self, system, user, model=None, max_tokens=1500, **kwargs):
                role = f"{prefix}_judge" if "EnvProbe judge" in system else f"{prefix}_actor"
                belief_instruction = (
                    "Use the upstream belief types that fit the public evidence, including spatial and "
                    "inventory types when applicable; infer facts only from public observations."
                    if context.get("action_schema", {}).get("classic_text") else
                    "Use tool_dep/subgoal/other for tool-workflow beliefs; infer facts only from public observations."
                )
                messages = [{"role": "system", "content": system + "\n" + schema
                             + "\nRepresent each native action as a JSON-encoded string in action/probe_action. "
                             + belief_instruction},
                            {"role": "user", "content": user}]
                if role.endswith("actor"):
                    if context.get("action_schema", {}).get("classic_text"):
                        candidate_instruction = (
                            "Also provide native_act_action: a string containing a concrete task action "
                            "from the same public history, for use when the controller selects act. "
                            "Generate this candidate in this response even when you propose a probe. "
                            "Use an executable native action from the full public task interface. "
                            "The controller's act/probe decision is separate from PRISM's I/E partition. "
                            "No reset or placeholder action exists."
                        )
                    else:
                        candidate_instruction = (
                            "Also provide native_act_action: a string containing a concrete execution action "
                            "from the same public history, for use when the controller selects act. "
                            "Generate this candidate in this response even when you propose a probe. "
                            "Use an executable native action. No reset or placeholder action exists."
                        )
                    messages[0]["content"] += (
                        "\nFor this native environment next_decision.type must be act or probe. " + candidate_instruction
                    )
                response = outer_model.call(messages, role=role, max_tokens=max_tokens)
                decision_generations["judge" if role.endswith("judge") else "actor"] = response["generation_id"]
                if role.endswith("actor"):
                    actor_candidates.clear()
                    actor_candidates["raw_output"] = response["text"]
                try:
                    result = _json_object(response["text"])
                    if role.endswith("judge"):
                        if result.get("decision") not in {"act", "probe"}:
                            raise ValueError("The judge must explicitly choose act or probe.")
                        if result["decision"] == "probe" and not result.get("probe_action"):
                            raise ValueError("The judge selected probe without an action.")
                    else:
                        proposed = result.get("next_decision")
                        if not isinstance(proposed, dict) or not proposed.get("action"):
                            raise ValueError("The actor returned no explicit native action.")
                        if proposed.get("type") not in {"act", "probe"}:
                            raise ValueError("The actor must explicitly choose act or probe.")
                except (TypeError, ValueError) as error:
                    raise InvalidNativeAction(response["text"], str(error)) from error
                if role.endswith("actor"):
                    actor_candidates["action"] = result.get("native_act_action")
                decision = result.get("next_decision")
                if isinstance(decision, dict) and isinstance(decision.get("action"), dict):
                    decision["action"] = _action_text(decision["action"])
                if isinstance(result.get("probe_action"), dict):
                    result["probe_action"] = _action_text(result["probe_action"])
                return result

        outer_model = model

        class NativeAgent(self.agent_module.LLMAgent):
            def step(self, *args, **kwargs):
                try:
                    return super().step(*args, **kwargs)
                except (TypeError, ValueError) as error:
                    # The official normalizer converts model-provided belief fields.
                    raise InvalidNativeAction(actor_candidates["raw_output"], str(error)) from error

        agent = NativeAgent(model="shared-native-backbone", client=Client(),
                            max_tokens=int(settings.get("actor_max_tokens", 1500)))
        agent.reset(str(context["benchmark"]), str(context["instruction"]))
        method = self.methods.get_method(self.method_name)
        if self.method_name == "envprobe_judge":
            method.attach_agent(agent)
            if getattr(method, "_agent", None) is None:
                raise RuntimeError("EnvProbe-Judge requires its complete judge callback.")
        method.reset_episode()

        def choose_native_action(ctx, belief=None):
            prompt = {"task": context["instruction"], "schema": schema,
                      "native_system_prompt": context.get("native_system_prompt", ""),
                      "observation": ctx.observation, "history": ctx.history,
                      "selected_belief": belief, "proposed_decision": ctx.agent_output.get("next_decision")}
            if context.get("action_schema", {}).get("classic_text"):
                prompt["allowed_probe_actions"] = _public_probe_actions(context)
                instruction = (
                    'Return JSON with key action: one native command string or declared tool object, or null. '
                    "Choose only from allowed_probe_actions to verify the selected belief with an actual "
                    "environment observation. Supply the declared arguments from public evidence, or generate "
                    "the question argument for an allowed clinical evidence tool. "
                    "Other information-purpose actions are not automatically probes. Omit any I/E mode field. "
                    "Return null when no allowed probe can verify the selected belief."
                )
            else:
                instruction = (
                    'Return JSON with key action: a string containing exactly one native action block, or null. '
                    + "Choose a legal read-only information query that can verify the selected belief. "
                      "Omit a mode field. Put executable read-only business API queries in a Python code block, "
                      "or an official retrieve_tools or read-only tool_call action block for PlanBench. "
                      "Retrieve documentation only when needed to discover the query; verify the actual business value. "
                      "Return null when no such query follows from the public schema and observations."
                )
            response = model.call([{"role": "system", "content": instruction},
                                   {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}],
                                  role=f"{prefix}_probe_mapping",
                                  max_tokens=int(config.get("max_new_tokens", 512)))
            try:
                action = _json_object(response["text"]).get("action")
            except (TypeError, ValueError) as error:
                raise InvalidNativeAction(response["text"], str(error)) from error
            if action is None or not isinstance(action, (str, dict)) or isinstance(action, str) and not action.strip():
                raise InvalidNativeAction(response["text"])
            return _action_text(action)

        # These are the two upstream environment-specific hooks. The scoring,
        # thresholds, judge and belief normalization remain upstream code.
        self.simple_module._probe_for_belief = lambda belief, ctx: choose_native_action(ctx, belief)

        def native_execution_candidate(ctx):
            action = actor_candidates.get("action")
            if not isinstance(action, (str, dict)) or not action:
                raise InvalidNativeAction(actor_candidates["raw_output"],
                                          "The actor omitted its native execution candidate.")
            return _action_text(action)

        self.base._heuristic_act = native_execution_candidate
        observation = {"native_observation": context["observation"]}
        history: list[dict[str, Any]] = []
        probe_count = 0
        done = success = False
        decisions = []
        public_env = type("NativePublicEnvironment", (), {"name": str(context["benchmark"])})()
        for step in range(horizon):
            output = {}
            schema = _schema_instruction(context)
            task_actions = _public_actions(context)
            probe_actions = _public_probe_actions(context)
            no_probe = bool(context.get("action_schema", {}).get("classic_text") and not probe_actions)
            try:
                output = agent.step(
                    observation, history, step, horizon, method_hint=method.method_hint,
                    task_action_spec=task_actions,
                    probe_action_spec=probe_actions,
                    probe_budget_remaining=probe_budget - probe_count,
                    probe_force_act=probe_count >= probe_budget or no_probe,
                )
                ctx = self.base.MethodContext(
                    step=step, horizon=horizon, observation=observation, agent_output=output,
                    task_action_spec=task_actions,
                    probe_action_spec=(probe_actions if context.get("action_schema", {}).get("classic_text") else [schema]),
                    env=public_env,
                    history=history, probe_budget_total=probe_budget, probes_used=probe_count,
                )
                decision = (method.force_act_from_agent(ctx, "no_public_probe_action") if no_probe
                            else method.decide(ctx))
                if not decision.action:
                    raise InvalidNativeAction(actor_candidates["raw_output"])
                probing = decision.decision_type == "probe"
                generation_id = model.last_generation_id if probing else decision_generations["actor"]
                action = decision.action
            except InvalidNativeAction as error:
                probing = False
                generation_id = model.last_generation_id
                action = {"kind": "invalid", "raw_text": getattr(error, "raw_output", model.last_text),
                          "validation_stage": "controller_output", "error": str(error)}
                decision = self.base.MethodDecision("invalid", action, reasoning=str(error))
            probe_count += int(probing)
            result = _rpc("env_step", action=action, probe=probing,
                          generation_id=generation_id)
            _update_public_schema(context, result)
            record = {"step": step, "action": decision.action, "type": decision.decision_type,
                      "target_belief": decision.target_belief_id, "reasoning": decision.reasoning,
                      "beliefs": output.get("beliefs"), "observation": result["observation"],
                      "agent_error": output.get("_llm_error")}
            decisions.append(record)
            history.append({"step": step, "action": decision.action, "type": decision.decision_type,
                            "observation": result["observation"]})
            observation = {"native_observation": result["observation"]}
            done, success = bool(result["done"]), bool(result["success"])
            if done:
                break
        return {"status": "completed", "success": success, "done": done,
                "native_execution_mapping": "same_actor_response_explicit_candidate_no_gold_state",
                "probe_budget": probe_budget, "probes_used": probe_count, "decisions": decisions,
                "enabled_components": ["belief_table", "four_dimensional_score" if self.method_name == "envprobe_simple"
                                       else "judge", "native_probe", "feedback_replanning"]}


def _provenance(repo: Path, controller: Any) -> dict[str, Any]:
    return {"repo": str(repo),
            "method": getattr(controller, "method_name", type(controller).__name__),
            "adaptation": "official-native-actions-and-shared-model-callbacks-v2"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                        choices=("pos", "worldevolver", "envprobe_simple", "envprobe_judge"))
    parser.add_argument("--repo", required=True, type=Path)
    args = parser.parse_args()
    # Upstream diagnostic print() output cannot corrupt the RPC stream.
    sys.stdout = sys.stderr
    controller = None
    for line in sys.stdin:
        command = json.loads(line)
        if command.get("kind") != "run":
            raise ValueError("Expected an episode command.")
        result: dict[str, Any]
        requests_before = _REQUEST_ID
        try:
            if controller is None:
                controller = (PoSController(args.repo) if args.method == "pos" else
                              WorldEvolverController(args.repo) if args.method == "worldevolver" else
                              EnvProbeController(args.repo, args.method))
            random.seed(int(command["seed"]))
            _rpc("begin_episode")
            result = controller.run(command["context"], command["config"], command.get("checkpoint"))
        except (InvalidNativeAction, UpstreamCallError) as exc:
            result = {"status": "method_error", "error": str(exc),
                      "error_type": type(exc).__name__, "raw_upstream_output": exc.raw_output}
        except RemoteBudgetExhausted as exc:
            result = {"status": "budget_exhausted", "reason": str(exc)}
            if args.method == "worldevolver" and getattr(controller, "wm", None) is not None:
                result["checkpoint"] = controller.wm.checkpoint_state()
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            result = {"status": "configuration_error" if controller is None or _REQUEST_ID == requests_before else "method_error",
                      "error": f"{type(exc).__name__}: {exc}", "error_type": type(exc).__name__}
        if controller is not None:
            result["provenance"] = _provenance(args.repo, controller)
        _emit({"kind": "result", "result": result})


if __name__ == "__main__":
    main()
