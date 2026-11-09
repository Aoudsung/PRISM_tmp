"""Method interface for choosing an action or probe at each step.

Methods may override the agent's proposal, but probing is capped at one quarter
of the episode horizon. Once that budget is exhausted, the method must act.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MethodDecision:
    """Final decision returned by a method."""
    decision_type: str          # "act", "probe", or "reset"
    action: str                 # Action string passed to the environment
    target_belief_id: Optional[str] = None
    reasoning: str = ""
    overrode_agent: bool = False  # Whether the method replaced the agent proposal


@dataclass
class MethodContext:
    """Context supplied to :meth:`Method.decide`."""
    step: int
    horizon: int
    observation: Dict[str, Any]
    agent_output: Dict[str, Any]    # Parsed agent response
    task_action_spec: List[str]
    probe_action_spec: List[str]
    env: Any                        # Environment, also used by oracle methods
    history: List[Dict[str, Any]] = field(default_factory=list)
    probe_budget_total: int = 0
    probes_used: int = 0


class Method:
    name: str = "base"
    method_hint: str = ""           # Optional guidance appended to the agent prompt

    def reset_episode(self) -> None:
        pass

    def decide(self, ctx: MethodContext) -> MethodDecision:
        """Choose the next step from the agent output, state, and probe budget."""
        raise NotImplementedError

    # Shared probe-budget helpers
    @staticmethod
    def is_budget_exhausted(ctx: MethodContext) -> bool:
        return ctx.probes_used >= ctx.probe_budget_total

    @staticmethod
    def take_agent_action(ctx: MethodContext) -> MethodDecision:
        """Return the agent's proposal unchanged."""
        nd = ctx.agent_output.get("next_decision", {})
        return MethodDecision(
            decision_type=str(nd.get("type", "act")),
            action=str(nd.get("action", "noop()")),
            target_belief_id=nd.get("target_belief"),
            reasoning="agent_decision",
            overrode_agent=False,
        )

    @staticmethod
    def force_act_from_agent(ctx: MethodContext, reason: str) -> MethodDecision:
        """Return an action, replacing a proposed probe with a fallback if needed."""
        nd = ctx.agent_output.get("next_decision", {})
        agent_type = str(nd.get("type", "act"))
        agent_action = str(nd.get("action", "noop()"))
        if agent_type == "act":
            return MethodDecision(decision_type="act", action=agent_action,
                                  reasoning=reason, overrode_agent=False)
        # The probe budget is exhausted, so select a valid task action.
        fallback = _heuristic_act(ctx)
        return MethodDecision(decision_type="act", action=fallback,
                              reasoning=f"{reason}|force_act_fallback",
                              overrode_agent=True)


def _heuristic_act(ctx: MethodContext) -> str:
    """Choose a deterministic fallback action from the current observation."""
    obs = ctx.observation or {}
    env_name = type(ctx.env).__name__
    if env_name == "ObjectStateWorld":
        # Prefer visible objects; otherwise move toward an adjacent room.
        if obs.get("objects_in_view"):
            return f"pick_up({obs['objects_in_view'][0]})"
        doors = obs.get("doors_in_view") or []
        if doors:
            # Door endpoints are not exposed, so infer the next room identifier.
            cur = obs.get("current_location", "room_0")
            if isinstance(cur, str) and cur.startswith("room_"):
                try:
                    idx = int(cur.split("_")[1])
                    return f"move_to(room_{idx + 1})"
                except Exception:
                    pass
        return "move_to(room_1)"
    if env_name == "ToolDAGWorld":
        completed = set(obs.get("completed_tools") or [])
        avail = set(obs.get("available_variables") or [])
        # Choose the first unfinished tool whose inputs are available.
        env_gold = ctx.env.get_gold_state() if hasattr(ctx.env, "get_gold_state") else {}
        for t in env_gold.get("tools", []):
            if t in completed:
                continue
            req = env_gold.get("tool_inputs", {}).get(t, [])
            if all(v in avail for v in req):
                return f"call_tool({t})"
        return "call_tool(t_0)"
    if env_name == "GraphNavWorld":
        neighbors = obs.get("neighbors") or []
        if neighbors:
            return f"move_to({neighbors[0]})"
        keys_here = obs.get("keys_in_view") or []
        if keys_here:
            return f"collect_key({keys_here[0]})"
        return "move_to(n_1)"
    return "noop()"
