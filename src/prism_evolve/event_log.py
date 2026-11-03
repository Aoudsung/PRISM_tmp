"""Module 1: Immutable Public Event Log and Public History Container.

Implements §1.4, §1.5, and §8.1 of the PRISM-Evolve design specification:
- Public interaction history h_t = (g, o_0, a_0, o_1, ..., a_{t-1}, o_t)
- Immutable EventRecord with unique occurrence_id, artifact_version, content_hash,
  and step time/resource cost Delta C_t.
- Remaining budget tracking B_t = B - C_t.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from typing import Any, Mapping
import uuid

INFORMATION_INTENTS = frozenset({"discover", "probe"})
EXECUTION_INTENTS = frozenset({"execute", "submit"})
ALL_INTENTS = INFORMATION_INTENTS | EXECUTION_INTENTS
VALID_MODES = frozenset({"I", "E"})


def mode_for_intent(intent: str) -> str:
    """Map public action intent to mode I={discover, probe} or E={execute, submit}."""
    if intent in INFORMATION_INTENTS:
        return "I"
    if intent in EXECUTION_INTENTS:
        return "E"
    raise ValueError(f"Unknown action intent {intent!r}; expected one of {sorted(ALL_INTENTS)}")


def compute_content_hash(action_text: str, observation: str, artifact_version: str) -> str:
    """Compute deterministic SHA-256 content hash for an event record."""
    hasher = hashlib.sha256()
    payload = json.dumps(
        {
            "action": action_text,
            "observation": observation,
            "artifact_version": artifact_version,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    hasher.update(payload.encode("utf-8"))
    return hasher.hexdigest()


@dataclass(frozen=True)
class CandidateAction:
    """A legal candidate action available at a public history state h_t."""

    action_id: str
    intent: str
    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    mode: str = ""

    def __post_init__(self) -> None:
        expected_mode = mode_for_intent(self.intent)
        if not self.mode:
            object.__setattr__(self, "mode", expected_mode)
        elif self.mode != expected_mode:
            raise ValueError(
                f"Mode {self.mode!r} contradicts intent {self.intent!r} (expected {expected_mode!r})"
            )
        if not self.raw_text:
            args_str = json.dumps(dict(self.arguments), sort_keys=True, ensure_ascii=False)
            object.__setattr__(self, "raw_text", f"[{self.mode}:{self.intent}] {self.tool_name}({args_str})")

    @property
    def is_terminal_submit(self) -> bool:
        return self.intent == "submit"


@dataclass(frozen=True)
class EventRecord:
    """Immutable record of a single environment interaction event at step t."""

    step_index: int
    action: CandidateAction
    observation: str
    step_cost: float
    occurrence_id: str = ""
    artifact_version: str = "v0"
    content_hash: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError(f"step_index must be >= 0, got {self.step_index}")
        if self.step_cost < 0.0:
            raise ValueError(f"step_cost must be >= 0.0, got {self.step_cost}")
        if not self.occurrence_id:
            object.__setattr__(
                self,
                "occurrence_id",
                f"occ-{self.step_index}-{uuid.uuid4().hex[:12]}",
            )
        if not self.content_hash:
            object.__setattr__(
                self,
                "content_hash",
                compute_content_hash(self.action.raw_text, self.observation, self.artifact_version),
            )

    @property
    def mode(self) -> str:
        return self.action.mode

    def with_masked_observation(self, placeholder: str = "[MASKED_FEEDBACK]") -> EventRecord:
        """Return a copy with masked observation payload while preserving step_cost and occurrence_id.

        Used strictly by the dev-set masked feedback diagnostic Delta_use(pi) (§6.2).
        """
        return replace(
            self,
            observation=placeholder,
            content_hash=compute_content_hash(self.action.raw_text, placeholder, self.artifact_version),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "occurrence_id": self.occurrence_id,
            "mode": self.mode,
            "intent": self.action.intent,
            "action_id": self.action.action_id,
            "tool_name": self.action.tool_name,
            "arguments": dict(self.action.arguments),
            "raw_text": self.action.raw_text,
            "observation": self.observation,
            "step_cost": self.step_cost,
            "artifact_version": self.artifact_version,
            "content_hash": self.content_hash,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PublicHistory:
    """Immutable public interaction history h_t = (g, o_0, a_0, o_1, ..., a_{t-1}, o_t).

    Reset completely at the start of every new task instance (§2.1).
    """

    task_id: str
    goal: str
    initial_observation: str
    total_budget: float
    scenario_id: str = ""
    events: tuple[EventRecord, ...] = ()

    def __post_init__(self) -> None:
        if self.total_budget <= 0.0:
            raise ValueError(f"total_budget B must be > 0, got {self.total_budget}")
        if not self.scenario_id:
            object.__setattr__(self, "scenario_id", self.task_id)

    @property
    def step_count(self) -> int:
        return len(self.events)

    @property
    def elapsed_cost(self) -> float:
        return float(sum(ev.step_cost for ev in self.events))

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self.total_budget - self.elapsed_cost)

    @property
    def budget_fraction_remaining(self) -> float:
        return max(0.0, min(1.0, self.remaining_budget / self.total_budget))

    def append(self, event: EventRecord) -> PublicHistory:
        """Return a new PublicHistory with `event` appended."""
        if event.step_index != len(self.events):
            raise ValueError(
                f"Expected event step_index={len(self.events)}, got {event.step_index}"
            )
        return replace(self, events=self.events + (event,))

    def with_masked_step(
        self,
        step_index: int,
        placeholder: str = "[MASKED_FEEDBACK]",
    ) -> PublicHistory:
        """Return a new PublicHistory with observation at `step_index` replaced by `placeholder`.

        Preserves identical elapsed_cost and remaining_budget (§6.2).
        """
        if not (0 <= step_index < len(self.events)):
            raise IndexError(f"step_index {step_index} out of range [0, {len(self.events)})")
        updated = list(self.events)
        updated[step_index] = updated[step_index].with_masked_observation(placeholder=placeholder)
        return replace(self, events=tuple(updated))

    def format_context(self, max_observation_chars: int = 2000) -> str:
        """Serialize public history h_t and remaining budget B_t into model-readable context."""
        lines = [
            f"[Task ID] {self.task_id}",
            f"[Goal] {self.goal}",
            f"[Budget] Remaining: {self.remaining_budget:.4f} / {self.total_budget:.4f} "
            f"({self.budget_fraction_remaining * 100.0:.1f}%)",
            f"[Initial Observation o_0] {self.initial_observation[:max_observation_chars]}",
        ]
        for ev in self.events:
            obs_snippet = ev.observation[:max_observation_chars]
            lines.append(
                f"[Step {ev.step_index} | occ={ev.occurrence_id} | cost={ev.step_cost:.4f} | "
                f"ver={ev.artifact_version}] Action: {ev.action.raw_text}"
            )
            lines.append(f"[Observation o_{ev.step_index + 1}] {obs_snippet}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "scenario_id": self.scenario_id,
            "goal": self.goal,
            "initial_observation": self.initial_observation,
            "total_budget": self.total_budget,
            "elapsed_cost": self.elapsed_cost,
            "remaining_budget": self.remaining_budget,
            "events": [ev.to_dict() for ev in self.events],
        }
