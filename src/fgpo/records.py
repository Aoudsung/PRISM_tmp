"""Shared F-GPO data contract: immutable integer segments and actual interaction records.

Actor segments alone own PPO responses.  Canonical full-history segments supply
reference conditioning and never add a second sampled action or response mask.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Mapping
import uuid


def monotonic_seconds() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC)


@dataclass(frozen=True)
class CostRecord:
    event_id: str
    kind: str
    role: str
    input_tokens: int | None
    output_tokens: int | None
    forward_tokens: int | None
    seconds: float | None
    gpu_ids: tuple[int, ...]
    details: Mapping[str, Any] = field(default_factory=dict)
    event_index: int = 0
    env_steps: int = 0
    session_id: str = ""
    session_elapsed_seconds: float | None = None
    session_device_seconds: float | None = None


def _last_event_index(path):
    """Read just the last complete journal row to number new append operations."""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        if not position:
            return 0
        handle.seek(position - 1)
        if handle.read(1) != b"\n":
            raise ValueError("The physical-cost journal ends with an incomplete event.")
        end, pieces = position - 1, []
        while end:
            start = max(0, end - 8192)
            handle.seek(start)
            block = handle.read(end - start)
            boundary = block.rfind(b"\n")
            if boundary >= 0:
                pieces.append(block[boundary + 1:])
                break
            pieces.append(block)
            end = start
        return int(json.loads(b"".join(reversed(pieces)))["event_index"])


class CostLedger:
    """Append-only actual work; no policy resource vector or training cutoff."""
    def __init__(self, path: str | Path | None = None, *, gpu_ids=()):
        self.path = None if path is None else Path(path)
        self.gpu_ids = tuple(int(device) for device in gpu_ids)
        self.records: list[CostRecord] = []
        self.started = monotonic_seconds()
        self.session_id = uuid.uuid4().hex
        self.event_cursor = (_last_event_index(self.path)
                             if self.path is not None and self.path.exists() else 0)
        self.durable_offset_bytes = 0
        self._elapsed_before = 0.
        self._device_seconds_before = 0.
        self._totals = {name: 0 for name in ("input_tokens", "output_tokens", "forward_tokens", "env_steps", "seconds")}
        self._usage_complete = True
        self.recovery = None
        self.active_group_id = None

    def record(self, kind, role, *, input_tokens=0, output_tokens=0, forward_tokens=0,
               seconds=0., gpu_ids=None, details=None, env_steps=0):
        self.event_cursor += 1
        elapsed = monotonic_seconds() - self.started
        event_details = {} if details is None else dict(details)
        if self.active_group_id is not None:
            event_details.setdefault("group_id", self.active_group_id)
        row = CostRecord(uuid.uuid4().hex, str(kind), str(role), input_tokens, output_tokens,
                         forward_tokens, seconds, self.gpu_ids if gpu_ids is None else tuple(gpu_ids),
                         event_details, self.event_cursor, int(env_steps),
                         self.session_id, elapsed, elapsed * len(self.gpu_ids))
        self.records.append(row)
        for name in self._totals:
            value = getattr(row, name)
            if value is None:
                self._usage_complete = False
            else:
                self._totals[name] += value
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("ab") as handle:
                handle.write((json.dumps(asdict(row), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
                self.durable_offset_bytes = handle.tell()
        return row

    def snapshot(self):
        elapsed = monotonic_seconds() - self.started
        return {"events": self.event_cursor, "wall_seconds": self._elapsed_before + elapsed,
                "allocated_device_seconds": self._device_seconds_before + elapsed * len(self.gpu_ids),
                **self._totals, "usage_complete": self._usage_complete,
                "durable_offset_bytes": self.durable_offset_bytes}

    def state_dict(self):
        current = self.snapshot()
        return {"totals": dict(self._totals), "usage_complete": self._usage_complete,
                "event_cursor": self.event_cursor, "durable_offset_bytes": self.durable_offset_bytes,
                "elapsed_active_seconds": current["wall_seconds"],
                "allocated_device_seconds": current["allocated_device_seconds"],
                "session_id": self.session_id,
                "session_elapsed_seconds": current["wall_seconds"] - self._elapsed_before,
                "session_device_seconds": current["allocated_device_seconds"] - self._device_seconds_before,
                "journal_path": None if self.path is None else str(self.path.resolve()),
                "active_group_id": self.active_group_id,
                "recovery": self.recovery}

    def load_state_dict(self, state):
        """Restore algorithm counters while adopting each actual durable tail row once.

        Current-process initialization is already in the same journal.  Read it
        through that journal, rather than adding its in-memory totals twice.
        Only session maxima and interrupted group identities are retained.
        """
        restore_started = monotonic_seconds()
        totals = dict(state["totals"])
        complete = bool(state["usage_complete"])
        cursor, offset = int(state["event_cursor"]), int(state["durable_offset_bytes"])
        sessions, groups, imported = {}, set(), 0
        if self.path is None:
            raise ValueError("Exact fee restoration requires the checkpoint's original durable physical-cost journal.")
        else:
            if state["journal_path"] != str(self.path.resolve()):
                raise ValueError("Resume must use the checkpoint's actual physical-cost journal.")
            with self.path.open("rb") as handle:
                handle.seek(0, 2)
                if handle.tell() < offset:
                    raise ValueError("The physical-cost journal is shorter than its saved checkpoint cursor.")
                handle.seek(offset)
                for line in handle:
                    if not line.endswith(b"\n"):
                        raise ValueError("The physical-cost journal contains an incomplete durable tail event.")
                    row = json.loads(line)
                    if int(row["event_index"]) != cursor + 1:
                        raise ValueError("The durable physical-cost event positions are discontinuous.")
                    cursor += 1
                    imported += 1
                    for name in totals:
                        value = row[name]
                        if value is None:
                            complete = False
                        else:
                            totals[name] += value
                    session = row["session_id"]
                    if session != self.session_id:
                        elapsed = row["session_elapsed_seconds"]
                        device = row["session_device_seconds"]
                        if elapsed is None or device is None:
                            raise ValueError("Durable tail events require their actual session timing coordinates.")
                        previous = sessions.get(session, (0., 0.))
                        sessions[session] = (max(previous[0], float(elapsed)), max(previous[1], float(device)))
                        group = row["details"].get("group_id")
                        if group is not None:
                            groups.add(str(group))
                offset = handle.tell()
        previous_wall, previous_device = 0., 0.
        for session, (elapsed, device) in sessions.items():
            if session == state["session_id"]:
                elapsed -= float(state["session_elapsed_seconds"])
                device -= float(state["session_device_seconds"])
            previous_wall += max(0., elapsed)
            previous_device += max(0., device)
        self._totals, self._usage_complete = totals, complete
        self.event_cursor, self.durable_offset_bytes = cursor, offset
        self._elapsed_before = float(state["elapsed_active_seconds"]) + previous_wall
        self._device_seconds_before = float(state["allocated_device_seconds"]) + previous_device
        if state["session_id"] == self.session_id:
            self._elapsed_before -= float(state["session_elapsed_seconds"])
            self._device_seconds_before -= float(state["session_device_seconds"])
        # Keep started: initialization and restore time are the current active
        # segment, and must not be restarted or added twice here.
        self.recovery = {"source_checkpoint_offset_bytes": int(state["durable_offset_bytes"]),
                         "adopted_through_offset_bytes": offset, "imported_events": imported,
                         "interrupted_group_ids": sorted(groups),
                         "checkpoint_active_group_id": state["active_group_id"],
                         "environment_continuation": "not_restored",
                         "unrecorded_interruption_wall_seconds": None,
                         "unrecorded_interruption_device_seconds": None,
                         "durable_tail_available": self.path is not None}
        self.record("controller", "ledger.restore", seconds=monotonic_seconds() - restore_started,
                    details=self.recovery)

    def begin_group(self, group_id=None):
        """Bound in-memory retention; aggregate counters and the complete JSONL remain intact."""
        self.records.clear()
        self.active_group_id = None if group_id is None else str(group_id)


@dataclass(frozen=True)
class TokenSpan:
    segment_id: str
    start: int
    length: int
    step_id: str

    @property
    def stop(self):
        return self.start + self.length


SpanRef = TokenSpan


@dataclass(frozen=True)
class PublicObservation:
    raw_observation: str
    pure_text: str
    pure_ids: tuple[int, ...]
    representation_ids: tuple[int, ...]
    user_prefix_ids: tuple[int, ...]
    user_body_ids: tuple[int, ...]
    observation_body_offset: int
    available_actions: tuple[str, ...] = ()


@dataclass(frozen=True)
class TurnBlock:
    step_id: str
    response_ids: tuple[int, ...]
    closure_ids: tuple[int, ...]
    user_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    observation_offset: int
    observation_ids: tuple[int, ...]


@dataclass(frozen=True)
class SegmentRecord:
    segment_id: str
    token_ids: tuple[int, ...]
    response_spans: tuple[TokenSpan, ...]
    observation_spans: tuple[TokenSpan, ...]
    behavior_closed_length: int
    scoring_tail_start: int | None
    kind: str = "actor"


@dataclass(frozen=True)
class StepRecord:
    step_id: str
    trajectory_id: str
    index: int
    segment_id: str
    prompt_length: int
    response_span: TokenSpan
    observation_span: TokenSpan
    response_ids: tuple[int, ...]
    old_token_logprobs: tuple[float, ...]
    observation_ids: tuple[int, ...]
    observation_repr_ids: tuple[int, ...]
    observation_text: str
    raw_observation: str
    action: Mapping[str, Any]
    environment_reward: float
    done: bool
    syntax_valid: bool
    turn_block: TurnBlock
    full_reference_segment_id: str
    full_reference_prompt_length: int
    full_reference_response_span: TokenSpan
    full_reference_observation_span: TokenSpan
    costs: tuple[CostRecord, ...] = ()
    environment_info: Mapping[str, Any] = field(default_factory=dict)
    shared_cost_event_ids: tuple[str, ...] = ()

    @property
    def old_token_log_probs(self):
        return self.old_token_logprobs


@dataclass(frozen=True)
class TrajectoryRecord:
    trajectory_id: str
    group_id: str
    slot: int
    task_id: str
    benchmark: str
    initial_seed: int
    sampling_seed: int
    goal: str
    goal_ids: tuple[int, ...]
    initial_observation_ids: tuple[int, ...]
    pinned_ids: tuple[int, ...]
    steps: tuple[StepRecord, ...]
    segments: tuple[SegmentRecord, ...]
    full_reference_segments: tuple[SegmentRecord, ...]
    official_success: bool
    official_metrics: Mapping[str, Any]
    costs: tuple[CostRecord, ...]
    history_length: int | None

    @property
    def success(self):
        return self.official_success


@dataclass(frozen=True)
class GroupRecord:
    group_id: str
    task_id: str
    benchmark: str
    initial_seed: int
    trajectories: tuple[TrajectoryRecord, ...]
    costs: tuple[CostRecord, ...]
