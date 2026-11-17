"""Shared F-GPO data contract: immutable integer segments and actual interaction records.

Actor segments alone own PPO responses. Reference and update computations use
those same actual bounded prefixes without creating additional response masks.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping
import uuid


_UNSET = object()


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
    """Keep complete physical events; discard only a torn final append."""
    with path.open("r+b") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        if not position:
            return 0, 0
        handle.seek(position - 1)
        discarded = 0
        if handle.read(1) != b"\n":
            original, end = position, position
            position = 0
            while end:
                start = max(0, end - 8192)
                handle.seek(start)
                boundary = handle.read(end - start).rfind(b"\n")
                if boundary >= 0:
                    position = start + boundary + 1
                    break
                end = start
            discarded = original - position
            handle.truncate(position)
            handle.flush()
            os.fsync(handle.fileno())
            if not position:
                return 0, discarded
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
        return int(json.loads(b"".join(reversed(pieces)))["event_index"]), discarded


def _merge_tail_observations(*sources):
    merged = {}
    for source in sources:
        for identity, size in source.items():
            merged[identity] = max(merged.get(identity, 0), int(size))
    return merged


def _cost_tail_id(writer_session_id, committed_bytes):
    # Older journals have one unlabelled observation at their monotonic prefix.
    origin = f"writer:{writer_session_id}" if writer_session_id else "legacy:costs-monotonic"
    return f"{origin}:{int(committed_bytes)}"


def _read_cost_tail_state(path: Path | None):
    empty = {"abandoned_costs_tail_bytes": 0, "consumed_tail_observations": {},
             "writer_session_id": None}
    if path is None:
        return empty
    state_file = path.parent / ".cost-tail-state.json"
    if not state_file.is_file():
        # Import the existing cumulative counter without making the GCS
        # publisher a writer of this cost-owner/recovery provenance file.
        state_file = path.parent / ".gcs-journal-state.json"
    if not state_file.is_file():
        return empty
    data = json.loads(state_file.read_text(encoding="utf-8"))
    return {"abandoned_costs_tail_bytes": int(data.get("abandoned_costs_tail_bytes", 0)),
            "consumed_tail_observations": dict(data.get("consumed_tail_observations", {})),
            "writer_session_id": data.get("writer_session_id")}


def _write_cost_tail_state(path: Path | None, *, count=0, consumed=None, writer_session_id=_UNSET):
    previous = _read_cost_tail_state(path)
    if path is None:
        return previous
    data = {"abandoned_costs_tail_bytes": max(previous["abandoned_costs_tail_bytes"], int(count)),
            "consumed_tail_observations": _merge_tail_observations(
                previous["consumed_tail_observations"], {} if consumed is None else consumed),
            "writer_session_id": previous["writer_session_id"]}
    if writer_session_id is not _UNSET:
        data["writer_session_id"] = writer_session_id
    state_file = path.parent / ".cost-tail-state.json"
    if data == previous and state_file.is_file():
        return data
    state_file.parent.mkdir(parents=True, exist_ok=True)
    pending = state_file.with_name(state_file.name + f".{uuid.uuid4().hex}.incoming")
    with pending.open("wb") as handle:
        handle.write((json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, state_file)
    directory = os.open(state_file.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return data


class CostLedger:
    """Append-only actual work; no policy resource vector or training cutoff."""
    def __init__(self, path: str | Path | None = None, *, gpu_ids=()):
        self.path = None if path is None else Path(path)
        self.gpu_ids = tuple(int(device) for device in gpu_ids)
        self.records: list[CostRecord] = []
        self.started = monotonic_seconds()
        self.session_id = uuid.uuid4().hex
        tail_state = _read_cost_tail_state(self.path)
        self.event_cursor, self._discarded_tail_bytes = (_last_event_index(self.path)
                             if self.path is not None and self.path.exists() else (0, 0))
        self._prior_abandoned_tail_bytes = tail_state["abandoned_costs_tail_bytes"]
        self._consumed_tail_observations = dict(tail_state["consumed_tail_observations"])
        newly_discarded = 0
        if self._discarded_tail_bytes:
            identity = _cost_tail_id(tail_state["writer_session_id"], self.path.stat().st_size)
            newly_discarded = max(0, self._discarded_tail_bytes
                                  - self._consumed_tail_observations.get(identity, 0))
            self._consumed_tail_observations[identity] = max(
                self._discarded_tail_bytes, self._consumed_tail_observations.get(identity, 0))
        self._cumulative_abandoned_tail_bytes = self._prior_abandoned_tail_bytes + newly_discarded
        self._abandoned_tail_bytes_before = 0
        _write_cost_tail_state(self.path, count=self._cumulative_abandoned_tail_bytes,
                               consumed=self._consumed_tail_observations, writer_session_id=self.session_id)
        self.durable_offset_bytes = 0
        self._elapsed_before = 0.
        self._device_seconds_before = 0.
        self._startup_wall_seconds = 0.
        self._startup_device_seconds = 0.
        self._startup_credited = False
        self._totals = {name: 0 for name in ("input_tokens", "output_tokens", "forward_tokens", "env_steps", "seconds")}
        self._usage_complete = not self._cumulative_abandoned_tail_bytes
        self.recovery = None
        self.active_group_id = None
        self.active_round = None
        self._model_scope_depth = 0
        self._model_calls = {}

    def credit_startup(self, *, pre_engine_seconds=0., startup_details=None, cold_asset_receipt=None):
        """Anchor session wall/device clocks at actual cell startup and record intervals once."""
        if self._startup_credited:
            raise RuntimeError("Startup intervals can only be credited once per ledger session")
        self._startup_credited = True
        cold_seconds = 0.
        if cold_asset_receipt is not None:
            cold_seconds = max(0., float(cold_asset_receipt.get("seconds") or 0.))
        pre_seconds = max(0., float(pre_engine_seconds))
        total_startup = cold_seconds + pre_seconds
        self._startup_wall_seconds = total_startup
        self._startup_device_seconds = total_startup * len(self.gpu_ids)
        if cold_asset_receipt is not None:
            self.record("initialization", "runner.cold_asset_restore",
                        seconds=cold_seconds, details=dict(cold_asset_receipt))
        self.record("initialization", "runner.cell_startup",
                    seconds=pre_seconds, details={} if startup_details is None else dict(startup_details))

    @contextmanager
    def context(self, *, group_id=_UNSET, round_index=_UNSET):
        """Set round/group attribution cleanly, flushing any prior model calls."""
        self._flush_model_calls()
        previous_group = self.active_group_id
        previous_round = self.active_round
        if group_id is not _UNSET:
            self.active_group_id = None if group_id is None else str(group_id)
        if round_index is not _UNSET:
            self.active_round = None if round_index is None else int(round_index)
        try:
            yield
        finally:
            self._flush_model_calls()
            self.active_group_id = previous_group
            self.active_round = previous_round

    @contextmanager
    def aggregate_model_calls(self):
        """Keep exact work totals with one journal row per role in an operation.

        Decode calls only need aggregate work for the manuscript's efficiency
        claim. Environment transitions and synchronized phases remain separate.
        Completed work is flushed on normal exit and on an exception.
        """
        self._model_scope_depth += 1
        try:
            yield
        finally:
            self._model_scope_depth -= 1
            if self._model_scope_depth == 0:
                self._flush_model_calls()

    def _flush_model_calls(self):
        if not self._model_calls:
            return
        pending, self._model_calls = self._model_calls, {}
        depth, self._model_scope_depth = self._model_scope_depth, 0
        try:
            for item in pending.values():
                self.record("model", **item)
        finally:
            self._model_scope_depth = depth

    def record(self, kind, role, *, input_tokens=0, output_tokens=0, forward_tokens=0,
               seconds=0., gpu_ids=None, details=None, env_steps=0):
        if kind == "model" and self._model_scope_depth:
            devices = self.gpu_ids if gpu_ids is None else tuple(gpu_ids)
            detail = {} if details is None else dict(details)
            key = (role, devices, detail.get("time_basis"), detail.get("status"),
                   detail.get("use_cache"), detail.get("executed_decoder_layers"),
                   detail.get("attention_mask_omitted"),
                   self.active_group_id, self.active_round)
            values = dict(input_tokens=input_tokens, output_tokens=output_tokens,
                          forward_tokens=forward_tokens, seconds=seconds, env_steps=env_steps)
            if key not in self._model_calls:
                detail.update(aggregation="model_calls_in_operation", calls=1)
                self._model_calls[key] = dict(role=role, gpu_ids=devices, details=detail, **values)
            else:
                item = self._model_calls[key]
                for name, value in values.items():
                    item[name] = None if item[name] is None or value is None else item[name] + value
                item["details"]["calls"] += 1
                for name in ("padded_compute_tokens", "positions", "cached_query_key_positions"):
                    if name in detail:
                        item["details"][name] = item["details"].get(name, 0) + detail[name]
            return None
        if kind != "model":
            self._flush_model_calls()
        self.event_cursor += 1
        active_seconds = monotonic_seconds() - self.started
        elapsed = self._startup_wall_seconds + active_seconds
        device_elapsed = self._startup_device_seconds + active_seconds * len(self.gpu_ids)
        event_details = {} if details is None else dict(details)
        if self.active_group_id is not None:
            event_details.setdefault("group_id", self.active_group_id)
        if self.active_round is not None:
            event_details.setdefault("round", self.active_round)
        row = CostRecord(uuid.uuid4().hex, str(kind), str(role), input_tokens, output_tokens,
                         forward_tokens, seconds, self.gpu_ids if gpu_ids is None else tuple(gpu_ids),
                         event_details, self.event_cursor, int(env_steps),
                         self.session_id, elapsed, device_elapsed)
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

    def _abandoned_tail_total(self):
        return max(self._abandoned_tail_bytes_before, self._cumulative_abandoned_tail_bytes)

    def snapshot(self):
        self._flush_model_calls()
        active_seconds = monotonic_seconds() - self.started
        elapsed = self._startup_wall_seconds + active_seconds
        device_elapsed = self._startup_device_seconds + active_seconds * len(self.gpu_ids)
        return {"events": self.event_cursor, "wall_seconds": self._elapsed_before + elapsed,
                "allocated_device_seconds": self._device_seconds_before + device_elapsed,
                **self._totals, "usage_complete": self._usage_complete,
                "discarded_incomplete_tail_bytes": self._abandoned_tail_total(),
                "durable_offset_bytes": self.durable_offset_bytes}

    def state_dict(self):
        current = self.snapshot()
        return {"totals": dict(self._totals), "usage_complete": self._usage_complete,
                "abandoned_tail_bytes": self._abandoned_tail_total(),
                "consumed_tail_observations": dict(self._consumed_tail_observations),
                "event_cursor": self.event_cursor, "durable_offset_bytes": self.durable_offset_bytes,
                "elapsed_active_seconds": current["wall_seconds"],
                "allocated_device_seconds": current["allocated_device_seconds"],
                "session_id": self.session_id,
                "session_elapsed_seconds": current["wall_seconds"] - self._elapsed_before,
                "session_device_seconds": current["allocated_device_seconds"] - self._device_seconds_before,
                "journal_path": None if self.path is None else str(self.path.resolve()),
                "active_group_id": self.active_group_id,
                "active_round": self.active_round,
                "recovery": self.recovery}

    def load_state_dict(self, state):
        """Restore algorithm counters while adopting each actual durable tail row once.

        Current-process initialization is already in the same journal.  Read it
        through that journal, rather than adding its in-memory totals twice.
        Only session maxima and interrupted group identities are retained.
        """
        restore_started = monotonic_seconds()
        tail_state = _read_cost_tail_state(self.path)
        sidecar_abandoned = tail_state["abandoned_costs_tail_bytes"]
        self._consumed_tail_observations = _merge_tail_observations(
            self._consumed_tail_observations, tail_state["consumed_tail_observations"],
            state.get("consumed_tail_observations", {}))
        if sidecar_abandoned > self._cumulative_abandoned_tail_bytes:
            self._cumulative_abandoned_tail_bytes = sidecar_abandoned
        totals = dict(state["totals"])
        abandoned = max(int(state.get("abandoned_tail_bytes", 0)), self._cumulative_abandoned_tail_bytes)
        complete = bool(state["usage_complete"]) and not abandoned
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
        self._abandoned_tail_bytes_before = abandoned
        _write_cost_tail_state(self.path, count=abandoned, consumed=self._consumed_tail_observations)
        self._totals, self._usage_complete = totals, complete
        self.event_cursor, self.durable_offset_bytes = cursor, offset
        self._elapsed_before = float(state["elapsed_active_seconds"]) + previous_wall
        self._device_seconds_before = float(state["allocated_device_seconds"]) + previous_device
        if state["session_id"] == self.session_id:
            self._elapsed_before -= float(state["session_elapsed_seconds"])
            self._device_seconds_before -= float(state["session_device_seconds"])
        # Keep started and _startup_*_seconds: initialization and restore time
        # are the current active segment, and must not be restarted or added twice here.
        self.recovery = {"source_checkpoint_offset_bytes": int(state["durable_offset_bytes"]),
                         "adopted_through_offset_bytes": offset, "imported_events": imported,
                         "discarded_incomplete_tail_bytes": self._discarded_tail_bytes,
                         "prior_abandoned_tail_bytes": self._prior_abandoned_tail_bytes,
                         "remote_abandoned_tail_bytes": max(0, sidecar_abandoned - (self._prior_abandoned_tail_bytes + self._discarded_tail_bytes)),
                         "cumulative_abandoned_tail_bytes": abandoned,
                         "interrupted_group_ids": sorted(groups),
                         "checkpoint_active_group_id": state["active_group_id"],
                         "checkpoint_active_round": state.get("active_round"),
                         "environment_continuation": "not_restored",
                         "unrecorded_interruption_wall_seconds": None,
                         "unrecorded_interruption_device_seconds": None,
                         "durable_tail_available": self.path is not None}
        self.record("controller", "ledger.restore", seconds=monotonic_seconds() - restore_started,
                    details=self.recovery)

    def begin_group(self, group_id=None):
        """Bound in-memory retention; aggregate counters and the complete JSONL remain intact."""
        self._flush_model_calls()
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
    user_suffix_ids: tuple[int, ...] = ()


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
    observation_span: TokenSpan | None
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
    full_reference_observation_span: TokenSpan | None
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
