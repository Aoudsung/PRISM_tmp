"""Canonical sufficient POMDP filtration and unique response ownership."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .public import (IM_END, NEWLINE, canonical_history_entry_ids, canonical_step_tail)
from .records import PublicObservation, SegmentRecord, TokenSpan, TurnBlock


@dataclass(frozen=True)
class Advance:
    prompt_length: int
    response_span: TokenSpan
    observation_span: TokenSpan | None
    block: TurnBlock
    next_prompt_ids: tuple[int, ...]
    cache_mode: str
    delta_ids: tuple[int, ...]


@dataclass
class _Segment:
    segment_id: str
    tokens: list[int]
    kind: str
    response_spans: list[TokenSpan] = field(default_factory=list)
    observation_spans: list[TokenSpan] = field(default_factory=list)
    behavior_closed_length: int = 0
    scoring_tail_start: int | None = None
    _frozen: SegmentRecord | None = field(default=None, init=False, repr=False, compare=False)

    def freeze(self):
        if self._frozen is None:
            self._frozen = SegmentRecord(
                self.segment_id, tuple(self.tokens), tuple(self.response_spans),
                tuple(self.observation_spans), self.behavior_closed_length,
                self.scoring_tail_start, self.kind,
            )
        return self._frozen


class TokenFiltration:
    """Canonical sufficient POMDP filtration (Task + compact executed history + current step).

    Strips expired action menus, raw <think> monologues, and invalid formatting clutter
    from retained history so every step prompt stays concise and uncontaminated while
    remaining an exact Markov-sufficient function of public observations and actions.
    """

    def __init__(self, tokenizer, root_prefix_ids, trajectory_id, *, benchmark, horizon,
                 initial_observation: PublicObservation, max_new_tokens,
                 history_length=None, max_context_tokens=32768, kind="actor"):
        if history_length not in (None, 2):
            raise ValueError("r6 uses complete bounded canonical history or its two-turn ablation.")
        self.tokenizer = tokenizer
        self.root_prefix_ids = tuple(root_prefix_ids)
        self.pinned_ids = self.root_prefix_ids
        self.trajectory_id = trajectory_id
        self.benchmark = str(benchmark)
        self.horizon = int(horizon)
        self.history_length = history_length
        self.kind = kind
        self.max_context_tokens = int(max_context_tokens)
        self.prompt_limit = self.max_context_tokens - int(max_new_tokens)
        self.max_tail_tokens = self.prompt_limit - len(self.root_prefix_ids)
        if not self.root_prefix_ids or self.max_tail_tokens <= 0:
            raise ValueError("The complete pinned goal/bootstrap exceeds the designated native prompt capacity.")
        self.blocks: list[TurnBlock] = []
        self.history_entry_ids: list[tuple[int, ...]] = []
        self._segments: list[_Segment] = []
        self._frozen_segments: tuple[SegmentRecord, ...] | None = None
        turn0_tail_ids, obs0_offset = canonical_step_tail(
            tokenizer, self.benchmark, step_index=0, horizon=self.horizon,
            history_entry_ids=(), current=initial_observation, max_tail_tokens=self.max_tail_tokens,
        )
        self.turn0_tail_ids = turn0_tail_ids
        self.prompt_ids = self.root_prefix_ids + self.turn0_tail_ids
        self.current = self._new_segment(self.prompt_ids, kind)
        self.current.observation_spans.append(TokenSpan(
            self.current.segment_id, len(self.root_prefix_ids) + obs0_offset,
            len(initial_observation.pure_ids), f"{trajectory_id}:step-0",
        ))

    def _new_segment(self, token_ids, kind):
        self._frozen_segments = None
        segment = _Segment(f"{self.trajectory_id}:{kind}:{len(self._segments)}", list(token_ids), kind)
        self._segments.append(segment)
        return segment

    def _block(self, step_id, response_ids, observation):
        closure = (NEWLINE,) if response_ids[-1] == IM_END else (IM_END, NEWLINE)
        return TurnBlock(
            step_id, tuple(response_ids), closure, observation.user_body_ids,
            tuple(response_ids) + closure + observation.user_body_ids,
            len(response_ids) + len(closure) + observation.observation_body_offset,
            observation.pure_ids,
        )

    def append(self, step_id: str, response_ids, observation: PublicObservation, *,
               action: Mapping | None = None, syntax_valid: bool = True, terminal: bool = False) -> Advance:
        response_ids = tuple(int(token) for token in response_ids)
        if not response_ids:
            raise ValueError("An actual sampled action contains at least one token, including any actual terminator.")
        prompt_length = len(self.prompt_ids)
        self._frozen_segments = None
        self.current._frozen = None
        response = TokenSpan(self.current.segment_id, prompt_length, len(response_ids), step_id)
        self.current.tokens.extend(response_ids)
        self.current.response_spans.append(response)
        self.current.behavior_closed_length = len(self.current.tokens)
        self.current.scoring_tail_start = None
        block = self._block(step_id, response_ids, observation)
        self.blocks.append(block)
        step_num = len(self.blocks)
        self.history_entry_ids.append(canonical_history_entry_ids(
            self.tokenizer, self.benchmark, step_num, action or {}, syntax_valid, observation,
        ))
        if terminal:
            return Advance(prompt_length, response, None, block, (), "terminal", ())
        entries = self.history_entry_ids if self.history_length is None else self.history_entry_ids[-2:]
        tail_ids, obs_offset_in_tail = canonical_step_tail(
            self.tokenizer, self.benchmark, step_index=step_num, horizon=self.horizon,
            history_entry_ids=entries, current=observation, max_tail_tokens=self.max_tail_tokens,
        )
        next_prompt = self.root_prefix_ids + tail_ids
        self.current = self._new_segment(next_prompt, self.kind)
        observation_span = TokenSpan(
            self.current.segment_id, len(self.root_prefix_ids) + obs_offset_in_tail,
            len(observation.pure_ids), step_id,
        )
        self.current.observation_spans.append(observation_span)
        self.prompt_ids = next_prompt
        return Advance(prompt_length, response, observation_span, block, next_prompt, "rebuild", tail_ids)

    def segments(self) -> tuple[SegmentRecord, ...]:
        if self._frozen_segments is None:
            self._frozen_segments = tuple(segment.freeze() for segment in self._segments)
        return self._frozen_segments
