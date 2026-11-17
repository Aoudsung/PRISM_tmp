"""Immutable ChatML turn blocks, whole-turn windows and unique response ownership."""
from __future__ import annotations

from dataclasses import dataclass, field

from .public import IM_START, IM_END, NEWLINE, encode_text
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
    """The actor's bounded history with unique ownership of sampled responses.

    Replayed whole turns in a new prompt own no PPO response. Feedback appears
    in a packed segment only when the actor consumes it at the next decision;
    terminal feedback remains in the step/turn record without a scoring pass.
    """
    def __init__(self, tokenizer, pinned_ids, trajectory_id, *, max_new_tokens,
                 history_length=None, max_context_tokens=32768, kind="actor"):
        if history_length not in (None, 2):
            raise ValueError("r3 uses bounded retained history or its two-turn ablation.")
        self.tokenizer, self.pinned_ids = tokenizer, tuple(pinned_ids)
        self.user_header_ids = (IM_START,) + encode_text(tokenizer, "user\n")
        self.assistant_footer_ids = (IM_END, NEWLINE, IM_START) + encode_text(tokenizer, "assistant\n")
        self.trajectory_id, self.history_length, self.kind = trajectory_id, history_length, kind
        self.max_context_tokens = int(max_context_tokens)
        self.prompt_limit = self.max_context_tokens - int(max_new_tokens)
        self.rollover_limit = int(.75 * self.prompt_limit)
        if not self.pinned_ids or len(self.pinned_ids) > self.prompt_limit:
            raise ValueError("The complete pinned goal/bootstrap exceeds the designated native prompt capacity.")
        self.blocks: list[TurnBlock] = []
        self._segments: list[_Segment] = []
        self._frozen_segments: tuple[SegmentRecord, ...] | None = None
        self.current = self._new_segment(self.pinned_ids, kind)
        self.prompt_ids = self.pinned_ids

    def _new_segment(self, token_ids, kind):
        self._frozen_segments = None
        segment = _Segment(f"{self.trajectory_id}:{kind}:{len(self._segments)}", list(token_ids), kind)
        self._segments.append(segment)
        return segment

    def _suffix(self, blocks, limit):
        length, selected = len(self.pinned_ids), []
        for block in reversed(blocks):
            if length + len(block.token_ids) > limit:
                break
            selected.append(block)
            length += len(block.token_ids)
        if blocks and not selected:
            raise ValueError("Pinned bootstrap and the complete newest atomic turn exceed the bounded context.")
        return self.pinned_ids + tuple(token for block in reversed(selected) for token in block.token_ids)

    def _block(self, step_id, response_ids, observation):
        closure = (NEWLINE,) if response_ids[-1] == IM_END else (IM_END, NEWLINE)
        header = self.user_header_ids
        footer = self.assistant_footer_ids
        user = header + observation.user_body_ids + footer
        offset = len(response_ids) + len(closure) + len(header) + observation.observation_body_offset
        return TurnBlock(step_id, tuple(response_ids), closure, user,
                         tuple(response_ids) + closure + user, offset, observation.pure_ids)

    def append(self, step_id: str, response_ids, observation: PublicObservation, *, terminal=False) -> Advance:
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
        if terminal:
            return Advance(prompt_length, response, None, block, (), "terminal", ())
        tail = block.closure_ids + block.user_ids
        newest_length = len(self.pinned_ids) + len(block.token_ids)
        if newest_length > self.prompt_limit:
            raise ValueError("Pinned bootstrap and the complete newest atomic turn exceed the bounded context.")
        # Reserve headroom after ordinary rollovers while retaining a legal
        # newest turn even when the pin itself exceeds the headroom target.
        window_limit = max(self.rollover_limit, newest_length)
        ordinary_next = self.prompt_ids + block.token_ids
        if self.history_length is None:
            next_prompt = ordinary_next if len(ordinary_next) <= self.prompt_limit else self._suffix(
                self.blocks, window_limit)
        else:
            recent = self.blocks[-2:]
            next_prompt = self.pinned_ids + tuple(token for item in recent for token in item.token_ids)
            if len(next_prompt) > self.prompt_limit:
                next_prompt = self._suffix(recent, window_limit)
        if next_prompt == ordinary_next:
            mode = "append"
            delta = (response_ids[-1],) + tail
            self.current.tokens.extend(tail)
        else:
            suffix_next = self.pinned_ids + tuple(token for item in self.blocks[-2:] for token in item.token_ids)
            mode = "suffix" if self.history_length == 2 and len(self.blocks) >= 3 and next_prompt == suffix_next else "rebuild"
            delta = block.token_ids if mode == "suffix" else next_prompt[len(self.pinned_ids):]
            self.current = self._new_segment(next_prompt, self.kind)
        observation_span = TokenSpan(self.current.segment_id,
            len(next_prompt) - len(block.token_ids) + block.observation_offset,
            len(observation.pure_ids), step_id)
        self.current.observation_spans.append(observation_span)
        self.prompt_ids = next_prompt
        return Advance(prompt_length, response, observation_span, block, next_prompt, mode, delta)

    def segments(self) -> tuple[SegmentRecord, ...]:
        if self._frozen_segments is None:
            self._frozen_segments = tuple(segment.freeze() for segment in self._segments)
        return self._frozen_segments
