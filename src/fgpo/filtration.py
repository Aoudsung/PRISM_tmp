"""Immutable ChatML turn blocks, whole-turn windows and unique response ownership."""
from __future__ import annotations

from dataclasses import dataclass, field

from .public import IM_START, IM_END, NEWLINE, encode_text
from .records import PublicObservation, SegmentRecord, TokenSpan, TurnBlock


@dataclass(frozen=True)
class Advance:
    prompt_length: int
    response_span: TokenSpan
    observation_span: TokenSpan
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

    def freeze(self):
        return SegmentRecord(self.segment_id, tuple(self.tokens), tuple(self.response_spans),
            tuple(self.observation_spans), self.behavior_closed_length, self.scoring_tail_start, self.kind)


class TokenFiltration:
    """A behavior view plus attached bounded feedback scoring suffixes.

    Replayed whole blocks in a new prompt have no owned response spans.  A
    feedback-only segment likewise owns no PPO response, even though its fixed
    context contains the actual generated action.
    """
    def __init__(self, tokenizer, pinned_ids, trajectory_id, *, max_new_tokens,
                 history_length=None, max_context_tokens=32768, kind="actor"):
        if history_length not in (None, 2):
            raise ValueError("F-GPO uses full filtration or its exact two-turn ablation.")
        self.tokenizer, self.pinned_ids = tokenizer, tuple(pinned_ids)
        self.trajectory_id, self.history_length, self.kind = trajectory_id, history_length, kind
        self.max_context_tokens = int(max_context_tokens)
        self.prompt_limit = self.max_context_tokens - int(max_new_tokens)
        self.rollover_limit = int(.75 * self.prompt_limit)
        if not self.pinned_ids or len(self.pinned_ids) > self.prompt_limit:
            raise ValueError("The complete pinned goal/bootstrap exceeds the designated native prompt capacity.")
        self.blocks: list[TurnBlock] = []
        self._segments: list[_Segment] = []
        self.current = self._new_segment(self.pinned_ids, kind)
        self.prompt_ids = self.pinned_ids

    def _new_segment(self, token_ids, kind):
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
        header = (IM_START,) + encode_text(self.tokenizer, "user\n")
        footer = (IM_END, NEWLINE, IM_START) + encode_text(self.tokenizer, "assistant\n")
        user = header + observation.user_body_ids + footer
        offset = len(response_ids) + len(closure) + len(header) + observation.observation_body_offset
        return TurnBlock(step_id, tuple(response_ids), closure, user,
                         tuple(response_ids) + closure + user, offset, observation.pure_ids)

    def append(self, step_id: str, response_ids, observation: PublicObservation, *, terminal=False) -> Advance:
        response_ids = tuple(int(token) for token in response_ids)
        if not response_ids:
            raise ValueError("An actual sampled action contains at least one token, including any actual terminator.")
        prompt_length = len(self.prompt_ids)
        response = TokenSpan(self.current.segment_id, prompt_length, len(response_ids), step_id)
        self.current.tokens.extend(response_ids)
        self.current.response_spans.append(response)
        self.current.behavior_closed_length = len(self.current.tokens)
        self.current.scoring_tail_start = None
        block = self._block(step_id, response_ids, observation)
        tail = block.closure_ids + block.user_ids
        if len(self.current.tokens) + len(tail) <= self.max_context_tokens:
            self.current.scoring_tail_start = len(self.current.tokens)
            self.current.tokens.extend(tail)
            observation_span = TokenSpan(self.current.segment_id,
                prompt_length + block.observation_offset, len(observation.pure_ids), step_id)
            self.current.observation_spans.append(observation_span)
        else:
            # The approved bounded scoring branch retains pin + newest COMPLETE turns.
            scoring = self._suffix([*self.blocks, block], self.rollover_limit)
            prefix_length = len(scoring) - len(block.token_ids)
            feedback = self._new_segment(scoring, self.kind + "_feedback")
            feedback.behavior_closed_length = prefix_length + len(response_ids)
            feedback.scoring_tail_start = feedback.behavior_closed_length
            observation_span = TokenSpan(feedback.segment_id,
                prefix_length + block.observation_offset, len(observation.pure_ids), step_id)
            feedback.observation_spans.append(observation_span)
        self.blocks.append(block)
        if terminal:
            return Advance(prompt_length, response, observation_span, block, (), "terminal", ())
        ordinary_next = self.prompt_ids + block.token_ids
        if self.history_length is None:
            next_prompt = ordinary_next if len(ordinary_next) <= self.prompt_limit else self._suffix(
                self.blocks, self.rollover_limit)
        else:
            recent = self.blocks[-2:]
            next_prompt = self.pinned_ids + tuple(token for item in recent for token in item.token_ids)
            if len(next_prompt) > self.prompt_limit:
                next_prompt = self._suffix(recent, self.rollover_limit)
        if next_prompt == ordinary_next:
            mode = "append"
            delta = (response_ids[-1],) + tail
        else:
            suffix_next = self.pinned_ids + tuple(token for item in self.blocks[-2:] for token in item.token_ids)
            mode = "suffix" if self.history_length == 2 and len(self.blocks) >= 3 and next_prompt == suffix_next else "rebuild"
            delta = block.token_ids if mode == "suffix" else next_prompt[len(self.pinned_ids):]
            self.current = self._new_segment(next_prompt, self.kind)
        self.prompt_ids = next_prompt
        return Advance(prompt_length, response, observation_span, block, next_prompt, mode, delta)

    def segments(self) -> tuple[SegmentRecord, ...]:
        return tuple(segment.freeze() for segment in self._segments)
