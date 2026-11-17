"""F-GPO public projection and native parsing, independent of privileged state."""
from __future__ import annotations

import ast
import json
import re
from typing import Mapping

from .records import PublicObservation


IM_START, IM_END, END_OF_TEXT, NEWLINE = 151644, 151645, 151643, 198
TERMINATORS = frozenset((IM_END, END_OF_TEXT))
HORIZONS = {"alfworld": 50, "appworld": 100}
OUTPUT_LIMITS = {"alfworld": 512, "appworld": 1024}
EMPTY_STDOUT = "[Empty stdout]"
_ALFWORLD_THINK_RE = re.compile(r"<think>(?:(?!</?think\b).)*?</think>", re.DOTALL | re.IGNORECASE)
_ALFWORLD_ACTION_RE = re.compile(r"<action>((?:(?!</?action\b).)*?)</action>", re.DOTALL | re.IGNORECASE)


def encode_text(tokenizer, text: str) -> tuple[int, ...]:
    return tuple(int(token) for token in tokenizer.encode(
        text, add_special_tokens=False, split_special_tokens=True))


def project_public_observation(tokenizer, benchmark: str, raw_observation: str, *,
                               action_schema: Mapping, goal: str, initial=False) -> PublicObservation:
    if benchmark == "appworld":
        pure = encode_text(tokenizer, raw_observation)
        if not initial and len(pure) > 1024:
            pure = pure[:512] + encode_text(tokenizer, "\n...[truncated]...\n") + pure[-512:]
        text = tokenizer.decode(pure, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        representation = pure if pure else encode_text(tokenizer, EMPTY_STDOUT)
        if initial:
            body = encode_text(tokenizer, json.dumps({"task": goal, "observation": raw_observation},
                                                     ensure_ascii=False))
            offset = -1  # Initial JSON bootstrap is not falsely identified as an unescaped pure-observation slice.
        else:
            body, offset = representation, 0
        return PublicObservation(raw_observation, text, pure, representation, (), body, offset)
    if benchmark != "alfworld":
        raise ValueError(f"Unsupported official F-GPO benchmark: {benchmark}")
    envelope = json.loads(raw_observation)
    text = str(envelope["native_observation"])
    if initial:
        text = re.sub(r"(?:^|\n)Your task is to:.*", "", text).strip()
    step_num = min(int(envelope.get("step", 0)) + 1, HORIZONS["alfworld"])
    commands = tuple(str(c) for c in action_schema["available_actions"] if str(c) != "help")
    prefix = f"Task: {goal}\nStep: {step_num}/{HORIZONS['alfworld']}\nCurrent observation:\n"
    command_lines = "\n".join(commands) if commands else "(none)"
    suffix = ("\n\nAdmissible commands (choose one exact command without quotes):\n"
              f"{command_lines}\nEmit concise <think>...</think> then <action>exact command</action>.")
    prefix_ids, pure, suffix_ids = encode_text(tokenizer, prefix), encode_text(tokenizer, text), encode_text(tokenizer, suffix)
    representation = pure if pure else encode_text(tokenizer, EMPTY_STDOUT)
    # Wrapper and observation are separate integer blocks; BPE cannot merge their boundary.
    return PublicObservation(raw_observation, text, pure, representation, prefix_ids,
                             prefix_ids + pure + suffix_ids, len(prefix_ids), commands, suffix_ids)


def pinned_prompt(tokenizer, system_prompt: str, initial: PublicObservation) -> tuple[int, ...]:
    """Use the actual Qwen ChatML template for structure, safe integer blocks for data."""
    system_header = (IM_START,) + encode_text(tokenizer, "system\n")
    user_header = (IM_START,) + encode_text(tokenizer, "user\n")
    assistant_header = (IM_START,) + encode_text(tokenizer, "assistant\n")
    skeleton = tuple(int(token) for token in tokenizer.apply_chat_template(
        [{"role": "system", "content": ""}, {"role": "user", "content": ""}],
        tokenize=True, add_generation_prompt=True))
    expected = system_header + (IM_END, NEWLINE) + user_header + (IM_END, NEWLINE) + assistant_header
    if skeleton != expected:
        raise ValueError("The F-GPO integer protocol requires the designated Qwen2.5 ChatML template.")
    return (system_header + encode_text(tokenizer, system_prompt) + (IM_END, NEWLINE) +
            user_header + initial.user_body_ids + (IM_END, NEWLINE) + assistant_header)


def parse_native_response(benchmark: str, text: str):
    """Return the actual native action and syntax/format validity, without action gating."""
    if benchmark == "alfworld":
        remainder = _ALFWORLD_THINK_RE.sub("", text)
        lowered_rem = remainder.lower()
        if "<think" in lowered_rem or "</think" in lowered_rem:
            return {"kind": "invalid", "text": ""}, False
        matches = list(_ALFWORLD_ACTION_RE.finditer(remainder))
        if len(matches) != 1:
            return {"kind": "invalid", "text": ""}, False
        outside = (remainder[:matches[0].start()] + remainder[matches[0].end():]).lower()
        if "<action" in outside or "</action" in outside:
            return {"kind": "invalid", "text": ""}, False
        command = matches[0].group(1).strip()
        while len(command) >= 2 and command[0] == command[-1] and command[0] in ("'", '"'):
            command = command[1:-1].strip()
        command = command.lower()
        if not command or "\n" in command or "\r" in command:
            return {"kind": "invalid", "text": ""}, False
        return {"kind": "native_text", "text": command}, True
    if benchmark != "appworld":
        raise ValueError(benchmark)
    fences = list(re.finditer(r"```", text))
    code, valid_fence = text.strip(), not fences
    if len(fences) == 2:
        opening = re.match(r"```([^\r\n`]*)\r?\n", text[fences[0].start():])
        if opening and opening.group(1).strip().lower() in ("", "python"):
            start, end = fences[0].start() + opening.end(), fences[1].start()
            code, valid_fence = text[start:end], start <= end
    syntax_valid = bool(code.strip()) and valid_fence
    try:
        ast.parse(code)
    except (SyntaxError, ValueError):
        syntax_valid = False
    return {"kind": "code", "code": code}, syntax_valid
