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
OUTPUT_LIMITS = {"alfworld": 256, "appworld": 1024}
EMPTY_STDOUT = "[Empty stdout]"


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
    commands = tuple(str(command) for command in action_schema["available_actions"])
    prefix = ((f"Task: {goal}\n" if initial else "") +
              "Admissible commands: " + json.dumps(commands, ensure_ascii=False) + "\nCurrent observation:\n")
    prefix_ids, pure = encode_text(tokenizer, prefix), encode_text(tokenizer, text)
    representation = pure if pure else encode_text(tokenizer, EMPTY_STDOUT)
    # Wrapper and observation are separate integer blocks; BPE cannot merge their boundary.
    return PublicObservation(raw_observation, text, pure, representation, prefix_ids,
                             prefix_ids + pure, len(prefix_ids), commands)


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
        raw = text.strip()
        match = re.fullmatch(r"(?:<think>.*?</think>\s*)?<action>(.*?)</action>", raw, re.DOTALL)
        command = match.group(1).strip() if match is not None else raw
        return {"kind": "native_text", "text": command}, bool(match is not None and command)
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
