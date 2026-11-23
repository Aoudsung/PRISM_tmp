"""F-GPO public projection and native parsing, independent of privileged state."""
from __future__ import annotations

import ast
import json
import re
from typing import Mapping, Sequence

from .records import PublicObservation
from .textcraft import (TEXTCRAFT_EVAL_HORIZON, TEXTCRAFT_HORIZON, TEXTCRAFT_OUTPUT_TOKENS,
                        textcraft_action, textcraft_observation, unpack_textcraft_observation)


IM_START, IM_END, END_OF_TEXT, NEWLINE = 151644, 151645, 151643, 198
TERMINATORS = frozenset((IM_END, END_OF_TEXT))
HORIZONS = {"alfworld": 50, "appworld": 100, "webshop": 15, "textcraft": TEXTCRAFT_HORIZON}
EVAL_HORIZONS = {**HORIZONS, "textcraft": TEXTCRAFT_EVAL_HORIZON}
OUTPUT_LIMITS = {"alfworld": 512, "appworld": 1024, "webshop": 512, "textcraft": TEXTCRAFT_OUTPUT_TOKENS}
EXACT_OBSERVATION_BENCHMARKS = frozenset(("alfworld", "webshop", "textcraft"))
EMPTY_STDOUT = "[Empty stdout]"
_TEXT_THINK_RE = re.compile(r"<think>(?:(?!</?think\b).)*?</think>", re.DOTALL | re.IGNORECASE)
_TEXT_ACTION_RE = re.compile(r"<action>((?:(?!</?action\b).)*?)</action>", re.DOTALL | re.IGNORECASE)
_ALF_CLEAN_RE = re.compile(r"(?:^|\n)You clean the (.+?) using ", re.IGNORECASE)
_ALF_HEAT_RE = re.compile(r"(?:^|\n)You heat the (.+?) using ", re.IGNORECASE)
_ALF_COOL_RE = re.compile(r"(?:^|\n)You cool the (.+?) using ", re.IGNORECASE)
_ALF_POWER_RE = re.compile(r"(?:^|\n)You turn (on|off) the (.+?)\.(?:\s|$)", re.IGNORECASE)
_ALF_SLICE_RE = re.compile(r"(?:^|\n)You slice the (.+?) (?:with|using) ", re.IGNORECASE)
_WEBSHOP_PREFIX_RE = re.compile(r"^WebShop\s*\[SEP\]\s*", re.IGNORECASE)
_WEBSHOP_INST_BACK_RE = re.compile(
    r"^Instruction:\s*(?:\[SEP\]\s*)?.*?(?=(?:\[button\]\s*Back to Search|Back to Search\s*\[SEP\]))",
    re.IGNORECASE,
)
_WEBSHOP_INST_SEP_RE = re.compile(r"^Instruction:\s*\[SEP\]\s*.*?\s*\[SEP\]\s*", re.IGNORECASE)
_WEBSHOP_NAV_RE = re.compile(
    r"^(?:\[button\]\s*Back to Search\s*\[button_\]|Back to Search\s*\[SEP\])\s*"
    r"(?:\[button\]\s*<\s*Prev\s*\[button_\]|<\s*Prev\s*\[SEP\])?\s*",
    re.IGNORECASE,
)


def update_alfworld_public_effects(effects: Mapping[str, str] | None, feedback_text: str) -> dict[str, str]:
    """Track non-observable ALFWorld object state changes solely from public simulator feedback."""
    updated = dict(effects or {})
    text = str(feedback_text or "")
    for match in _ALF_CLEAN_RE.finditer(text):
        obj = " ".join(match.group(1).strip().lower().split())
        if obj:
            updated[f"clean:{obj}"] = "clean"
    for match in _ALF_HEAT_RE.finditer(text):
        obj = " ".join(match.group(1).strip().lower().split())
        if obj:
            updated[f"temp:{obj}"] = "hot"
    for match in _ALF_COOL_RE.finditer(text):
        obj = " ".join(match.group(1).strip().lower().split())
        if obj:
            updated[f"temp:{obj}"] = "cool"
    for match in _ALF_POWER_RE.finditer(text):
        state, obj = match.group(1).lower(), " ".join(match.group(2).strip().lower().split())
        if obj:
            if state == "on":
                updated[f"power:{obj}"] = "on"
            else:
                updated.pop(f"power:{obj}", None)
    for match in _ALF_SLICE_RE.finditer(text):
        obj = " ".join(match.group(1).strip().lower().split())
        if obj:
            updated[f"sliced:{obj}"] = "sliced"
    return dict(sorted(updated.items()))


def strip_webshop_repeated_header(obs_text: str, goal: str = "") -> str:
    """Strip WebShop repeated banner/instruction/navigation boilerplate without discarding page state."""
    clean = " ".join(str(obs_text).split())
    stripped = _WEBSHOP_PREFIX_RE.sub("", clean, count=1).strip()
    goal_clean = " ".join(str(goal).split())
    if goal_clean:
        for prefix in (
            f"Instruction: [SEP] {goal_clean} [SEP]",
            f"Instruction: {goal_clean} [SEP]",
            f"Instruction: {goal_clean}",
        ):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):].lstrip()
                break
    if stripped.startswith("Instruction:"):
        if re.search(r"(?:\[button\]\s*Back to Search|Back to Search\s*\[SEP\])", stripped, re.IGNORECASE):
            stripped = _WEBSHOP_INST_BACK_RE.sub("", stripped, count=1).lstrip()
        else:
            stripped = _WEBSHOP_INST_SEP_RE.sub("", stripped, count=1).lstrip()
    stripped = _WEBSHOP_NAV_RE.sub("", stripped, count=1).strip()
    return stripped if stripped else clean


def encode_text(tokenizer, text: str) -> tuple[int, ...]:
    return tuple(int(token) for token in tokenizer.encode(
        text, add_special_tokens=False, split_special_tokens=True))


def project_public_observation(tokenizer, benchmark: str, raw_observation: str, *,
                               action_schema: Mapping, goal: str, initial=False,
                               published_baseline=False,
                               alfworld_effects: Mapping[str, str] | None = None) -> PublicObservation:
    if benchmark == "textcraft":
        if published_baseline:
            text = textcraft_observation(raw_observation)
            pure = encode_text(tokenizer, text)
            return PublicObservation(raw_observation, text, pure, pure, (), pure, 0)
        feedback, inventory = unpack_textcraft_observation(raw_observation)
        text = f"Status: Ready to craft.\n{inventory}" if initial else f"Latest feedback: {feedback}\n{inventory}"
        pure = encode_text(tokenizer, text)
        inv_ids = encode_text(tokenizer, inventory)
        representation = inv_ids if inv_ids else encode_text(tokenizer, EMPTY_STDOUT)
        suffix = (
            "\nEnclose concise reasoning in literal <think> and </think> tags, "
            "then enclose one exact lowercase command in literal <action> and </action> tags."
        )
        suffix_ids = encode_text(tokenizer, suffix)
        return PublicObservation(raw_observation, text, pure, representation, (),
                                 pure + suffix_ids, 0, (), suffix_ids)
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
    if benchmark == "webshop":
        text = raw_observation
        step_num = min(int(action_schema["step"]) + 1, HORIZONS[benchmark])
        commands = tuple(action_schema["available_actions"])
        suffix = "\nAdmissible actions:\n" + ("\n".join(commands) if commands else "(none)")
        ws_clean = strip_webshop_repeated_header(text, goal=goal)
        ws_repr = (ws_clean + "\n||" + "\n".join(commands)).strip()
        representation = encode_text(tokenizer, ws_repr if ws_repr else EMPTY_STDOUT)
    elif benchmark == "alfworld":
        envelope = json.loads(raw_observation)
        text = str(envelope["native_observation"])
        if initial:
            text = re.sub(r"(?:^|\n)Your task is to:.*", "", text).strip()
        step_num = min(int(envelope.get("step", 0)) + 1, HORIZONS[benchmark])
        commands = tuple(str(c) for c in action_schema["available_actions"] if str(c) != "help")
        command_lines = "\n".join(commands) if commands else "(none)"
        suffix = (f"\n\nAdmissible commands (choose one exact command without quotes):\n"
                  f"{command_lines}\n"
                  "Enclose concise reasoning in literal <think> and </think> tags, "
                  "then enclose one exact command in literal <action> and </action> tags.")
        effects = dict(alfworld_effects) if alfworld_effects is not None else dict(envelope.get("public_object_effects") or {})
        repr_lines = list(commands)
        if effects:
            repr_lines.append("||" + "|".join(f"{k}={v}" for k, v in sorted(effects.items())))
        alf_repr = "\n".join(repr_lines).strip()
        representation = encode_text(tokenizer, alf_repr if alf_repr else EMPTY_STDOUT)
    else:
        raise ValueError(f"Unsupported official F-GPO benchmark: {benchmark}")
    prefix = f"Task: {goal}\nStep: {step_num}/{HORIZONS[benchmark]}\nCurrent observation:\n"
    prefix_ids, pure, suffix_ids = encode_text(tokenizer, prefix), encode_text(tokenizer, text), encode_text(tokenizer, suffix)
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


def _validate_chat_template(tokenizer) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    system_header = (IM_START,) + encode_text(tokenizer, "system\n")
    user_header = (IM_START,) + encode_text(tokenizer, "user\n")
    assistant_header = (IM_START,) + encode_text(tokenizer, "assistant\n")
    skeleton = tuple(int(token) for token in tokenizer.apply_chat_template(
        [{"role": "system", "content": ""}, {"role": "user", "content": ""}],
        tokenize=True, add_generation_prompt=True))
    expected = system_header + (IM_END, NEWLINE) + user_header + (IM_END, NEWLINE) + assistant_header
    if skeleton != expected:
        raise ValueError("The F-GPO integer protocol requires the designated Qwen2.5 ChatML template.")
    return system_header, user_header, assistant_header


def canonical_root_prefix(tokenizer, benchmark: str, system_prompt: str, goal: str,
                          initial: PublicObservation) -> tuple[int, ...]:
    """Episode-invariant ChatML prefix (system instructions + pinned task/recipes) for r8 shared KV."""
    system_header, user_header, _ = _validate_chat_template(tokenizer)
    if benchmark == "alfworld":
        task_header_text = f"Task: {goal}\nInitial observation: {' '.join(initial.pure_text.split())}\n"
    elif benchmark == "textcraft":
        task_header_text = f"{unpack_textcraft_observation(initial.raw_observation)[0]}\n"
    elif benchmark == "webshop":
        task_header_text = f"Task: {goal}\n"
    elif benchmark == "appworld":
        task_header_text = json.dumps({"task": goal, "observation": initial.raw_observation}, ensure_ascii=False) + "\n"
    else:
        raise ValueError(f"Unsupported official F-GPO benchmark: {benchmark}")
    return (system_header + encode_text(tokenizer, system_prompt) + (IM_END, NEWLINE) +
            user_header + encode_text(tokenizer, task_header_text))


def canonical_history_entry_ids(tokenizer, benchmark: str, step_num: int, action: Mapping,
                                syntax_valid: bool, observation: PublicObservation) -> tuple[int, ...]:
    """Compact tag-preserving canonical executed (<action> -> Observation) record, stripped of boilerplate and <think>."""
    if benchmark in ("alfworld", "webshop", "textcraft"):
        cmd = str(action.get("text") or "").strip() if syntax_valid else ""
        clean_cmd = " ".join(cmd.split())
        action_repr = (
            f"<action>{clean_cmd}</action>"
            if clean_cmd
            else "(invalid format: missing <think>...</think><action>...</action>)"
        )
        if benchmark == "alfworld":
            obs_clean = " ".join(observation.pure_text.split())
            line = f"[Step {step_num}] {action_repr} -> Observation: {obs_clean}\n"
        elif benchmark == "textcraft":
            fb, inv = unpack_textcraft_observation(observation.raw_observation)
            line = f"[Step {step_num}] {action_repr} -> {' '.join(fb.split())} ({' '.join(inv.split())})\n"
        else:
            obs_clean = strip_webshop_repeated_header(observation.pure_text)
            if len(obs_clean) > 320:
                obs_clean = obs_clean[:320] + " ..."
            line = f"[Step {step_num}] {action_repr} -> Observation: {obs_clean}\n"
    elif benchmark == "appworld":
        cmd = str(action.get("code") or "").strip() if syntax_valid else ""
        action_repr = " ".join(cmd.split()) if cmd else "(invalid python syntax)"
        obs_clean = " ".join(observation.pure_text.split())
        if len(obs_clean) > 320:
            obs_clean = obs_clean[:320] + " ..."
        line = f"[Step {step_num}] Code: {action_repr} -> Observation: {obs_clean}\n"
    else:
        raise ValueError(f"Unsupported official F-GPO benchmark: {benchmark}")
    return encode_text(tokenizer, line)


def canonical_step_tail(tokenizer, benchmark: str, *, step_index: int, horizon: int,
                        history_entry_ids: Sequence[tuple[int, ...]], current: PublicObservation,
                        max_tail_tokens: int) -> tuple[tuple[int, ...], int]:
    """Build the canonical tail (prior executed steps + current observation + action menu + assistant header)."""
    step_num = min(int(step_index) + 1, int(horizon))
    assistant_footer_ids = (IM_END, NEWLINE, IM_START) + encode_text(tokenizer, "assistant\n")
    if benchmark == "alfworld":
        current_prefix_ids = encode_text(tokenizer, f"\nCurrent step: {step_num}/{horizon}\nCurrent observation:\n")
        cmd_lines = "\n".join(current.available_actions) if current.available_actions else "(none)"
        current_suffix_ids = encode_text(
            tokenizer,
            f"\n\nAdmissible commands (choose one exact command without quotes):\n{cmd_lines}\n"
            "Enclose concise reasoning in literal <think> and </think> tags, "
            "then enclose one exact command in literal <action> and </action> tags.",
        )
    elif benchmark == "textcraft":
        current_prefix_ids = encode_text(tokenizer, f"\nCurrent step: {step_num}/{horizon}\n")
        current_suffix_ids = current.user_suffix_ids
    elif benchmark == "webshop":
        current_prefix_ids = encode_text(tokenizer, f"\nCurrent step: {step_num}/{horizon}\nCurrent observation:\n")
        cmd_lines = "\n".join(current.available_actions) if current.available_actions else "(none)"
        current_suffix_ids = encode_text(
            tokenizer,
            f"\nAdmissible actions:\n{cmd_lines}\n"
            "Enclose concise reasoning in literal <think> and </think> tags, "
            "then enclose one exact command in literal <action> and </action> tags.",
        )
    elif benchmark == "appworld":
        current_prefix_ids = encode_text(tokenizer, f"\nCurrent step: {step_num}/{horizon}\nCurrent observation:\n")
        current_suffix_ids = ()
    else:
        raise ValueError(f"Unsupported official F-GPO benchmark: {benchmark}")

    current_block_ids = current_prefix_ids + current.pure_ids + current_suffix_ids + assistant_footer_ids
    if len(current_block_ids) > max_tail_tokens:
        raise ValueError("Pinned root and newest canonical step exceed prompt capacity.")

    history_section_ids: tuple[int, ...] = ()
    if history_entry_ids:
        header = encode_text(tokenizer, f"\nCompleted steps ({len(history_entry_ids)}/{horizon}):\n")
        budget = max_tail_tokens - len(current_block_ids) - len(header)
        selected = []
        used = 0
        for entry in reversed(history_entry_ids):
            if used + len(entry) > budget:
                break
            selected.append(entry)
            used += len(entry)
        if selected:
            history_section_ids = header + tuple(token for entry in reversed(selected) for token in entry)

    tail_ids = history_section_ids + current_block_ids
    obs_offset_in_tail = len(history_section_ids) + len(current_prefix_ids)
    return tail_ids, obs_offset_in_tail


def parse_native_response(benchmark: str, text: str, *, published_baseline: bool = False):
    """Return the actual native action and syntax/format validity, without action gating."""
    if benchmark == "textcraft":
        command, valid = textcraft_action(text, strict_r5=not published_baseline)
        return {"kind": "native_text" if command else "invalid", "text": command}, valid
    if benchmark in ("alfworld", "webshop"):
        remainder = _TEXT_THINK_RE.sub("", text)
        lowered_rem = remainder.lower()
        if "<think" in lowered_rem or "</think" in lowered_rem:
            return {"kind": "invalid", "text": ""}, False
        matches = list(_TEXT_ACTION_RE.finditer(remainder))
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
