"""Public native-command languages and their exact sampling/scoring masks.

Household tasks use plain commands; tool environments use their original JSON calls.
Neither path constructs or compiles a PRISM program AST.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any, Mapping, Sequence

from .native_program_decoder import ProgramDecoder, ProgramLogitsProcessor


TEXT_ACTOR_CONTRACT = "prism-native-text-v1"


def text_actions(schema: Mapping[str, Any]) -> dict[str, list[str]]:
    pools = schema.get("mode_actions", {})
    if not isinstance(pools, Mapping) or set(pools) - {"I", "E"}:
        raise ValueError("Native action pools require only I/E mode keys.")
    result: dict[str, list[str]] = {}
    for mode in ("I", "E"):
        values = pools.get(mode, [])
        if not isinstance(values, (list, tuple)) or any(
            not isinstance(value, str) or not value or "\n" in value or "\r" in value
            for value in values
        ):
            raise ValueError("Public native candidates must be nonempty single-line strings.")
        result[mode] = list(dict.fromkeys(values))
    if set(result["I"]) & set(result["E"]):
        raise ValueError("I and E native action sets must be disjoint.")
    return result


def open_text_actions(schema: Mapping[str, Any]) -> dict[str, list[str]]:
    values = schema.get("open_text_modes", {})
    if not isinstance(values, Mapping) or set(values) - {"I", "E"}:
        raise ValueError("Open native actions require only I/E mode keys.")
    result: dict[str, list[str]] = {}
    for mode in ("I", "E"):
        names = values.get(mode, [])
        if not isinstance(names, (list, tuple)) or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Open native actions need public operation names.")
        result[mode] = list(dict.fromkeys(names))
    if set(result["I"]) & set(result["E"]):
        raise ValueError("Open I/E operation names must be disjoint.")
    return result


def text_available_modes(schema: Mapping[str, Any]) -> tuple[str, ...]:
    candidates, opened = text_actions(schema), open_text_actions(schema)
    search_available = bool(schema.get("search_available", False))
    native_schemas = schema.get("native_action_schemas_by_mode", {})
    if not isinstance(native_schemas, Mapping) or set(native_schemas) - {"I", "E"}:
        raise ValueError("Native tool schemas require only I/E mode keys.")
    return tuple(mode for mode in ("I", "E")
                 if candidates[mode] or opened[mode] or native_schemas.get(mode)
                 or (mode == "I" and search_available))


def native_text_language(schema: Mapping[str, Any], mode: str | None) -> dict[str, Any]:
    modes = text_available_modes(schema) if mode is None else (mode,)
    if not modes or any(value not in text_available_modes(schema) for value in modes):
        raise ValueError("Requested native action class is unavailable.")
    native_schemas = schema.get("native_action_schemas_by_mode")
    if native_schemas:
        variants = [deepcopy(native_schemas[value]) for value in modes]
        return {"kind": "json", "schema": variants[0] if len(variants) == 1 else {"anyOf": variants}}
    candidates, opened = text_actions(schema), open_text_actions(schema)
    alternatives = [json.dumps(value, ensure_ascii=False)
                    for selected in modes for value in candidates[selected]]
    if "I" in modes and bool(schema.get("search_available", False)):
        alternatives.append('"search[" [^\\[\\]\\r\\n]+ "]"')
    if any(opened[selected] for selected in modes):
        raise ValueError("Open native tools require their original JSON argument schemas.")
    grammar = "root ::= " + " | ".join(alternatives)
    return {"kind": "grammar", "grammar": grammar}


def extract_native_text(text: str) -> str:
    """Accept a complete action envelope while preserving invalid model output."""
    raw = text.strip()
    match = re.fullmatch(r"(?:<think>.*?</think>\s*)?<action>(.*?)</action>", raw, re.DOTALL)
    return match.group(1).strip() if match is not None else raw


def native_text_mode(text: str, schema: Mapping[str, Any]) -> str | None:
    native_schemas = schema.get("native_action_schemas_by_mode")
    if native_schemas:
        from jsonschema import Draft202012Validator
        try:
            action = json.loads(text)
        except (TypeError, ValueError):
            return None
        matches = [mode for mode in text_available_modes(schema)
                   if Draft202012Validator(native_schemas[mode]).is_valid(action)]
    else:
        candidates = text_actions(schema)
        matches = [mode for mode in ("I", "E") if text in candidates[mode]]
        if (bool(schema.get("search_available", False))
                and "I" not in matches
                and bool(re.fullmatch(r"search\[[^\[\]\r\n]+\]", text))):
            matches.append("I")
    if len(matches) > 1:
        raise ValueError("A native action matched both I and E.")
    return matches[0] if matches else None


class NativeTextDecoder(ProgramDecoder):
    """Reuse XGrammar's token accounting without the PRISM JSON language."""

    def native_contract(self, schema: Mapping[str, Any], mode: str | None) -> dict[str, Any]:
        language = native_text_language(schema, mode)
        support = {key: deepcopy(schema[key]) for key in (
            "benchmark", "mode_actions", "search_available", "open_text_modes",
            "native_action_schemas_by_mode", "static_native_tools",
        ) if key in schema}
        result = super().contract(language, support, mode)
        result.update(decoder_kind="native_text", decoder_version=TEXT_ACTOR_CONTRACT)
        return result

    def _compiled(self, contract: Mapping[str, Any]) -> Any:
        key = (contract["schema_json"], "native_text")
        if key not in self.compiled:
            language = json.loads(contract["schema_json"])
            if language["kind"] == "json":
                options = dict(contract["compile_options"])
                options["separators"] = tuple(options["separators"])
                compiled = self.compiler.compile_json_schema(language["schema"], **options)
            else:
                compiled = self.compiler.compile_grammar(language["grammar"])
            self.compiled[key] = compiled
            if len(self.compiled) > 256:
                self.compiled.popitem(last=False)
        self.compiled.move_to_end(key)
        return self.compiled[key]

    def processor(self, contract: Mapping[str, Any], prompt_length: int) -> "NativeTextProcessor":
        return NativeTextProcessor(self, contract, prompt_length)


class NativeTextProcessor(ProgramLogitsProcessor):
    def completed(self, response_ids: Sequence[int]) -> bool:
        # A command may prefix another command (e.g. object 1/object 10).
        # Require the sampled EOS; stopping at the first complete string would
        # silently remove the longer command from the conditional distribution.
        self._accept(response_ids[self.consumed:])
        return bool(self.matcher.is_terminated())
