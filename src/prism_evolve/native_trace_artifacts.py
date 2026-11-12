"""Gate 0 public projection and content-addressed trace export (contract §5.2).

Legacy trace bytes remain immutable. Diagnostic metadata is deliberately absent
from public_view; saved actor tokens and pre-policy history are distinct evidence.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

PUBLIC_SCHEMA_FIELDS = (
    "benchmark", "classic_text", "native_format", "native_interface",
    "native_system_prompt", "available_actions", "feasible_modes", "mode_actions",
    "probe_actions", "search_available", "native_action_schemas_by_mode",
)
PUBLIC_STATE_FIELDS = (
    "available_actions", "benchmark", "feasible_modes", "mode_actions",
    "probe_actions", "search_available", "native_action_schemas_by_mode",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def public_observation(text: str) -> dict[str, Any]:
    """Whitelist the adapter's public envelope, preserving native return text."""
    value = json.loads(text)
    if not isinstance(value, dict) or "native_observation" not in value:
        raise ValueError("Missing native public observation envelope")
    state = value.get("classic_text_action_state", {})
    return {
        "feedback_type": value.get("feedback_type"),
        "native_observation": deepcopy(value["native_observation"]),
        "classic_text_action_state": {
            key: deepcopy(state[key]) for key in PUBLIC_STATE_FIELDS if key in state
        },
    }


def reset_material(trace: Mapping[str, Any]) -> tuple[str, str]:
    if isinstance(trace.get("initial_observation"), str) and trace.get("goal") is not None:
        return str(trace["goal"]), trace["initial_observation"]
    for event in trace.get("events", ()):
        for message in event.get("messages_before", ()):
            if message.get("role") != "user":
                continue
            try:
                value = json.loads(message["content"])
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict) and value.get("feedback_type") == "task":
                observation = value.get("observation")
                text = observation if isinstance(observation, str) else canonical_bytes(observation).decode("utf-8")
                return str(value.get("instruction", "")), text
    raise ValueError("Source has no complete public reset message")


def public_view(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Only this projection may enter compilation, binding or predicates.

    Terminal success, private facts, state_changed, receipts, cost diagnostics,
    old DCC estimates and actor-history metadata are excluded by construction.
    """
    goal, before = reset_material(trace)
    events = []
    for index, event in enumerate(trace.get("events", ())):
        schema = event.get("sampling", {}).get("native_action_schema", {})
        events.append({
            "event_id": f"e{index:05d}",
            "obs_before": public_observation(before),
            "obs_after": public_observation(event["observation"]),
            "action": deepcopy({key: event.get("action", {}).get(key)
                                for key in ("kind", "text", "name", "arguments")
                                if key in (event.get("action") or {})}),
            "response": str(event.get("response", "")),
            "schema": {key: deepcopy(schema[key]) for key in PUBLIC_SCHEMA_FIELDS if key in schema},
        })
        before = event["observation"]
    return {"public_contract": "prism-gate0-public-v1", "goal": goal, "events": events}


def put_blob(root: Path, text: str, *, payload_complete: bool = True) -> dict[str, Any]:
    data = text.encode("utf-8")
    digest = sha256(data)
    path = root / "blobs" / (digest + ".txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("Content-address collision")
    else:
        path.write_bytes(data)
    return {"path": path.relative_to(root).as_posix(), "sha256": digest,
            "utf8_bytes": len(data), "payload_complete": payload_complete}


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def export_trace(root: Path, trace: Mapping[str, Any], episode_id: str, *,
                 tokenizer: Any = None) -> dict[str, Any]:
    """Export complete observations, exact token evidence and separate history.

    Old traces predate physical call IDs. These stay null rather than inventing
    calls or reconstructing a physical bill from action-token lengths.
    """
    goal, before = reset_material(trace)
    view = public_view(trace)
    events = []
    for index, event in enumerate(trace["events"]):
        prompt_ids = list(event.get("prompt_token_ids", ()))
        actor_messages = event.get("actor_messages")
        actor_ref = None
        actor_evidence = "MISSING"
        execution = event.get("action_execution", {})
        if execution.get("origin") == "program" and execution.get("model_call_id") is None:
            actor_evidence = "DETERMINISTIC_ACTION_NO_MODEL_CALL"
        elif actor_messages is not None:
            actor_ref = put_blob(root, canonical_bytes(actor_messages).decode("utf-8"))
            actor_evidence = "RECORDED_ACTUAL_MESSAGES"
        elif prompt_ids and tokenizer is not None:
            text = tokenizer.decode(prompt_ids, skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False)
            if tokenizer.encode(text, add_special_tokens=False) != prompt_ids:
                raise ValueError(f"Actor token roundtrip failed: {episode_id}/{index}")
            actor_ref = put_blob(root, text)
            actor_evidence = "LOSSLESS_DECODE_OF_RECORDED_PROMPT_IDS"
        schema = view["events"][index]["schema"]
        events.append({
            "episode_id": episode_id, "event_id": f"e{index:05d}",
            "obs_before_ref": put_blob(root, before),
            "obs_before_origin": ("preceding_native_return" if index else
                                  "recorded_native_reset" if isinstance(trace.get("initial_observation"), str)
                                  else "decoded_public_reset_message"),
            "obs_after_ref": put_blob(root, event["observation"]),
            "actor_input_ref": actor_ref, "actor_input_evidence": actor_evidence,
            "prompt_token_ids_ref": put_blob(root, canonical_bytes(prompt_ids).decode("utf-8")),
            "history_before_ref": put_blob(root, canonical_bytes(event["messages_before"]).decode("utf-8")),
            "schema_ref": put_blob(root, canonical_bytes(schema).decode("utf-8")),
            "action": deepcopy(event.get("action")), "response": event.get("response"),
            "action_execution": deepcopy(event.get("action_execution", {
                "origin": "UNKNOWN_LEGACY", "model_call_id": None,
                "cost_event_ids": [], "program_version": None, "node_version": None,
            })),
        })
        before = event["observation"]
    result = {"episode_id": episode_id, "task_id": trace["task_id"],
              "goal_ref": put_blob(root, goal), "events": events,
              "public_input_ref": put_blob(root, canonical_bytes(view).decode("utf-8")),
              "diagnostics_are_compiler_input": False}
    write_json_exclusive(root / "trace_exports" / (episode_id + ".json"), result)
    return result


def verify_blob_references(root: Path, value: Any) -> int:
    if isinstance(value, dict):
        count = 0
        if {"path", "sha256", "utf8_bytes"} <= value.keys():
            path = (root / value["path"]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Blob reference escapes run root")
            data = path.read_bytes()
            data.decode("utf-8")
            if sha256(data) != value["sha256"] or len(data) != value["utf8_bytes"]:
                raise ValueError("Blob integrity mismatch")
            count += 1
        return count + sum(verify_blob_references(root, item) for item in value.values())
    if isinstance(value, list):
        return sum(verify_blob_references(root, item) for item in value)
    return 0
