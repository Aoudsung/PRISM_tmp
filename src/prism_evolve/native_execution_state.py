"""Public, committed execution state for native-r3.

This module knows neither task answers nor benchmark trusted-state tables. Its
inputs are the public task, public API documents and receipts from actual native
requests. Expression evaluation is pure; it never performs a business call.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import posixpath
import re
from typing import Any, Iterable, Mapping

from .native_semantics import (
    infer_output_semantic as _public_output_semantic,
    semantic_key as _public_semantic_key,
)


STATE_VERSION = "native-r3-public-state"

_CAPITALIZED_STOPWORDS = frozenset({
    "a", "an", "and", "any", "all", "after", "also", "are", "as", "at", "be",
    "before", "between", "both", "but", "by", "can", "check", "complete", "compute",
    "copy", "count", "create", "current", "delete", "do", "does", "each", "either",
    "ensure", "every", "find", "first", "File", "file", "files", "folder", "folders",
    "for", "from", "get", "give", "given", "go", "have", "her", "here", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "keep", "last", "let", "list", "look",
    "make", "may", "me", "move", "my", "need", "new", "no", "not", "now", "of", "on",
    "one", "only", "or", "other", "our", "out", "over", "please", "put", "read",
    "remove", "return", "run", "same", "save", "search", "see", "select", "send",
    "set", "should", "show", "so", "some", "such", "take", "task", "tell", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "this", "those",
    "through", "to", "total", "transfer", "under", "up", "update", "use", "using",
    "verify", "view", "want", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whose", "why", "will", "with", "within", "without", "would",
    "write", "you", "your",
})


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    raise TypeError(f"Native public values must be JSON values, got {type(value).__name__}.")


def pointer_get(value: Any, pointer: str = "") -> Any:
    if pointer == "":
        return value
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("A binding path must be an RFC 6901 JSON pointer.")
    current = value
    for part in pointer[1:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", key):
                raise ValueError("Array paths require a nonnegative canonical index.")
            current = current[int(key)]
        elif isinstance(current, Mapping):
            current = current[key]
        else:
            raise ValueError("A JSON pointer traversed a scalar value.")
    return current


def walk_values(value: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    yield path, value
    if isinstance(value, Mapping):
        for key, child in value.items():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            yield from walk_values(child, path + "/" + escaped)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk_values(child, path + "/" + str(index))


def semantic_key(name: str, description: str = "") -> str:
    """Resolve identity from its public definition, through the shared module."""
    return _public_semantic_key(name, description)


def infer_output_semantic(tool: str, doc: Mapping[str, Any] | None = None) -> str:
    return _public_output_semantic(tool, doc)


def _task_spans(instruction: str) -> list[dict[str, Any]]:
    """Extract typed, source-located values explicitly present in the task.

    Universal task lexer: extracts quoted strings verbatim, paths, identifiers,
    emails, currency amounts, phone numbers, standalone digit strings (emitting
    both integer and string views so schema type compatibility decides), and
    capitalized named task tokens. Every value retains its character span and
    local surrounding context text for soft lexical ranking.
    """
    candidates: list[tuple[int, int, str, str | None]] = []
    patterns = (
        (r"`([^`]+)`", "quoted"),
        (r'"([^"\n]+)"', "quoted"),
        (r"(?<![A-Za-z0-9])((?:~/|/home/)[^\s,;\"'`]+)", "path"),
        (r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b", "email"),
        (r"\b([A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+)\b", "identifier"),
        (r"\b([A-Z]{2,}-[A-Za-z0-9]+)\b", "identifier"),
    )
    for pattern, kind in patterns:
        for match in re.finditer(pattern, instruction):
            start, end = match.span(1)
            candidates.append((start, end, kind, None))

    number = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    for match in re.finditer(
        r"(?:[$€£¥]|\b(?:USD|EUR|GBP|JPY|CNY)\b)\s*(" + number + r")(?!\d)",
        instruction,
        flags=re.IGNORECASE,
    ):
        candidates.append((match.start(), match.end(), "amount", match.group(1)))

    phone_pattern = (
        r"\b(?:phone(?:[ _-]+number)?|telephone(?:[ _-]+number)?|mobile(?:[ _-]+number)?)"
        r"\s*(?:(?:is|:|=)\s*)?[\"'`]?(\+?\d[\d ()-]{5,}\d)"
    )
    for match in re.finditer(phone_pattern, instruction, flags=re.IGNORECASE):
        if 7 <= sum(character.isdigit() for character in match.group(1)) <= 15:
            start, end = match.span(1)
            candidates.append((start, end, "phone", None))

    for match in re.finditer(r"(?<![A-Za-z0-9_/@.-])(\d+(?:\.\d+)?)(?![A-Za-z0-9_.-])", instruction):
        start, end = match.span(1)
        candidates.append((start, end, "number", match.group(1)))

    for match in re.finditer(r"\b([A-Z][A-Za-z0-9_-]{2,})\b", instruction):
        word = match.group(1)
        if word.lower() in _CAPITALIZED_STOPWORDS:
            continue
        start, end = match.span(1)
        candidates.append((start, end, "named_token", None))

    priority = {
        "amount": 0,
        "phone": 0,
        "email": 1,
        "path": 1,
        "quoted": 2,
        "identifier": 3,
        "number": 4,
        "named_token": 5,
    }
    spans: list[dict[str, Any]] = []
    seen_values: set[tuple[str, str]] = set()
    occupied_ranges: list[tuple[int, int, str]] = []

    for start, end, kind, numeric in sorted(candidates, key=lambda item: (item[0], priority[item[2]], -item[1])):
        if kind != "quoted":
            end = start + len(instruction[start:end].rstrip(".,;"))
        if end <= start:
            continue
        if any(
            s <= start and end <= e and (kind in {"number", "named_token"} or (s == start and e == end and k == kind))
            for s, e, k in occupied_ranges
        ):
            continue
        source_text = instruction[start:end]
        if not source_text:
            continue
        before = instruction[max(0, start - 180):start].strip(" =:`'\"")
        context_text = instruction[max(0, start - 96):min(len(instruction), end + 96)]
        segments = [part.strip() for part in re.split(r"[,;\n]|\band\b|\bwith\b", before) if part.strip()]
        raw_label = segments[-1] if segments else ""
        raw_label = re.split(r"[(\[{]", raw_label)[-1].strip()
        label = re.sub(
            r"^.*\b(?:given|using|the|for|of|with|is|as)\s+",
            "",
            raw_label,
            flags=re.IGNORECASE,
        )
        value: Any = source_text
        raw_semantic = semantic_key(label)
        is_typed_label = bool(
            raw_semantic
            and len(raw_semantic.split("_")) <= 4
            and (
                raw_semantic.endswith(("_id", "_code", "_number", "_token", "_path", "_status", "_amount", "_timestamp"))
                or raw_semantic in {"email", "password", "size", "color", "amount", "username", "message"}
            )
        )
        semantic = raw_semantic if is_typed_label else ""
        normalization = "verbatim"
        extra_string_span: str | None = None

        if kind == "amount":
            digits = str(numeric).replace(",", "")
            value = float(digits) if "." in digits else int(digits)
            semantic, normalization = "amount", "currency_to_json_number"
        elif kind == "phone":
            semantic = "phone_number"
        elif kind == "email":
            semantic = "email"
        elif kind == "number":
            digits = str(numeric).replace(",", "")
            value = float(digits) if "." in digits else int(digits)
            semantic = ("phone_number" if len(digits) >= 7 and "." not in digits else (semantic or "number"))
            normalization = "numeric_token"
            if "." not in digits:
                extra_string_span = digits
        elif source_text.startswith(("~/", "/home/")):
            semantic = "file_path" if re.search(r"\.[A-Za-z0-9]+$", source_text) else "directory_path"
        elif kind == "quoted":
            message_context = re.search(r"\b(?:message|sms)\b", before, flags=re.IGNORECASE)
            recipient_context = re.search(
                r"\b(?:to|named|called|recipient|contact)\s*$",
                before,
                flags=re.IGNORECASE,
            )
            identity_semantic = (
                semantic.endswith(("_id", "_code"))
                or semantic in {"access_token", "password", "email", "phone_number"}
            ) and re.search(
                r"\b(?:id|identifier|code|token|password|email|phone(?:[_ -]number)?)\b",
                label,
                flags=re.IGNORECASE,
            )
            if message_context and not recipient_context and not identity_semantic:
                semantic = "message"
            elif not semantic or semantic in {"text", "quote", "quoted_text", "string", "content"}:
                semantic = "task_text"
        elif kind == "named_token":
            semantic = semantic if semantic in {"size", "color", "fulfillment_status"} else "task_text"
            normalization = "named_token"

        occupied_ranges.append((start, end, kind))
        key = (canonical(value), json_type(value))
        if key not in seen_values:
            seen_values.add(key)
            spans.append({
                "value": value,
                "semantic_type": semantic or "task_text",
                "task_span": {
                    "start": start,
                    "end": end,
                    "text": source_text,
                    "label": label,
                    "context_text": context_text,
                    "normalization": normalization,
                },
            })
        if extra_string_span is not None:
            str_key = (canonical(extra_string_span), "string")
            if str_key not in seen_values:
                seen_values.add(str_key)
                str_semantic = "phone_number" if len(extra_string_span) >= 7 else (semantic or "task_text")
                spans.append({
                    "value": extra_string_span,
                    "semantic_type": str_semantic,
                    "task_span": {
                        "start": start,
                        "end": end,
                        "text": source_text,
                        "label": label,
                        "context_text": context_text,
                        "normalization": "digit_string",
                    },
                })
    return spans


def value_origins_hash(state_or_context: Any, arguments: Mapping[str, Any]) -> str:
    """Compute a deterministic hash of the currently committed public origins of arguments.

    Used by per-tuple precondition rejection memory so unrelated reads do not
    re-enable a blocked call while a new upstream binding for one of its input
    values does.
    """
    bindings = binding_table(state_or_context)
    origins: dict[str, list[str]] = {}
    for arg_name, arg_val in sorted(arguments.items()):
        target_canon = canonical(arg_val)
        matched: list[str] = []
        for identifier, binding in sorted(bindings.items()):
            if binding.get("source") == "public_document" or not binding.get("usable_for_binding", True):
                continue
            for path, candidate in walk_values(binding["value"]):
                if canonical(candidate) == target_canon:
                    matched.append(f"{binding.get('source', '')}:{identifier}:{path}:{binding.get('version', 0)}")
        origins[str(arg_name)] = sorted(set(matched))
    payload = canonical(origins)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class ExecutionState:
    benchmark: str
    task_id: str
    instruction: str
    initial_public_observation: str = ""
    version: int = 0
    binding_version: int = 0
    mutation_version: int = 0
    bindings: dict[str, dict[str, Any]] = field(default_factory=dict)
    receipts: list[dict[str, Any]] = field(default_factory=list)
    retrievals: list[dict[str, Any]] = field(default_factory=list)
    batches: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_task(
        cls,
        benchmark: str,
        task_id: str,
        instruction: str,
        initial_public_observation: str = "",
    ) -> "ExecutionState":
        state = cls(str(benchmark), str(task_id), str(instruction), str(initial_public_observation))
        if initial_public_observation:
            public = json.loads(initial_public_observation)
            if isinstance(public, Mapping):
                # Documentation examples are syntax, not executed business data.
                initial_value = {
                    key: deepcopy(value)
                    for key, value in public.items()
                    if key not in {"bootstrap_api_docs", "bootstrap_apis"}
                }
                state.bindings["initial_state"] = {
                    "id": "initial_state",
                    "value": initial_value,
                    "json_type": "object",
                    "semantic_type": "initial_public_state",
                    "source": "initial_observation",
                    "tool": None,
                    "arguments": {},
                    "path": "",
                    "scope": state.task_id,
                    "version": 0,
                    "committed": True,
                    "value_status": "usable",
                    "usable_for_binding": True,
                }
        for index, span in enumerate(_task_spans(state.instruction)):
            value, semantic = span["value"], span["semantic_type"]
            identifier = f"task_{index}"
            state.bindings[identifier] = {
                "id": identifier,
                "value": value,
                "json_type": json_type(value),
                "semantic_type": semantic,
                "source": "task_span",
                "tool": None,
                "arguments": {},
                "path": "",
                "scope": state.task_id,
                "version": 0,
                "committed": True,
                "value_status": "usable",
                "usable_for_binding": True,
                "task_span": deepcopy(span["task_span"]),
            }
        return state

    def dump(self) -> dict[str, Any]:
        return deepcopy({
            "benchmark": self.benchmark,
            "task_id": self.task_id,
            "instruction": self.instruction,
            "initial_public_observation": self.initial_public_observation,
            "version": self.version,
            "binding_version": self.binding_version,
            "mutation_version": self.mutation_version,
            "bindings": self.bindings,
            "receipts": self.receipts,
            "retrievals": self.retrievals,
            "batches": self.batches,
        })

    @classmethod
    def load(cls, data: Mapping[str, Any]) -> "ExecutionState":
        fields = {key: deepcopy(data[key]) for key in cls.__dataclass_fields__ if key in data}
        if isinstance(fields.get("bindings"), list):
            fields["bindings"] = {item["id"]: item for item in fields["bindings"]}
        fields.pop("task_instruction", None)
        return cls(**fields)

    def apply_receipt(self, receipt: Mapping[str, Any]) -> list[str]:
        item = deepcopy(dict(receipt))
        canonical(item)  # Commit only serializable actual native values.
        call_id = str(item["call_id"])
        if any(old["call_id"] == call_id for old in self.receipts):
            raise ValueError("A native request receipt can be committed only once.")
        kind = item.get("kind", "call")
        binding_before, mutation_before = self.binding_version, self.mutation_version
        args_origin_hash = value_origins_hash(self, item.get("arguments", {}))
        item["novel_public_fact"] = False
        item.setdefault("value_status", "usable")
        item.setdefault("rejection_kind", "")

        # Every successful state-changing write or authentication transition advances mutation_version.
        if (
            kind == "call"
            and item.get("succeeded", False)
            and (
                str(item.get("method", "")).upper() in {"POST", "PUT", "PATCH", "DELETE"}
                or str(item.get("tool", "")).endswith((".login", ".logout", ".refresh_access_token"))
            )
        ):
            self.mutation_version += 1

        def stamp_transition() -> None:
            metadata = {
                "binding_version_before": binding_before,
                "binding_version_after": self.binding_version,
                "mutation_version_before": mutation_before,
                "mutation_version_after": self.mutation_version,
                "novel_public_fact": bool(item["novel_public_fact"]),
                "value_status": item["value_status"],
                "args_origin_hash": args_origin_hash,
                "rejection_kind": str(item.get("rejection_kind", "")),
            }
            item.update(metadata)
            if isinstance(receipt, dict):
                receipt.update(metadata)

        stamp_transition()
        if kind == "discover":
            self.retrievals.append({
                "tool": item["tool"],
                "arguments": item.get("arguments", {}),
                "binding_version": self.binding_version,
                "mutation_version": self.mutation_version,
                "succeeded": bool(item.get("succeeded", True)),
            })
        self.receipts.append(item)
        self.version += 1
        if not item.get("succeeded", False) or "output" not in item or kind == "finish":
            return []
        identifier = str(item.get("bind") or f"result_{len(self.receipts)}")
        if identifier in self.bindings:
            raise ValueError(f"Binding {identifier!r} was already committed; choose a new bind name.")
        output = deepcopy(item["output"])
        semantic = str(item.get("semantic_type") or infer_output_semantic(item["tool"], item.get("public_doc")))
        # All executed business call returns remain in the public constructible support;
        # value_status is preserved as soft quality metadata for ranking/prompt display.
        usable = kind != "discover"
        novel = usable and not any(
            binding.get("source") == "native_return"
            and binding.get("usable_for_binding", True)
            and binding.get("semantic_type") == semantic
            and canonical(binding["value"]) == canonical(output)
            for binding in self.bindings.values()
        )
        if novel:
            self.binding_version += 1
        item["novel_public_fact"] = bool(novel)
        stamp_transition()
        self.bindings[identifier] = {
            "id": identifier,
            "value": output,
            "json_type": json_type(output),
            "semantic_type": semantic,
            "source": "public_document" if kind == "discover" else "native_return",
            "tool": item["tool"],
            "arguments": deepcopy(item.get("arguments", {})),
            "path": "",
            "scope": self.task_id,
            "version": self.version,
            "call_id": call_id,
            "committed": True,
            "value_status": item["value_status"],
            "usable_for_binding": usable,
            "novel_public_fact": bool(novel),
        }
        committed = [identifier]
        batch_name = item.get("batch_bind")
        if batch_name:
            batch = self.batches.setdefault(str(batch_name), {"size": int(item["batch_size"]), "items": {}})
            if batch["size"] != int(item["batch_size"]):
                raise ValueError("Batch receipt size changed during one native action.")
            batch["items"][str(int(item["batch_index"]))] = identifier
            if len(batch["items"]) == batch["size"]:
                if str(batch_name) in self.bindings:
                    raise ValueError("Batch output binding must be new.")
                members = [self.bindings[batch["items"][str(i)]] for i in range(batch["size"])]
                values = [member["value"] for member in members]
                statuses = {member.get("value_status", "usable") for member in members}
                batch_status = next(
                    (status for status in ("inapplicable", "unreliable", "provisional") if status in statuses),
                    "usable",
                )
                self.bindings[str(batch_name)] = {
                    **self.bindings[identifier],
                    "id": str(batch_name),
                    "value": values,
                    "json_type": "array",
                    "semantic_type": semantic + "_collection",
                    "member_bindings": [batch["items"][str(i)] for i in range(batch["size"])],
                    "value_status": batch_status,
                    "usable_for_binding": all(member.get("usable_for_binding", True) for member in members),
                    "novel_public_fact": False,
                }
                committed.append(str(batch_name))
        return committed

    def public_context(self) -> dict[str, Any]:
        result = self.dump()
        result["state_version"] = STATE_VERSION
        result["task_instruction"] = self.instruction
        result["bindings"] = list(result["bindings"].values())
        return result


def state_context(state_or_context: Any) -> dict[str, Any]:
    if isinstance(state_or_context, ExecutionState):
        return state_or_context.public_context()
    if not isinstance(state_or_context, Mapping):
        raise TypeError("A public state or public program context is required.")
    if "program_context" in state_or_context:
        return state_context(state_or_context["program_context"])
    if "execution_state" in state_or_context:
        return dict(state_or_context["execution_state"])
    return dict(state_or_context)


def binding_table(state_or_context: Any) -> dict[str, dict[str, Any]]:
    bindings = state_context(state_or_context).get("bindings", [])
    if isinstance(bindings, Mapping):
        return dict(bindings)
    return {str(item["id"]): dict(item) for item in bindings}


PURE_OPERATIONS = {
    "equal": (2, 2), "not_equal": (2, 2), "less": (2, 2), "less_equal": (2, 2),
    "greater": (2, 2), "greater_equal": (2, 2), "and": (2, 2), "or": (2, 2), "not": (1, 1),
    "add": (2, 2), "subtract": (2, 2), "multiply": (2, 2), "divide": (2, 2),
    "length": (1, 1), "sum": (1, 1), "min": (1, 1), "max": (1, 1),
    "all": (1, 1), "any": (1, 1), "contains": (2, 2), "unique": (1, 1),
    "set_difference": (2, 2), "concat": (2, 8), "slice": (3, 3),
    "get": (2, 2), "project": (2, 2), "filter": (2, 2), "prefix_through": (3, 3),
    "path_join": (2, 8), "basename": (1, 1), "dirname": (1, 1),
    "date_field": (2, 2), "string": (1, 1), "integer": (1, 1), "float": (1, 1),
    "lower": (1, 1), "upper": (1, 1), "strip": (1, 1), "join": (2, 2),
}


def evaluate_expression(
    expression: Any,
    state_or_context: Any,
    *,
    item: Any = None,
    depth: int = 0,
) -> Any:
    """Evaluate a finite value expression over committed public values.

    ``filter`` and ``project`` take an expression containing ``{"item":"/field"}``
    as their second argument. ``prefix_through`` takes collection, identity JSON
    pointer, and the actually observed current identity; it requires one match.
    """
    if depth > 12:
        raise ValueError("Public value expression nesting exceeds twelve.")
    if not isinstance(expression, Mapping):
        raise TypeError("Values use explicit ref, literal, item or op expression objects.")
    if "ref" in expression:
        if set(expression) - {"ref", "path"}:
            raise ValueError("Unknown fields in a value reference.")
        binding = binding_table(state_or_context)[str(expression["ref"])]
        if not binding.get("committed", True):
            raise ValueError("An uncommitted value is not an execution argument.")
        return deepcopy(pointer_get(binding["value"], str(expression.get("path", ""))))
    if set(expression) == {"literal"}:
        canonical(expression["literal"])
        return deepcopy(expression["literal"])
    if set(expression) == {"item"}:
        if item is None:
            raise ValueError("An item reference requires an observed collection element.")
        return deepcopy(pointer_get(item, str(expression["item"])))
    if set(expression) != {"op", "args"} or expression["op"] not in PURE_OPERATIONS:
        raise ValueError("Unknown public value expression.")
    operation, operands = str(expression["op"]), expression["args"]
    low, high = PURE_OPERATIONS[operation]
    if not isinstance(operands, list) or not low <= len(operands) <= high:
        raise ValueError("Pure operation arity does not match its contract.")
    ev = lambda x, current=item: evaluate_expression(x, state_or_context, item=current, depth=depth + 1)
    if operation in {"filter", "project"}:
        sequence = ev(operands[0])
        if not isinstance(sequence, list):
            raise TypeError("Collection expressions require an observed array.")
        if operation == "filter":
            return [deepcopy(value) for value in sequence if ev(operands[1], value) is True]
        return [ev(operands[1], value) for value in sequence]
    values = [ev(operand) for operand in operands]
    if operation == "equal": return values[0] == values[1]
    if operation == "not_equal": return values[0] != values[1]
    if operation == "less": return values[0] < values[1]
    if operation == "less_equal": return values[0] <= values[1]
    if operation == "greater": return values[0] > values[1]
    if operation == "greater_equal": return values[0] >= values[1]
    if operation == "and": return bool(values[0]) and bool(values[1])
    if operation == "or": return bool(values[0]) or bool(values[1])
    if operation == "not": return not values[0]
    if operation == "add": return values[0] + values[1]
    if operation == "subtract": return values[0] - values[1]
    if operation == "multiply": return values[0] * values[1]
    if operation == "divide": return values[0] / values[1]
    if operation == "length": return len(values[0])
    if operation == "sum": return sum(values[0])
    if operation == "min": return min(values[0])
    if operation == "max": return max(values[0])
    if operation == "all": return all(value is True for value in values[0])
    if operation == "any": return any(value is True for value in values[0])
    if operation == "contains": return values[1] in values[0]
    if operation == "unique":
        return list({canonical(value): value for value in values[0]}.values())
    if operation == "set_difference":
        excluded = {canonical(value) for value in values[1]}
        return [value for value in values[0] if canonical(value) not in excluded]
    if operation == "concat":
        if all(isinstance(v, list) for v in values):
            merged: list[Any] = []
            for v in values:
                merged.extend(v)
            return merged
        return "".join(str(v) for v in values)
    if operation == "slice": return values[0][values[1]:values[2]]
    if operation == "get": return pointer_get(values[0], values[1])
    if operation == "prefix_through":
        sequence, pointer, current = values
        positions = [i for i, value in enumerate(sequence) if pointer_get(value, pointer) == current]
        if len(positions) != 1:
            raise ValueError("Current-item identity must have exactly one position in the observed ordered collection.")
        return sequence[:positions[0] + 1]
    if operation == "path_join": return posixpath.join(*(str(v) for v in values))
    if operation == "basename": return posixpath.basename(str(values[0]))
    if operation == "dirname": return posixpath.dirname(str(values[0]))
    if operation == "date_field":
        if values[1] not in {"year", "month", "day", "hour", "minute", "second"}:
            raise ValueError("Unsupported calendar component.")
        parsed = datetime.fromisoformat(str(values[0]).replace("Z", "+00:00"))
        return getattr(parsed, values[1])
    if operation == "string": return str(values[0])
    if operation == "integer": return int(values[0])
    if operation == "float": return float(values[0])
    if operation == "lower": return str(values[0]).lower()
    if operation == "upper": return str(values[0]).upper()
    if operation == "strip": return str(values[0]).strip()
    if operation == "join": return str(values[0]).join(str(v) for v in values[1])
    raise ValueError("Pure operation has no implementation.")


def expression_references(expression: Any) -> set[str]:
    if not isinstance(expression, Mapping):
        return set()
    if "ref" in expression:
        return {str(expression["ref"])}
    return set().union(*(expression_references(arg) for arg in expression.get("args", [])))


def reference_for_value(
    state_or_context: Any,
    value: Any,
    *,
    semantic_type: str = "",
    allow_public_literals: bool = True,
) -> dict[str, Any]:
    """Resolve a teacher argument through Depth <= 2 bottom-up synthesis over public state.

    Every atom in the synthesized expression comes from a committed binding or a
    public task span/literal present in the instruction. Target ``value`` is used
    strictly for bottom-up equality matching and never injected as an ungrounded
    literal.
    """
    context = state_context(state_or_context)
    target_canon = canonical(value)
    preferred, ordinary = [], []
    bindings = binding_table(context)

    # Depth 0 (A_0): Direct committed binding references
    atom_refs: list[tuple[Any, dict[str, Any], str]] = []
    for identifier, binding in bindings.items():
        if binding.get("source") == "public_document" or not binding.get("usable_for_binding", True):
            continue
        source = str(binding.get("source", ""))
        for path, candidate in walk_values(binding["value"]):
            expression = {"ref": identifier, "path": path}
            atom_refs.append((candidate, expression, source))
            if canonical(candidate) != target_canon:
                continue
            key = semantic_key(path.rsplit("/", 1)[-1]) if path else binding.get("semantic_type", "")
            (preferred if not semantic_type or key == semantic_type else ordinary).append(expression)
    if preferred:
        return preferred[-1]
    if ordinary:
        return ordinary[-1]

    instruction = str(context.get("instruction", context.get("task_instruction", "")))
    if allow_public_literals and isinstance(value, str) and value and value in instruction:
        return {"literal": value}
    if allow_public_literals and isinstance(value, (int, float)) and not isinstance(value, bool):
        if re.search(r"(?<![\w.])" + re.escape(str(value)) + r"(?![\w.])", instruction):
            return {"literal": value}

    # Depth 1 (A_1): Unary pure operations over grounded references
    unary_candidates: list[tuple[Any, dict[str, Any], str]] = []
    for candidate, ref_expr, source in atom_refs:
        if isinstance(candidate, str) and candidate:
            for unary_op in ("basename", "dirname", "strip", "lower", "upper"):
                try:
                    derived = evaluate_expression({"op": unary_op, "args": [ref_expr]}, context)
                except Exception:
                    continue
                op_expr = {"op": unary_op, "args": [ref_expr]}
                unary_candidates.append((derived, op_expr, source))
                if canonical(derived) == target_canon:
                    return op_expr
            if isinstance(value, int) and not isinstance(value, bool) and re.fullmatch(r"[+-]?\d+", candidate.strip()):
                op_expr = {"op": "integer", "args": [ref_expr]}
                if canonical(int(candidate.strip())) == target_canon:
                    return op_expr
            if isinstance(value, float) and re.fullmatch(r"[+-]?\d+(?:\.\d+)?", candidate.strip()):
                op_expr = {"op": "float", "args": [ref_expr]}
                if canonical(float(candidate.strip())) == target_canon:
                    return op_expr
        elif isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            if isinstance(value, str):
                op_expr = {"op": "string", "args": [ref_expr]}
                derived = str(candidate)
                unary_candidates.append((derived, op_expr, source))
                if canonical(derived) == target_canon:
                    return op_expr
        elif isinstance(candidate, list):
            for unary_op in ("length", "sum", "min", "max", "unique"):
                try:
                    derived = evaluate_expression({"op": unary_op, "args": [ref_expr]}, context)
                except Exception:
                    continue
                if canonical(derived) == target_canon:
                    return {"op": unary_op, "args": [ref_expr]}

    # Depth 2 (A_2): N-ary path_join and concat synthesis over grounded string pieces
    if isinstance(value, str) and value:
        # Build grounded string piece map: string_value -> (priority_rank, expr)
        # Prefer ref / unary(ref) over task literals
        string_pieces: dict[str, tuple[int, dict[str, Any]]] = {}

        def _register_piece(text: str, rank: int, expr: dict[str, Any]) -> None:
            if not text:
                return
            existing = string_pieces.get(text)
            if existing is None or rank < existing[0]:
                string_pieces[text] = (rank, expr)

        for candidate, ref_expr, source in atom_refs:
            if isinstance(candidate, str) and candidate:
                rank = 0 if source == "native_return" else 1
                _register_piece(candidate, rank, ref_expr)
                if candidate.startswith(("~/", "/")):
                    _register_piece(candidate.rstrip("/"), rank, ref_expr)
        for derived, op_expr, source in unary_candidates:
            if isinstance(derived, str) and derived:
                rank = 2 if source == "native_return" else 3
                _register_piece(derived, rank, op_expr)
                if derived.startswith(("~/", "/")):
                    _register_piece(derived.rstrip("/"), rank, op_expr)

        if allow_public_literals:
            # Grounded substrings of value that also appear verbatim in instruction
            if "/" in value:
                for part in value.split("/"):
                    if part and part in instruction:
                        _register_piece(part, 4, {"literal": part})
                target_dir = posixpath.dirname(value)
                if target_dir and target_dir in instruction:
                    _register_piece(target_dir.rstrip("/"), 4, {"literal": target_dir})
            for token in re.findall(r"[A-Za-z0-9_.-]+", value):
                if token and token in instruction:
                    _register_piece(token, 4, {"literal": token})
            for sep in ("/", "_", "-", ".", " "):
                if sep in value:
                    _register_piece(sep, 5, {"literal": sep})

        # 2a. Path synthesis via path_join (2..8 args)
        if "/" in value:
            norm_target = value.rstrip("/") if value != "/" else "/"
            base_candidates: list[tuple[int, str, dict[str, Any]]] = []
            for text, (rank, expr) in string_pieces.items():
                if text.startswith(("~/", "/")) and norm_target.startswith(text.rstrip("/") + "/"):
                    base_candidates.append((rank, text.rstrip("/"), expr))
            base_candidates.sort(key=lambda item: (item[0], -len(item[1])))
            for _, base_text, base_expr in base_candidates:
                remainder = norm_target[len(base_text) + 1:]
                segments = [seg for seg in remainder.split("/") if seg]
                if 1 <= len(segments) <= 7 and all(seg in string_pieces for seg in segments):
                    seg_exprs = [string_pieces[seg][1] for seg in segments]
                    candidate_expr = {"op": "path_join", "args": [base_expr, *seg_exprs]}
                    try:
                        if evaluate_expression(candidate_expr, context) == value:
                            return candidate_expr
                    except Exception:
                        pass

        # 2b. General substring DP synthesis via concat (2..8 args)
        n = len(value)
        if 2 <= n <= 256:
            # dp[i] holds best (num_args, total_rank, expr_list) covering value[:i]
            dp: list[tuple[int, int, list[dict[str, Any]]] | None] = [None] * (n + 1)
            dp[0] = (0, 0, [])
            for i in range(n):
                state_i = dp[i]
                if state_i is None:
                    continue
                count_i, rank_i, exprs_i = state_i
                if count_i >= 8:
                    continue
                for piece_text, (piece_rank, piece_expr) in string_pieces.items():
                    end_idx = i + len(piece_text)
                    if end_idx <= n and value.startswith(piece_text, i):
                        cand_state = (count_i + 1, rank_i + piece_rank, [*exprs_i, piece_expr])
                        prev = dp[end_idx]
                        if prev is None or (cand_state[0], cand_state[1]) < (prev[0], prev[1]):
                            dp[end_idx] = cand_state
            final_state = dp[n]
            if final_state is not None and 2 <= final_state[0] <= 8:
                concat_expr = {"op": "concat", "args": final_state[2]}
                if expression_references(concat_expr) or allow_public_literals:
                    try:
                        if evaluate_expression(concat_expr, context) == value:
                            return concat_expr
                    except Exception:
                        pass

    raise ValueError(
        f"Teacher argument has no public binding or supported public derivation ({json_type(value)})."
    )
