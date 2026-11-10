"""One public JSON contract and a deterministic native-r3 action compiler.

The model selects every discovery, API call and submission. Compilation resolves
public references and pure expressions only; it never retries or selects a
business operation. Both benchmark evaluators remain native.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from functools import lru_cache
from itertools import count, product
import json
import posixpath
import re
from typing import Any, Mapping

from .native_execution_state import (
    PURE_OPERATIONS,
    binding_table,
    canonical,
    evaluate_expression,
    expression_references,
    infer_output_semantic,
    json_type,
    pointer_get,
    semantic_key,
    state_context,
    value_origins_hash,
    walk_values,
)
from .native_semantics import (
    PublicSemanticIndex,
    candidate_lexical_score,
    document_description_length,
    public_document_quality,
    tool_cluster_key,
)

PROGRAM_VERSION = "native-r3"
CONTEXT_VERSION = "native-r3-public-context"
READ_METHODS = frozenset({"GET", "HEAD"})
WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
VARIABLE_PATTERN = r"^[A-Za-z][A-Za-z0-9_]*$"
MAX_EXPRESSION_DEPTH = 6
_PROGRAM_IDS = count(1)
_KNOWN_SEMANTICS = {
    "access_token", "password", "authorization_code", "payment_auth_id", "payment_intent_id",
    "payment_status", "payment_id", "refund_amount", "refund_id", "return_request_id", "return_audit_id", "pricing_snapshot_id",
    "order_draft_item_id", "order_item_id", "order_draft_id", "warehouse_request_id", "order_id",
    "created_timestamp", "user_id", "email", "product_id", "song_id", "size", "color",
    "file_path", "directory_path", "destination_path", "page_index", "page_size", "app_name", "api_name",
}
_IDENTITY_SEMANTICS = frozenset({
    "access_token", "password", "authorization_code", "payment_auth_id", "payment_id",
    "payment_intent_id", "refund_id", "return_request_id", "return_audit_id", "pricing_snapshot_id",
    "order_item_id", "order_id", "user_id", "song_id", "product_id",
})


def _is_identity_semantic(semantic: str) -> bool:
    return semantic.endswith("_id") or semantic in _IDENTITY_SEMANTICS or semantic in {"email", "phone_number"}


def next_binding_name(context: Mapping[str, Any]) -> str:
    """Allocate the one new result namespace exposed by the current grammar.

    The name is fixed before decoding and teacher target construction. A map
    owns both the base name and all ``base_index`` member names. Failed requests
    still advance the receipt count; existing commitments are never renamed.
    """
    state, occupied = state_context(context), binding_table(context)
    index = len(state.get("receipts", [])) + 1
    while True:
        name = f"r3_result_{index}"
        if not any(identifier == name or identifier.startswith(name + "_") for identifier in occupied):
            return name
        index += 1


def _object(properties: Mapping[str, Any], required=None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(properties if required is None else required),
        "additionalProperties": False,
    }


def _choice(choices: list[dict[str, Any]]) -> dict[str, Any]:
    if not choices:
        raise ValueError("The public state contains no feasible action of the selected class.")
    return choices[0] if len(choices) == 1 else {"anyOf": choices}


def _constant(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return _object({key: _constant(child) for key, child in value.items()})
    if isinstance(value, list):
        return {
            "type": "array",
            "prefixItems": [_constant(child) for child in value],
            "minItems": len(value),
            "maxItems": len(value),
        }
    return {"const": value}


def _parameters(api: Mapping[str, Any]) -> list[dict[str, Any]]:
    schema = api.get("arguments_schema", api.get("parameters", []))
    if isinstance(schema, Mapping) and "properties" in schema:
        required = set(schema.get("required", []))
        return [
            {**deepcopy(value), "name": name, "required": name in required}
            for name, value in schema["properties"].items()
        ]
    if isinstance(schema, Mapping):
        return [{**deepcopy(value), "name": name} for name, value in schema.items()]
    return deepcopy(list(schema))


def _types(parameter: Mapping[str, Any]) -> set[str]:
    source = parameter.get("type", "string")
    if isinstance(source, list):
        return set(source)
    aliases = {
        "str": "string", "int": "integer", "float": "number", "bool": "boolean",
        "list": "array", "dict": "object", "none": "null",
    }
    return {aliases.get(part.strip().lower(), part.strip().lower()) for part in str(source).split("|")}


def _enum_values(parameter: Mapping[str, Any]) -> list[Any]:
    values = list(parameter.get("enum", []))
    for constraint in parameter.get("constraints", []):
        match = re.fullmatch(r"value in (\[.*\])", str(constraint))
        if match:
            try:
                parsed = ast.literal_eval(match.group(1))
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                values.extend(parsed)
    return values


def _type_matches(value: Any, parameter: Mapping[str, Any]) -> bool:
    if value is None and not parameter.get("required", False) and parameter.get("default") is None:
        return True
    actual, types = json_type(value), _types(parameter)
    if not (actual in types or (actual == "integer" and "number" in types) or not types):
        return False
    enums = _enum_values(parameter)
    if enums and not any(canonical(value) == canonical(item) for item in enums):
        return False
    return True


def _parameter_semantic(
    parameter: Mapping[str, Any],
    index: PublicSemanticIndex | None = None,
    tool: str = "",
) -> str:
    if index is not None:
        return index.parameter_semantic(parameter, tool)
    name, description = str(parameter["name"]), str(parameter.get("description", ""))
    if name in {"id", "identifier", "value", "path", "record"}:
        interpreted = semantic_key(description.split(".", 1)[0])
        if interpreted in _KNOWN_SEMANTICS:
            return interpreted
    return semantic_key(name, description)


def _leaf_semantic(
    binding: Mapping[str, Any],
    path: str,
    value: Any,
    index: PublicSemanticIndex | None = None,
) -> str:
    if index is not None:
        return index.binding_semantic(binding, path, value)
    if path:
        fields = [
            part.replace("~1", "/").replace("~0", "~")
            for part in path.split("/")[1:]
            if not part.isdigit()
        ]
        if fields and semantic_key(fields[-1]):
            return semantic_key(fields[-1])
    if isinstance(value, str) and value.startswith(("~/", "/home/")):
        return "file_path" if re.search(r"\.[A-Za-z0-9]+$", value) else "directory_path"
    return str(binding.get("semantic_type", ""))


def _credential_scope(binding: Mapping[str, Any], path: str) -> str:
    producer = str(binding.get("tool") or "")
    if producer.endswith(".login"):
        return producer.split(".", 1)[0]
    parts = path.split("/")
    if len(parts) > 2:
        try:
            parent = pointer_get(binding["value"], "/".join(parts[:-1]))
        except Exception:
            parent = None
        if isinstance(parent, Mapping):
            for key in ("account_name", "app_name", "account", "app"):
                if isinstance(parent.get(key), str):
                    return parent[key]
    return ""


def _semantics_match(actual: str, expected: str, parameter: Mapping[str, Any] | None = None) -> bool:
    """Soft semantic alignment predicate used for ranking, never for hard exclusion."""
    if not actual or not expected:
        return False
    if actual == expected:
        return True
    if {actual, expected} <= {"file_path", "directory_path", "destination_path"}:
        return True
    if actual.endswith("_id") or expected.endswith("_id"):
        return False
    if actual in _KNOWN_SEMANTICS and expected in _KNOWN_SEMANTICS:
        return False
    a = set(actual.split("_")) - {"record", "reference", "value"}
    e = set(expected.split("_")) - {"record", "reference", "value"}
    return bool(a and e and (a <= e or e <= a))


def _synthesized_path_candidates(
    context: Mapping[str, Any],
    parameter: Mapping[str, Any],
    expected: str,
) -> list[dict[str, Any]]:
    """Construct Depth <= 2 grounded path expressions (dirname, path_join) for path parameters."""
    if "string" not in _types(parameter):
        return []
    param_name = str(parameter.get("name", "")).lower()
    if expected not in {"file_path", "directory_path", "destination_path"} and "path" not in param_name and "dir" not in param_name:
        return []
    bindings = binding_table(context)
    base_dirs: list[tuple[str, dict[str, Any], int]] = []
    segment_tokens: list[tuple[str, dict[str, Any], int]] = []
    file_basenames: list[tuple[str, dict[str, Any], int]] = []

    for identifier, binding in bindings.items():
        if binding.get("source") == "public_document" or not binding.get("usable_for_binding", True):
            continue
        version = int(binding.get("version", 0))
        source = str(binding.get("source", ""))
        for path, value in walk_values(binding["value"]):
            if not isinstance(value, str) or not value:
                continue
            ref_expr = {"ref": identifier, "path": path}
            if value.startswith(("~/", "/")):
                base_dirs.append((value.rstrip("/"), ref_expr, version))
                if source == "native_return" and "/" in value.rstrip("/"):
                    parent = posixpath.dirname(value.rstrip("/"))
                    if parent and parent != "/":
                        base_dirs.append((parent, {"op": "dirname", "args": [ref_expr]}, version))
                    bname = posixpath.basename(value.rstrip("/"))
                    if bname and "." in bname:
                        file_basenames.append((bname, {"op": "basename", "args": [ref_expr]}, version))
            elif source == "task_span" and re.fullmatch(r"[A-Za-z0-9_.-]{1,40}", value):
                segment_tokens.append((value, ref_expr, version))

    synthesized: list[dict[str, Any]] = []
    seen_vals: set[str] = set()
    for base_val, base_expr, base_ver in base_dirs[:6]:
        for seg_val, seg_expr, seg_ver in segment_tokens[:8]:
            joined_dir = posixpath.join(base_val, seg_val)
            if joined_dir not in seen_vals:
                seen_vals.add(joined_dir)
                synthesized.append({
                    "expression": {"op": "path_join", "args": [base_expr, seg_expr]},
                    "value": joined_dir,
                    "semantic_type": "directory_path",
                    "source": "synthesized_path",
                    "scope": state_context(context).get("task_id"),
                    "score": 2.5,
                    "_version": max(base_ver, seg_ver),
                })
            for fname_val, fname_expr, fname_ver in file_basenames[:6]:
                joined_file = posixpath.join(base_val, seg_val, fname_val)
                if joined_file not in seen_vals:
                    seen_vals.add(joined_file)
                    synthesized.append({
                        "expression": {"op": "path_join", "args": [base_expr, seg_expr, fname_expr]},
                        "value": joined_file,
                        "semantic_type": "file_path",
                        "source": "synthesized_path",
                        "scope": state_context(context).get("task_id"),
                        "score": 2.4,
                        "_version": max(base_ver, seg_ver, fname_ver),
                    })
    return synthesized


_STRICT_FORMAT_SORTS = {
    "access_token", "password", "authorization_code",
    "file_path", "directory_path", "destination_path",
    "email", "phone_number", "page_index", "page_size",
    "app_name", "api_name", "created_timestamp", "timestamp",
}


def _schema_compatible(
    actual: str,
    expected: str,
    parameter: Mapping[str, Any],
    value: Any,
    *,
    benchmark: str = "",
    source: str = "",
    index: PublicSemanticIndex | None = None,
) -> bool:
    """Many-sorted compatibility check for tool call parameters.

    Enforces sort disjointness across entity identifiers (*_id), credentials,
    paths, and structured relational fields so that unready downstream tools
    remain in missing_inputs until their upstream sort is observed, while
    admitting grounded task-span quantities/texts for AppWorld parameters.
    """
    if _semantics_match(actual, expected, parameter):
        return True
    if index is not None and index.equivalent(actual, expected):
        return True
    if benchmark == "planbench_xl":
        return False
    if expected in {"access_token", "password", "authorization_code"} or actual in {
        "access_token", "password", "authorization_code"
    }:
        return actual == expected
    if expected in {"file_path", "directory_path", "destination_path"} or actual in {
        "file_path", "directory_path", "destination_path"
    }:
        return {actual, expected} <= {"file_path", "directory_path", "destination_path"}
    if expected.endswith("_id") or actual.endswith("_id"):
        if expected == "record_id" and (actual.endswith("_id") or isinstance(value, int)):
            return True
        return actual == expected
    if expected == "email" or actual == "email":
        return isinstance(value, str) and "@" in value and expected in {
            "email", "username", "user", "recipient", "sender"
        }
    if expected == "phone_number" or actual == "phone_number":
        return (
            actual == "phone_number"
            or (isinstance(value, str) and re.fullmatch(r"\+?\d[\d -]{6,14}\d", value) is not None)
        ) and expected in {"phone_number", "username", "user", "recipient", "sender"}
    if expected in _STRICT_FORMAT_SORTS or actual in _STRICT_FORMAT_SORTS:
        return False
    if source == "task_span":
        if _types(parameter) <= {"integer", "number"} and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if "string" in _types(parameter) and isinstance(value, str) and actual in {"message", "task_text", "text"}:
            return True
    return False


def public_value_candidates(
    context: Mapping[str, Any],
    parameter: Mapping[str, Any],
    tool: str = "",
    *,
    semantic_index: PublicSemanticIndex | None = None,
    allow_any_scalar: bool = False,
) -> list[dict[str, Any]]:
    """Return all publicly grounded, many-sorted schema-compatible value candidates.

    Hard support depends on public provenance and formal many-sorted schema/scope
    compatibility. Semantic similarity and recency rank the returned list softly.
    """
    index = semantic_index or _semantic_index(context)
    expected = _parameter_semantic(parameter, index, tool)
    benchmark = str(context.get("benchmark", ""))
    scored: list[tuple[float, int, int, dict[str, Any]]] = []
    ordinal = 0

    for identifier, binding in binding_table(context).items():
        if binding.get("source") == "public_document" or not binding.get("usable_for_binding", True):
            continue
        version = int(binding.get("version", 0))
        source = str(binding.get("source", ""))
        for path, value in walk_values(binding["value"]):
            if not _type_matches(value, parameter):
                continue
            # Do not bind whole composite containers to scalar parameters
            if isinstance(value, (Mapping, list)) and not (_types(parameter) & {"object", "array"}):
                continue
            actual = _leaf_semantic(binding, path, value, index)
            if expected in {"access_token", "password"} and "." in tool:
                scope = _credential_scope(binding, path)
                if scope and scope.lower() != tool.split(".", 1)[0].lower():
                    continue
            if not allow_any_scalar and not _schema_compatible(
                actual, expected, parameter, value, benchmark=benchmark, source=source, index=index
            ):
                continue
            score = candidate_lexical_score(
                parameter,
                binding,
                path,
                value,
                tool=tool,
                idf=index.idf,
                expected_semantic=expected,
                actual_semantic=actual,
            )
            if _semantics_match(actual, expected, parameter):
                score += 4.0
            score += {
                "usable": 2.0,
                "provisional": -1.0,
                "unreliable": -2.5,
                "inapplicable": -4.0,
            }.get(str(binding.get("value_status", "usable")), 0.0)
            ordinal += 1
            scored.append((
                score,
                version,
                -ordinal,
                {
                    "expression": {"ref": identifier, "path": path},
                    "value": deepcopy(value),
                    "semantic_type": actual,
                    "source": binding.get("source"),
                    "scope": binding.get("scope"),
                    "score": round(score, 4),
                },
            ))

    if context.get("benchmark") == "appworld":
        for item in _synthesized_path_candidates(context, parameter, expected):
            ordinal += 1
            ver = int(item.pop("_version", 0))
            scored.append((float(item["score"]), ver, -ordinal, item))

    scored.sort(key=lambda entry: (entry[0], entry[1], entry[2]), reverse=True)
    return [entry[3] for entry in scored]


def _documented_literals(parameter: Mapping[str, Any]) -> list[Any]:
    values = list(parameter.get("enum", []))
    if "default" in parameter:
        values.append(parameter["default"])
    for constraint in parameter.get("constraints", []):
        match = re.fullmatch(r"value in (\[.*\])", str(constraint))
        if match:
            try:
                parsed = ast.literal_eval(match.group(1))
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                values.extend(parsed)
    return list({canonical(value): value for value in values if _type_matches(value, parameter)}.values())


def _control_parameter(parameter: Mapping[str, Any]) -> bool:
    return (
        _parameter_semantic(parameter) in {"page_index", "page_size"}
        or str(parameter["name"]) in {
            "offset", "limit", "page", "sort_by", "sort_direction", "ascending", "descending",
            "recursive", "overwrite", "exist_ok", "show_hidden", "include_hidden", "compress",
        }
        or _types(parameter) <= {"boolean", "null"}
    )


def _task_literal_candidates(context: Mapping[str, Any], parameter: Mapping[str, Any]) -> list[Any]:
    instruction, values = str(state_context(context).get("instruction", "")), _documented_literals(parameter)
    if _control_parameter(parameter):
        if _types(parameter) & {"integer", "number"}:
            values.extend([0, 1])
        if "boolean" in _types(parameter):
            values.extend([False, True])
    if "string" in _types(parameter) and _parameter_semantic(parameter) in {
        "file_path", "directory_path", "destination_path",
    }:
        values.extend(match.group(0).rstrip(".,;") for match in re.finditer(r"(?:~/|/home/)[^\s`]+", instruction))
    return list({canonical(value): value for value in values if _type_matches(value, parameter)}.values())


def _tool_catalog(context: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if context["benchmark"] == "appworld":
        return {
            name: deepcopy(api)
            for name, api in context.get("apis", {}).items()
            if not name.startswith("api_docs.") and name != "supervisor.complete_task"
        }
    return {str(tool["name"]): deepcopy(tool) for tool in context.get("tools", [])}


@lru_cache(maxsize=256)
def _cached_semantic_index(catalog_json: str) -> PublicSemanticIndex:
    return PublicSemanticIndex(json.loads(catalog_json))


def _semantic_index(context: Mapping[str, Any]) -> PublicSemanticIndex:
    return _cached_semantic_index(canonical(_tool_catalog(context)))


def _public_aw_discoveries(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    apis = context.get("apis", {})
    apps = {name.split(".")[0] for name in apis if "." in name and not name.startswith("api_docs.")}
    names: set[tuple[str, str]] = set()
    for binding in binding_table(context).values():
        if binding.get("source") == "initial_observation" and isinstance(binding["value"], Mapping):
            app_descriptions = binding["value"].get("apps", {})
            if isinstance(app_descriptions, Mapping):
                apps.update(str(app) for app in app_descriptions)
        if binding.get("source") != "public_document":
            continue
        producer, value = binding.get("tool"), binding["value"]
        if producer == "api_docs.show_app_descriptions" and isinstance(value, list):
            apps.update(str(item["name"]) for item in value if isinstance(item, Mapping) and "name" in item)
        if producer == "api_docs.show_api_descriptions" and isinstance(value, list):
            app = binding.get("arguments", {}).get("app_name")
            if app:
                names.update(
                    (str(app), str(item["name"]))
                    for item in value
                    if isinstance(item, Mapping) and "name" in item
                )
    candidates = [{"tool": "api_docs.show_app_descriptions", "arguments": {}}]
    candidates.extend(
        {"tool": "api_docs.show_api_descriptions", "arguments": {"app_name": app}}
        for app in sorted(apps)
    )
    candidates.extend(
        {"tool": "api_docs.show_api_doc", "arguments": {"app_name": app, "api_name": api}}
        for app, api in sorted(names)
        if f"{app}.{api}" not in apis
    )
    # Deduplicate all attempted discovery queries across the episode
    seen = {
        (str(r["tool"]), canonical(r.get("arguments", {})))
        for r in state_context(context).get("retrievals", [])
    }
    return [
        candidate
        for candidate in candidates
        if (candidate["tool"], canonical(candidate["arguments"])) not in seen
    ]


def _public_pb_discoveries(context: Mapping[str, Any], missing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    state = state_context(context)
    index = _semantic_index(context)
    instruction = str(state.get("instruction", ""))
    target = _target_phrase(instruction)
    needs = list(
        dict.fromkeys(
            [target]
            + [index.resolve(item["semantic_type"]).replace("_", " ") for item in missing if item.get("semantic_type")]
        )
    )
    known = list(
        dict.fromkeys(
            _leaf_semantic(binding, path, value, index).replace("_", " ")
            for binding in binding_table(context).values()
            if binding.get("source") in {"task_span", "native_return"} and binding.get("usable_for_binding", True)
            for path, value in walk_values(binding["value"])
            if value is not None
            and not isinstance(value, (Mapping, list))
            and _leaf_semantic(binding, path, value, index)
            and _leaf_semantic(binding, path, value, index)
            not in {"task_text", "text", "message", "number", "named_token"}
        )
    )
    candidates = [{"meta_tool": "by_output_info", "output_information": need} for need in needs if need]
    candidates.extend({"meta_tool": "by_inputs", "input_information": [value]} for value in known if value)
    if 1 < len(known) <= 2:
        candidates.append({"meta_tool": "by_inputs", "input_information": known})

    def query_key(query: Mapping[str, Any]) -> str:
        return canonical({
            "meta_tool": query.get("meta_tool", ""),
            "inputs": sorted({index.resolve(str(value)) for value in query.get("input_information", [])}),
            "output": index.resolve(str(query.get("output_information", ""))),
        })

    # Deduplicate all completed concept queries across the episode, including 0-tool returns
    seen = {
        query_key(item["arguments"])
        for item in state.get("retrievals", [])
        if isinstance(item.get("arguments"), Mapping) and "meta_tool" in item["arguments"]
    }
    unique = {}
    for candidate in candidates:
        key = query_key(candidate)
        if key not in seen:
            unique.setdefault(key, candidate)
    return list(unique.values())


def _target_phrase(instruction: str) -> str:
    sentences = [sentence.strip() for sentence in re.split(r"[.!?]\s*", instruction) if sentence.strip()]
    meaningful = [sentence for sentence in sentences if re.search(r"[A-Za-z0-9]", sentence)]
    return meaningful[-1] if meaningful else instruction


def _normalized_arguments(arguments: Mapping[str, Any], doc: Mapping[str, Any]) -> dict[str, Any]:
    values = {}
    for parameter in _parameters(doc):
        name = parameter["name"]
        if name in arguments:
            values[name] = arguments[name]
        elif "default" in parameter:
            values[name] = parameter["default"]
    return values


def _is_read_call(context: Mapping[str, Any], tool: str, doc: Mapping[str, Any]) -> bool:
    if context.get("benchmark") == "planbench_xl":
        return True
    method = str(doc.get("method", "")).upper()
    if method in WRITE_METHODS or tool.endswith((".login", ".logout", ".refresh_access_token")):
        return False
    return True


def _is_tuple_blocked(
    context: Mapping[str, Any],
    tool: str,
    doc: Mapping[str, Any],
    resolved_arguments: Mapping[str, Any],
) -> bool:
    """Check whether a single resolved argument tuple (T, v) is operationally redundant.

    1. Completed deterministic reads at the current mutation_version are blocked
       (independent of binding_version, so unrelated reads do not revive old reads).
    2. Input-precondition rejections are blocked at (mutation_version, args_origin_hash)
       until a new upstream binding alters the provenance of at least one input.
    3. Deterministic terminal rejections (e.g., 422 / null tool execution) are
       blocked at the current mutation_version.
    """
    state = state_context(context)
    current_mutation = int(state.get("mutation_version", 0))
    norm_canon = canonical(_normalized_arguments(resolved_arguments, doc))
    is_read = _is_read_call(context, tool, doc)
    origin_hash: str | None = None

    for receipt in state.get("receipts", []):
        if receipt.get("kind") != "call" or receipt.get("tool") != tool:
            continue
        if int(receipt.get("mutation_version_after", 0)) != current_mutation:
            continue
        if canonical(_normalized_arguments(receipt.get("arguments", {}), doc)) != norm_canon:
            continue
        if receipt.get("succeeded", False):
            if is_read:
                return True
        else:
            rejection_kind = str(receipt.get("rejection_kind", ""))
            if rejection_kind == "input_precondition":
                if origin_hash is None:
                    origin_hash = value_origins_hash(context, resolved_arguments)
                if receipt.get("args_origin_hash", "") == origin_hash:
                    return True
            else:
                return True
    return False


def _filter_unblocked_arguments(
    context: Mapping[str, Any],
    tool: str,
    doc: Mapping[str, Any],
    arguments: dict[str, dict[str, Any]],
) -> bool:
    """Prune per-tuple exhausted values from argument domains and return whether all tuples are exhausted."""
    state = state_context(context)
    current_mutation = int(state.get("mutation_version", 0))
    has_prior_for_tool = any(
        r.get("kind") == "call"
        and r.get("tool") == tool
        and int(r.get("mutation_version_after", 0)) == current_mutation
        for r in state.get("receipts", [])
    )
    if not has_prior_for_tool:
        return False

    absent = object()
    names: list[str] = []
    domains: list[list[Any]] = []
    expr_by_param: dict[str, list[tuple[dict[str, Any], Any]]] = {}
    lit_by_param: dict[str, list[Any]] = {}
    count_values = 1

    for parameter in _parameters(doc):
        name = parameter["name"]
        support = arguments[name]
        ref_pairs: list[tuple[dict[str, Any], Any]] = []
        values: list[Any] = []
        for reference in support["references"]:
            try:
                val = evaluate_expression(reference, context)
            except Exception:
                continue
            ref_pairs.append((reference, val))
            values.append(val)
        lit_vals = list(support["literals"])
        values.extend(lit_vals)
        unique_values = list({canonical(value): value for value in values}.values())
        if not parameter.get("required", False):
            if "default" in parameter:
                default = parameter["default"]
                if not any(canonical(value) == canonical(default) for value in unique_values):
                    unique_values.append(default)
            else:
                unique_values.append(absent)
        if not unique_values:
            return False
        expr_by_param[name] = ref_pairs
        lit_by_param[name] = lit_vals
        count_values *= len(unique_values)
        if count_values > 128:
            # Domain cross-product exceeds enumeration bound; compile-time _is_tuple_blocked still guards
            return False
        names.append(name)
        domains.append(unique_values)

    unblocked_tuples: list[dict[str, Any]] = []
    for combination in product(*domains):
        resolved = {name: value for name, value in zip(names, combination) if value is not absent}
        if not _is_tuple_blocked(context, tool, doc, resolved):
            unblocked_tuples.append(resolved)

    if not unblocked_tuples:
        return True

    # Prune individual parameter candidate expressions whose values only occur in blocked tuples
    for parameter in _parameters(doc):
        name = parameter["name"]
        allowed_canons = {canonical(tup[name]) for tup in unblocked_tuples if name in tup}
        support = arguments[name]
        support["references"] = [
            ref for ref, val in expr_by_param[name] if canonical(val) in allowed_canons
        ]
        support["literals"] = [
            lit for lit in lit_by_param[name] if canonical(lit) in allowed_canons
        ]
    return False


def terminal_answer_candidates(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return all grounded scalar native_return candidates, softly ranked by target relevance."""
    instruction = str(state_context(context).get("instruction", ""))
    target_phrase = _target_phrase(instruction)
    target_sem = semantic_key(target_phrase)
    parameter = {
        "name": target_sem or "answer",
        "description": target_phrase,
        "type": ["string", "number", "integer", "boolean"],
        "required": True,
    }
    return [
        candidate
        for candidate in public_value_candidates(context, parameter, allow_any_scalar=True)
        if candidate["source"] == "native_return"
    ]


def _business_write_receipts(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        receipt
        for receipt in state_context(context).get("receipts", [])
        if receipt.get("kind") == "call"
        and receipt.get("succeeded")
        and str(receipt.get("method", "")).upper() in WRITE_METHODS
        and not str(receipt.get("tool", "")).endswith((".login", ".logout", ".refresh_access_token"))
    ]


def _map_frontier(context: Mapping[str, Any], catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    if context["benchmark"] != "appworld":
        return []
    semantic_index = _semantic_index(context)
    collections = [
        (identifier, path, value, binding)
        for identifier, binding in binding_table(context).items()
        if binding.get("source") != "public_document" and binding.get("usable_for_binding", True)
        for path, value in walk_values(binding["value"])
        if isinstance(value, list) and value
    ]
    result = []
    for tool, doc in catalog.items():
        parameters = _parameters(doc)
        supported = []
        for identifier, path, values, binding in collections:
            per_item = {}
            for parameter in parameters:
                if (
                    public_value_candidates(context, parameter, tool, semantic_index=semantic_index)
                    or _task_literal_candidates(context, parameter)
                ):
                    continue
                common = None
                for index, value in enumerate(values):
                    paths = {
                        part
                        for part, child in walk_values(value)
                        if _type_matches(child, parameter)
                        and not isinstance(child, (Mapping, list))
                    }
                    common = paths if common is None else common & paths
                if parameter.get("required", False) and not common:
                    break
                if common:
                    per_item[parameter["name"]] = [{"item": part} for part in sorted(common)]
            else:
                if per_item:
                    supported.append({"items": {"ref": identifier, "path": path}, "item_arguments": per_item})
        if supported:
            result.append({"tool": tool, "collections": supported})
    return result


def _dependency_focus(
    context: Mapping[str, Any],
    catalog: Mapping[str, Any],
    ready: list[dict[str, Any]],
    missing: list[dict[str, Any]],
) -> dict[str, Any]:
    """Rank stored public documents by task dependencies, document quality, and current data.

    All candidates remain in the executable frontier. This order only allocates
    prompt space and selects the missing-input discovery vocabulary.
    """
    state = state_context(context)
    index = _semantic_index(context)
    instruction = str(state.get("instruction", ""))
    target = index.resolve(instruction if context["benchmark"] == "appworld" else _target_phrase(instruction))
    outputs = {name: set(index.produced_semantics(name, doc)) for name, doc in catalog.items()}
    inputs = {
        name: {
            _parameter_semantic(p, index, name)
            for p in _parameters(doc)
            if p.get("required", False) and not _control_parameter(p)
        }
        for name, doc in catalog.items()
    }
    distance = {target: 0} if target else {}
    queue = list(distance)
    for desired in queue:
        for name, output in outputs.items():
            if desired not in output:
                continue
            for dependency in sorted(inputs[name]):
                if dependency and dependency not in distance:
                    distance[dependency] = distance[desired] + 1
                    queue.append(dependency)
    known, latest = set(), set()
    business = [
        binding
        for binding in binding_table(context).values()
        if binding.get("source") == "native_return" and binding.get("usable_for_binding", True)
    ]
    latest_version = max((int(binding.get("version", 0)) for binding in business), default=-1)
    for binding in binding_table(context).values():
        if binding.get("source") not in {"task_span", "native_return"} or not binding.get("usable_for_binding", True):
            continue
        semantics = {
            _leaf_semantic(binding, path, value, index)
            for path, value in walk_values(binding["value"])
            if not isinstance(value, (Mapping, list)) and value is not None
        }
        known.update(semantics)
        if binding.get("source") == "native_return" and int(binding.get("version", 0)) == latest_version:
            latest.update(semantics)
    ready_names = {call["tool"] for call in ready}
    used = {
        str(receipt["tool"])
        for receipt in state.get("receipts", [])
        if receipt.get("kind") == "call" and receipt.get("succeeded")
    }
    task_words = set(re.findall(r"[a-z][a-z0-9]*", str(state.get("instruction", "")).lower()))
    quality_rank = {"usable": 0, "provisional": 1, "unreliable": 2, "inapplicable": 3}

    def priority(name: str) -> tuple[Any, ...]:
        doc = catalog[name]
        qual = quality_rank.get(public_document_quality(doc), 1)
        is_ready, on_path = name in ready_names, bool(outputs[name] & distance.keys())
        fresh_output = bool(outputs[name] - known)
        next_edge = bool(inputs[name] & latest)
        if is_ready and on_path and fresh_output:
            tier = 0
        elif is_ready and fresh_output and next_edge and name not in used:
            tier = 1
        elif is_ready and fresh_output and name not in used:
            tier = 2
        elif is_ready and next_edge and name not in used:
            tier = 3
        elif is_ready and name not in used:
            tier = 4
        elif is_ready:
            tier = 5
        elif on_path and fresh_output:
            tier = 6
        elif on_path:
            tier = 7
        else:
            tier = 8
        words = set(re.findall(r"[a-z][a-z0-9]*", name.replace("_", " ").lower()))
        return (
            qual,
            tier,
            not next_edge,
            name in used,
            -len(inputs[name] & known),
            min((distance.get(output, len(catalog) + 1) for output in outputs[name]), default=len(catalog) + 1),
            -len(words & task_words),
            document_description_length(doc),
            name,
        )

    ranked = sorted(catalog, key=priority)
    # Diversify top focus across operation clusters, placing ready tools before unready tools
    focus: list[str] = []
    seen_clusters: set[str] = set()
    for require_ready in (True, False):
        for name in ranked:
            if (name in ready_names) != require_ready:
                continue
            cluster = tool_cluster_key(name, catalog[name], index)
            if cluster not in seen_clusters and len(focus) < 12:
                seen_clusters.add(cluster)
                focus.append(name)
        for name in ranked:
            if (name in ready_names) != require_ready:
                continue
            if len(focus) >= 12:
                break
            if name not in focus:
                focus.append(name)

    needed = sorted(distance, key=lambda semantic: (distance[semantic], semantic))
    focused_missing = [
        item
        for item in missing
        if item["tool"] in focus
        or outputs[item["tool"]] & distance.keys()
        or public_document_quality(catalog[item["tool"]]) == "usable"
    ]
    return {
        "relevant_tools": ranked,
        "focus_tools": focus,
        "dependency_semantics": needed,
        "discovery_missing_inputs": focused_missing,
    }


def _context_cache_key(context: Mapping[str, Any]) -> str:
    relevant = {
        key: context[key]
        for key in (
            "benchmark", "apis", "tools", "retrieval_schema", "use_tool_ids",
            "execution_state", "generation_token_budget",
        )
        if key in context
    }
    return canonical(relevant)


def build_frontier(context: Mapping[str, Any]) -> dict[str, Any]:
    return deepcopy(_build_frontier_cached(_context_cache_key(context)))


@lru_cache(maxsize=256)
def _build_frontier_cached(context_json: str) -> dict[str, Any]:
    context = json.loads(context_json)
    ready, missing, bindings = [], [], binding_table(context)
    stored_catalog = _tool_catalog(context)
    # Invariant 1: Never delete tools from the executable catalog based on heuristic document caveats
    catalog = stored_catalog
    semantic_index = _semantic_index(context)
    exhausted = []
    for name, tool in sorted(catalog.items()):
        arguments, required_missing = {}, []
        for parameter in _parameters(tool):
            candidates = public_value_candidates(context, parameter, name, semantic_index=semantic_index)
            literals = _task_literal_candidates(context, parameter)
            arguments[parameter["name"]] = {
                "semantic_type": _parameter_semantic(parameter, semantic_index, name),
                "references": [candidate["expression"] for candidate in candidates],
                "literals": literals,
                "required": bool(parameter.get("required", False)),
            }
            if parameter.get("required", False) and not candidates and not literals:
                required_missing.append({
                    "tool": name,
                    "parameter": parameter["name"],
                    "semantic_type": _parameter_semantic(parameter, semantic_index, name),
                    "description": parameter.get("description", ""),
                })
        missing.extend(required_missing)
        if not required_missing:
            if _filter_unblocked_arguments(context, name, tool, arguments):
                exhausted.append(name)
            else:
                ready.append({"tool": name, "arguments": arguments})

    focus = _dependency_focus(context, catalog, ready, missing)
    discovery_missing = focus.pop("discovery_missing_inputs")
    discoveries = (
        _public_aw_discoveries(context)
        if context["benchmark"] == "appworld"
        else _public_pb_discoveries(context, discovery_missing)
    )
    observed = [
        binding
        for binding in bindings.values()
        if binding.get("source") == "native_return" and binding.get("usable_for_binding", True)
    ]
    target_answers = terminal_answer_candidates(context)
    if context["benchmark"] == "planbench_xl":
        finish_ready = bool(target_answers)
    else:
        finish_ready = bool(observed)
    ready_maps = _map_frontier(context, {name: doc for name, doc in catalog.items() if name not in exhausted})
    modes = (["I"] if discoveries else []) + (["E"] if ready or ready_maps or finish_ready else [])

    state = state_context(context)
    retrievals = state.get("retrievals", [])
    recent_disc_receipts = [r for r in state.get("receipts", []) if r.get("kind") == "discover"][-3:]
    recent_new_tools = sum(
        int(
            r.get(
                "new_tool_count",
                len(r.get("output", [])) if isinstance(r.get("output"), list) else (1 if r.get("succeeded") else 0),
            )
        )
        for r in recent_disc_receipts
    )
    discovery_telemetry = {
        "remaining_queries": len(discoveries),
        "executed_queries": len(retrievals),
        "recent_new_tools": recent_new_tools,
        "ready_business_calls": len(ready),
    }

    return {
        "version": state.get("version", 0),
        "binding_version": state.get("binding_version", 0),
        "mutation_version": state.get("mutation_version", 0),
        "feasible_modes": modes,
        "ready_calls": ready,
        "ready_maps": ready_maps,
        "missing_inputs": missing,
        "discovery_queries": discoveries,
        "discovery_telemetry": discovery_telemetry,
        "target_answer_references": [x["expression"] for x in target_answers],
        "finish_ready": finish_ready,
        "next_binding": next_binding_name(context),
        "exhausted_calls_at_current_state": exhausted,
        "qualified_out_tools": {
            name: public_document_quality(doc)
            for name, doc in stored_catalog.items()
            if public_document_quality(doc) != "usable"
        },
        **focus,
    }


def available_modes(context: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(build_frontier(context)["feasible_modes"])


def program_instruction(context: Mapping[str, Any], mode: str | None) -> str:
    return (
        "Emit one complete native-r3 JSON action. discover requests public documentation/tool retrieval; "
        "call executes every business API, including authentication and read-only business APIs. "
        "Use immutable ref/path expressions for actual returned data or task spans. Credentials, record identifiers and "
        "authorization codes retain their documented business meaning and app scope. Bind every required input. "
        "Use exactly the next_binding supplied by the current public frontier for this action; it is fixed before decoding. "
        "When ready_calls already contain the required upstream or target business tool, prefer executing call over redundant discovery; "
        "never repeat an identical completed retrieval or an already exhausted read tuple at the same mutation version. "
        "Follow actual JSON paths, list shapes and full tool qualifications. "
        "AppWorld map_call applies the chosen API to an already observed finite items array; item pointers "
        "address each element. Use file_system APIs for task files and pure path_join/dirname/basename expressions to compose paths. "
        "Finish only after the requested task operations or final target values are established through actual native returns; "
        "omit answer for AppWorld operation tasks and use a grounded answer reference for questions. "
        "The native evaluator determines terminal success. "
        + ("Selected action class: " + mode + ". " if mode else "Choose discovery or execution directly. ")
        + "Pure expressions have ref/path, literal, item, or op/args form and perform no external action. "
        "Return JSON only; follow the supplied schema with no Markdown, XML or explanatory text."
    )


def _expression_definitions(context: Mapping[str, Any]) -> dict[str, Any]:
    scalar = {"type": ["null", "boolean", "number", "string"]}
    definitions: dict[str, Any] = {
        "json": {
            "anyOf": [
                scalar,
                {"type": "array", "items": {"$ref": "#/$defs/json"}},
                {"type": "object", "additionalProperties": {"$ref": "#/$defs/json"}},
            ]
        }
    }
    refs, item_paths = [], set()
    for identifier, binding in binding_table(context).items():
        if binding.get("source") == "public_document" or not binding.get("usable_for_binding", True):
            continue
        paths = [path for path, _ in walk_values(binding["value"])]
        refs.append(_object({"ref": {"const": identifier}, "path": {"enum": paths}}, required=["ref"]))
        for _, value in walk_values(binding["value"]):
            if isinstance(value, list):
                for item in value:
                    item_paths.update(path for path, _ in walk_values(item))
    atoms = refs + [_object({"literal": {"$ref": "#/$defs/json"}})]
    for depth in range(MAX_EXPRESSION_DEPTH + 1):
        for family, item_scope in (("value", False), ("item_value", True)):
            choices = deepcopy(atoms)
            if item_scope and item_paths:
                choices.append(_object({"item": {"enum": sorted(item_paths)}}))
            if depth:
                for operation, (low, high) in PURE_OPERATIONS.items():
                    child = {"$ref": f"#/$defs/{family}_{depth-1}"}
                    if operation in {"filter", "project"}:
                        args = {
                            "type": "array",
                            "prefixItems": [child, {"$ref": f"#/$defs/item_value_{depth-1}"}],
                            "minItems": 2,
                            "maxItems": 2,
                        }
                    else:
                        args = {"type": "array", "items": child, "minItems": low, "maxItems": high}
                    choices.append(_object({"op": {"const": operation}, "args": args}))
            definitions[f"{family}_{depth}"] = _choice(choices)
    return definitions


def _group_reference_schemas(
    references: Sequence[Mapping[str, Any]],
    *,
    include_ops: bool = True,
) -> list[dict[str, Any]]:
    by_ref: dict[str, list[str]] = {}
    op_refs: list[Mapping[str, Any]] = []
    for ref in references:
        if "ref" in ref and set(ref.keys()) <= {"ref", "path"}:
            rid = str(ref["ref"])
            rpath = str(ref.get("path", ""))
            paths = by_ref.setdefault(rid, [])
            if rpath not in paths:
                paths.append(rpath)
        elif include_ops:
            op_refs.append(ref)
    result = [
        _object({"ref": {"const": rid}, "path": {"enum": paths}}, required=["ref", "path"])
        for rid, paths in by_ref.items()
    ]
    if include_ops:
        result.extend(_constant(ref) for ref in op_refs)
    return result


def program_schema(context: Mapping[str, Any], mode: str | None) -> dict[str, Any]:
    relevant = {
        key: context[key]
        for key in (
            "benchmark", "apis", "tools", "retrieval_schema", "use_tool_ids",
            "execution_state", "generation_token_budget",
        )
        if key in context
    }
    return deepcopy(_program_schema(canonical(relevant), mode))


@lru_cache(maxsize=256)
def _program_schema(context_json: str, mode: str | None) -> dict[str, Any]:
    context = json.loads(context_json)
    if mode not in {None, "I", "E"}:
        raise ValueError("The action mode must be I, E, or None.")
    frontier, definitions, variants = build_frontier(context), _expression_definitions(context), []
    bind = {"const": next_binding_name(context)}
    if mode in {None, "I"}:
        for query in frontier["discovery_queries"]:
            if context["benchmark"] == "appworld":
                core = {
                    "kind": {"const": "discover"},
                    "tool": {"const": query["tool"]},
                    "arguments": _object(
                        {key: _constant({"literal": value}) for key, value in query["arguments"].items()}
                    ),
                }
            else:
                core = {"kind": {"const": "discover"}, "arguments": _constant(query)}
            core["bind"] = bind
            variants.append(_object(core))
    if mode in {None, "E"}:
        for call in frontier["ready_calls"]:
            properties, required = {}, []
            for name, argument in call["arguments"].items():
                has_general_expr = (
                    context["benchmark"] == "appworld"
                    and not _is_identity_semantic(argument["semantic_type"])
                )
                choices = _group_reference_schemas(
                    argument["references"],
                    include_ops=not has_general_expr,
                )
                choices += [_constant({"literal": value}) for value in argument["literals"]]
                if has_general_expr:
                    choices.append({"$ref": "#/$defs/value_6"})
                if choices:
                    properties[name] = _choice(choices)
                    if argument["required"]:
                        required.append(name)
            core = {
                "kind": {"const": "call"},
                "tool": {"const": call["tool"]},
                "arguments": _object(properties, required),
                "bind": bind,
            }
            variants.append(_object(core))
        if context["benchmark"] == "appworld":
            for mapped in frontier["ready_maps"]:
                name, tool = mapped["tool"], _tool_catalog(context)[mapped["tool"]]
                arguments = _object(
                    {p["name"]: {"$ref": "#/$defs/item_value_6"} for p in _parameters(tool)},
                    [p["name"] for p in _parameters(tool) if p.get("required", False)],
                )
                core = {
                    "kind": {"const": "map_call"},
                    "tool": {"const": name},
                    "items": {"$ref": "#/$defs/value_6"},
                    "arguments": arguments,
                    "bind": bind,
                }
                variants.append(_object(core))
        if frontier["finish_ready"]:
            answer_schema = (
                _choice(_group_reference_schemas(frontier["target_answer_references"]))
                if context["benchmark"] == "planbench_xl"
                else {"$ref": "#/$defs/value_6"}
            )
            core = {"kind": {"const": "finish"}, "answer": answer_schema}
            variants.append(
                _object(
                    core,
                    required=["kind", "answer"] if context["benchmark"] == "planbench_xl" else ["kind"],
                )
            )
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", **_choice(variants), "$defs": definitions}


def _check_literals(
    expression: Any,
    context: Mapping[str, Any],
    parameter: Mapping[str, Any],
    *,
    discover: bool = False,
    in_operation: bool = False,
) -> None:
    if "literal" in expression:
        value = expression["literal"]
        if discover:
            return
        instruction = str(state_context(context).get("instruction", ""))
        if any(canonical(value) == canonical(item) for item in _documented_literals(parameter)):
            return
        if isinstance(value, str) and value and value in instruction:
            return
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if re.search(r"(?<![\w.])" + re.escape(str(value)) + r"(?![\w.])", instruction):
                return
        if _control_parameter(parameter) and isinstance(value, (int, float, bool)):
            return
        if value is None and not parameter.get("required", False):
            return
        if in_operation and (
            (
                isinstance(value, str)
                and (
                    value.startswith("/")
                    or value in {"year", "month", "day", "hour", "minute", "second", "/", "", ".", "..", "_", "-", " "}
                )
            )
            or isinstance(value, (int, bool))
        ):
            return
        for binding in binding_table(context).values():
            if (
                binding.get("source") != "public_document"
                and binding.get("usable_for_binding", True)
                and any(canonical(candidate) == canonical(value) for _, candidate in walk_values(binding["value"]))
            ):
                return
        raise ValueError("Factual literals must occur in the public task, native returns or documented constants.")
    for child in expression.get("args", []):
        _check_literals(child, context, parameter, discover=discover, in_operation=True)


def _resolve_arguments(
    program: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    item: Any = None,
    discover: bool = False,
) -> dict[str, Any]:
    tool = program["tool"]
    semantic_index = _semantic_index(context)
    doc = context["apis"][tool] if context["benchmark"] == "appworld" else _tool_catalog(context)[tool]
    parameters, result = {p["name"]: p for p in _parameters(doc)}, {}
    for name, expression in program["arguments"].items():
        parameter = parameters[name]
        _check_literals(expression, context, parameter, discover=discover)
        value = evaluate_expression(expression, context, item=item)
        if not _type_matches(value, parameter):
            raise TypeError(f"{tool}.{name} requires {parameter.get('type')}; observed {json_type(value)}.")
        if "ref" in expression and not discover:
            identifier, path = str(expression["ref"]), str(expression.get("path", ""))
            binding = binding_table(context)[identifier]
            expected = _parameter_semantic(parameter, semantic_index, tool)
            if expected in {"access_token", "password"} and "." in tool:
                scope = _credential_scope(binding, path)
                if scope and scope.lower() != tool.split(".", 1)[0].lower():
                    raise ValueError("Credential reference belongs to another app.")
        result[name] = value
    for parameter in parameters.values():
        if parameter.get("required", False) and parameter["name"] not in result:
            raise ValueError(f"Unbound required input {tool}.{parameter['name']}.")
    if item is None and not discover and _is_tuple_blocked(context, tool, doc, result):
        raise ValueError(f"Call tuple for {tool} was already exhausted at the current mutation version.")
    return result


def _check_fields(value: Any, required: set[str], optional: set[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object.")
    missing, extra = required - value.keys(), value.keys() - required - optional
    if missing or extra:
        raise ValueError(f"{label}: missing fields {sorted(missing)}, unknown fields {sorted(extra)}.")


def _check_pointer(pointer: Any) -> None:
    if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
        raise ValueError("Expression paths must be RFC 6901 JSON pointers.")
    if re.search(r"~(?![01])", pointer):
        raise ValueError("JSON pointers use only ~0 and ~1 escapes.")


def _validate_expression(
    expression: Any,
    bindings: Mapping[str, Any],
    *,
    item_scope: bool = False,
    depth: int = 0,
) -> None:
    """Validate each tagged AST node once; no schema-alternative backtracking.

    The six-operation depth limit matches value_6/item_value_6 in XGrammar.
    Item scope is introduced only by map_call or filter/project's lambda body.
    Every reference points to a currently usable public value.
    """
    if depth > MAX_EXPRESSION_DEPTH:
        raise ValueError(f"Expression nesting exceeds {MAX_EXPRESSION_DEPTH} operations.")
    if not isinstance(expression, Mapping):
        raise TypeError("An expression must use ref, literal, item, or op/args form.")
    tags = set(expression) & {"ref", "literal", "item", "op"}
    if len(tags) != 1:
        raise ValueError("An expression must have exactly one expression tag.")
    tag = next(iter(tags))
    if tag == "ref":
        _check_fields(expression, {"ref"}, {"path"}, "Reference")
        identifier = expression["ref"]
        if not isinstance(identifier, str) or re.fullmatch(VARIABLE_PATTERN, identifier) is None:
            raise ValueError("A reference must name a public result identifier.")
        path = expression.get("path", "")
        _check_pointer(path)
        if identifier not in bindings:
            raise ValueError(f"The reference {identifier!r} has no committed public value.")
        binding = bindings[identifier]
        if (
            binding.get("source") == "public_document"
            or not binding.get("committed", True)
            or not binding.get("usable_for_binding", True)
        ):
            raise ValueError("Execution expressions require committed task data or actual native returns.")
        try:
            pointer_get(binding["value"], path)
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ValueError(f"Reference {identifier!r} has no observed value at {path!r}.") from error
        return
    if tag == "literal":
        _check_fields(expression, {"literal"}, set(), "Literal")
        return
    if tag == "item":
        _check_fields(expression, {"item"}, set(), "Item reference")
        _check_pointer(expression["item"])
        if not item_scope:
            raise ValueError("An item reference requires map_call or a filter/project lambda.")
        return
    _check_fields(expression, {"op", "args"}, set(), "Pure operation")
    operation, arguments = expression["op"], expression["args"]
    if not isinstance(operation, str) or operation not in PURE_OPERATIONS:
        raise ValueError("Unknown pure operation.")
    if depth == MAX_EXPRESSION_DEPTH:
        raise ValueError("The deepest expression level admits atoms only.")
    low, high = PURE_OPERATIONS[operation]
    if not isinstance(arguments, list) or not low <= len(arguments) <= high:
        raise ValueError(f"{operation} requires between {low} and {high} operands.")
    for index, child in enumerate(arguments):
        _validate_expression(
            child,
            bindings,
            item_scope=item_scope or (operation in {"filter", "project"} and index == 1),
            depth=depth + 1,
        )


def validate_program(program: Any, context: Mapping[str, Any], mode: str | None) -> None:
    """Validate the selected action and tagged AST without expanding anyOf.

    JSON Schema remains the sampling grammar. Runtime validation dispatches by
    kind/tool and expression tag, traverses actual nodes, and checks membership
    in the same public finite frontier. It never builds nested error trees.
    """
    canonical(program)
    if mode not in {None, "I", "E"}:
        raise ValueError("The action mode must be I, E, or None.")
    if not isinstance(program, Mapping) or not isinstance(program.get("kind"), str):
        raise TypeError("A native action must be an object with a string kind.")
    kind = program["kind"]
    if kind not in {"discover", "call", "map_call", "finish"}:
        raise ValueError("Unknown native action kind.")
    selected_mode = "I" if kind == "discover" else "E"
    if mode is not None and mode != selected_mode:
        raise ValueError("Program kind contradicts the selected feasible action class.")
    if kind == "map_call" and context["benchmark"] != "appworld":
        raise ValueError("map_call is available only in AppWorld.")
    required, optional = {"kind"}, set()
    if kind == "finish":
        (required if context["benchmark"] == "planbench_xl" else optional).add("answer")
    else:
        required.update({"arguments", "bind"})
        if kind != "discover" or context["benchmark"] == "appworld":
            required.add("tool")
        if kind == "map_call":
            required.add("items")
    _check_fields(program, required, optional, "Native action")
    bindings, frontier = binding_table(context), build_frontier(context)
    if kind != "finish" and program["bind"] != next_binding_name(context):
        raise ValueError("bind must equal the current public frontier's preallocated next_binding.")
    if kind == "finish":
        if not frontier["finish_ready"]:
            raise ValueError("The public terminal frontier is not ready.")
        if "answer" in program:
            _validate_expression(program["answer"], bindings)
            if context["benchmark"] == "planbench_xl" and canonical(program["answer"]) not in {
                canonical(reference) for reference in frontier["target_answer_references"]
            }:
                raise ValueError("A PlanBench answer must select an actual scalar target binding.")
        return
    if not isinstance(program["arguments"], Mapping):
        raise TypeError("Action arguments must be an object.")
    if kind == "discover":
        if context["benchmark"] == "appworld":
            if not isinstance(program["tool"], str):
                raise TypeError("A discovery tool must have a public string name.")
            arguments = {}
            for name, expression in program["arguments"].items():
                _check_fields(expression, {"literal"}, set(), "Discovery argument")
                arguments[name] = expression["literal"]
            query = {"tool": program["tool"], "arguments": arguments}
        else:
            query = program["arguments"]
        if canonical(query) not in {canonical(candidate) for candidate in frontier["discovery_queries"]}:
            raise ValueError("Discovery must select an unexecuted query in the current public frontier.")
        return
    tool = program["tool"]
    if not isinstance(tool, str):
        raise TypeError("A business tool must have a public string name.")
    calls = {call["tool"]: call for call in frontier["ready_calls"]}
    if kind == "call":
        if tool not in calls:
            raise ValueError("The selected tool has unresolved public inputs, is exhausted, or is undiscovered.")
        arguments = calls[tool]["arguments"]
        allowed = {
            name: value
            for name, value in arguments.items()
            if value["references"]
            or value["literals"]
            or (context["benchmark"] == "appworld" and not _is_identity_semantic(value["semantic_type"]))
        }
        _check_fields(
            program["arguments"],
            {name for name, value in allowed.items() if value["required"]},
            {name for name, value in allowed.items() if not value["required"]},
            "Call arguments",
        )
        for name, expression in program["arguments"].items():
            _validate_expression(expression, bindings)
            argument = allowed[name]
            general_expression = context["benchmark"] == "appworld" and not _is_identity_semantic(
                argument["semantic_type"]
            )
            choices = [*argument["references"], *({"literal": value} for value in argument["literals"])]
            if not general_expression and canonical(expression) not in {canonical(choice) for choice in choices}:
                raise ValueError(f"{tool}.{name} must select a documented literal or compatible public reference.")
        return
    if tool not in {mapped["tool"] for mapped in frontier["ready_maps"]}:
        raise ValueError("The selected mapped tool has no compatible observed collection.")
    parameters = _parameters(_tool_catalog(context)[tool])
    _check_fields(
        program["arguments"],
        {p["name"] for p in parameters if p.get("required", False)},
        {p["name"] for p in parameters if not p.get("required", False)},
        "Mapped arguments",
    )
    _validate_expression(program["items"], bindings)
    for expression in program["arguments"].values():
        _validate_expression(expression, bindings, item_scope=True)


def declared_variables(program: Mapping[str, Any]) -> list[str]:
    return [str(program["bind"])] if "bind" in program else []


def definitely_assigned_variables(program: Mapping[str, Any]) -> list[str]:
    return declared_variables(program)


def compile_program(program: Any, context: Mapping[str, Any], mode: str | None) -> dict[str, Any]:
    validate_program(program, context, mode)
    kind = program["kind"]
    selected_mode = "I" if kind == "discover" else "E"
    if mode is not None and mode != selected_mode:
        raise ValueError("Program kind contradicts the selected feasible action class.")
    program_id, plan = f"program-{next(_PROGRAM_IDS)}", []
    binding = next_binding_name(context) if kind == "finish" else program["bind"]
    common = {
        "mode": selected_mode,
        "r3_program": deepcopy(program),
        "typed_program": deepcopy(program),
        "typed_program_version": PROGRAM_VERSION,
        "program_id": program_id,
        "generation_token_budget": context.get("generation_token_budget"),
        "receipt_plan": plan,
    }
    if kind == "finish":
        if not build_frontier(context)["finish_ready"]:
            raise ValueError("The public terminal frontier is not ready for submission.")
        answer = None
        if "answer" in program:
            references = expression_references(program["answer"])
            if not references or not any(
                binding_table(context)[ref].get("source") == "native_return" for ref in references
            ):
                raise ValueError("A final answer must depend on an actual observed business return.")
            answer = evaluate_expression(program["answer"], context)
        if context["benchmark"] == "planbench_xl":
            if isinstance(answer, (dict, list)):
                raise ValueError("Select the actual scalar target value before final submission.")
            return {**common, "kind": "final_answer", "answer": str(answer)}
        arguments = {"answer": answer} if "answer" in program else {}
        plan.append({
            "tool": "supervisor.complete_task",
            "arguments": arguments,
            "kind": "finish",
            "bind": binding,
        })
    elif context["benchmark"] == "planbench_xl":
        if kind == "discover":
            query = deepcopy(program["arguments"])
            plan.append({
                "tool": "retrieve_tools",
                "arguments": query,
                "kind": "discover",
                "bind": binding,
            })
            return {**common, "kind": "retrieve_tools", "arguments": query}
        arguments, tool = _resolve_arguments(program, context), _tool_catalog(context)[program["tool"]]
        identifier = "tool_id" if context.get("use_tool_ids") else "name"
        plan.append({
            "tool": program["tool"],
            "arguments": arguments,
            "kind": "call",
            "bind": binding,
            "public_doc": deepcopy(tool),
            "semantic_type": _semantic_index(context).output_semantic(program["tool"], tool),
        })
        return {**common, "kind": "tool_call", identifier: tool[identifier], "arguments": arguments}
    else:
        if kind == "map_call":
            if not expression_references(program["items"]):
                raise ValueError("A mapped call requires an actually observed finite collection.")
            items = evaluate_expression(program["items"], context)
            if not isinstance(items, list) or not items:
                raise ValueError("map_call requires a nonempty observed array.")
            observed_values = {
                canonical(value)
                for fact in binding_table(context).values()
                if fact.get("source") != "public_document"
                for _, value in walk_values(fact["value"])
            }
            if any(canonical(item) not in observed_values for item in items):
                raise ValueError("Every map item must come from the actual observed collection.")
        else:
            items = [None]
        for index, item in enumerate(items):
            arguments = _resolve_arguments(program, context, item=item, discover=kind == "discover")
            entry = {
                "tool": program["tool"],
                "arguments": arguments,
                "kind": "discover" if kind == "discover" else "call",
                "bind": f"{binding}_{index}" if kind == "map_call" else binding,
                "public_doc": deepcopy(context["apis"][program["tool"]]),
            }
            if kind == "map_call":
                entry.update(batch_bind=binding, batch_index=index, batch_size=len(items))
            plan.append(entry)
    lines = ["import json as _r3_json"]
    for entry in plan:
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*\.[A-Za-z][A-Za-z0-9_]*", entry["tool"]) is None:
            raise ValueError("Native AppWorld calls require one public app and API identifier.")
        lines.append(f"print(_r3_json.dumps(apis.{entry['tool']}(**{entry['arguments']!r}), ensure_ascii=False))")
    return {
        **common,
        "kind": "code",
        "code": "\n".join(lines),
        "declared_variables": [],
        "definitely_assigned_variables": [],
    }
