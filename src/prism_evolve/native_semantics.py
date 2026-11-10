"""Public provenance labels and soft lexical/IDF ranking for ``native-r3.3``.

Under the First-Principles architecture (Invariant 1: Public Constructibility),
hard action/argument support depends ONLY on public provenance and machine-readable
JSON Schema compatibility. Entity labels, natural-language token overlap, IDF
weights, operation clusters, description conciseness, and document reliability
caveats are used exclusively for soft candidate ranking and prompt presentation—
never to delete tools from ``catalog`` or exclude schema-compatible grounded
candidates from the grammar or compiler.
"""
from __future__ import annotations

from collections import defaultdict
import math
import posixpath
import re
from typing import Any, Iterable, Mapping, Sequence


_GENERIC_FIELDS = frozenset({"id", "identifier", "record", "value", "result", "data", "path"})
_JSON_TYPES = frozenset({"string", "integer", "number", "boolean", "object", "array", "null"})
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "get", "has", "have", "if", "in", "into", "is", "it",
    "its", "may", "not", "of", "on", "or", "out", "please", "set", "should",
    "that", "the", "their", "them", "then", "these", "this", "those", "to",
    "up", "use", "using", "was", "were", "what", "when", "where", "which",
    "while", "who", "will", "with", "would", "your", "my", "me",
})


def _words(value: Any) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value))
    text = re.sub(r"[_\-]", " ", text).lower()
    return re.sub(r"\s+", " ", text).strip(" `\"'.,:;")


def _label(value: Any) -> str:
    return "_".join(re.findall(r"[a-z][a-z0-9]*", _words(value)))


def _stem(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ses") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize_words(value: Any, *, keep_generic: bool = True) -> list[str]:
    """Tokenize identifiers, paths, and descriptions into stemmed lowercase words."""
    words = re.findall(r"[a-z0-9]+", _words(value))
    out: list[str] = []
    for w in words:
        st = _stem(w)
        if not st or st in _STOPWORDS:
            continue
        if not keep_generic and st in _GENERIC_FIELDS:
            continue
        out.append(st)
    return out


def _first_sentence(description: Any) -> str:
    return re.split(r"[.!?](?:\s|$)", str(description).strip(), maxsplit=1)[0].strip()


def _identity_definition(description: str) -> str:
    """Return the entity head defined by an explicit public identity sentence."""
    first = _words(_first_sentence(description))
    match = re.match(r"^(?:(?:a|an|the) )?(?:unique )?(?:id|identifier|record id) (?:of|for) (.+)$", first)
    if match is None:
        return ""
    entity = re.split(r",|;|\bmeaning\b|\bwhich\b|\bthat\b", match.group(1), maxsplit=1)[0].strip()
    return re.sub(r"^(?:a|an|the|specific|particular|exact)\s+", "", entity)


def _identity_label(entity: str) -> str:
    """Name the defined entity for provenance display and soft scoring."""
    text = _words(entity)
    if re.search(r"\b(?:line )?item (?:inside|in|within) (?:an? |the )?(?:order draft|draft order)\b", text):
        return "order_draft_item_id"
    if re.search(r"\bline item (?:inside|in|within) (?:an? |the )?(?:final |placed |completed )*order\b", text):
        return "order_item_id"
    if re.match(r"^(?:(?:the|a|an|internal) )*(?:review|audit) (?:record|id|ticket) for (?:an? |the )?return (?:request|case|ticket)\b", text):
        return "return_audit_id"
    head = re.split(r"\b(?:for|belonging to|created for|used for|associated with|linked to)\b", text, maxsplit=1)[0]
    head = re.sub(r"^(?:(?:a|an|the|specific|particular|exact|final|placed|completed|original|recorded|actual)\s+)+", "", head).strip()
    head = re.sub(r"\s+itself$", "", head)
    known = _name_semantic(head)
    if known.endswith("_id") or known in {"authorization_code", "access_token", "password"}:
        return known
    head = re.sub(r"\b(?:record|identifier|id)\b", " ", head)
    noun = _label(head)
    return noun + "_id" if noun else ""


_NON_ID_PROPERTY_RE = re.compile(
    r"\b(?:status|state|outcome|decision|stage|standing|time|timestamp|date|"
    r"recorded at|filed at|updated at|created at|when|whether|where|how|what|"
    r"email|e mail|phone|telephone|mobile|amount|total|price|fee|cost|"
    r"size|sizing|color|colour|count|quantity|reason|carrier|category|kind|type)\b"
)


def _name_semantic(name: str) -> str:
    """Normalize surface nouns into concise provenance labels for soft ranking."""
    raw = _words(name)
    has = lambda pattern: re.search(pattern, raw) is not None
    if has(r"\b(?:access|bearer) token\b"):
        return "access_token"
    if has(r"\b(?:password|passwd)\b"):
        return "password"
    if has(r"\b(?:authorization|authorisation|auth|approval).*\bcode\b"):
        return "authorization_code"
    if has(r"\bemail\b|\be mail\b"):
        return "email"
    if has(r"\bphone(?: number)?\b|\btelephone(?: number)?\b|\bmobile number\b"):
        return "phone_number"
    if has(r"\busername\b|\buser name\b"):
        return "username"
    if has(r"\b(?:account on file|masked account(?: number)?|payment account(?: number)?|card or account(?: number)?)\b"):
        return "payment_account_number"
    if has(r"(?:refund|money returned|money back).*(?:amount|total)|\bmoney returned\b"):
        return "refund_amount"
    if has(r"payment.*(?:status|state)|settlement (?:status|state)"):
        return "payment_status"
    if has(r"\bshipment.*(?:status|state)\b"):
        return "shipment_status"
    if has(r"\bdelivery.*(?:status|state)\b"):
        return "delivery_status"
    if has(r"order.*(?:created|creation)|created (?:timestamp|time)|creation time"):
        return "created_timestamp"
    if re.match(r"^(?:(?:selected|chosen|item|variant|product|sizing) )*size(?:$| (?:of|choice|option|label|value|name)\b)", raw) or has(r"\bsize$|\bsizing choice\b"):
        return "size"
    if re.match(r"^(?:(?:selected|chosen|item|variant|product|finish) )*(?:color|colour)(?:$| (?:of|choice|option|label|value|name)\b)", raw) or has(r"\b(?:color|colour)$"):
        return "color"
    if has(r"\bdestination\b|\btarget path\b"):
        return "destination_path"
    if has(r"\bfile (?:path|name)\b|\bsource file\b|\bsource path\b"):
        return "file_path"
    if has(r"\bdirectory path\b|\bfolder path\b"):
        return "directory_path"
    if has(r"\bpage size\b|\bper page\b|\blimit\b"):
        return "page_size"
    if has(r"\bpage(?: index| number)?\b"):
        return "page_index"
    if has(r"\bapp name\b"):
        return "app_name"
    if has(r"\bapi name\b"):
        return "api_name"
    # Non-ID property phrases (state, status, time, amount, category, etc.) are not *_id sorts
    if _NON_ID_PROPERTY_RE.search(raw) and not re.search(r"\b(?:id|identifier)\s*$", raw):
        words = re.findall(r"[a-z][a-z0-9]*", raw)
        stop = {"the", "a", "an", "of", "for", "this", "that", "unique", "get", "show", "retrieve"}
        return "_".join(word for word in words if word not in stop)
    if has(r"\breturn(?: request| case| ticket)? (?:review|audit|approval)\b|\breview ticket\b"):
        return "return_audit_id"
    if has(r"\breturn (?:request|case|ticket)\b"):
        return "return_request_id"
    if has(r"\brefund.*\b(?:id|record|reference)\b") or raw in {"refund", "finance refund"}:
        return "refund_id"
    if has(r"\b(?:payment )?(?:authorization|authorisation|auth|approval).*\b(?:id|record|reference)\b") or raw in {
            "payment authorization", "payment authorisation", "payment auth"}:
        return "payment_auth_id"
    if has(r"\bpayment intent\b|\bcheckout intent\b"):
        return "payment_intent_id"
    if has(r"\b(?:payment|billing) (?:method|instrument)\b"):
        return "payment_method_id"
    if has(r"\bpayment.*\b(?:id|record|reference)\b") or raw in {"payment", "completed payment"}:
        return "payment_id"
    if has(r"\bpricing (?:snapshot|quote|result)\b|\bsaved quote\b"):
        return "pricing_snapshot_id"
    if has(r"\border draft (?:line )?item\b|\bdraft (?:line|item)\b|\bprecheckout line item\b|\bcart line item\b|\bcart item\b"):
        return "order_draft_item_id"
    if has(r"\border (?:line|(?:line |record )?item)\b|\bplaced order (?:line|item)\b|\bpurchased item\b|\bfinal order line\b"):
        return "order_item_id"
    if has(r"\border draft\b|\bdraft order\b|\bdraft cart\b|\bin progress order\b"):
        return "order_draft_id"
    if has(r"\b(?:pick pack|warehouse|fulfillment|fulfilment) (?:request|(?:work )?order|job)\b"):
        return "warehouse_request_id"
    if has(r"\b(?:delivery|dropoff|doorstep) (?:attempt|try).*\b(?:id|record|reference)\b") or raw in {
            "delivery attempt", "delivery try", "dropoff attempt", "doorstep attempt"}:
        return "delivery_attempt_id"
    if has(r"\bshipment.*\b(?:id|record|reference)\b") or raw in {"shipment", "outbound shipment"}:
        return "shipment_id"
    if has(r"\border.*\b(?:id|record|reference)\b") or raw in {"order", "placed order", "completed order"}:
        return "order_id"
    if has(r"\b(?:account|user|customer|shopper).*\b(?:id|record|reference)\b") or raw in {
            "customer", "customer or user account", "user account", "shopper", "user"}:
        return "user_id"
    if has(r"\b(?:product |purchasable )?(?:variant|sku)\b|\bproduct option\b"):
        return "variant_id"
    if has(r"\b(?:product|parent product).*\b(?:id|record|reference)\b") or raw in {"product", "parent product"}:
        return "product_id"
    if has(r"\b(?:song|track).*\b(?:id|record|reference)\b") or raw in {"song", "track"}:
        return "song_id"
    if has(r"\bsize\b|\bsizing choice\b"):
        return "size"
    if has(r"\bcolor\b|\bcolour\b"):
        return "color"
    words = re.findall(r"[a-z][a-z0-9]*", raw)
    stop = {"the", "a", "an", "of", "for", "this", "that", "unique", "get", "show", "retrieve"}
    return "_".join(word for word in words if word not in stop)


def semantic_key(name: str, description: str = "") -> str:
    """Interpret a field using its public definition, then its surface name."""
    raw, first = _words(name), _words(_first_sentence(description))
    if raw in {"username", "user name"}:
        phone = re.search(r"\b(?:phone|telephone|mobile)(?: number)?\b", first) is not None
        email = re.search(r"\b(?:email|e mail)\b", first) is not None
        if phone != email:
            return "phone_number" if phone else "email"
    entity = _identity_definition(description)
    if entity:
        return _identity_label(entity)
    if re.match(r"^(?:(?:the|a|an) )?email address\b", first):
        return "email"
    if re.match(r"^(?:(?:the|a|an) )?(?:phone|telephone|mobile) number\b", first):
        return "phone_number"
    if re.match(r"^(?:(?:the|a|an) )?(?:selected )?size\b", first):
        return "size"
    if re.match(r"^(?:(?:the|a|an) )?(?:selected )?(?:color|colour)\b", first):
        return "color"
    return _name_semantic(raw or first)


def _output_name(tool: str) -> str:
    operation = str(tool).rsplit(".", 1)[-1]
    output = re.split(r"_from_", operation, maxsplit=1)[0]
    return re.sub(r"^(?:get|show|retrieve|fetch)_", "", output)


def _described_output(description: str) -> str:
    """Extract the output noun from the public operation clause, excluding input."""
    first = _first_sentence(description)
    match = re.search(
        r"\b(?:this(?:\s+(?:endpoint|route|path|tool|function|call|lookup|helper))?|it)\s+"
        r"(?:returns|retrieves|fetches|surfaces|resolves|pulls|looks\s+up|gives|provides|tells\s+you)\s+(.+)",
        first,
        re.I,
    )
    if match is None:
        match = re.match(
            r"^(?:returns|retrieves|fetches|surfaces|resolves|pulls|looks\s+up|gives|provides)\s+(.+)",
            first,
            re.I,
        )
    if match is None:
        match = re.search(r"\btraces?\b.+?\bback to\s+(.+)", first, re.I)
    if match is None:
        return ""
    phrase = match.group(1).strip()
    entity = _identity_definition(phrase)
    if entity:
        return _identity_label(entity)
    if re.match(r"^(?:when|whether|where|how|why|what)\b", phrase, re.I):
        return ""
    noun = re.split(
        r"\b(?:that|which|who|belonging to|associated with|linked to|created after|created from|behind|saved on|set as|used for|created for)\b",
        phrase,
        maxsplit=1,
    )[0]
    noun = re.sub(r"^(?:(?:a|an|the|exact|original|corresponding|linked)\s+)+", "", noun, flags=re.I).strip()
    if re.match(r"^(?:internal )?(?:review|audit) (?:record|id|ticket) for (?:an? |the )?return (?:request|case|ticket)\b", _words(noun)):
        return "return_audit_id"
    if _NON_ID_PROPERTY_RE.search(_words(noun)):
        head = re.split(r"\b(?:of|for|on|in|from|after|before|under|within)\b", noun, maxsplit=1)[0].strip()
        candidate = semantic_key(head)
        if candidate in {
            "email", "phone_number", "size", "color", "refund_amount",
            "payment_status", "shipment_status", "delivery_status", "created_timestamp",
            "file_path", "directory_path", "destination_path",
        }:
            return candidate
        return ""
    head = re.split(r"\b(?:for|created for|used for)\b", noun, maxsplit=1)[0].strip()
    candidate = semantic_key(head)
    if candidate.endswith("_id") or candidate in {
            "access_token", "password", "authorization_code", "email", "phone_number",
            "size", "color", "refund_amount", "payment_status", "shipment_status", "delivery_status",
            "created_timestamp", "file_path", "directory_path", "destination_path"}:
        return candidate
    return ""


def infer_output_semantic(tool: str, doc: Mapping[str, Any] | None = None) -> str:
    document = doc or {}
    surface = document.get("surface_output")
    if isinstance(surface, str) and surface.strip():
        return semantic_key(surface)
    if isinstance(surface, Mapping):
        definition = str(surface.get("description", ""))
        if isinstance(surface.get("name"), str) and surface["name"]:
            return semantic_key(surface["name"], definition)
        if _identity_definition(definition):
            return semantic_key("", definition)
        if definition:
            return semantic_key(definition)
    described = _described_output(str(document.get("description", "")))
    if described:
        return described
    return semantic_key(_output_name(tool))


def public_document_quality(doc: Mapping[str, Any] | None) -> str:
    """Classify explicit public reliability caveats for soft ordering and prompt display.

    Caveat statuses ('usable', 'provisional', 'unreliable', 'inapplicable') never
    delete a tool from the catalog or suppress a grounded binding.
    """
    text = _words((doc or {}).get("description", ""))
    if re.search(r"\b(?:not (?:usable|suitable|appropriate|applicable|available) (?:here|now)|"
                 r"no longer (?:returns? (?:valid|usable)(?: production)? data|valid)|"
                 r"(?:data|results?) (?:is|are) no longer valid|"
                 r"does not return valid(?: production)? data|not suitable for (?:this|the current) (?:case|task)|"
                 r"not expected in (?:this|the current) setting)\b", text):
        return "inapplicable"
    if re.search(r"\b(?:unreliable|noisy|miswired|mapping is unstable|not dependable|possibly (?:wrong|incorrect)|"
                 r"(?:wrong|incorrect) adjacent (?:field|value)|"
                 r"may (?:map|read|return|surface|pull).{0,70}(?:wrong|nearby|semantically nearby) (?:field|value)|"
                 r"mapping can slip to a related field)\b", text):
        return "unreliable"
    if re.search(r"\b(?:preview value|predicted or preview|estimated value|non authoritative preview|"
                 r"default or preview|outdated cache|stale cache|lagging cache|"
                 r"older (?:cache snapshot|cached value)|cached historical value|"
                 r"may (?:return|serve|surface) (?:an? )?(?:old|older|outdated) values?)\b", text):
        return "provisional"
    return "usable"


def document_description_length(doc: Mapping[str, Any] | None) -> int:
    """Return word count of a tool's description for soft conciseness tie-breaking."""
    if not isinstance(doc, Mapping):
        return 0
    return len(re.findall(r"[a-z0-9]+", _words(doc.get("description", ""))))


def tool_cluster_key(
    tool: str,
    doc: Mapping[str, Any] | None = None,
    index: "PublicSemanticIndex | None" = None,
) -> str:
    """Group tools representing similar input-to-output transitions for soft ordering."""
    document = doc or {}
    out_sem = index.output_semantic(tool, document) if index is not None else infer_output_semantic(tool, document)
    in_sems = sorted(
        (index.parameter_semantic(p, tool) if index is not None else semantic_key(str(p.get("name", "")), str(p.get("description", ""))))
        for p in _parameters(document)
        if p.get("required", False)
    )
    app = str(tool).split(".", 1)[0] if "." in str(tool) else ""
    return f"{app}:{'+'.join(in_sems)}->{out_sem}"


def _parameters(doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    schema = doc.get("arguments_schema", doc.get("parameters", []))
    if isinstance(schema, list):
        return [dict(value) for value in schema if isinstance(value, Mapping) and "name" in value]
    if not isinstance(schema, Mapping):
        return []
    properties = schema.get("properties", schema)
    required = set(schema.get("required", []))
    return [{**dict(value), "name": name, "required": value.get("required", name in required)}
            for name, value in properties.items() if isinstance(value, Mapping)]


def _response_fields(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Mapping[str, Any]]]:
    """Read public JSON Schema or AppWorld's example-shaped success schema."""
    if isinstance(value, list):
        for child in value:
            yield from _response_fields(child, path + ("*",))
        return
    if not isinstance(value, Mapping):
        if path:
            yield path, {}
        return
    if isinstance(value.get("properties"), Mapping):
        for name, child in value["properties"].items():
            yield from _response_fields(child, path + (str(name),))
        return
    if value.get("type") == "array" and "items" in value:
        yield from _response_fields(value["items"], path + ("*",))
        return
    for key in ("anyOf", "oneOf", "allOf"):
        if isinstance(value.get(key), list):
            for child in value[key]:
                yield from _response_fields(child, path)
            return
    schema_keys = {"type", "enum", "description", "format", "$ref", "title", "default", "examples",
                   "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
                   "minLength", "maxLength", "pattern", "const", "additionalProperties", "nullable"}
    kind = value.get("type")
    typed_schema = (isinstance(kind, str) and kind in _JSON_TYPES) or (
        isinstance(kind, list) and bool(kind) and all(isinstance(item, str) and item in _JSON_TYPES for item in kind))
    if set(value) <= schema_keys and (typed_schema or "$ref" in value or "description" in value):
        if path:
            yield path, value
        return
    for name, child in value.items():
        yield from _response_fields(child, path + (str(name),))


def _success_schema(doc: Mapping[str, Any]) -> Any:
    schemas = doc.get("response_schemas")
    if isinstance(schemas, Mapping) and "success" in schemas:
        return schemas["success"]
    return doc.get("output_schema", doc.get("return_schema"))


def produced_semantics(tool: str, doc: Mapping[str, Any] | None = None) -> set[str]:
    """Return public root and leaf types without treating schema examples as data."""
    document = doc or {}
    result = {infer_output_semantic(tool, document)}
    for path, spec in _response_fields(_success_schema(document)):
        names = [part for part in path if part != "*"]
        if names:
            result.add(semantic_key(names[-1], str(spec.get("description", ""))))
    return {value for value in result if value and value not in _GENERIC_FIELDS and value not in _JSON_TYPES}


def compute_catalog_idf(
    documents: Mapping[str, Mapping[str, Any]] | None = None,
    instruction: str = "",
    bindings: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, float]:
    """Compute token IDF weights over the public instruction, catalog, and bindings."""
    corpus: list[set[str]] = []
    if instruction:
        inst_toks = set(tokenize_words(instruction))
        if inst_toks:
            corpus.append(inst_toks)
    for name, doc in (documents or {}).items():
        if not isinstance(doc, Mapping):
            continue
        parts = [str(name), str(doc.get("description", "")), str(doc.get("surface_output", ""))]
        for p in _parameters(doc):
            parts.append(str(p.get("name", "")))
            parts.append(str(p.get("description", "")))
        toks = set(tokenize_words(" ".join(parts)))
        if toks:
            corpus.append(toks)
    for bname, binding in (bindings or {}).items():
        if not isinstance(binding, Mapping):
            continue
        parts = [str(bname), str(binding.get("semantic_type", "")), str(binding.get("tool", ""))]
        span = binding.get("task_span")
        if isinstance(span, Mapping):
            parts.append(str(span.get("context_text", "")))
            parts.append(str(span.get("label", "")))
        toks = set(tokenize_words(" ".join(parts)))
        if toks:
            corpus.append(toks)

    n_docs = max(1, len(corpus))
    df: dict[str, int] = defaultdict(int)
    for doc_set in corpus:
        for tok in doc_set:
            df[tok] += 1

    idf: dict[str, float] = {}
    for tok, count in df.items():
        w = math.log(1.0 + (n_docs + 1.0) / (1.0 + count))
        if tok in _GENERIC_FIELDS:
            w *= 0.35
        idf[tok] = w
    return idf


def candidate_lexical_score(
    parameter: Mapping[str, Any],
    binding: Mapping[str, Any],
    path: str,
    value: Any,
    *,
    tool: str = "",
    tool_doc: Mapping[str, Any] | None = None,
    index: "PublicSemanticIndex | None" = None,
    idf: Mapping[str, float] | None = None,
    expected_semantic: str = "",
    actual_semantic: str = "",
) -> float:
    """Compute soft lexical/IDF similarity s(p, e) between parameter p and candidate e.

    Used strictly for candidate ordering; a score of 0.0 never excludes a
    schema-compatible grounded candidate from the support set.
    """
    default_idf = math.log(1.0 + 4.0)

    def weight_of(tok: str) -> float:
        if idf is not None and tok in idf:
            return float(idf[tok])
        return default_idf * (0.35 if tok in _GENERIC_FIELDS else 1.0)

    param_name = str(parameter.get("name", ""))
    param_desc = _first_sentence(parameter.get("description", ""))
    expected_sem = expected_semantic or (
        index.parameter_semantic(parameter, tool)
        if index is not None
        else semantic_key(param_name, str(parameter.get("description", "")))
    )
    actual_sem = actual_semantic or (
        index.binding_semantic(binding, path, value)
        if index is not None
        else str(binding.get("semantic_type", ""))
    )

    score = 0.0
    # 1. Public semantic label alignment bonus (soft preference, never hard filter)
    if expected_sem and actual_sem:
        if expected_sem == actual_sem or (index is not None and index.equivalent(actual_sem, expected_sem)):
            score += 6.0
        elif {expected_sem, actual_sem} <= {"file_path", "directory_path", "destination_path"}:
            score += 4.0
        else:
            exp_toks = set(tokenize_words(expected_sem, keep_generic=False))
            act_toks = set(tokenize_words(actual_sem, keep_generic=False))
            if exp_toks and act_toks:
                overlap = exp_toks & act_toks
                if overlap:
                    score += sum(1.5 * weight_of(t) for t in overlap)
                    # Penalize partial ID mismatch (e.g. order_item_id vs order_id) in soft ranking
                    if expected_sem.endswith("_id") and actual_sem.endswith("_id") and exp_toks != act_toks:
                        score -= 1.0

    # 2. Token IDF overlap between parameter p and binding provenance d(e)
    p_primary = set(tokenize_words(f"{param_name} {expected_sem}", keep_generic=False))
    p_context = set(tokenize_words(f"{param_desc} {tool}", keep_generic=False))

    leaf_field = path.rsplit("/", 1)[-1] if path else ""
    producer_tool = str(binding.get("tool") or "")
    producer_doc = (index.documents.get(producer_tool) if index is not None and producer_tool else None) or {}
    producer_desc = _first_sentence(producer_doc.get("description", ""))
    span = binding.get("task_span") if isinstance(binding.get("task_span"), Mapping) else {}
    span_ctx = str(span.get("context_text", "")) + " " + str(span.get("label", ""))

    e_primary = set(tokenize_words(f"{leaf_field} {actual_sem} {_output_name(producer_tool)}", keep_generic=False))
    e_context = set(tokenize_words(f"{producer_desc} {span_ctx}", keep_generic=False))

    for tok in p_primary & e_primary:
        score += 2.0 * weight_of(tok)
    for tok in (p_primary & e_context) - e_primary:
        score += 1.0 * weight_of(tok)
    for tok in p_context & (e_primary | e_context):
        score += 0.6 * weight_of(tok)

    return round(float(score), 6)


class PublicSemanticIndex:
    """Per-public-catalog entity definitions and soft provenance index."""
    def __init__(self, catalog: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> None:
        if isinstance(catalog, Mapping) and "benchmark" in catalog:
            catalog = catalog.get("apis", catalog.get("tools", {}))
        if isinstance(catalog, Mapping):
            self.documents = {str(name): dict(doc) for name, doc in catalog.items() if isinstance(doc, Mapping)}
        else:
            self.documents = {str(doc.get("name") or f"{doc.get('app_name')}.{doc.get('api_name')}"): dict(doc)
                              for doc in catalog if isinstance(doc, Mapping)}
        self._claims: dict[str, set[str]] = defaultdict(set)
        self._parameters: dict[tuple[str, str], str] = {}
        self._fields: dict[tuple[str, tuple[str, ...]], str] = {}
        self._outputs: dict[str, str] = {}
        for tool, doc in self.documents.items():
            for parameter in _parameters(doc):
                name, description = str(parameter["name"]), str(parameter.get("description", ""))
                semantic = semantic_key(name, description)
                self._parameters[(tool, name)] = semantic
                if _identity_definition(description):
                    self._claim(name, semantic)
            for path, spec in _response_fields(_success_schema(doc)):
                names = [part for part in path if part != "*"]
                if not names:
                    continue
                description = str(spec.get("description", ""))
                semantic = semantic_key(names[-1], description)
                self._fields[(tool, path)] = semantic
                if _identity_definition(description):
                    self._claim(names[-1], semantic)
        for tool, doc in self.documents.items():
            semantic = infer_output_semantic(tool, doc)
            self._outputs[tool] = semantic
            described = _described_output(str(doc.get("description", "")))
            if described:
                self._claim(_output_name(tool), described)
        self.idf = compute_catalog_idf(self.documents)

    def _claim(self, alias: str, semantic: str) -> None:
        if semantic:
            if _label(alias) not in _GENERIC_FIELDS:
                self._claims[_label(alias)].add(semantic)
            self._claims[_label(semantic)].add(semantic)

    def resolve(self, label: str) -> str:
        raw = _label(label)
        claims = self._claims.get(raw, set())
        if len(claims) == 1:
            return next(iter(claims))
        normalized = semantic_key(label)
        claims = self._claims.get(_label(normalized), set())
        return next(iter(claims)) if len(claims) == 1 else normalized

    def parameter_semantic(self, parameter: Mapping[str, Any], tool: str = "") -> str:
        name, description = str(parameter["name"]), str(parameter.get("description", ""))
        if (tool, name) in self._parameters:
            return self._parameters[(tool, name)]
        interpreted = semantic_key(name, description)
        return interpreted if _identity_definition(description) else self.resolve(interpreted)

    def output_semantic(self, tool: str, doc: Mapping[str, Any] | None = None) -> str:
        interpreted = infer_output_semantic(tool, doc) if doc is not None else self._outputs.get(tool)
        return self.resolve(interpreted if interpreted is not None else infer_output_semantic(tool))

    def produced_semantics(self, tool: str, doc: Mapping[str, Any] | None = None) -> set[str]:
        document = doc if doc is not None else self.documents.get(tool, {})
        return {self.resolve(value) for value in produced_semantics(tool, document)}

    def binding_semantic(self, binding: Mapping[str, Any], path: str = "", value: Any = None) -> str:
        tool = str(binding.get("tool") or "")
        if path:
            parts = tuple(part.replace("~1", "/").replace("~0", "~") for part in path.split("/")[1:])
            shape = tuple("*" if part.isdigit() else part for part in parts)
            declared = self._fields.get((tool, shape))
            if declared and declared not in _GENERIC_FIELDS:
                return self.resolve(declared)
            fields = [part for part in parts if not part.isdigit()]
            if fields:
                field = self.resolve(fields[-1])
                if field not in _GENERIC_FIELDS:
                    return field
                if field in {"id", "identifier"} and tool in self.documents:
                    output = self.output_semantic(tool)
                    if output.endswith("_id"):
                        return output
        if isinstance(value, str) and value.startswith(("~/", "/home/")):
            return "file_path" if posixpath.splitext(posixpath.basename(value))[1] else "directory_path"
        if tool in self.documents:
            return self.output_semantic(tool)
        return self.resolve(str(binding.get("semantic_type", "")))

    def equivalent(self, actual: str, expected: str) -> bool:
        first, second = self.resolve(actual), self.resolve(expected)
        return bool(first and second and first == second)
