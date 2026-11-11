"""Autoregressive LoRA policy for the shared native-r3 public action contract.

The gate and both conditional action policies share one causal language model.
Every recorded response keeps the exact sampled token IDs. Teacher forcing
scores only those response tokens, as required for RL and step-level DPO.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import inspect
import json
import math
from pathlib import Path
import re
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .native_core import BudgetExhausted, BudgetMeter, NativeGeneration, monotonic_seconds


PROMPT_VERSION = "native-causal-r3.3-public-constructible"


class NoFeasibleAction(RuntimeError):
    """The current public state has no supported action; no substitute is sampled."""


def _canonical_program(program: Any, schema: Mapping[str, Any]) -> str:
    """Serialize a teacher action in the same property order as its JSON grammar.

    This only orders an already validated target. It never supplies a missing
    field, changes a value or extends an incomplete action. Discriminator keys
    are checked before child expressions, and each (value, resolved schema)
    pair is visited once. Nested expression unions therefore do not expand into
    the validation-error tree produced by a generic anyOf validator.
    """
    resolved: dict[int, Mapping[str, Any]] = {}
    matched: dict[tuple[int, int], bool] = {}

    def resolve(node: Mapping[str, Any]) -> Mapping[str, Any]:
        original = id(node)
        if original in resolved:
            return resolved[original]
        while "$ref" in node:
            pointer = str(node["$ref"])
            if not pointer.startswith("#/"):
                raise ValueError("The action grammar requires local schema references.")
            node = schema
            for key in pointer[2:].split("/"):
                node = node[key.replace("~1", "/").replace("~0", "~")]
        resolved[original] = node
        return node

    def same_json(first: Any, second: Any) -> bool:
        # JSON Schema distinguishes booleans from numbers even though Python
        # considers True == 1. Numeric 1 and 1.0 retain JSON Schema equality.
        if isinstance(first, bool) or isinstance(second, bool):
            return type(first) is type(second) and first == second
        if isinstance(first, Mapping) or isinstance(second, Mapping):
            return (isinstance(first, Mapping) and isinstance(second, Mapping)
                    and set(first) == set(second)
                    and all(same_json(first[key], second[key]) for key in first))
        if isinstance(first, list) or isinstance(second, list):
            return (isinstance(first, list) and isinstance(second, list)
                    and len(first) == len(second)
                    and all(same_json(a, b) for a, b in zip(first, second)))
        return first == second

    def shallow_match(value: Any, node: Mapping[str, Any]) -> bool:
        """Reject a different tagged production before visiting its children."""
        node = resolve(node)
        if "const" in node and not same_json(value, node["const"]):
            return False
        if "enum" in node and not any(same_json(value, candidate) for candidate in node["enum"]):
            return False
        kind = node.get("type")
        kinds = [kind] if isinstance(kind, str) else kind or []
        actual = ("null" if value is None else "boolean" if isinstance(value, bool)
                  else "object" if isinstance(value, dict) else "array" if isinstance(value, list)
                  else "string" if isinstance(value, str) else "integer" if isinstance(value, int)
                  else "number" if isinstance(value, float) else "unknown")
        if kinds and actual not in kinds and not (actual == "integer" and "number" in kinds):
            return False
        if isinstance(value, dict):
            properties = node.get("properties", {})
            if any(key not in value for key in node.get("required", [])):
                return False
            if node.get("additionalProperties") is False and any(key not in properties for key in value):
                return False
            # Action and expression variants are distinguished by kind/tool,
            # op, ref, literal or item. Checking scalar constraints here also
            # handles finite reference and retrieval-query productions.
            for key, item in value.items():
                if key not in properties:
                    continue
                child = resolve(properties[key])
                if "const" in child and not same_json(item, child["const"]):
                    return False
                if "enum" in child and not any(same_json(item, entry) for entry in child["enum"]):
                    return False
        if isinstance(value, list):
            if len(value) < node.get("minItems", 0) or len(value) > node.get("maxItems", math.inf):
                return False
        if isinstance(value, str):
            if len(value) < node.get("minLength", 0) or len(value) > node.get("maxLength", math.inf):
                return False
            if "pattern" in node and re.search(node["pattern"], value) is None:
                return False
        return True

    def matches(value: Any, node: Mapping[str, Any]) -> bool:
        node = resolve(node)
        key = (id(value), id(node))
        if key in matched:
            return matched[key]
        result = match_children(value, node) if shallow_match(value, node) else False
        matched[key] = result
        return result

    def match_children(value: Any, node: Mapping[str, Any]) -> bool:
        variants = node.get("anyOf", node.get("oneOf"))
        if variants is not None:
            return any(matches(value, candidate) for candidate in variants
                       if shallow_match(value, candidate))
        if isinstance(value, dict):
            properties = node.get("properties", {})
            additional = node.get("additionalProperties", {})
            return all(matches(item, properties[key]) if key in properties else
                       matches(item, additional) if isinstance(additional, Mapping) else bool(additional)
                       for key, item in value.items())
        if isinstance(value, list):
            prefix = node.get("prefixItems", [])
            if any(not matches(item, prefix[index]) for index, item in enumerate(value[:len(prefix)])):
                return False
            remainder = value[len(prefix):]
            items = node.get("items", {})
            if items is False and remainder:
                return False
            if isinstance(items, Mapping):
                return all(matches(item, items) for item in remainder)
        return True

    def ordered(value: Any, node: Mapping[str, Any]) -> Any:
        node = resolve(node)
        variants = node.get("anyOf", node.get("oneOf"))
        if variants is not None:
            selected = next((candidate for candidate in variants
                             if shallow_match(value, candidate) and matches(value, candidate)), None)
            if selected is None:
                raise ValueError("Teacher action has no matching public grammar variant.")
            return ordered(value, selected)
        if isinstance(value, dict):
            properties = node.get("properties", {})
            result = {key: ordered(value[key], definition) for key, definition in properties.items() if key in value}
            additional = node.get("additionalProperties", {})
            for key, item in value.items():
                if key not in result:
                    result[key] = ordered(item, additional) if isinstance(additional, Mapping) else item
            return result
        if isinstance(value, list):
            prefix, items = node.get("prefixItems", []), node.get("items", {})
            return [ordered(item, prefix[index]) if index < len(prefix) else
                    ordered(item, items) if isinstance(items, Mapping) else item
                    for index, item in enumerate(value)]
        return value

    return json.dumps(ordered(program, schema), ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _model_loader_for_config(config: Any) -> Any:
    """Select the official AutoModel mapping without changing checkpoint config."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    architectures = tuple(getattr(config, "architectures", None) or ())
    if (type(config) in AutoModelForImageTextToText._model_mapping
            and any("ConditionalGeneration" in name for name in architectures)):
        return AutoModelForImageTextToText
    if type(config) in AutoModelForCausalLM._model_mapping:
        return AutoModelForCausalLM
    if type(config) in AutoModelForImageTextToText._model_mapping:
        return AutoModelForImageTextToText
    raise ValueError(
        f"Unsupported native generation config {type(config).__name__} "
        f"(model_type={getattr(config, 'model_type', None)!r}); "
        "install the official Transformers version supporting this checkpoint."
    )


def parse_native_action(
    text: str, mode: str | None = None, *,
    action_schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse one complete r3 JSON action and compile its selected native call."""
    if action_schema is not None and action_schema.get("classic_text"):
        from .native_text_actions import native_text_mode
        selected = native_text_mode(text, action_schema)
        if selected is None or (mode is not None and selected != mode):
            return {"kind": "invalid", "mode": mode, "raw_text": text,
                    "validation_stage": "native_text_contract",
                    "error": "The generated command is outside its public action class."}
        return {"kind": "native_text", "text": text, "mode": mode, "raw_text": text}
    from .native_program import compile_program
    from jsonschema.exceptions import ValidationError
    if mode not in (None, "I", "E"):
        raise ValueError("The action mode must be I, E, or None.")
    if action_schema is None or "program_context" not in action_schema:
        raise ValueError("Parsing a native-r3 action requires its public state and grammar.")
    try:
        program = json.loads(text)
        if not isinstance(program, dict):
            raise ValueError("An action is exactly one JSON object.")
        action = compile_program(program, action_schema["program_context"], mode)
    except (ValueError, TypeError, KeyError, ValidationError) as error:
        return {"kind": "invalid", "mode": mode, "raw_text": text,
                "validation_stage": "public_action_contract", "error": str(error)}
    action["raw_text"] = text
    return action


class NativeCausalPolicy(nn.Module):
    """One causal LM, one trainable LoRA and an optional frozen warmstart LoRA.

    The reference shares base-model storage. Its own adapter tensors keep the
    warmstart distribution fixed while later PRISM updates change the actor.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str | torch.device = "cuda:0",
        *,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        target_modules: str | Sequence[str] = "all-linear",
        meter: BudgetMeter | None = None,
        max_new_tokens: int = 512,
        max_context_tokens: int | None = None,
        temperature: float = 1.0,
        mode_labels: Mapping[str, str] | None = None,
        dtype: torch.dtype | None = None,
        device_map: Mapping[str, Any] | None = None,
        gradient_checkpointing: bool = True,
        local_files_only: bool = True,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        activation_offload_cpu: bool = False,
        activation_offload_min_bytes: int = 1048576,
    ) -> None:
        super().__init__()
        if lora_rank <= 0 or lora_alpha <= 0 or max_new_tokens <= 0:
            raise ValueError("LoRA rank/alpha and generation length must be positive.")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Sampling temperature must be finite and positive.")
        from transformers import AutoConfig, AutoTokenizer
        from peft import LoraConfig, TaskType, get_peft_model

        self.model_path = str(model_path)
        self.meter = meter
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self._gradient_checkpointing = bool(gradient_checkpointing)
        self._reference_depth = 0
        self._reference_adapter: str | None = None
        self._active_reference: str | None = None
        self._reference_label = "raw_base"
        self.activation_offload_cpu = bool(activation_offload_cpu)
        self.activation_offload_min_bytes = int(activation_offload_min_bytes)
        if self.activation_offload_min_bytes < 1:
            raise ValueError("Activation offload threshold must be positive.")
        self.offloaded_activation_bytes = 0
        checkpoint_config = AutoConfig.from_pretrained(
            self.model_path, local_files_only=local_files_only, trust_remote_code=False,
        )
        model_loader = _model_loader_for_config(checkpoint_config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=local_files_only, trust_remote_code=False,
        )
        if not self.tokenizer.chat_template:
            raise ValueError("The selected checkpoint needs its native chat template.")
        requested_device = torch.device(device)
        if dtype is None:
            dtype = (torch.bfloat16 if requested_device.type == "cuda"
                     and torch.cuda.is_bf16_supported() else torch.float32)
        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype, "local_files_only": local_files_only,
            "trust_remote_code": False, "config": checkpoint_config,
        }
        if device_map is not None:
            load_kwargs["device_map"] = dict(device_map)
        elif requested_device.type == "cuda":
            load_kwargs["device_map"] = {"": str(requested_device)}
        base = model_loader.from_pretrained(self.model_path, **load_kwargs)
        if device_map is None and requested_device.type != "cuda":
            base.to(requested_device)
        text_config = getattr(base.config, "text_config", None) or base.config
        self.model_metadata = {
            "checkpoint_path": self.model_path,
            "config_model_type": getattr(base.config, "model_type", None),
            "config_architectures": list(getattr(base.config, "architectures", None) or ()),
            "auto_model_class": model_loader.__name__,
            "loaded_model_class": type(base).__name__,
            "text_model_type": getattr(text_config, "model_type", None),
            "hidden_size": getattr(text_config, "hidden_size", None),
            "num_hidden_layers": getattr(text_config, "num_hidden_layers", None),
            "max_position_embeddings": getattr(text_config, "max_position_embeddings", None),
            "language_model_only": getattr(base.config, "language_model_only", None),
            "input_modalities_used": ["text"],
        }
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer needs an existing EOS or padding token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = get_peft_model(base, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=lora_rank, lora_alpha=lora_alpha,
            lora_dropout=0.0, target_modules=target_modules, bias="none",
            init_lora_weights=True,
        ))
        native_eos = base.generation_config.eos_token_id
        if native_eos is None:
            native_eos = self.tokenizer.eos_token_id
        self.eos_token_ids = ([int(native_eos)] if isinstance(native_eos, int)
                             else [int(token) for token in native_eos] if native_eos is not None else [])
        self.model_metadata["generation_eos_token_ids"] = list(self.eos_token_ids)
        output_embedding = self.model.get_output_embeddings()
        if output_embedding is None:
            raise ValueError("The native generation model must expose its causal LM output head.")
        hidden_size = int(output_embedding.weight.shape[1])
        self.value_head = nn.Linear(hidden_size, 1, device=output_embedding.weight.device, dtype=torch.float32)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        from .native_dcc import ControlledResponseEnsemble, ProgramExecutionState, ProgramLibrary
        self.dcc_ensemble = ControlledResponseEnsemble().to(
            device=output_embedding.weight.device, dtype=torch.float32
        )
        self.dcc_libraries: dict[str, ProgramLibrary] = {}
        self.dcc_active_belief: torch.Tensor | None = None
        self.dcc_active_program_state: ProgramExecutionState | None = None
        self.dcc_active_member_index: int | None = None
        self.dcc_mode_enabled: bool = False
        self.dcc_eval_variant: str = "full"
        self.dcc_controller_mode: str = "expected_utility"
        self.dcc_episode_events: list[Any] = []
        self.dcc_task_instruction: str = ""
        self.dcc_remaining_budget_ratio: float = 1.0
        self._dcc_action_embed_cache: dict[str, tuple[float, ...]] = {}
        self._lora_trainable_names = {
            name for name, param in self.model.named_parameters() if param.requires_grad
        }
        # A deterministic actor distribution is also used for training scores.
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
            if hasattr(module, "attention_dropout"):
                module.attention_dropout = 0.0
        if self._gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            self.model.enable_input_require_grads()
        self.model.config.use_cache = False
        text_config.use_cache = False
        self.model.eval()
        model_window = int(getattr(text_config, "max_position_embeddings", 32768))
        self.max_context_tokens = int(max_context_tokens or model_window)
        if not 0 < self.max_context_tokens <= model_window:
            raise ValueError("Context limit must fit the checkpoint's declared context window.")
        self.mode_labels = dict(mode_labels or {"I": "I", "E": "E"})
        if set(self.mode_labels) != {"I", "E"}:
            raise ValueError("Exactly the I and E mode labels must be configured.")
        self.mode_token_ids: dict[str, int] = {}
        for mode, label in self.mode_labels.items():
            token_ids = self.tokenizer.encode(label, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(f"Mode label {label!r} must be exactly one existing token.")
            self.mode_token_ids[mode] = int(token_ids[0])
        if self.mode_token_ids["I"] == self.mode_token_ids["E"]:
            raise ValueError("Mode labels must have distinct token IDs.")
        parameters = inspect.signature(base.forward).parameters
        self._keep_logits_arg = next(
            (name for name in ("logits_to_keep", "num_logits_to_keep") if name in parameters),
            None,
        )
        self._cuda_devices = tuple(sorted(
            {parameter.device for parameter in self.parameters() if parameter.device.type == "cuda"},
            key=str,
        ))
        self._parameter_storages = {
            tensor.untyped_storage().data_ptr()
            for tensor in (*self.model.parameters(), *self.model.buffers())
            if tensor.device.type == "cuda"
        }

    @property
    def device(self) -> torch.device:
        return self.model.get_input_embeddings().weight.device

    def _synchronize(self) -> None:
        devices = getattr(self, "_cuda_devices", (self.device,) if self.device.type == "cuda" else ())
        for device in devices:
            torch.cuda.synchronize(device)

    @contextmanager
    def _activation_saving(self):
        """Offload saved activations bit-for-bit; retain parameter storage on GPU."""
        if (not getattr(self, "activation_offload_cpu", False)
                or not torch.is_grad_enabled() or self.device.type != "cuda"):
            yield
            return

        def pack(tensor):
            size = tensor.numel() * tensor.element_size()
            if (tensor.device.type == "cuda" and tensor.layout == torch.strided
                    and size >= self.activation_offload_min_bytes
                    and tensor.untyped_storage().data_ptr() not in self._parameter_storages):
                cpu = torch.empty_like(tensor, device="cpu", pin_memory=True)
                cpu.copy_(tensor.detach())
                self.offloaded_activation_bytes += size
                return ("native_activation", cpu, tensor.device)
            return tensor.detach()

        def unpack(value):
            if isinstance(value, tuple):
                return value[1].to(device=value[2])
            return value

        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            yield

    def _tokenize_messages(self, messages: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
        if not messages:
            raise ValueError("A model request needs public messages.")
        coalesced: list[dict[str, Any]] = []
        for message in messages:
            item = dict(message)
            if coalesced and coalesced[-1].get("role") == item.get("role"):
                prev = str(coalesced[-1].get("content", ""))
                curr = str(item.get("content", ""))
                coalesced[-1]["content"] = f"{prev}\n\n{curr}" if prev and curr else (prev or curr)
            else:
                coalesced.append(item)
        encoded = self.tokenizer.apply_chat_template(
            coalesced, tokenize=True,
            add_generation_prompt=True, return_tensors="pt", return_dict=False, **self.chat_template_kwargs,
        )
        if isinstance(encoded, Mapping):
            encoded = encoded["input_ids"]
        ids = tuple(int(item) for item in encoded.reshape(-1).tolist())
        if not ids:
            raise ValueError("Chat template produced an empty prompt.")
        return ids

    def _check_context(self, prompt_length: int, response_length: int) -> None:
        if prompt_length <= 0:
            raise ValueError("A model prompt must contain at least one token.")
        if prompt_length + response_length > self.max_context_tokens:
            raise BudgetExhausted("context_window")

    @staticmethod
    def _with_instruction(
        messages: Sequence[Mapping[str, Any]], instruction: str,
    ) -> list[dict[str, Any]]:
        result = [dict(message) for message in messages]
        if result and result[0].get("role") == "system":
            result[0]["content"] = str(result[0].get("content", "")) + "\n\n" + instruction
        else:
            result.insert(0, {"role": "system", "content": instruction})
        return result

    @staticmethod
    def _schema_text(action_schema: Mapping[str, Any] | str) -> str:
        return (action_schema if isinstance(action_schema, str)
                else json.dumps(dict(action_schema), ensure_ascii=False, sort_keys=True,
                                allow_nan=False, separators=(",", ":")))

    @staticmethod
    def _catalog_documents(catalog: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        """Retain complete public documents, including types and reliability caveats."""
        if catalog.get("benchmark") == "appworld":
            return deepcopy(dict(catalog.get("apis", {})))
        id_key = "tool_id" if catalog.get("use_tool_ids") else "name"
        documents = {str(tool[id_key]): deepcopy(dict(tool)) for tool in catalog.get("tools", [])}
        for source in catalog.get("sources", []):
            for document in source.get("api_docs", []):
                identifier = str(document.get(id_key, ""))
                if identifier not in documents:
                    continue
                current = documents[identifier]
                if document.get("parameters") != current.get("arguments_schema", current.get("parameters")):
                    continue
                # parameters and arguments_schema denote this same public JSON Schema.
                merged = {**current, **deepcopy(dict(document))}
                merged.pop("arguments_schema", None)
                documents[identifier] = merged
        return documents

    @staticmethod
    def _receipt_for_prompt(receipt: Mapping[str, Any]) -> dict[str, Any]:
        """Retain actual request/error metadata; successful values live in bindings."""
        result = {key: deepcopy(value) for key, value in receipt.items()
                  if key not in {"public_doc", "output"}}
        if receipt.get("succeeded"):
            if receipt.get("kind") != "finish" and receipt.get("bind"):
                key = "discovery_binding_ref" if receipt.get("kind") == "discover" else "result_binding_ref"
                result[key] = str(receipt["bind"])
        elif "output" in receipt:
            # A failed result has no committed binding. Preserve its actual
            # error payload rather than pointing at an unavailable value.
            result["error_output"] = deepcopy(receipt["output"])
        return result

    @classmethod
    def _state_for_prompt(cls, catalog: Mapping[str, Any], documents: Mapping[str, Any]) -> dict[str, Any]:
        state = deepcopy(catalog["execution_state"])
        state.pop("initial_public_observation", None)
        # The task remains verbatim in the first public message. initial_state
        # retains the native calendar and other actual starting state fields.
        state.pop("task_instruction", None)
        state.pop("instruction", None)
        bindings = state.get("bindings", [])
        bindings = list(bindings.values()) if isinstance(bindings, Mapping) else bindings
        business_bindings, discoveries, summaries = [], [], {}
        id_key = "tool_id" if catalog.get("use_tool_ids") else "name"

        def references(value: Any) -> list[str]:
            found: list[str] = []
            if isinstance(value, Mapping):
                if "app_name" in value and "api_name" in value:
                    name = f"{value['app_name']}.{value['api_name']}"
                    if name in documents:
                        found.append(name)
                elif isinstance(value.get(id_key), str) and value[id_key] in documents:
                    found.append(value[id_key])
                for child in value.values():
                    found.extend(references(child))
            elif isinstance(value, list):
                for child in value:
                    found.extend(references(child))
            return list(dict.fromkeys(found))

        for binding in bindings:
            if binding.get("source") != "public_document":
                # Values and their identity/type/scope remain exact. Request
                # arguments and transport bookkeeping are already in receipts;
                # repeating long access tokens in every old binding serves no
                # parameter-resolution purpose in the actor prompt.
                business_bindings.append({key: deepcopy(binding[key]) for key in
                    ("id", "value", "json_type", "semantic_type", "source", "tool", "scope", "member_bindings",
                     "value_status", "usable_for_binding")
                    if key in binding})
                continue
            compact = {key: deepcopy(binding[key]) for key in
                       ("id", "source", "tool", "arguments", "call_id", "version") if key in binding}
            compact["catalog_refs"] = references(binding.get("value"))
            # App/API name-and-description lists precede full API discovery.
            # Their short public descriptions stay available once so the
            # actor can choose which full document to request next.
            if str(binding.get("tool", "")).endswith(("show_app_descriptions", "show_api_descriptions")):
                value = binding.get("value")
                if isinstance(value, list):
                    rows = [{key: deepcopy(item[key]) for key in ("name", "app_name", "api_name", "description")
                             if key in item} for item in value if isinstance(item, Mapping)]
                    if rows:
                        summary_id = str(binding["id"])
                        summaries[summary_id] = rows
                        compact["discovery_summary_ref"] = summary_id
            discoveries.append(compact)
        state["bindings"] = business_bindings
        state["discovery_bindings"] = discoveries
        state["discovery_summaries"] = summaries
        state["receipt_count"] = len(state.get("receipts", []))
        state["receipts"] = [cls._receipt_for_prompt(item) for item in state.get("receipts", [])[-4:]]
        retrievals = state.get("retrievals", [])
        state["retrieval_count"] = len(retrievals)
        state["retrievals"] = [item for item in retrievals
                               if item.get("binding_version", 0) == state.get("binding_version", 0)]
        return state

    @staticmethod
    def _action_forms(catalog: Mapping[str, Any], mode: str | None) -> dict[str, Any]:
        """Generic syntax examples; all factual choices come from the public frontier."""
        from .native_execution_state import PURE_OPERATIONS
        from .native_program import next_binding_name
        reference = {"ref": "<committed_binding_id>", "path": "<actual_JSON_pointer_or_empty>"}
        next_binding = next_binding_name(catalog)
        forms: dict[str, Any] = {
            "notation": (
                "Angle-bracket strings name syntax roles. Replace them with supported frontier entries, "
                "exact documented parameter names and actual binding IDs/JSON paths. "
                "Use frontier.next_binding verbatim for every discover, call or map_call; the grammar fixes this fresh name. "
                "The path key may be omitted only where the supplied grammar permits it; explicit empty path selects the whole value. "
                "An item expression is scoped to map_call arguments or the second argument of filter/project."
            ),
            "value_forms": {
                "reference": reference,
                "literal": {"literal": "<task_span_or_documented_constant>"},
                "operation": {"op": "<operation_name>", "args": ["<ValueExpr>"]},
                "project": {"op": "project", "args": [
                    {"ref": "<array_binding_id>", "path": ""}, {"item": "/actual_field"}]},
                "filter": {"op": "filter", "args": [
                    {"ref": "<array_binding_id>", "path": ""},
                    {"op": "equal", "args": [{"item": "/actual_field"},
                                               {"ref": "<task_binding_id>", "path": ""}]}]},
            },
            "pure_operation_arity": {name: list(arity) for name, arity in PURE_OPERATIONS.items()},
        }
        if mode in {None, "I"}:
            if catalog["benchmark"] == "appworld":
                forms["discover"] = {
                    "kind": "discover", "tool": "api_docs.show_api_doc",
                    "arguments": {"app_name": {"literal": "<discovered_app_name>"},
                                  "api_name": {"literal": "<discovered_api_name>"}},
                    "bind": next_binding,
                }
            else:
                forms["discover"] = {
                    "kind": "discover", "arguments": {"meta_tool": "by_output_info",
                    "output_information": "<frontier_missing_input_concept>"}, "bind": next_binding,
                }
        if mode in {None, "E"}:
            forms["call"] = {"kind": "call", "tool": "<ready_public_tool>",
                             "arguments": {"<documented_parameter>": reference}, "bind": next_binding}
            forms["finish_question"] = {"kind": "finish", "answer": {
                "ref": "<executed_target_result_binding>", "path": "<actual_scalar_target_pointer_or_empty>"}}
            if catalog["benchmark"] == "appworld":
                forms["map_call"] = {"kind": "map_call", "tool": "<public_business_API>",
                    "items": {"ref": "<observed_array_binding>", "path": ""},
                    "arguments": {"<element_parameter>": {"item": "<actual_element_pointer_or_empty>"},
                                  "<shared_parameter>": reference}, "bind": next_binding}
                forms["finish_operation"] = {"kind": "finish"}
        return forms

    @classmethod
    def _model_history(
        cls, messages: Sequence[Mapping[str, Any]], catalog: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Deduplicate documents only after raw history has authorized the catalog.

        A catalog reference replaces an identical JSON document. Task text, API
        results, errors, action order and all non-document fields remain visible.
        The caller's authoritative history and provenance objects are unchanged.
        """
        documents = cls._catalog_documents(catalog)
        references = {
            json.dumps(doc, sort_keys=True, ensure_ascii=False, separators=(",", ":")): identifier
            for identifier, doc in documents.items()
        }
        for source in catalog.get("sources", []):
            for doc in source.get("api_docs", []):
                identifier = (f"{doc.get('app_name')}.{doc.get('api_name')}"
                              if catalog.get("benchmark") == "appworld"
                              else str(doc.get("tool_id" if catalog.get("use_tool_ids") else "name", "")))
                current = documents.get(identifier)
                if current is not None and all(current.get(key) == value for key, value in doc.items()):
                    references[json.dumps(doc, sort_keys=True, ensure_ascii=False,
                                          separators=(",", ":"))] = identifier
        decoder = json.JSONDecoder()

        def transform(value: Any) -> Any:
            if isinstance(value, dict):
                if value.get("feedback_type") == "native_transition" and isinstance(value.get("r3_receipts"), list):
                    receipts = value["r3_receipts"]
                    result = {key: transform(item) for key, item in value.items()
                              if key not in {"native_observation", "r3_receipts"}}
                    result["r3_receipts"] = [cls._receipt_for_prompt(item) for item in receipts]
                    if receipts and all(item.get("succeeded") for item in receipts):
                        result["native_observation"] = (
                            "Actual request results are recorded in the referenced public bindings, "
                            "with value_status describing any documented reliability caveats."
                        )
                    elif "native_observation" in value:
                        result["native_observation"] = compact_text(str(value["native_observation"]))
                    return result
                key = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                if key in references:
                    return {"public_catalog_ref": references[key]}
                return {key: compact_text(item) if key in {"observation", "native_observation"} and isinstance(item, str)
                        else transform(item) for key, item in value.items()}
            if isinstance(value, list):
                return [transform(item) for item in value]
            return value

        def compact_text(text: str) -> str:
            parts: list[str] = []
            cursor = 0
            for candidate in re.finditer(r"[\[{]", text):
                start = candidate.start()
                if start < cursor:
                    continue
                try:
                    value, length = decoder.raw_decode(text[start:])
                except ValueError:
                    continue
                end = start + length
                replaced = transform(value)
                parts.append(text[cursor:start])
                parts.append(json.dumps(replaced, ensure_ascii=False, separators=(",", ":"))
                             if replaced != value else text[start:end])
                cursor = end
            parts.append(text[cursor:])
            return "".join(parts)

        result = deepcopy(list(messages))
        for message in result:
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                message["content"] = compact_text(message["content"])
        return result

    @classmethod
    def _compact_catalog_for_prompt(
        cls, catalog: Mapping[str, Any], messages: Sequence[Mapping[str, Any]], *,
        mode: str | None = None, for_gate: bool = False,
    ) -> dict[str, Any]:
        """Build public context units for a bounded, dependency-ordered view.

        The authoritative catalog and action support stay complete. Only the
        current public dependency window supplies full documents to this view;
        other executable tools receive compact parameter/reference summaries.
        _protocol_messages packs whole units against the real tokenizer budget.
        """
        from .native_execution_state import binding_table, canonical, walk_values
        from .native_program import _leaf_semantic, _semantic_index
        from .native_semantics import public_document_quality
        frontier = catalog["frontier"]
        semantic_index = _semantic_index(catalog)
        documents = cls._catalog_documents(catalog)
        focus = list(frontier["focus_tools"])
        if catalog["benchmark"] == "appworld":
            focus.extend(name for name in documents if name.startswith("api_docs."))
        focus = list(dict.fromkeys(name for name in focus if name in documents))
        relevant = set(focus)
        state = cls._state_for_prompt(catalog, documents)
        discoveries = state["discovery_bindings"]
        state["discovery_binding_count"] = len(discoveries)
        for discovery in discoveries:
            refs = discovery.pop("catalog_refs", [])
            discovery["catalog_count"] = len(refs)
            discovery["visible_catalog_refs"] = [name for name in refs if name in relevant]
        # Full discovery history is retained in the immutable ledger. A recent
        # receipt and the current feasible queries expose its effect on action
        # support without repeating the growing catalog at every step.
        state["discovery_bindings"] = discoveries[-4:]

        def call_summary(call: Mapping[str, Any]) -> dict[str, Any]:
            arguments = {}
            for name, argument in call["arguments"].items():
                paths: dict[str, list[str]] = {}
                expressions: list[dict[str, Any]] = []
                for reference in argument["references"]:
                    if "ref" in reference:
                        paths.setdefault(reference["ref"], []).append(reference.get("path", ""))
                    else:
                        expressions.append(deepcopy(reference))
                entry: dict[str, Any] = {"semantic_type": argument["semantic_type"],
                    "required": argument["required"], "reference_paths": paths,
                    "literals": deepcopy(argument["literals"])}
                if expressions:
                    entry["synthesized_expressions"] = expressions[:4]
                arguments[name] = entry
            return {"tool": call["tool"], "arguments": arguments}

        ready = {item["tool"]: call_summary(item) for item in frontier["ready_calls"]}
        ordered_names = list(frontier["relevant_tools"])
        observed_bindings = list(binding_table(catalog).values())
        recent_values = [binding for binding in observed_bindings if binding.get("source") == "native_return"
                         and binding.get("usable_for_binding", False)]

        def observed_semantics(binding: Mapping[str, Any]) -> set[str]:
            return {_leaf_semantic(binding, path, value, semantic_index)
                    for path, value in walk_values(binding["value"])
                    if value is not None and not isinstance(value, (Mapping, list))}

        # The rendered bindings omit transport versions. Read the authoritative
        # public ledger so this window uses the same latest evidence as the frontier.
        latest_version = max((int(binding.get("version", 0)) for binding in recent_values), default=-1)
        latest_semantics = set().union(*(observed_semantics(binding) for binding in recent_values
                                        if int(binding.get("version", 0)) == latest_version))
        known = set().union(*(observed_semantics(binding) for binding in observed_bindings
                              if binding.get("source") in {"task_span", "native_return"}
                              and binding.get("usable_for_binding", False)))
        dependencies = set(frontier["dependency_semantics"])

        def query_priority(query: Mapping[str, Any]) -> tuple[int, int, int]:
            if catalog["benchmark"] == "appworld":
                arguments = query.get("arguments", {})
                name = f"{arguments.get('app_name', '')}.{arguments.get('api_name', '')}"
                latest_app = str(recent_values[-1].get("tool", "")).split(".", 1)[0] if recent_values else ""
                return (int(name in relevant), int(arguments.get("app_name") == latest_app),
                        int(query.get("tool") == "api_docs.show_api_doc"))
            inputs = {semantic_index.resolve(str(value)) for value in query.get("input_information", [])}
            output = semantic_index.resolve(str(query.get("output_information", "")))
            return (int(bool(inputs & latest_semantics)),
                    int(bool(inputs) and inputs <= known), int(output in dependencies))

        queries = sorted(frontier["discovery_queries"], key=query_priority, reverse=True)
        # Sort is stable: equivalent public queries retain the frontier's
        # semantic order. No private teacher action influences this window.
        queries = list({canonical(query): deepcopy(query) for query in queries}.values())
        exhausted = set(frontier["exhausted_calls_at_current_state"])
        qualified = frontier["qualified_out_tools"]
        recent_tools = {str(receipt["tool"]) for receipt in state["receipts"] if receipt.get("tool")}
        qualified_names = sorted(qualified, key=lambda name: (
            name not in recent_tools,
            not bool(semantic_index.produced_semantics(name) & dependencies), name))[:6]
        visible_frontier = {key: deepcopy(frontier[key]) for key in
            ("version", "binding_version", "mutation_version", "feasible_modes", "finish_ready", "next_binding",
             "dependency_semantics", "target_answer_references", "discovery_telemetry") if key in frontier}
        visible_frontier.update({
            "exhausted_calls_at_current_state": [name for name in ordered_names if name in exhausted][:6],
            "qualified_out_tools": {name: qualified[name] for name in qualified_names},
            "catalog_compact_index": [
                {"tool": name, "quality": public_document_quality(documents[name]),
                 "ready": name in ready, "exhausted": name in exhausted}
                for name in ordered_names if name not in relevant and name in documents
            ][:32],
            "ready_calls": [ready[name] for name in focus if name in ready],
            "other_ready_calls": [ready[name] for name in ordered_names if name in ready and name not in relevant],
            "ready_maps": [deepcopy(item) for item in frontier["ready_maps"] if item["tool"] in relevant],
            "missing_inputs": [{key: deepcopy(item[key]) for key in ("tool", "parameter", "semantic_type")}
                               for item in frontier["missing_inputs"] if item["tool"] in relevant],
            "discovery_queries": queries[:24],
        })
        result: dict[str, Any] = {
            "benchmark": catalog["benchmark"],
            "execution_state": state,
            "frontier": visible_frontier,
            "documents": {name: documents[name] for name in focus},
            "catalog_counts": {"stored_documents": len(documents),
                "stored_ready_calls": len(frontier["ready_calls"]),
                "stored_ready_maps": len(frontier["ready_maps"]),
                "stored_missing_inputs": len(frontier["missing_inputs"]),
                "stored_discovery_queries": len(frontier["discovery_queries"]),
                "stored_exhausted_calls_at_current_state": len(exhausted),
                "stored_qualified_out_tools": len(qualified)},
            "document_reference_rule": (
                "public_catalog_ref refers to the same public document. Read the exact parameter names, "
                "semantic input types, return JSON paths, applicability and reliability conditions. "
                "Bindings preserve actual task spans and native observations with exact JSON values and scopes. "
                "value_status and qualified_out_tools report public applicability or reliability caveats as advisory context; "
                "prefer usable tools with concise direct specifications over provisional or unreliable alternatives. "
                "reference_paths maps each binding ID to its actual JSON pointers; "
                "encode a selected entry as {ref: binding_ID, path: pointer}. Documents in this view are complete. "
                "The complete catalog and feasible support remain stored; catalog_counts and catalog_compact_index report their state. "
                "Already executed deterministic read tuples and terminal rejections are blocked at the current mutation_version. "
                "A generated variable is available only after a receipt commits it. "
                "Use frontier.next_binding verbatim; names from earlier actions are immutable."
            ),
            "decision_rule": (
                "Choose I to discover a missing public API or the producer of an unresolved input. "
                "Choose E to execute a ready business call, including login and reads, or finish when the task is complete. "
                "Use discovery_telemetry to switch to E once ready_business_calls are available and recent discovery adds no new tools. "
                "A ready tool consuming the latest typed business value can advance the dependency immediately. "
                "Use binding references for identities and returned values. Keep password, access_token, order ID, "
                "return_request_id, return_audit_id, refund ID, authorization code, authorization-record ID, "
                "snapshot ID and payment ID distinct. "
                "Submit finish only after all required business reads or state-changing writes requested by the task have succeeded."
            ),
        }
        if not for_gate:
            result["action_forms"] = cls._action_forms(catalog, mode)
        if catalog["benchmark"] == "planbench_xl":
            result["use_tool_ids"] = bool(catalog.get("use_tool_ids"))
            if mode in (None, "I") or for_gate:
                result["retrieval_schema"] = deepcopy(catalog["retrieval_schema"])
        return result

    def _protocol_messages(
        self, messages: Sequence[Mapping[str, Any]], catalog: Mapping[str, Any], instruction: str,
        public: Mapping[str, Any], *, response_reserve: int | None = None,
    ) -> list[dict[str, Any]]:
        """Pack complete public units before the one model request is made.

        Task, committed business values and last real feedback are fixed.
        Documents, candidate summaries and discovery descriptions are admitted
        as whole units in public dependency order using the actual chat-template
        tokenizer. This is deterministic prompt construction, with no generated
        response, retry, truncated schema or substituted action involved.
        """
        history = self._model_history(messages, catalog)
        users = [index for index, message in enumerate(history) if message.get("role") == "user"]
        if not users:
            raise ValueError("A public task message is required.")
        task_index = users[0]
        following = [dict(message) for message in history[task_index + 1:]
                     if message.get("role") in {"assistant", "user"}]
        reserve = int(response_reserve if response_reserve is not None
                      else catalog.get("generation_token_budget", self.max_new_tokens))
        if reserve < 1:
            raise ValueError("The prompt window must reserve a positive complete-action output budget.")
        prompt_limit = self.max_context_tokens - reserve
        chosen = deepcopy(dict(public))
        documents = chosen.pop("documents")
        chosen["documents"] = {}
        summaries = chosen["execution_state"].pop("discovery_summaries")
        chosen["execution_state"]["discovery_summaries"] = {}
        frontier = chosen["frontier"]
        pools = {key: frontier.pop(key) for key in
                 ("ready_calls", "other_ready_calls", "ready_maps", "missing_inputs", "discovery_queries")}
        frontier.update({key: [] for key in pools})
        counts = chosen["catalog_counts"]
        counts.update({"visible_documents": 0, "visible_ready_calls": 0,
                       "summarized_ready_calls": 0, "visible_ready_maps": 0,
                       "visible_missing_inputs": 0, "visible_discovery_queries": 0,
                       "visible_discovery_descriptions": 0,
                       "stored_discovery_descriptions": sum(len(rows) for rows in summaries.values())})
        chosen["context_window"] = {"prompt_token_limit": prompt_limit,
                                     "reserved_response_tokens": reserve}
        discovery_refs = {item["id"]: tuple(item["visible_catalog_refs"])
                          for item in chosen["execution_state"]["discovery_bindings"]}

        def render(view: Mapping[str, Any]) -> list[dict[str, Any]]:
            return [
                {"role": "system", "content": instruction},
                dict(history[task_index]),
                {"role": "user", "content": "Current public execution state:\n" + self._schema_text(view)},
                *following[-2:],
            ]

        def refresh_counts(view: dict[str, Any]) -> None:
            visible = view["frontier"]
            for discovery in view["execution_state"]["discovery_bindings"]:
                discovery["visible_catalog_refs"] = [name for name in discovery_refs[discovery["id"]]
                                                       if name in view["documents"]]
            view["catalog_counts"].update({
                "visible_documents": len(view["documents"]),
                "visible_ready_calls": len(visible["ready_calls"]),
                "summarized_ready_calls": len(visible["other_ready_calls"]),
                "visible_ready_maps": len(visible["ready_maps"]),
                "visible_missing_inputs": len(visible["missing_inputs"]),
                "visible_discovery_queries": len(visible["discovery_queries"]),
                "visible_discovery_descriptions": sum(len(rows) for rows in
                    view["execution_state"]["discovery_summaries"].values()),
            })

        def admit(candidate: dict[str, Any]) -> bool:
            nonlocal chosen
            refresh_counts(candidate)
            if len(self._tokenize_messages(render(candidate))) > prompt_limit:
                return False
            chosen = candidate
            return True

        # The immutable business state is never cut at a token boundary. If it
        # alone exceeds the declared model window, report the actual resource
        # limit before any model generation or external API call.
        refresh_counts(chosen)
        if len(self._tokenize_messages(render(chosen))) > prompt_limit:
            raise BudgetExhausted("context_window")

        full_candidate = deepcopy(chosen)
        full_candidate["documents"] = deepcopy(documents)
        for key in ("ready_calls", "ready_maps", "missing_inputs"):
            full_candidate["frontier"][key] = [
                deepcopy(item) for name in documents for item in pools[key] if item["tool"] == name
            ]
        full_candidate["execution_state"]["discovery_summaries"] = {
            identifier: deepcopy(rows) for identifier, rows in summaries.items() if rows
        }
        full_candidate["frontier"]["other_ready_calls"] = [
            *deepcopy(pools["other_ready_calls"]),
            *(deepcopy(item) for item in pools["ready_calls"] if item["tool"] not in documents),
        ][:12]
        full_candidate["frontier"]["discovery_queries"] = deepcopy(pools["discovery_queries"])[:12]
        if admit(full_candidate):
            result = render(chosen)
            self._check_context(len(self._tokenize_messages(result)), reserve)
            return result

        # A short feasible discovery window stays visible even when execution
        # documents occupy most of the prompt; all remaining queries retain
        # their original grammar support.
        for query in pools["discovery_queries"][:8]:
            candidate = deepcopy(chosen)
            candidate["frontier"]["discovery_queries"].append(query)
            if not admit(candidate):
                break

        for name, document in documents.items():
            candidate = deepcopy(chosen)
            candidate["documents"][name] = document
            for key in ("ready_calls", "ready_maps", "missing_inputs"):
                candidate["frontier"][key].extend(item for item in pools[key] if item["tool"] == name)
            if not admit(candidate):
                # This whole document is outside this turn's display window;
                # its name and bindings can still enter as a compact ready call.
                continue

        # App/API description rows are independent public discovery choices.
        # Keep each selected row intact, including its description.
        for identifier, rows in summaries.items():
            for row in rows:
                candidate = deepcopy(chosen)
                candidate["execution_state"]["discovery_summaries"].setdefault(identifier, []).append(row)
                if not admit(candidate):
                    break

        other_calls = [*pools["other_ready_calls"],
                       *(item for item in pools["ready_calls"] if item["tool"] not in chosen["documents"])][:12]
        for call in other_calls:
            candidate = deepcopy(chosen)
            candidate["frontier"]["other_ready_calls"].append(call)
            if not admit(candidate):
                break
        for query in pools["discovery_queries"][8:12]:
            candidate = deepcopy(chosen)
            candidate["frontier"]["discovery_queries"].append(query)
            if not admit(candidate):
                break

        result = render(chosen)
        self._check_context(len(self._tokenize_messages(result)), reserve)
        return result

    def _mode_messages(self, messages: Sequence[Mapping[str, Any]], action_schema: Any) -> list[dict[str, Any]]:
        if action_schema.get("classic_text"):
            return self._text_action_messages(messages, action_schema, None, for_gate=True)
        catalog = action_schema["program_context"]
        public = self._compact_catalog_for_prompt(catalog, messages, for_gate=True)
        instruction = (
            "Select the next action class from the supplied public feasible frontier. "
            f"Reply with the single token {self.mode_labels['I']} for interface discovery, or "
            f"{self.mode_labels['E']} for actual business execution. "
            "Execution includes authentication, business reads, writes and verified final submission. "
            "Use the next unmet dependency, discovery_telemetry and currently usable input bindings to decide."
        )
        return self._protocol_messages(messages, catalog, instruction, public, response_reserve=1)

    def _action_messages(
        self, messages: Sequence[Mapping[str, Any]], action_schema: Any, mode: str,
    ) -> list[dict[str, Any]]:
        if not isinstance(action_schema, Mapping) or mode not in {"I", "E"}:
            raise ValueError("A conditional action needs a public I/E schema.")
        if action_schema.get("classic_text"):
            return self._text_action_messages(messages, action_schema, mode)
        from .native_program import program_instruction
        context = action_schema["program_context"]
        public = self._compact_catalog_for_prompt(context, messages, mode=mode)
        instruction = (
            "Emit exactly one complete JSON action in the shared native-r3 contract. "
            "The runtime compiles your selected operation and resolves its explicit references. "
            "Use the actual public return structure, including whether each value is a scalar, list or object. "
            "Choose the next dependency from the current public frontier. "
            "A finish action refers to an executed target answer or completes the requested operation.\n"
            + program_instruction(context, mode)
        )
        return self._protocol_messages(messages, context, instruction, public)

    def _program_decoder_engine(self):
        from .native_program_decoder import ProgramDecoder
        decoder = getattr(self, "_program_decoder", None)
        if decoder is None:
            decoder = ProgramDecoder(self.tokenizer,
                                     int(self.model.get_output_embeddings().weight.shape[0]),
                                     self.eos_token_ids)
            self._program_decoder = decoder
        return decoder

    def _text_decoder_engine(self):
        from .native_text_actions import NativeTextDecoder
        decoder = getattr(self, "_text_decoder", None)
        if decoder is None:
            decoder = NativeTextDecoder(self.tokenizer,
                                       int(self.model.get_output_embeddings().weight.shape[0]),
                                       self.eos_token_ids)
            self._text_decoder = decoder
        return decoder

    def _action_decoder_engine(self, contract: Mapping[str, Any]):
        return (self._text_decoder_engine() if contract.get("decoder_kind") == "native_text"
                else self._program_decoder_engine())

    def _text_action_messages(
        self, messages: Sequence[Mapping[str, Any]], schema: Mapping[str, Any],
        mode: str | None, *, for_gate: bool = False,
    ) -> list[dict[str, Any]]:
        from .native_text_actions import open_text_actions, text_actions
        candidates, opened = text_actions(schema), open_text_actions(schema)
        selected = ("I", "E") if mode is None else (mode,)
        public = {
            "mode_actions": {key: candidates[key] for key in selected},
            "open_text_modes": {key: opened[key] for key in selected},
            "search_available": bool("I" in selected and schema.get("search_available", False)),
        }
        history = [dict(message) for message in messages if message.get("role") != "system"]
        if schema.get("benchmark") == "loca" and schema.get("native_tools"):
            # LOCA's fixed tool catalog is serialized once, with original descriptions.
            selected_names = {name for key in selected for name in opened[key]}
            public["native_tools"] = [deepcopy(tool) for tool in schema["native_tools"]
                                      if tool["function"]["name"] in selected_names]
            for message in history:
                if message.get("role") != "user" or not isinstance(message.get("content"), str):
                    continue
                try:
                    content = json.loads(message["content"])
                except ValueError:
                    continue
                if not isinstance(content, dict):
                    continue
                observation = content.get("observation") if content.get("feedback_type") == "task" else content
                if not isinstance(observation, dict):
                    continue
                state = observation.get("classic_text_action_state")
                if not isinstance(state, dict) or state.get("benchmark") != "loca":
                    continue
                state.pop("native_tools", None)
                state.pop("native_action_schemas_by_mode", None)
                message["content"] = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
        elif schema.get("native_action_schemas_by_mode"):
            public["native_action_schemas_by_mode"] = {
                key: schema["native_action_schemas_by_mode"][key]
                for key in selected if key in schema["native_action_schemas_by_mode"]
            }
        if for_gate:
            instruction = (
                f"Choose {self.mode_labels['I']} for information acquisition or "
                f"{self.mode_labels['E']} for goal execution/submission. Reply with that single token. "
                "Information acquisition may change navigation or visible interaction state. "
                "Choose using the current public observations and the listed native operations."
            )
        else:
            syntax = ("one native JSON tool call matching the provided schema"
                      if schema.get("native_action_schemas_by_mode") else "one native text action")
            instruction = (
                f"Emit exactly {syntax}" + (f" for mode {mode}. " if mode is not None else ". ") +
                "Do not add reasoning, wrappers, or another action. "
                "Choose from this mode's visible commands. For an open operation, generate its argument "
                "from the task and public observations using its original parameter schema."
            )
        system = str(schema.get("native_system_prompt") or schema.get("native_interface") or "")
        result = [{"role": "system", "content": system + "\n" + instruction},
                  *history,
                  {"role": "user", "content": "Current public action interface:\n" +
                   json.dumps(public, ensure_ascii=False, separators=(",", ":"))}]
        self._check_context(len(self._tokenize_messages(result)), 1 if for_gate else self.max_new_tokens)
        return result

    def _generate_native_text(
        self, messages: Sequence[Mapping[str, Any]], schema: Mapping[str, Any], *,
        mode: str, greedy: bool, role: str, max_new_tokens: int | None,
    ) -> NativeGeneration:
        from .native_text_actions import TEXT_ACTOR_CONTRACT, text_available_modes
        public_schema = deepcopy(dict(schema))
        contract = self._text_decoder_engine().native_contract(public_schema, mode)
        prompt = self._text_action_messages(messages, public_schema, mode)
        generation = self.complete(prompt, role=role, max_new_tokens=max_new_tokens,
                                   greedy=greedy, program_contract=contract)
        complete = generation.metadata["program_complete"]
        action = (parse_native_action(generation.text, mode=mode, action_schema=public_schema)
                  if complete else
                  {"kind": "invalid", "mode": mode, "attempted_mode": mode,
                   "raw_text": generation.text, "validation_stage": "native_text_truncation",
                   "error": "Generation ended before the native action's sampled EOS."})
        metadata = dict(generation.metadata, actor_contract=TEXT_ACTOR_CONTRACT, classic_text=True,
                        actor_gated=True, action_schema=public_schema, native_action_schema=public_schema,
                        text_action_complete=complete, text_action_contract=contract,
                        feasible_modes=list(text_available_modes(public_schema)))
        metadata.pop("program_contract", None)
        return replace(generation, action=action, mode=mode, metadata=metadata)

    def _native_action_messages(
        self, messages: Sequence[Mapping[str, Any]], action_schema: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """The ungated actor uses the same union of supported JSON operations."""
        from .native_program import program_instruction
        context = action_schema["program_context"]
        public = self._compact_catalog_for_prompt(context, messages)
        instruction = (
            "Emit one complete JSON action from the public feasible frontier. "
            "Select discovery or business execution directly; all operations use the same native-r3 syntax. "
            "Resolve the next unmet dependency with usable observed references and the exact public tool document. "
            "Retain the semantic distinction between different record IDs and between credentials and tokens.\n"
            + program_instruction(context, None)
        )
        return self._protocol_messages(messages, context, instruction, public)

    def _generate_program(
        self, messages: Sequence[Mapping[str, Any]], action_schema: Mapping[str, Any], *,
        mode: str | None, greedy: bool, role: str, max_new_tokens: int | None,
    ) -> NativeGeneration:
        from .native_program import PROGRAM_VERSION, available_modes, program_schema
        context = deepcopy(action_schema["program_context"])
        feasible = tuple(available_modes(context))
        if not feasible:
            raise NoFeasibleAction("No public discovery, ready business call or valid finish is available.")
        if mode is not None and mode not in feasible:
            raise NoFeasibleAction(f"Mode {mode} has no public feasible action.")
        limit = self.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        if limit < 1:
            raise ValueError("Generation length must be positive.")
        context["generation_token_budget"] = limit
        schema = deepcopy(dict(action_schema))
        schema["program_context"] = context
        contract = self._program_decoder_engine().contract(program_schema(context, mode), context, mode)
        prompt = (self._native_action_messages(messages, schema) if mode is None
                  else self._action_messages(messages, schema, mode))
        generation = self.complete(prompt, role=role, max_new_tokens=limit,
                                   greedy=greedy, program_contract=contract)
        metadata = dict(generation.metadata, actor_contract=PROGRAM_VERSION,
                        actor_gated=mode is not None, raw_program_text=generation.text,
                        action_schema=schema, native_action_schema=schema,
                        program_context=context, feasible_modes=list(feasible))
        compile_started = monotonic_seconds()
        if metadata["program_complete"]:
            action = parse_native_action(generation.text, mode=mode, action_schema=schema)
        else:
            action = {"kind": "invalid", "mode": mode, "attempted_mode": mode,
                      "raw_text": generation.text, "validation_stage": "program_truncation",
                      "error": "The resource limit ended generation before a complete action."}
        metadata.update(program_compile_seconds=monotonic_seconds() - compile_started,
                        compiled_action=deepcopy(action))
        return replace(generation, action=action, mode=mode, metadata=metadata)

    @torch.no_grad()
    def act_native(
        self, messages: Sequence[Mapping[str, Any]], action_schema: Mapping[str, Any], *,
        greedy: bool = False, role: str = "actor", max_new_tokens: int | None = None,
    ) -> NativeGeneration:
        benchmark = action_schema.get("benchmark", action_schema.get("program_context", {}).get("benchmark"))
        instruction = (
            "Execute one Python code block in AppWorld. The apis object is already available; "
            "print API results you need to observe. Variables and database state persist across turns. "
            "Discover exact signatures through api_docs and finish through supervisor.complete_task."
            if benchmark == "appworld" else
            "Emit one PlanBench action: <retrieve_tools>JSON</retrieve_tools>, "
            "<tool_call>JSON</tool_call>, or <final_answer>answer</final_answer>. "
            "Retrieval meta_tool is by_output_info (output_information), by_inputs (input_information, "
            "one or two phrases), or by_io_info (both fields). Tool calls use a discovered name or tool_id "
            "and arguments. Choose the information phrases, tool, arguments and answer yourself."
        )
        if action_schema.get("classic_text"):
            instruction = (
                "Select one native action from the current public observations. Return the native JSON "
                "tool call with name and arguments."
                if action_schema.get("native_action_schemas_by_mode") else
                "Select one native text action using the current observations and public action list."
            )
        prompt = [dict(message) for message in messages if message.get("role") != "system"]
        instruction = str(action_schema.get("native_system_prompt") or instruction)
        prompt.insert(0, {"role": "system", "content": instruction})
        if benchmark == "loca" and action_schema.get("classic_text"):
            # Ungated baselines retain the full union, with each public tool defined once.
            prompt = self._text_action_messages(messages, action_schema, None)
            instruction = str(prompt[0]["content"])
        generation = self.complete(prompt, greedy=greedy, role=role, max_new_tokens=max_new_tokens)
        raw = generation.text.strip()
        if action_schema.get("classic_text"):
            from .native_text_actions import extract_native_text
            action = {"kind": "native_text", "text": extract_native_text(generation.text)}
        elif benchmark == "appworld":
            match = re.fullmatch(r"```(?:python)?\s*\n?(.*?)\n?```", raw, flags=re.DOTALL)
            action = {"kind": "code", "code": match.group(1) if match else raw}
        elif benchmark == "planbench_xl":
            action = {"raw_text": raw}
        else:
            raise ValueError(f"Unsupported native benchmark: {benchmark}")
        return replace(generation, mode=None, mode_prompt_ids=(), mode_token_id=None,
                       action=action,
                       metadata=dict(generation.metadata, actor_gated=False,
                                     classic_text=bool(action_schema.get("classic_text")),
                                     gate_active=False, mode_sampled=False, forced_mode=None,
                                     actor_contract="official-native-action-v1",
                                     native_instruction=instruction, native_messages=prompt,
                                     native_text_prompt_schema=(deepcopy(dict(action_schema))
                                         if benchmark == "loca" and action_schema.get("classic_text") else None)))

    def teacher_forced_generation(
        self, messages: Sequence[Mapping[str, Any]], action_schema: Mapping[str, Any],
        program: Mapping[str, Any], *, mode: str | None = None, gated: bool = True,
        role: str = "teacher.target",
    ) -> NativeGeneration:
        """Encode a complete public teacher target without a model generation.

        Reference solutions choose the action outside this function. Only the
        current public state and this action enter the student distribution.
        Model cost is recorded when score_tokens performs teacher forcing.
        """
        from .native_program import PROGRAM_VERSION, available_modes, compile_program, program_schema
        context = deepcopy(action_schema["program_context"])
        context["generation_token_budget"] = self.max_new_tokens
        feasible = tuple(available_modes(context))
        selected_mode = "I" if program.get("kind") == "discover" else "E"
        if selected_mode not in feasible:
            raise NoFeasibleAction("The teacher target is outside the current public action support.")
        if mode is not None and mode != selected_mode:
            raise ValueError("Teacher mode and action class disagree.")
        conditional_mode = selected_mode if gated else None
        schema = deepcopy(dict(action_schema))
        schema["program_context"] = context
        compiled = compile_program(dict(program), context, conditional_mode)
        grammar = program_schema(context, conditional_mode)
        text = _canonical_program(dict(program), grammar)
        response_ids = tuple(self.tokenizer.encode(text, add_special_tokens=False))
        if not response_ids or len(response_ids) > self.max_new_tokens:
            raise ValueError("A complete teacher action must fit the actor's unchanged output-token budget.")
        contract = self._program_decoder_engine().contract(grammar, context, conditional_mode)
        # Verify target token membership as part of target encoding. No model
        # call or environment operation is performed here.
        self._program_decoder_engine().scoring_mask(contract, response_ids)
        prompt = (self._action_messages(messages, schema, selected_mode) if gated
                  else self._native_action_messages(messages, schema))
        prompt_ids = self._tokenize_messages(prompt)
        self._check_context(len(prompt_ids), len(response_ids))
        gate_active = bool(gated and len(feasible) == 2)
        mode_prompt = (self._tokenize_messages(self._mode_messages(messages, schema)) if gate_active else ())
        if mode_prompt:
            self._check_context(len(mode_prompt), 1)
        metadata = {
            "prompt_version": PROMPT_VERSION, "actor_contract": PROGRAM_VERSION,
            "actor_gated": bool(gated), "teacher_forced": True,
            "temperature": 1.0, "mode_temperature": 1.0,
            "gate_active": gate_active, "mode_sampled": False, "forced_mode": selected_mode,
            "feasible_modes": list(feasible), "program_complete": True,
            "program_eos_accepted": False, "program_contract": contract,
            "program_context": context, "action_schema": schema, "native_action_schema": schema,
            "raw_program_text": text, "compiled_action": deepcopy(compiled),
            "reference_mode": self._reference_label,
        }
        return NativeGeneration(text=text, prompt_ids=prompt_ids, response_ids=response_ids,
                                action=compiled, mode=conditional_mode,
                                mode_prompt_ids=mode_prompt,
                                mode_token_id=self.mode_token_ids[selected_mode] if gate_active else None,
                                role=role, metadata=metadata)

    def rebind_native_generation(
        self, generation: NativeGeneration, messages: Sequence[Mapping[str, Any]],
        action_schema: Mapping[str, Any] | None = None,
    ) -> NativeGeneration:
        """Rebind an offline target to public messages, retaining its sampled tokens."""
        from .native_program import PROGRAM_VERSION, available_modes, program_schema, compile_program
        if generation.metadata.get("actor_contract") != PROGRAM_VERSION:
            raise ValueError("Only a recorded native-r3 action can be rebound.")
        schema = deepcopy(dict(action_schema or generation.metadata["action_schema"]))
        from .native_envs import public_action_schema
        schema = public_action_schema(schema, messages)
        context = schema["program_context"]
        mode = generation.mode if generation.metadata.get("actor_gated") else None
        compile_program(json.loads(generation.text), context, mode)
        contract = self._program_decoder_engine().contract(program_schema(context, mode), context, mode)
        self._program_decoder_engine().scoring_mask(contract, generation.response_ids, counterfactual=True)
        gate_active = bool(mode is not None and len(available_modes(context)) == 2)
        prompt = (self._native_action_messages(messages, schema) if mode is None
                  else self._action_messages(messages, schema, mode))
        metadata = {key: value for key, value in generation.metadata.items()
                    if key not in {"sampling_log_probs", "mode_log_probability", "gate_probability"}}
        metadata.update(target_rebound_to_public_prefix=True, action_schema=schema,
                        native_action_schema=schema, program_context=context, program_contract=contract,
                        gate_active=gate_active, mode_sampled=False)
        return replace(generation, prompt_ids=self._tokenize_messages(prompt),
                       mode_prompt_ids=self._tokenize_messages(self._mode_messages(messages, schema)) if gate_active else (),
                       mode_token_id=self.mode_token_ids[mode] if gate_active else None, metadata=metadata)

    def _install_reference(self, weights: Mapping[str, torch.Tensor], label: str) -> None:
        from peft.utils.save_and_load import set_peft_model_state_dict
        if not isinstance(label, str) or not label or label == "raw_base":
            raise ValueError("A warmstart snapshot requires its own explicit reference label.")
        adapter = "prism_warmstart_reference"
        trainable = {name: parameter.requires_grad for name, parameter in self.model.named_parameters()}
        if adapter not in self.model.peft_config:
            self.model.add_adapter(adapter, deepcopy(self.model.peft_config["default"]))
        result = set_peft_model_state_dict(self.model, dict(weights), adapter_name=adapter)
        unexpected = list(getattr(result, "unexpected_keys", ()))
        missing = [name for name in getattr(result, "missing_keys", ()) if f".{adapter}." in name]
        if unexpected or missing:
            raise ValueError(f"Warmstart reference mismatch: missing={missing}, unexpected={unexpected}")
        self.model.set_adapter("default")
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(trainable.get(name, False) and f".{adapter}." not in name)
        self._reference_adapter = adapter
        self._reference_label = str(label)
        self._parameter_storages = {
            tensor.untyped_storage().data_ptr()
            for tensor in (*self.model.parameters(), *self.model.buffers())
            if tensor.device.type == "cuda"
        }

    def freeze_warmstart_reference(self, label: str = "warmstart_reference") -> None:
        """Copy LoRA tensors once; the reference never follows optimizer updates."""
        from peft.utils.save_and_load import get_peft_model_state_dict
        if self._reference_depth:
            raise RuntimeError("Freeze the warmstart reference outside a reference context.")
        weights = {key: value.detach().cpu().clone() for key, value in
                   get_peft_model_state_dict(self.model, adapter_name="default").items()}
        self._install_reference(weights, label)

    @contextmanager
    def raw_base_reference(self) -> Iterator[NativeCausalPolicy]:
        """Explicit original-backbone control used by the R experiment arm."""
        if self._reference_depth and self._active_reference != "raw_base":
            raise RuntimeError("Different reference distributions cannot be nested.")
        previous_depth = self._reference_depth
        self._reference_depth += 1
        self._active_reference = "raw_base"
        try:
            with self.model.disable_adapter(), torch.no_grad():
                yield self
        finally:
            self._reference_depth = previous_depth
            if not previous_depth:
                self._active_reference = None

    @contextmanager
    def frozen_reference(self) -> Iterator[NativeCausalPolicy]:
        """Use the registered immutable warmstart, or the defined initial base."""
        if self._reference_adapter is None:
            with self.raw_base_reference():
                yield self
            return
        if self._reference_depth:
            if self._active_reference != self._reference_label:
                raise RuntimeError("Different reference distributions cannot be nested.")
            self._reference_depth += 1
            try:
                yield self
            finally:
                self._reference_depth -= 1
            return
        active = self.model.active_adapter
        trainable = {name: parameter.requires_grad for name, parameter in self.model.named_parameters()}
        self._reference_depth = 1
        self._active_reference = self._reference_label
        try:
            self.model.set_adapter(self._reference_adapter)
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
            with torch.no_grad():
                yield self
        finally:
            self.model.set_adapter(active)
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad_(trainable[name])
            self._reference_depth = 0
            self._active_reference = None

    @contextmanager
    def _model_mode(self, scoring: bool = False) -> Iterator[None]:
        previous = self.model.training
        self.model.train(bool(scoring and torch.is_grad_enabled() and self._gradient_checkpointing))
        try:
            yield
        finally:
            self.model.train(previous)

    def _forward_logits(self, input_ids: tuple[int, ...], keep: int) -> torch.Tensor:
        tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        kwargs: dict[str, Any] = {"input_ids": tensor, "attention_mask": torch.ones_like(tensor),
                                 "use_cache": False}
        if self._keep_logits_arg is not None:
            kwargs[self._keep_logits_arg] = keep
        return self.model(**kwargs).logits[0, -keep:, :]

    def _mode_log_probs(
        self, prompt_ids: tuple[int, ...], *, role: str, generating: bool = False,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, float]:
        self._check_context(len(prompt_ids), 1)
        if self.meter:
            if generating:
                self.meter.ensure_model_call(len(prompt_ids), 1, role)
            else:
                self.meter.ensure_forward(len(prompt_ids), role)
        self._synchronize()
        started = monotonic_seconds()
        with self._activation_saving(), self._model_mode(scoring=not generating):
            logits = self._forward_logits(prompt_ids, 1)[-1]
            pair = logits[[self.mode_token_ids["I"], self.mode_token_ids["E"]]].float()
            if not torch.isfinite(pair).all():
                raise FloatingPointError("The model emitted non-finite I/E gate logits.")
            log_probs = F.log_softmax(pair / temperature, dim=0)
        self._synchronize()
        elapsed = max(0.0, monotonic_seconds() - started)
        if self.meter:
            self.meter.record_model_call(
                len(prompt_ids) if generating else 0, 1 if generating else 0,
                elapsed, role, forward_tokens=0 if generating else len(prompt_ids),
            )
        return log_probs, elapsed

    @torch.no_grad()
    def complete(
        self, messages: Sequence[Mapping[str, Any]], role: str = "actor", *,
        max_new_tokens: int | None = None, temperature: float | None = None,
        greedy: bool = False, program_contract: Mapping[str, Any] | None = None,
    ) -> NativeGeneration:
        """Generate exact response tokens and record the physical model request."""
        from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

        limit = self.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        requested_limit = limit
        temp = self.temperature if temperature is None else float(temperature)
        if limit <= 0 or not math.isfinite(temp) or temp <= 0:
            raise ValueError("Generation length and temperature must be positive.")
        prompt_ids = self._tokenize_messages(messages)
        context_remaining = self.max_context_tokens - len(prompt_ids)
        limit = min(limit, context_remaining)
        if limit <= 0:
            raise BudgetExhausted("context_window")
        if self.meter:
            limit = min(limit, self.meter.remaining_model_tokens - len(prompt_ids))
            if limit <= 0:
                raise BudgetExhausted("model_tokens")
        self._check_context(len(prompt_ids), limit)
        if self.meter:
            self.meter.ensure_model_call(len(prompt_ids), limit, role)
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        processors = None
        program_processor = None
        program_setup_seconds = 0.0
        if program_contract is not None:
            setup_started = monotonic_seconds()
            program_processor = self._action_decoder_engine(program_contract).processor(program_contract, len(prompt_ids))
            program_setup_seconds = monotonic_seconds() - setup_started
            processors = LogitsProcessorList([program_processor])
        eos_ids = list(self.eos_token_ids)
        settings: dict[str, Any] = {
            "max_new_tokens": limit, "do_sample": not greedy, "num_beams": 1,
            "repetition_penalty": 1.0, "no_repeat_ngram_size": 0, "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": eos_ids or None,
            "bos_token_id": self.tokenizer.bos_token_id,
            "return_dict_in_generate": True, "output_scores": True,
        }
        if not greedy:
            settings.update(temperature=temp, top_k=0, top_p=1.0)
        stopping = StoppingCriteriaList()
        action_stop = None
        time_stop = None
        if program_processor is not None:
            prompt_length = len(prompt_ids)

            class NativeActionStop(StoppingCriteria):
                triggered = False

                def __call__(self, input_ids, scores, **kwargs):
                    emitted = input_ids[0, prompt_length:].tolist()
                    self.triggered = program_processor.completed(emitted)
                    return torch.full((input_ids.shape[0],), self.triggered,
                                      device=input_ids.device, dtype=torch.bool)

            action_stop = NativeActionStop()
            stopping.append(action_stop)
        if self.meter:
            cursor: BudgetMeter | None = self.meter
            deadlines = []
            while cursor is not None:
                deadlines.append(cursor.started + cursor.limits.max_wall_seconds)
                cursor = cursor.parent
            deadline = min(deadlines)

            class RealTimeBudgetStop(StoppingCriteria):
                triggered = False

                def __call__(self, input_ids, scores, **kwargs):
                    self.triggered = monotonic_seconds() >= deadline
                    return torch.full((input_ids.shape[0],), self.triggered,
                                      device=input_ids.device, dtype=torch.bool)

            time_stop = RealTimeBudgetStop()
            stopping.append(time_stop)
        if self.meter:
            self.meter.ensure_model_call(len(prompt_ids), limit, role)
        self._synchronize()
        started = monotonic_seconds()
        with self._model_mode(), torch.no_grad():
            output = self.model.generate(
                input_ids=prompt, attention_mask=torch.ones_like(prompt),
                generation_config=GenerationConfig(**settings),
                stopping_criteria=stopping, logits_processor=processors,
            )
            response = output.sequences[0, len(prompt_ids):]
            response_ids = tuple(int(item) for item in response.tolist())
            if len(output.scores) != len(response_ids):
                raise RuntimeError("Generation scores do not align with emitted response tokens.")
            token_scores = [
                F.log_softmax(scores[0].float(), dim=-1)[response[index]]
                for index, scores in enumerate(output.scores)
            ]
            if not token_scores:
                raise RuntimeError("The model returned no response tokens.")
            sampled_scores = torch.stack(token_scores)
            if not torch.isfinite(sampled_scores).all():
                raise FloatingPointError("An emitted response token has non-finite sampling log probability.")
            sampling_log_probs = sampled_scores.cpu().tolist()
        self._synchronize()
        elapsed = max(0.0, monotonic_seconds() - started)
        program_metadata = {}
        if program_processor is not None:
            program_metadata = {**program_processor.finish(response_ids),
                                "program_contract": deepcopy(dict(program_contract)),
                                "program_setup_seconds": program_setup_seconds}
        eos_reached = bool(response_ids and response_ids[-1] in eos_ids)
        wall_reached = bool(time_stop is not None and time_stop.triggered)
        length_reached = len(response_ids) >= limit
        action_complete = bool(action_stop is not None and action_stop.triggered)
        limit_sources = []
        if limit == requested_limit:
            limit_sources.append("max_new_tokens")
        if limit == context_remaining:
            limit_sources.append("context_window")
        if self.meter and limit == self.meter.remaining_model_tokens - len(prompt_ids):
            limit_sources.append("model_token_budget")
        stop_reason = ("action_complete" if action_complete else "eos" if eos_reached else
                       "wall_time_budget" if wall_reached else
                       "+".join(limit_sources) if length_reached else "generation_returned")
        if self.meter:
            self.meter.record_model_call(len(prompt_ids), len(response_ids), elapsed, role)
        return NativeGeneration(
            text=self.tokenizer.decode(response_ids, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False),
            prompt_ids=prompt_ids, response_ids=response_ids, input_tokens=len(prompt_ids),
            output_tokens=len(response_ids), elapsed_seconds=elapsed, role=role,
            metadata={"prompt_version": PROMPT_VERSION, "temperature": temp if not greedy else 1.0,
                      **program_metadata,
                      "greedy": bool(greedy), "sampling_log_probs": sampling_log_probs,
                      "max_new_tokens": limit, "requested_max_new_tokens": requested_limit,
                      "output_limit_reached": length_reached and not (eos_reached or action_complete),
                      "action_boundary_reached": action_complete,
                      "response_tokens_finite": True,
                      "eos_token_ids": eos_ids, "eos_reached": eos_reached,
                      "wall_deadline_reached": wall_reached,
                      "generation_limit_sources": limit_sources,
                      "generation_stop_reason": stop_reason,
                      "model": dict(self.model_metadata),
                      "reference_mode": self._active_reference or self._reference_label, "reference_active": bool(self._reference_depth)},
        )

    def get_dcc_library(self, benchmark: str):
        from .native_dcc import ProgramLibrary
        name = str(benchmark).split(":", 1)[0]
        if name not in self.dcc_libraries:
            self.dcc_libraries[name] = ProgramLibrary(name)
        return self.dcc_libraries[name]

    def get_active_dcc_model_view(self):
        """Return the immutable episode-fixed ModelView (phi_0 for reset_parameters/k0_planner, phi_T otherwise)."""
        use_init = self.dcc_eval_variant in ("reset_parameters", "k0_planner")
        return self.dcc_ensemble.active_model_view(use_initial_params=use_init)

    def reset_dcc_episode(
        self,
        benchmark: str,
        *,
        task_instruction: str = "",
        member_index: int | None = None,
    ) -> None:
        """Reset intra-task state s_{0,phi}(h_0) and active program state while retaining cross-task phi and Xi."""
        self.get_dcc_library(benchmark)
        view = self.get_active_dcc_model_view()
        self.dcc_active_belief = view.reset_intra_task_belief(device=self.device)
        self.dcc_active_program_state = None
        # Always assign `member_index` unconditionally (including None during deployment/evaluation, fixes A)
        self.dcc_active_member_index = member_index
        self.dcc_episode_events = []
        self.dcc_task_instruction = str(task_instruction)
        self.dcc_remaining_budget_ratio = 1.0
        self._dcc_context_cache_key: str | None = None
        self._dcc_context_cache_vec: tuple[float, ...] = ()

    @torch.no_grad()
    def extract_frozen_semantic_context(
        self,
        goal_text: str,
        public_fields: Mapping[str, str],
    ) -> tuple[float, ...]:
        """Extract 32-dim shared semantic representation from the frozen Qwen input embedding matrix."""
        from .native_dcc import FROZEN_HISTORY_DIM, _canonical_json

        cache_key = f"{goal_text}||{_canonical_json(dict(public_fields))}"
        if getattr(self, "_dcc_context_cache_key", None) == cache_key and getattr(self, "_dcc_context_cache_vec", ()):
            return self._dcc_context_cache_vec

        emb_layer = self.model.get_input_embeddings()
        text = f"Goal: {goal_text}\nObserved: {_canonical_json(dict(public_fields))}"
        tok_ids = self.tokenizer.encode(text, add_special_tokens=False)[:256]
        if not tok_ids or emb_layer is None:
            from .native_dcc import ControlledResponseEnsemble
            vec = ControlledResponseEnsemble.deterministic_semantic_projection(text, FROZEN_HISTORY_DIM)
            self._dcc_context_cache_key = cache_key
            self._dcc_context_cache_vec = vec
            return vec

        ids_t = torch.tensor(tok_ids, dtype=torch.long, device=emb_layer.weight.device)
        embeds = emb_layer(ids_t).float()  # (L, D)
        mean_emb = embeds.mean(dim=0)      # (D,)
        chunks = mean_emb.tensor_split(FROZEN_HISTORY_DIM)
        pooled = torch.stack([c.mean() for c in chunks])
        norm = pooled.norm().clamp(min=1e-6)
        vec = tuple(float(x) for x in (pooled / norm).detach().cpu().tolist())
        self._dcc_context_cache_key = cache_key
        self._dcc_context_cache_vec = vec
        return vec

    @torch.no_grad()
    def extract_frozen_action_embeddings(
        self,
        action_texts: Sequence[str],
        *,
        role_bindings: Mapping[str, str] | None = None,
    ) -> list[tuple[float, ...]]:
        """Batch-extract 16-dim semantic action embeddings from the frozen Qwen input embedding matrix.

        Applies `PublicBenchmarkBinder.normalize_action_for_semantic_encoding` and memoizes immutable
        frozen embeddings so full-action-set scoring and tail NLL share identical vectors at minimal cost.
        """
        from .native_dcc import FROZEN_ACTION_DIM, ControlledResponseEnsemble, PublicBenchmarkBinder

        cache: dict[str, tuple[float, ...]] = getattr(self, "_dcc_action_embed_cache", {})
        if not hasattr(self, "_dcc_action_embed_cache"):
            self._dcc_action_embed_cache = cache

        norm_texts = [
            PublicBenchmarkBinder.normalize_action_for_semantic_encoding(act, role_bindings)
            for act in action_texts
        ]
        emb_layer = self.model.get_input_embeddings()
        if emb_layer is None:
            return [
                ControlledResponseEnsemble.deterministic_semantic_projection(nt, FROZEN_ACTION_DIM)
                for nt in norm_texts
            ]

        dev = emb_layer.weight.device
        missing = [nt for nt in dict.fromkeys(norm_texts) if nt not in cache]
        if missing:
            if len(cache) > 8192:
                cache.clear()
            for nt in missing:
                tok_ids = self.tokenizer.encode(nt, add_special_tokens=False)[:64]
                if not tok_ids:
                    cache[nt] = tuple(0.0 for _ in range(FROZEN_ACTION_DIM))
                    continue
                ids_t = torch.tensor(tok_ids, dtype=torch.long, device=dev)
                embeds = emb_layer(ids_t).float()
                mean_emb = embeds.mean(dim=0)
                chunks = mean_emb.tensor_split(FROZEN_ACTION_DIM)
                pooled = torch.stack([c.mean() for c in chunks])
                norm = pooled.norm().clamp(min=1e-6)
                cache[nt] = tuple(float(x) for x in (pooled / norm).detach().cpu().tolist())

        return [cache[nt] for nt in norm_texts]

    @torch.no_grad()
    def score_frozen_action_logprobs(
        self,
        messages: Sequence[Mapping[str, Any]],
        action_schema: Mapping[str, Any],
        action_texts: Sequence[str],
        *,
        prompt_ids: Sequence[int] | None = None,
        prefill_already_charged: bool = False,
        role: str = "dcc.candidate_suffix_prior",
    ) -> dict[str, float]:
        """Compute exact length-normalized frozen Qwen log-probabilities l_0(a | g, h) via shared KV prefill.

        Prefills the decision prompt once under `frozen_reference()` and evaluates candidate action suffixes
        in small batches against the cached KV state, charging only incremental suffix tokens when prefill
        was already metered in `prior_generation`.
        """
        unique_acts = [a for a in dict.fromkeys(str(x).strip() for x in action_texts) if a]
        if not unique_acts:
            return {}

        if prompt_ids:
            p_ids = tuple(int(x) for x in prompt_ids)
        elif action_schema.get("classic_text"):
            from .native_text_envs import public_text_action_schema
            pub_schema = public_text_action_schema(action_schema, messages)
            p_ids = self._tokenize_messages(self._text_action_messages(messages, pub_schema, None))
        else:
            p_ids = self._tokenize_messages(self._native_action_messages(messages, action_schema))

        if len(p_ids) + 32 > self.max_context_tokens:
            p_ids = p_ids[-(self.max_context_tokens - 32) :]

        tokenized: list[tuple[str, list[int]]] = []
        suffix_tokens = 0
        for act in unique_acts:
            t_ids = list(self.tokenizer.encode(act, add_special_tokens=False)[:32])
            if t_ids:
                tokenized.append((act, t_ids))
                suffix_tokens += max(0, len(t_ids) - 1)
        if not tokenized:
            return {a: -6.0 for a in unique_acts}

        charged_tokens = suffix_tokens + (0 if prefill_already_charged else len(p_ids))
        if self.meter and charged_tokens > 0:
            if self.meter.remaining_model_tokens < charged_tokens:
                return {a: 0.0 for a in unique_acts}
            self.meter.ensure_forward(charged_tokens, role)

        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(self.device) if torch.cuda.is_available() else None
        self._synchronize()
        started = monotonic_seconds()
        lp_out: dict[str, float] = {}
        try:
            with self.frozen_reference(), self._model_mode(), torch.no_grad():
                prompt_t = torch.tensor([p_ids], dtype=torch.long, device=self.device)
                prefill = self.model(
                    input_ids=prompt_t,
                    attention_mask=torch.ones_like(prompt_t),
                    use_cache=True,
                )
                first_lp = F.log_softmax(prefill.logits[0, -1, :].float(), dim=-1)
                base_cache = prefill.past_key_values

                multi_items: list[tuple[str, list[int]]] = []
                for act, t_ids in tokenized:
                    if len(t_ids) == 1:
                        lp_out[act] = float(first_lp[t_ids[0]].item())
                    else:
                        multi_items.append((act, t_ids))

                pad_id = int(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0)
                p_len = len(p_ids)
                chunk_size = 1 if p_len > 4096 else (2 if p_len > 2048 else 8)
                for start_i in range(0, len(multi_items), chunk_size):
                    chunk = multi_items[start_i : start_i + chunk_size]
                    b_sz = len(chunk)
                    max_suf = max(len(t_ids) - 1 for _, t_ids in chunk)
                    suf_mat = torch.full((b_sz, max_suf), pad_id, dtype=torch.long, device=self.device)
                    for b_i, (_, t_ids) in enumerate(chunk):
                        s_len = len(t_ids) - 1
                        suf_mat[b_i, :s_len] = torch.tensor(t_ids[:-1], dtype=torch.long, device=self.device)
                    chunk_cache = deepcopy(base_cache)
                    if b_sz > 1:
                        chunk_cache.batch_repeat_interleave(b_sz)
                    attn_mask = torch.ones((b_sz, p_len + max_suf), dtype=torch.long, device=self.device)
                    cache_pos = torch.arange(p_len, p_len + max_suf, dtype=torch.long, device=self.device)
                    suf_res = self.model(
                        input_ids=suf_mat,
                        attention_mask=attn_mask,
                        past_key_values=chunk_cache,
                        cache_position=cache_pos,
                        use_cache=False,
                    )
                    suf_lps = F.log_softmax(suf_res.logits.float(), dim=-1)
                    for b_i, (act, t_ids) in enumerate(chunk):
                        tot_lp = float(first_lp[t_ids[0]].item())
                        for k_pos in range(1, len(t_ids)):
                            tot_lp += float(suf_lps[b_i, k_pos - 1, t_ids[k_pos]].item())
                        lp_out[act] = tot_lp / float(len(t_ids))
                    del chunk_cache, suf_res, suf_lps
                del prefill, base_cache
        finally:
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, self.device)

        self._synchronize()
        elapsed = max(0.0, monotonic_seconds() - started)
        if self.meter and charged_tokens > 0:
            self.meter.record_model_call(0, 0, elapsed, role, forward_tokens=charged_tokens)

        for act in unique_acts:
            lp_out.setdefault(act, -6.0)
        return lp_out

    @torch.no_grad()
    def build_dcc_decision_context(
        self,
        *,
        benchmark: str,
        goal_text: str,
        events: Sequence[Any],
        action_schema: Mapping[str, Any],
        remaining_budget_ratio: float,
        remaining_steps_ratio: float,
        max_env_steps: int,
        member_index: int | None = None,
        controller_mode: str | None = None,
    ):
        """Construct the single canonical read-only `DecisionContext` shared across training, diagnostics, and deployment."""
        from .native_dcc import TAIL_POLICY_VERSION, DecisionContext, PublicBenchmarkBinder

        clean_bench = str(benchmark).split(":", 1)[0]
        pub_fields = PublicBenchmarkBinder.extract_public_fields(
            clean_bench, goal_text, events, action_schema
        )
        r_bindings = PublicBenchmarkBinder.extract_role_bindings(
            clean_bench, goal_text, events, action_schema
        )
        t_slots = PublicBenchmarkBinder.encode_typed_field_slots(pub_fields)
        rem_sec = float(min(1.0, max(0.0, remaining_budget_ratio)))
        eff_max = max(1, int(max_env_steps))
        step_frac = 1.0 / float(eff_max)
        rem_steps = float(min(1.0, max(0.0, remaining_steps_ratio)))
        frozen_ctx = self.extract_frozen_semantic_context(goal_text, pub_fields)
        frozen_tail_vec = self.extract_frozen_action_embeddings([TAIL_POLICY_VERSION])[0]
        eff_mode = str(controller_mode or getattr(self, "dcc_controller_mode", "expected_utility"))
        return DecisionContext(
            benchmark=clean_bench,
            goal_text=str(goal_text),
            public_fields=dict(pub_fields),
            typed_slots=t_slots,
            role_bindings=dict(r_bindings),
            action_schema=action_schema,
            remaining_budget_ratio=rem_sec,
            spent_seconds_ratio=float(1.0 - rem_sec),
            remaining_steps_ratio=rem_steps,
            step_budget_fraction=step_frac,
            frozen_context_vec=frozen_ctx,
            frozen_tail_action_vec=frozen_tail_vec,
            member_index=member_index,
            controller_mode=eff_mode,
        )

    def snapshot_dcc_controller_state(self) -> dict[str, Any]:
        """Capture complete intra-task controller state (belief, residual program state, bindings, events)."""
        return {
            "belief": (
                self.dcc_active_belief.detach().clone()
                if self.dcc_active_belief is not None else None
            ),
            "program_state": (
                self.dcc_active_program_state.clone()
                if self.dcc_active_program_state is not None else None
            ),
            "member_index": self.dcc_active_member_index,
            "episode_events": list(self.dcc_episode_events),
            "task_instruction": str(self.dcc_task_instruction),
            "remaining_budget_ratio": float(self.dcc_remaining_budget_ratio),
            "eval_variant": str(self.dcc_eval_variant),
        }

    def restore_dcc_controller_state(self, state: Mapping[str, Any]) -> None:
        """Restore complete intra-task controller state at a post-q branch checkpoint."""
        belief = state.get("belief")
        self.dcc_active_belief = belief.to(self.device).clone() if isinstance(belief, torch.Tensor) else None
        prog_state = state.get("program_state")
        self.dcc_active_program_state = prog_state.clone() if prog_state is not None else None
        self.dcc_active_member_index = state.get("member_index")
        self.dcc_episode_events = list(state.get("episode_events", ()))
        self.dcc_task_instruction = str(state.get("task_instruction", self.dcc_task_instruction))
        self.dcc_remaining_budget_ratio = float(state.get("remaining_budget_ratio", 1.0))

    @torch.no_grad()
    def observe_dcc_transition(
        self,
        benchmark: str,
        task_instruction: str,
        events: Sequence[Any],
        action_schema: Mapping[str, Any],
        *,
        actual_action: str,
        remaining_budget_ratio: float = 1.0,
        spent_seconds_ratio: float | None = None,
        remaining_steps_ratio: float = 1.0,
        had_error: bool = False,
        terminal_done: bool = False,
        terminal_success: bool = False,
    ) -> dict[str, Any]:
        """Unified intra-task update `next_state = observe(previous_state, actual_action, actual_public_response)`.

        Uses the episode's fixed `ModelView` and full pre-action `ResourceState` + `role_bindings`
        so online filtering matches offline NLL training and lookahead bit-for-bit.
        """
        from .native_dcc import PublicBenchmarkBinder

        if self.dcc_active_belief is None:
            self.reset_dcc_episode(benchmark, task_instruction=task_instruction)
        pre_events = list(events[:-1]) if events else []
        pre_fields = PublicBenchmarkBinder.extract_public_fields(
            benchmark, task_instruction, pre_events, action_schema
        )
        pre_bindings = PublicBenchmarkBinder.extract_role_bindings(
            benchmark, task_instruction, pre_events, action_schema
        )
        post_fields = PublicBenchmarkBinder.extract_public_fields(
            benchmark, task_instruction, events, action_schema
        )
        rem_sec = float(min(1.0, max(0.0, remaining_budget_ratio)))
        spent_sec = float(1.0 - rem_sec if spent_seconds_ratio is None else min(1.0, max(0.0, spent_seconds_ratio)))
        rem_steps = float(min(1.0, max(0.0, remaining_steps_ratio)))
        frozen_ctx = self.extract_frozen_semantic_context(task_instruction, pre_fields)
        frozen_act = self.extract_frozen_action_embeddings([actual_action], role_bindings=pre_bindings)[0]
        active_view = self.get_active_dcc_model_view()
        next_belief, outcome_bin, outcome_probs = active_view.observe(
            self.dcc_active_belief,
            goal_text=task_instruction,
            pre_public_fields=pre_fields,
            actual_action=actual_action,
            post_public_fields=post_fields,
            remaining_budget_ratio=rem_sec,
            spent_seconds_ratio=spent_sec,
            remaining_steps_ratio=rem_steps,
            role_bindings=pre_bindings,
            frozen_context_vec=frozen_ctx,
            frozen_action_vec=frozen_act,
            had_error=had_error,
            terminal_done=terminal_done,
            terminal_success=terminal_success,
        )
        self.dcc_active_belief = next_belief
        self.dcc_episode_events = list(events)
        self.dcc_task_instruction = str(task_instruction)
        self.dcc_remaining_budget_ratio = float(remaining_budget_ratio)
        if self.dcc_active_program_state is not None:
            raw_obs = str(getattr(events[-1], "observation", "")) if events else ""
            self.dcc_active_program_state.record_call_return(raw_obs, post_fields)
            self.dcc_active_program_state.bound_variables.update(
                PublicBenchmarkBinder.extract_role_bindings(
                    benchmark, task_instruction, events, action_schema
                )
            )
            if not self.dcc_active_program_state.remaining_templates():
                self.dcc_active_program_state.completed = True
        return {
            "pre_public_fields": pre_fields,
            "post_public_fields": post_fields,
            "outcome_bin": int(outcome_bin),
            "mean_predicted_prob": float(outcome_probs[:, outcome_bin].mean().item()),
            "frozen_context_vec": frozen_ctx,
            "frozen_action_vec": frozen_act,
        }

    def configure_dcc_trainable_mode(self, dcc_only: bool) -> None:
        """Freeze 7B base/LoRA when training DCC shared response model K_phi; restore for other baselines."""
        for name, param in self.model.named_parameters():
            param.requires_grad_(False if dcc_only else (name in self._lora_trainable_names))
        for param in self.value_head.parameters():
            param.requires_grad_(not dcc_only)
        for param in self.dcc_ensemble.parameters():
            param.requires_grad_(bool(dcc_only))

    def _build_bound_native_generation(
        self,
        messages: Sequence[Mapping[str, Any]],
        action_schema: Mapping[str, Any],
        concrete_action_text: str,
        *,
        role: str,
        greedy: bool,
        planning_elapsed: float = 0.0,
    ) -> NativeGeneration:
        """Construct a valid NativeGeneration for a concrete action bound by a program or candidate planner."""
        from .native_text_actions import TEXT_ACTOR_CONTRACT, native_text_mode, text_available_modes

        public_schema = deepcopy(dict(action_schema))
        inferred_mode = native_text_mode(concrete_action_text, public_schema)
        feasible_modes = list(text_available_modes(public_schema)) if public_schema.get("classic_text") else []
        selected_mode = inferred_mode or (feasible_modes[0] if feasible_modes else "I")
        contract = (
            self._text_decoder_engine().native_contract(public_schema, selected_mode)
            if public_schema.get("classic_text") else None
        )
        prompt = (
            self._text_action_messages(messages, public_schema, selected_mode)
            if public_schema.get("classic_text")
            else self._native_action_messages(messages, public_schema)
        )
        prompt_ids = self._tokenize_messages(prompt)
        resp_ids = list(self.tokenizer.encode(concrete_action_text, add_special_tokens=False))
        if self.eos_token_ids and (not resp_ids or resp_ids[-1] not in self.eos_token_ids):
            resp_ids.append(int(self.eos_token_ids[0]))
        response_ids = tuple(int(t) for t in resp_ids)
        action = parse_native_action(concrete_action_text, mode=selected_mode, action_schema=public_schema)
        if action.get("kind") == "invalid" and public_schema.get("classic_text"):
            action = {
                "kind": "native_text",
                "text": concrete_action_text,
                "mode": selected_mode,
                "raw_text": concrete_action_text,
            }
        gate_active = len(feasible_modes) == 2
        mode_prompt_ids = (
            self._tokenize_messages(self._mode_messages(messages, public_schema))
            if gate_active else ()
        )
        metadata = {
            "prompt_version": PROMPT_VERSION,
            "temperature": 1.0 if greedy else self.temperature,
            "greedy": bool(greedy),
            "sampling_log_probs": [0.0 for _ in response_ids],
            "max_new_tokens": self.max_new_tokens,
            "requested_max_new_tokens": self.max_new_tokens,
            "output_limit_reached": False,
            "action_boundary_reached": True,
            "response_tokens_finite": True,
            "eos_token_ids": list(self.eos_token_ids),
            "eos_reached": True,
            "wall_deadline_reached": False,
            "generation_limit_sources": ["max_new_tokens"],
            "generation_stop_reason": "action_complete",
            "model": dict(self.model_metadata),
            "reference_mode": self._active_reference or self._reference_label,
            "reference_active": bool(self._reference_depth),
            "actor_contract": TEXT_ACTOR_CONTRACT if public_schema.get("classic_text") else "prism-dcc-v2",
            "classic_text": bool(public_schema.get("classic_text")),
            "actor_gated": True,
            "gate_active": gate_active,
            "mode_sampled": False,
            "mode_temperature": 1.0 if greedy else self.temperature,
            "mode_log_probability": math.log(0.5) if gate_active else 0.0,
            "gate_probability": 0.5 if gate_active else float(selected_mode == "I"),
            "forced_mode": None,
            "action_schema": public_schema,
            "native_action_schema": public_schema,
            "text_action_complete": True,
            "text_action_contract": contract,
            "feasible_modes": feasible_modes,
            "program_bound_execution": True,
        }
        return NativeGeneration(
            text=concrete_action_text,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            action=action,
            mode=selected_mode,
            mode_prompt_ids=mode_prompt_ids,
            mode_token_id=self.mode_token_ids.get(selected_mode) if gate_active else None,
            input_tokens=len(prompt_ids) + len(mode_prompt_ids),
            output_tokens=len(response_ids) + int(gate_active),
            elapsed_seconds=planning_elapsed,
            role=role,
            metadata=metadata,
        )

    @torch.no_grad()
    def act(
        self, messages: Sequence[Mapping[str, Any]], action_schema: Any, *,
        forced_mode: str | None = None, greedy: bool = False, role: str = "actor",
        max_new_tokens: int | None = None,
        events: Sequence[Any] | None = None,
        task_instruction: str | None = None,
        remaining_budget_ratio: float | None = None,
        remaining_steps_ratio: float | None = None,
        max_env_steps: int | None = None,
        forced_program: Any | None = None,
        ungated_union: bool = False,
    ) -> NativeGeneration:
        """Select and execute the best action or condition program under unified terminal Q_m^rho."""
        if forced_mode not in (None, "I", "E"):
            raise ValueError("forced_mode must be I, E, or None.")
        if action_schema.get("classic_text"):
            from .native_text_actions import text_available_modes
            feasible = text_available_modes(action_schema)
        else:
            from .native_program import available_modes
            feasible = tuple(available_modes(action_schema["program_context"]))
        if not feasible:
            raise NoFeasibleAction("No public action is currently supported.")
        if forced_mode is not None and forced_mode not in feasible:
            raise NoFeasibleAction(f"Forced mode {forced_mode} has no ready public action.")

        gate_active = len(feasible) == 2
        mode_prompt_ids: tuple[int, ...] = ()
        mode_elapsed = 0.0
        mode_temp = 1.0 if greedy else self.temperature
        dcc_decision: dict[str, Any] | None = None

        if self.dcc_mode_enabled:
            from .native_dcc import (
                DCC_CONTRACT_VERSION,
                TAIL_POLICY_VERSION,
                ExecutableProgram,
                PlanningDecision,
                ProgramExecutionState,
                PublicBenchmarkBinder,
                _short_hash,
            )

            planning_started = monotonic_seconds()
            benchmark = str(
                action_schema.get(
                    "benchmark",
                    action_schema.get("program_context", {}).get("benchmark", "alfworld"),
                )
            ).split(":", 1)[0]
            library = self.get_dcc_library(benchmark)
            if self.dcc_active_belief is None:
                self.reset_dcc_episode(benchmark)

            hist_events = list(events) if events is not None else list(self.dcc_episode_events)
            goal_text = (
                str(task_instruction)
                if task_instruction
                else self.dcc_task_instruction
                or PublicBenchmarkBinder.extract_task_instruction(messages)
            )
            rem_ratio = (
                float(remaining_budget_ratio)
                if remaining_budget_ratio is not None
                else float(self.dcc_remaining_budget_ratio)
            )
            eff_max_steps = max(1, int(max_env_steps) if max_env_steps is not None else 25)
            rem_steps_ratio = (
                float(remaining_steps_ratio)
                if remaining_steps_ratio is not None
                else max(0.0, 1.0 - (len(hist_events) / float(eff_max_steps)))
            )
            step_budget_frac = 1.0 / float(eff_max_steps)
            public_fields = PublicBenchmarkBinder.extract_public_fields(
                benchmark, goal_text, hist_events, action_schema
            )
            role_bindings = PublicBenchmarkBinder.extract_role_bindings(
                benchmark, goal_text, hist_events, action_schema
            )
            use_init_params = self.dcc_eval_variant in ("reset_parameters", "k0_planner")
            starter_only = self.dcc_eval_variant in ("no_program_growth", "k0_planner")

            # If a forced_program is actively executing during probe collection, step its AST frame directly (fixes F24)
            if forced_program is not None:
                prog_frame = (
                    forced_program
                    if isinstance(forced_program, ProgramExecutionState)
                    else ProgramExecutionState(program=forced_program, bound_variables=dict(role_bindings))
                )
                step_res = PublicBenchmarkBinder.step_ast_frame(
                    prog_frame, action_schema, goal_text, hist_events
                )
                self.dcc_active_program_state = step_res.frame
                if step_res.status == "Action" and step_res.concrete_action is not None and action_schema.get("classic_text"):
                    planning_elapsed = max(0.0, monotonic_seconds() - planning_started)
                    gen = self._build_bound_native_generation(
                        messages,
                        action_schema,
                        step_res.concrete_action,
                        role=f"{role}.action",
                        greedy=greedy,
                        planning_elapsed=planning_elapsed,
                    )
                    gen.metadata["dcc_decision"] = {
                        "selected_program_id": step_res.frame.program.program_id,
                        "selected_program_origin": step_res.frame.program.origin,
                        "selected_program_steps": list(step_res.frame.program.step_templates),
                        "selected_program_fields": list(step_res.frame.program.readout_fields),
                        "executed_concrete_action": step_res.concrete_action,
                        "program_interpreter_executed": True,
                        "program_interpreter_status": "Action",
                        "program_attributed": True,
                        "residual_program_state": step_res.frame.to_dict(),
                        "dcc_eval_variant": self.dcc_eval_variant,
                    }
                    return gen

            forced_program_failed = bool(
                forced_program is not None
                and self.dcc_active_program_state is not None
                and self.dcc_active_program_state.binding_failed
            )

            # Sample one concrete proposal from the frozen backbone prior p_0 over the full action union (mode=None, fixes F08)
            prior_generation: NativeGeneration | None = None
            if action_schema.get("classic_text"):
                prior_generation = self._generate_native_text(
                    messages,
                    action_schema,
                    mode=forced_mode,
                    greedy=greedy,
                    role=f"{role}.action",
                    max_new_tokens=max_new_tokens,
                )
            else:
                prior_generation = self._generate_program(
                    messages,
                    action_schema,
                    mode=forced_mode,
                    greedy=greedy,
                    role=f"{role}.action",
                    max_new_tokens=max_new_tokens,
                )

            prior_action_text = prior_generation.text.strip()
            controller_mode = str(getattr(self, "dcc_controller_mode", "expected_utility"))
            decision_ctx = self.build_dcc_decision_context(
                benchmark=benchmark,
                goal_text=goal_text,
                events=hist_events,
                action_schema=action_schema,
                remaining_budget_ratio=rem_ratio,
                remaining_steps_ratio=rem_steps_ratio,
                max_env_steps=eff_max_steps,
                member_index=self.dcc_active_member_index,
                controller_mode=controller_mode,
            )
            public_fields = decision_ctx.public_fields
            role_bindings = decision_ctx.role_bindings

            # Diagnostic tracking of historical zero-field-change actions (only excludes in archived `legacy_lcb_v3`)
            observed_self_loops: set[str] = set()
            for ev in hist_events:
                ev_meta = getattr(ev, "metadata", None)
                if isinstance(ev_meta, Mapping):
                    pre_f = ev_meta.get("dcc_pre_public_fields")
                    post_f = ev_meta.get("dcc_public_fields")
                    if (
                        isinstance(pre_f, Mapping)
                        and isinstance(post_f, Mapping)
                        and dict(pre_f) == dict(public_fields)
                        and dict(post_f) == dict(public_fields)
                    ):
                        ev_act = getattr(getattr(ev, "generation", None), "text", "")
                        if ev_act:
                            observed_self_loops.add(str(ev_act).strip())

            use_legacy_filter = controller_mode == "legacy_lcb_v3"

            # Build unified candidate pool E(h) = {admissible native actions} U {executable condition programs}
            # Each entry carries `(candidate_id, concrete_action, program_or_None, advanced_ExecutionFrame_or_None)`.
            candidates: list[tuple[str, str, ExecutableProgram | None, ProgramExecutionState | None]] = []
            seen_atomic_actions: set[str] = set()
            has_valid_prior = False

            # 1. Anchor prior-proposed atomic action at index 0 (`a_{pi_0} -> pi_0`)
            if prior_action_text and (not use_legacy_filter or prior_action_text not in observed_self_loops):
                candidates.append(
                    (f"native_prior:{_short_hash(prior_action_text)}", prior_action_text, None, None)
                )
                seen_atomic_actions.add(prior_action_text)
                has_valid_prior = True

            # 2. Residual program from previous step (preserving `local_env`, return bindings, and `node_stack`)
            if self.dcc_active_program_state is not None and not forced_program_failed:
                res_prog = self.dcc_active_program_state.residual_program()
                if res_prog is not None:
                    res_step = PublicBenchmarkBinder.step_ast_frame(
                        self.dcc_active_program_state, action_schema, goal_text, hist_events
                    )
                    if res_step.status == "Action" and res_step.concrete_action and (
                        not use_legacy_filter or res_step.concrete_action not in observed_self_loops
                    ):
                        candidates.append(
                            (res_prog.program_id, res_step.concrete_action, res_prog, res_step.frame)
                        )

            # 3. Active library programs bound in current public state
            seen_prog_ids: set[str] = {c[0] for c in candidates}
            progs = library.active_programs(starter_only=starter_only)
            for prog in progs:
                if prog.program_id in seen_prog_ids:
                    continue
                temp_frame = ProgramExecutionState(program=prog, bound_variables=dict(role_bindings))
                step_res = PublicBenchmarkBinder.step_ast_frame(
                    temp_frame, action_schema, goal_text, hist_events
                )
                if step_res.status == "Action" and step_res.concrete_action and (
                    not use_legacy_filter or step_res.concrete_action not in observed_self_loops
                ):
                    # Single-Call starter programs identical to an already-included atomic action add no distinct trajectory;
                    # multi-step or Case programs are always evaluated alongside atomic candidates!
                    is_multistep_or_case = (
                        len(step_res.frame.node_stack) > 0 or bool(prog.case_condition_field) or prog.origin != "starter"
                    )
                    if is_multistep_or_case or step_res.concrete_action not in seen_atomic_actions:
                        candidates.append((prog.program_id, step_res.concrete_action, prog, step_res.frame))
                        seen_prog_ids.add(prog.program_id)

            # 4. Complete unbiased admissible native actions (no [:32] truncation on finite ALFWorld/WebShop action sets;
            #    LOCA open parameter space discloses bounded proposal support & tool coverage)
            adm_full = PublicBenchmarkBinder.enumerate_admissible_actions(
                benchmark, action_schema, goal_text, role_bindings
            )
            adm_list = adm_full[:32] if use_legacy_filter else adm_full
            for adm_act in adm_list:
                if adm_act not in seen_atomic_actions and (
                    not use_legacy_filter or adm_act not in observed_self_loops
                ):
                    candidates.append((f"native_atomic:{_short_hash(adm_act)}", adm_act, None, None))
                    seen_atomic_actions.add(adm_act)
            if not candidates and prior_action_text:
                candidates.append((f"native_prior:{_short_hash(prior_action_text)}", prior_action_text, None, None))

            norm_cand_texts = [
                PublicBenchmarkBinder.normalize_action_for_semantic_encoding(c[1], role_bindings)
                for c in candidates
            ]
            cand_action_vecs = self.extract_frozen_action_embeddings(norm_cand_texts)
            embed_cache: dict[str, tuple[float, ...]] = dict(zip(norm_cand_texts, cand_action_vecs))
            embed_cache[TAIL_POLICY_VERSION] = decision_ctx.frozen_tail_action_vec

            prior_lp_map = self.score_frozen_action_logprobs(
                messages,
                action_schema,
                [c[1] for c in candidates],
                prompt_ids=prior_generation.prompt_ids,
                prefill_already_charged=True,
            )

            def _lookup_action_embed(a_str: str) -> tuple[float, ...]:
                norm_a = PublicBenchmarkBinder.normalize_action_for_semantic_encoding(a_str, role_bindings)
                if norm_a not in embed_cache:
                    embed_cache[norm_a] = self.extract_frozen_action_embeddings([norm_a])[0]
                return embed_cache[norm_a]

            active_view = self.get_active_dcc_model_view()
            scored = active_view.evaluate_decision_candidates(
                decision_ctx,
                self.dcc_active_belief,
                candidates,
                action_embedder=_lookup_action_embed,
                precomputed_action_vecs=embed_cache,
                precomputed_prior_logprobs=prior_lp_map,
                use_initial_params=False,
            )

            cand_ids = [str(item["candidate_id"]) for item in scored]
            cand_acts = [str(item["candidate_action"]) for item in scored]
            cand_qs = [float(item["q_selected"]) for item in scored]
            cand_q_means = [float(item["q_mean"]) for item in scored]
            cand_member_qs = [tuple(item["q_members"]) for item in scored]

            # Primary `expected_utility` controller selects argmax_rho (1/M) sum_m Q_m^rho (1e-7 float32 tie epsilon
            # preserves index-0 prior proposal under exact algebraic equality at phi_0 without arbitrary safety walls).
            adv_margin = (
                0.002
                if (controller_mode == "legacy_lcb_v3" and self.dcc_active_member_index is None and has_valid_prior)
                else 1e-7
            )
            best_idx = 0
            best_eff_q = cand_qs[0] if cand_qs else -1e9
            for idx in range(1, len(cand_qs)):
                eff_score = cand_qs[idx] - adv_margin
                if eff_score > best_eff_q:
                    best_eff_q = cand_qs[idx]
                    best_idx = idx
            best_q = cand_qs[best_idx] if cand_qs else 0.0

            sel_id, sel_act, sel_prog, sel_next_frame = candidates[best_idx]
            if sel_next_frame is not None and not sel_next_frame.completed and not sel_next_frame.binding_failed:
                self.dcc_active_program_state = sel_next_frame
            else:
                self.dcc_active_program_state = None

            planning_elapsed = max(0.0, monotonic_seconds() - planning_started)
            loca_tools_total = len(PublicBenchmarkBinder._extract_loca_tools(action_schema)) if benchmark == "loca" else 0
            plan_record = PlanningDecision(
                decision_id=_short_hash((goal_text, len(hist_events), sel_id, sel_act)),
                task_id=goal_text[:48],
                step_index=len(hist_events),
                candidate_ids=tuple(cand_ids),
                candidate_actions=tuple(cand_acts),
                predicted_q_values=tuple(cand_qs),
                member_q_values=tuple(cand_member_qs),
                model_version="phi_0" if use_init_params else DCC_CONTRACT_VERSION,
                tail_policy_version=TAIL_POLICY_VERSION,
                planning_compute_cost={
                    "candidates_evaluated": len(candidates),
                    "admissible_native_actions_total": len(adm_full),
                    "admissible_native_actions_scored": len(adm_list),
                    "loca_schema_tools_total": loca_tools_total,
                    "planning_seconds": planning_elapsed,
                    "backbone_input_tokens": int(prior_generation.input_tokens) + len(mode_prompt_ids),
                    "backbone_output_tokens": int(prior_generation.output_tokens) + int(bool(mode_prompt_ids)),
                },
                selected_candidate_id=sel_id,
                executed_action=sel_act,
                residual_program_state=(
                    self.dcc_active_program_state.to_dict()
                    if self.dcc_active_program_state is not None else None
                ),
            )
            dcc_decision = {
                "selected_program_id": sel_prog.program_id if sel_prog is not None else sel_id,
                "selected_program_origin": sel_prog.origin if sel_prog is not None else "native_atomic",
                "selected_program_steps": list(sel_prog.step_templates) if sel_prog is not None else [sel_act],
                "selected_program_fields": list(sel_prog.readout_fields) if sel_prog is not None else [],
                "executed_concrete_action": sel_act,
                "prior_action": prior_action_text,
                "prior_q_mean": cand_q_means[0] if cand_q_means else 0.0,
                "selected_q_mean": cand_q_means[best_idx] if cand_q_means else 0.0,
                "selected_prior_logprob": float(prior_lp_map.get(sel_act, 0.0)),
                "controller_mode": controller_mode,
                "observed_field_unchanged_actions": sorted(observed_self_loops),
                "program_interpreter_executed": bool(sel_prog is not None),
                "program_interpreter_status": "BindingFailureFallback" if forced_program_failed else (
                    "Action" if sel_prog is not None else "AtomicAction"
                ),
                "program_attributed": bool(sel_prog is not None and not forced_program_failed),
                "dcc_eval_variant": self.dcc_eval_variant,
                "planning_decision": plan_record.to_dict(),
                "unified_q": best_q,
            }

            if sel_act != prior_action_text and action_schema.get("classic_text"):
                bound_gen = self._build_bound_native_generation(
                    messages,
                    action_schema,
                    sel_act,
                    role=role,
                    greedy=greedy,
                    planning_elapsed=prior_generation.elapsed_seconds + mode_elapsed,
                )
                bound_gen.metadata["dcc_decision"] = dcc_decision
                return replace(
                    bound_gen,
                    input_tokens=prior_generation.input_tokens + len(mode_prompt_ids),
                    output_tokens=prior_generation.output_tokens + int(bool(mode_prompt_ids)),
                )

            metadata = dict(
                prior_generation.metadata,
                actor_gated=True,
                gate_active=gate_active,
                mode_sampled=gate_active and forced_mode is None,
                mode_temperature=mode_temp,
                mode_log_probability=0.0,
                gate_probability=float(prior_generation.mode == "I"),
                forced_mode=forced_mode,
                dcc_decision=dcc_decision,
            )
            return replace(
                prior_generation,
                mode_prompt_ids=mode_prompt_ids,
                mode_token_id=self.mode_token_ids.get(prior_generation.mode) if gate_active else None,
                input_tokens=prior_generation.input_tokens + len(mode_prompt_ids),
                output_tokens=prior_generation.output_tokens + int(bool(mode_prompt_ids)),
                elapsed_seconds=prior_generation.elapsed_seconds + mode_elapsed,
                role=role,
                metadata=metadata,
            )

        if ungated_union:
            if action_schema.get("classic_text"):
                generation = self._generate_native_text(
                    messages,
                    action_schema,
                    mode=None,
                    greedy=greedy,
                    role=f"{role}.action",
                    max_new_tokens=max_new_tokens,
                )
            else:
                generation = self._generate_program(
                    messages,
                    action_schema,
                    mode=None,
                    greedy=greedy,
                    role=f"{role}.action",
                    max_new_tokens=max_new_tokens,
                )
            metadata = dict(
                generation.metadata,
                actor_gated=False,
                gate_active=False,
                mode_sampled=False,
                mode_temperature=mode_temp,
                mode_log_probability=0.0,
                gate_probability=0.5,
                forced_mode=None,
            )
            return replace(
                generation,
                mode_prompt_ids=(),
                mode_token_id=None,
                role=role,
                metadata=metadata,
            )

        if gate_active:
            mode_prompt_ids = self._tokenize_messages(self._mode_messages(messages, action_schema))
            mode_log_probs, mode_elapsed = self._mode_log_probs(
                mode_prompt_ids, role=f"{role}.mode", generating=True, temperature=mode_temp,
            )
            if forced_mode is not None:
                mode = forced_mode
            elif greedy:
                mode = "I" if int(mode_log_probs.argmax().item()) == 0 else "E"
            else:
                mode = "I" if int(torch.multinomial(mode_log_probs.exp(), 1).item()) == 0 else "E"
            probability = float(mode_log_probs[0].exp().item())
            mode_log_probability = float(mode_log_probs[0 if mode == "I" else 1].item())
        else:
            mode = feasible[0]
            probability = float(mode == "I")
            mode_log_probability = 0.0
        generator = self._generate_native_text if action_schema.get("classic_text") else self._generate_program
        generation = generator(messages, action_schema, mode=mode, greedy=greedy,
                               role=f"{role}.action", max_new_tokens=max_new_tokens)
        metadata = dict(generation.metadata, actor_gated=True, gate_active=gate_active,
                        mode_sampled=gate_active and forced_mode is None,
                        mode_temperature=mode_temp, mode_log_probability=mode_log_probability,
                        gate_probability=probability, forced_mode=forced_mode)
        if dcc_decision is not None:
            metadata["dcc_decision"] = dcc_decision
        return replace(generation, mode=mode, mode_prompt_ids=mode_prompt_ids,
                       mode_token_id=self.mode_token_ids[mode] if gate_active else None,
                       input_tokens=generation.input_tokens + len(mode_prompt_ids),
                       output_tokens=generation.output_tokens + int(gate_active),
                       elapsed_seconds=generation.elapsed_seconds + mode_elapsed,
                       role=role, metadata=metadata)

    @staticmethod
    def _counterfactual_schema(generation: NativeGeneration, messages) -> dict[str, Any]:
        if generation.metadata.get("classic_text"):
            from .native_text_envs import public_text_action_schema
            from .native_text_actions import text_available_modes
            from .native_program_decoder import ProgramOutOfSupport
            schema = public_text_action_schema(generation.metadata["action_schema"], messages)
            feasible = text_available_modes(schema)
            if not feasible or (generation.mode is not None and generation.mode not in feasible):
                token = int(generation.response_ids[0]) if generation.response_ids else -1
                error = ProgramOutOfSupport(0, token, generation.metadata["text_action_contract"])
                error.details.update(support_scope="public_native_action_class", feasible_modes=list(feasible))
                raise error
            return schema
        from .native_envs import public_action_schema
        from .native_program import available_modes
        from .native_program_decoder import ProgramOutOfSupport
        schema = public_action_schema(generation.metadata["action_schema"], messages)
        context = schema["program_context"]
        context["generation_token_budget"] = generation.metadata["program_context"]["generation_token_budget"]
        feasible = tuple(available_modes(context))
        if not feasible or (generation.mode is not None and generation.mode not in feasible):
            token = int(generation.response_ids[0]) if generation.response_ids else -1
            error = ProgramOutOfSupport(0, token, generation.metadata["program_contract"])
            error.details["support_scope"] = "public_action_class"
            error.details["feasible_modes"] = list(feasible)
            raise error
        return schema

    def _score_prompts(
        self, generation: NativeGeneration, messages: Sequence[Mapping[str, Any]] | None,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if messages is None:
            return tuple(generation.prompt_ids), tuple(generation.mode_prompt_ids)
        if generation.metadata.get("actor_contract") == "official-native-action-v1":
            schema = generation.metadata.get("native_text_prompt_schema")
            if schema is not None:
                # Counterfactual scoring must use the sampling prompt serializer.
                prompt = self._text_action_messages(messages, schema, None)
                return self._tokenize_messages(prompt), ()
            prompt = [{"role": "system", "content": generation.metadata["native_instruction"]},
                      *[dict(message) for message in messages if message.get("role") != "system"]]
            return self._tokenize_messages(prompt), ()
        if "action_schema" not in generation.metadata:
            return self._tokenize_messages(messages), ()
        if generation.metadata.get("classic_text"):
            from .native_text_actions import text_available_modes
            schema = self._counterfactual_schema(generation, messages)
            action_prompt = self._tokenize_messages(self._text_action_messages(messages, schema, generation.mode))
            gate_prompt = (self._tokenize_messages(self._mode_messages(messages, schema))
                           if len(text_available_modes(schema)) == 2 else ())
            return action_prompt, gate_prompt
        from .native_program import available_modes
        schema = self._counterfactual_schema(generation, messages)
        if not generation.metadata.get("actor_gated", generation.mode is not None):
            return self._tokenize_messages(self._native_action_messages(messages, schema)), ()
        action_prompt = self._tokenize_messages(self._action_messages(messages, schema, generation.mode))
        gate_prompt = (self._tokenize_messages(self._mode_messages(messages, schema))
                       if len(available_modes(schema["program_context"])) == 2 else ())
        return action_prompt, gate_prompt

    def gate_probability(
        self, generation: NativeGeneration | Sequence[int], *,
        messages: Sequence[Mapping[str, Any]] | None = None,
        reference: bool = False, role: str = "score.gate",
    ) -> torch.Tensor:
        if reference:
            with self.frozen_reference(), torch.no_grad():
                return self.gate_probability(generation, messages=messages, role=f"reference.{role}")
        if isinstance(generation, NativeGeneration):
            if not generation.metadata.get("actor_gated", generation.mode is not None):
                raise ValueError("An ungated action has no gate probability.")
            _, prompt = self._score_prompts(generation, messages)
            temp = float(generation.metadata.get("mode_temperature", self.temperature))
            if not prompt:
                if generation.metadata.get("classic_text"):
                    from .native_text_actions import text_available_modes
                    schema = (generation.metadata["action_schema"] if messages is None else
                              self._counterfactual_schema(generation, messages))
                    modes = text_available_modes(schema)
                else:
                    from .native_program import available_modes
                    context = (generation.metadata["program_context"] if messages is None else
                               self._counterfactual_schema(generation, messages)["program_context"])
                    modes = tuple(available_modes(context))
                if len(modes) != 1:
                    raise ValueError("A missing gate prompt requires a single feasible action class.")
                return torch.tensor(float(modes[0] == "I"), device=self.device, dtype=torch.float32)
        else:
            if messages is not None:
                raise ValueError("Changed messages require a generation with its public schema.")
            prompt = tuple(int(item) for item in generation)
            temp = self.temperature
        if not prompt:
            raise ValueError("This state does not contain a gate prompt.")
        values, _ = self._mode_log_probs(prompt, role=role, temperature=temp)
        return values[0].exp()

    def score_tokens(
        self, generation: NativeGeneration, messages: Sequence[Mapping[str, Any]] | None = None,
        include_mode: bool = True, reference: bool = False, role: str = "score",
    ) -> torch.Tensor:
        """Return response-token log probabilities, optionally preceded by mode log-p.

        The score has no prompt-token terms, appended EOS, length normalization,
        clipping or intrinsic reward. ``messages`` replaces only public context;
        the scored next-action tokens remain exactly those originally sampled.
        """
        if reference:
            with self.frozen_reference(), torch.no_grad():
                return self.score_tokens(generation, messages, include_mode, role=f"reference.{role}")
        prompt_ids, mode_prompt = self._score_prompts(generation, messages)
        response_ids = tuple(generation.response_ids)
        if not response_ids:
            raise ValueError("Cannot score a generation without its exact response token IDs.")
        self._check_context(len(prompt_ids), len(response_ids))
        body_inputs = prompt_ids + response_ids[:-1]
        temp = float(generation.metadata.get("temperature", 1.0))
        gate_active = bool(generation.metadata.get("gate_active", bool(generation.mode_prompt_ids)))
        mode_tokens = len(mode_prompt) if include_mode and gate_active else 0
        program_contract = generation.metadata.get("text_action_contract", generation.metadata.get("program_contract"))
        program_mask = None
        if program_contract is not None:
            from .native_program_decoder import ProgramOutOfSupport
            decoder = self._action_decoder_engine(program_contract)
            counterfactual = False
            if messages is not None:
                schema = self._counterfactual_schema(generation, messages)
                if generation.metadata.get("classic_text"):
                    replacement = decoder.native_contract(schema, generation.mode)
                else:
                    from .native_program import program_schema
                    context = schema["program_context"]
                    replacement = decoder.contract(program_schema(context, generation.mode), context, generation.mode)
                counterfactual = replacement["schema_json"] != program_contract["schema_json"]
                program_contract = replacement
            try:
                program_mask = decoder.scoring_mask(program_contract, response_ids,
                                                     counterfactual=counterfactual)
            except ProgramOutOfSupport as error:
                generation.metadata.setdefault("counterfactual_support", []).append(dict(error.details))
                raise
        if self.meter:
            self.meter.ensure_forward(len(body_inputs) + mode_tokens, role)
        self._synchronize()
        started = monotonic_seconds()
        with self._activation_saving(), self._model_mode(scoring=True):
            logits = self._forward_logits(body_inputs, len(response_ids)).float() / temp
            if program_mask is not None:
                logits = decoder.mask_logits(logits, program_mask)
            targets = torch.tensor(response_ids, dtype=torch.long, device=logits.device)
            token_log_probs = -F.cross_entropy(logits, targets, reduction="none")
        self._synchronize()
        elapsed = max(0.0, monotonic_seconds() - started)
        if self.meter:
            self.meter.record_model_call(0, 0, elapsed, role, forward_tokens=len(body_inputs))
        if include_mode and gate_active:
            if generation.mode not in self.mode_token_ids:
                raise ValueError("Mode metadata must match the recorded action class.")
            if generation.mode_token_id != self.mode_token_ids[generation.mode]:
                raise ValueError("Recorded mode token does not match this policy tokenizer.")
            if mode_prompt:
                mode_temp = float(generation.metadata.get("mode_temperature", self.temperature))
                log_modes, _ = self._mode_log_probs(mode_prompt, role=f"{role}.gate", temperature=mode_temp)
                selected = log_modes[0 if generation.mode == "I" else 1]
            else:
                # A masked history may leave just this class feasible. Its
                # probability is exactly one under that changed public support.
                selected = torch.zeros((), device=token_log_probs.device, dtype=token_log_probs.dtype)
            token_log_probs = torch.cat((selected.reshape(1), token_log_probs))
        if not torch.isfinite(token_log_probs).all():
            raise FloatingPointError("Recorded response tokens have non-finite policy log probabilities.")
        return token_log_probs

    def mode_score(
        self, generation: NativeGeneration, *, reference: bool = False, role: str = "score.gate",
    ) -> torch.Tensor:
        if reference:
            with self.frozen_reference(), torch.no_grad():
                return self.mode_score(generation, role=f"reference.{role}")
        if not generation.metadata.get("gate_active", bool(generation.mode_prompt_ids)):
            return torch.zeros((), device=self.device, dtype=torch.float32)
        _, mode_prompt = self._score_prompts(generation, None)
        if generation.mode not in self.mode_token_ids or not mode_prompt:
            raise ValueError("Mode metadata must match a recorded mode prompt.")
        if generation.mode_token_id != self.mode_token_ids[generation.mode]:
            raise ValueError("Recorded mode token does not match this policy tokenizer.")
        mode_temp = float(generation.metadata.get("mode_temperature", self.temperature))
        log_modes, _ = self._mode_log_probs(mode_prompt, role=role, temperature=mode_temp)
        return log_modes[0 if generation.mode == "I" else 1]

    def score(
        self, generation: NativeGeneration, messages: Sequence[Mapping[str, Any]] | None = None,
        include_mode: bool = True, reference: bool = False, role: str = "score",
    ) -> torch.Tensor:
        if messages is not None:
            return self.score_counterfactual(generation, messages, include_mode=include_mode,
                                             reference=reference, role=role)["log_probability"]
        return self.score_tokens(generation, None, include_mode, reference, role).sum()

    def score_counterfactual(
        self, generation: NativeGeneration, messages: Sequence[Mapping[str, Any]], *,
        include_mode: bool = True, reference: bool = False, role: str = "score_counterfactual",
    ) -> dict[str, Any]:
        """Expose a changed-history support event before any consumer forms a loss."""
        from .native_program_decoder import ProgramOutOfSupport
        try:
            tokens = self.score_tokens(generation, messages, include_mode, reference, role)
        except ProgramOutOfSupport as error:
            return {"status": "out_of_support",
                    "log_probability": torch.full((), -torch.inf, device=self.device, dtype=torch.float32),
                    "first_disallowed_token": error.details["first_disallowed_token"],
                    "token_log_probs": None, "support_event": dict(error.details)}
        return {"status": "supported", "log_probability": tokens.sum(),
                "first_disallowed_token": None, "token_log_probs": tokens}

    @torch.no_grad()
    def score_no_grad(
        self, generation: NativeGeneration, messages: Sequence[Mapping[str, Any]] | None = None,
        include_mode: bool = True, reference: bool = False, role: str = "score_no_grad",
    ) -> torch.Tensor:
        return self.score(generation, messages, include_mode, reference, role)

    def action_log_prob(self, generation: NativeGeneration, **kwargs: Any) -> torch.Tensor:
        return self.score(generation, **kwargs)

    def state_value(self, generation: NativeGeneration, role: str = "critic") -> torch.Tensor:
        """Evaluate V(h) before action selection; only the value head receives gradients."""
        prompt_ids = tuple(generation.mode_prompt_ids or generation.prompt_ids)
        self._check_context(len(prompt_ids), 0)
        if self.meter:
            self.meter.ensure_forward(len(prompt_ids), role)
        hidden: list[torch.Tensor] = []

        def capture_last_hidden(module: nn.Module, arguments: tuple[torch.Tensor, ...]) -> None:
            # LM-head input is the final causal hidden state. Capture only one
            # vector, avoiding materializing all transformer-layer activations.
            hidden.append(arguments[0][0, -1, :].detach().clone())

        hook = self.model.get_output_embeddings().register_forward_pre_hook(capture_last_hidden)
        self._synchronize()
        started = monotonic_seconds()
        try:
            with self._model_mode(), torch.no_grad():
                self._forward_logits(prompt_ids, 1)
        finally:
            hook.remove()
        self._synchronize()
        elapsed = max(0.0, monotonic_seconds() - started)
        if self.meter:
            self.meter.record_model_call(0, 0, elapsed, role, forward_tokens=len(prompt_ids))
        if len(hidden) != 1:
            raise RuntimeError("The causal model must expose one final LM-head hidden state.")
        state = hidden[0].to(device=self.value_head.weight.device, dtype=self.value_head.weight.dtype)
        return self.value_head(state).squeeze(-1)

    def save_adapter(self, directory: str | Path) -> None:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=False)
        self.model.save_pretrained(output, selected_adapters=["default"])
        self.tokenizer.save_pretrained(output)
        torch.save(self.value_head.state_dict(), output / "native_value_head.pt")
        torch.save(self.dcc_ensemble.state_dict(), output / "dcc_response_ensemble.pt")
        torch.save(
            self.dcc_ensemble.initial_state_dict_snapshot(),
            output / "dcc_initial_response_ensemble.pt",
        )
        (output / "dcc_program_libraries.json").write_text(
            json.dumps(
                {name: lib.to_dict() for name, lib in self.dcc_libraries.items()},
                indent=2,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
        if self._reference_adapter is not None:
            from peft.utils.save_and_load import get_peft_model_state_dict
            reference = {key: value.detach().cpu().clone() for key, value in
                         get_peft_model_state_dict(self.model, adapter_name=self._reference_adapter).items()}
            torch.save(reference, output / "warmstart_reference.pt")
        (output / "native_policy.json").write_text(json.dumps({
            "model_path": self.model_path, "prompt_version": PROMPT_VERSION,
            "mode_labels": self.mode_labels, "mode_token_ids": self.mode_token_ids,
            "temperature": self.temperature, "max_new_tokens": self.max_new_tokens,
            "max_context_tokens": self.max_context_tokens,
            "chat_template_kwargs": self.chat_template_kwargs,
            "model": self.model_metadata,
            "reference_mode": self._reference_label,
            "has_warmstart_reference": self._reference_adapter is not None,
            "activation_offload_cpu": self.activation_offload_cpu,
            "activation_offload_min_bytes": self.activation_offload_min_bytes,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def load_adapter(self, directory: str | Path) -> None:
        """Load weights into the existing adapter, preserving optimizer parameter identities."""
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

        source = Path(directory)
        weights = load_peft_weights(str(source), device=str(self.device))
        result = set_peft_model_state_dict(self.model, weights, adapter_name="default")
        unexpected = getattr(result, "unexpected_keys", ())
        missing_adapter = [key for key in getattr(result, "missing_keys", ())
                           if "lora_" in key and ".default." in key]
        if unexpected or missing_adapter:
            raise ValueError(f"Adapter weights mismatch: missing={missing_adapter}, unexpected={unexpected}")
        self.value_head.load_state_dict(torch.load(
            source / "native_value_head.pt", map_location=self.value_head.weight.device, weights_only=True,
        ))
        dcc_pt = source / "dcc_response_ensemble.pt"
        if dcc_pt.is_file():
            loaded_dcc = torch.load(
                dcc_pt, map_location=self.dcc_ensemble.transition_logits.device, weights_only=True,
            )
            current_dcc = self.dcc_ensemble.state_dict()
            compatible_dcc = {
                k: v for k, v in loaded_dcc.items()
                if k in current_dcc and current_dcc[k].shape == v.shape
            }
            current_dcc.update(compatible_dcc)
            self.dcc_ensemble.load_state_dict(current_dcc, strict=True)
        dcc_init_pt = source / "dcc_initial_response_ensemble.pt"
        if dcc_init_pt.is_file():
            loaded_init = torch.load(
                dcc_init_pt, map_location=self.dcc_ensemble.transition_logits.device, weights_only=True,
            )
            self.dcc_ensemble.load_initial_state_dict_snapshot(loaded_init)
        dcc_lib_path = source / "dcc_program_libraries.json"
        if dcc_lib_path.is_file():
            from .native_dcc import ProgramLibrary
            raw_libs = json.loads(dcc_lib_path.read_text(encoding="utf-8"))
            self.dcc_libraries.clear()
            for name, payload in raw_libs.items():
                lib = ProgramLibrary(name)
                lib.load_dict(payload)
                self.dcc_libraries[name] = lib
        # Clear intra-task DCC state whenever switching checkpoints so resident evaluation never leaks across cells (fixes A)
        self.dcc_active_belief = None
        self.dcc_active_program_state = None
        self.dcc_active_member_index = None
        self.dcc_episode_events = []
        self._dcc_context_cache_key = None
        self._dcc_context_cache_vec = ()
        metadata = json.loads((source / "native_policy.json").read_text(encoding="utf-8"))
        has_reference = metadata["has_warmstart_reference"]
        if not isinstance(has_reference, bool):
            raise ValueError("Checkpoint reference presence must be recorded explicitly.")
        if has_reference:
            weights = torch.load(source / "warmstart_reference.pt", map_location="cpu", weights_only=True)
            self._install_reference(weights, metadata["reference_mode"])
        else:
            if metadata["reference_mode"] != "raw_base":
                raise ValueError("A checkpoint declaring a warmstart reference must include its weights.")
            self._reference_adapter = None
            self._reference_label = "raw_base"
        self.model.set_adapter("default")

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "value_head"):
            del self.value_head
        if hasattr(self, "dcc_ensemble"):
            del self.dcc_ensemble
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
