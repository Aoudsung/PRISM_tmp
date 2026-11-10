"""XGrammar decoding and differentiable scoring of the same public program language.

API basis: https://xgrammar.mlc.ai/docs/latest/api/python/grammar_matcher.html
and mlc-ai/xgrammar v0.2.8 python/xgrammar/contrib/hf.py. A fresh matcher is
used for each token stream; only immutable compiled public contracts are cached.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from itertools import count
from importlib.metadata import version
import hashlib
import json
from typing import Any, Mapping, Sequence

import torch


DECODER_VERSION = "prism-xgrammar-r3-executed-state"
XGRAMMAR_VERSION = "0.2.8"


class ProgramOutOfSupport(ValueError):
    """A counterfactual public history excludes a previously sampled token."""
    def __init__(self, index: int, token_id: int, contract: Mapping[str, Any]) -> None:
        self.details = {"status": "out_of_support", "first_invalid_token_index": index,
                        "token_id": token_id, "contract_id": contract["contract_id"],
                        "context_id": contract["context_id"],
                        "first_disallowed_token": {"index": index, "token_id": token_id},
                        "sequence_probability": 0.0, "log_probability": None,
                        "log_probability_kind": "negative_infinity"}
        super().__init__(f"Counterfactual program support excludes response token {index} ({token_id}).")


class ProgramDecoder:
    def __init__(self, tokenizer: Any, vocab_size: int, eos_token_ids: Sequence[int]) -> None:
        import xgrammar as xgr
        self.xgrammar_version = version("xgrammar")
        if not eos_token_ids:
            raise ValueError("Program decoding requires the checkpoint's native EOS tokens.")
        self.xgr = xgr
        self.vocab_size = int(vocab_size)
        self.eos_token_ids = tuple(int(token) for token in eos_token_ids)
        info = xgr.TokenizerInfo.from_huggingface(
            tokenizer, vocab_size=self.vocab_size, stop_token_ids=list(self.eos_token_ids))
        self.compiler = xgr.GrammarCompiler(
            info, max_threads=4, cache_enabled=True, cache_limit_bytes=268435456,
        )
        self.compiled: OrderedDict[tuple, Any] = OrderedDict()
        self.contracts: OrderedDict[tuple, dict[str, Any]] = OrderedDict()
        self.masks: OrderedDict[tuple, torch.Tensor] = OrderedDict()
        self.contract_ids = count(1)

    def contract(self, schema: Mapping[str, Any], context: Mapping[str, Any], mode: str | None) -> dict[str, Any]:
        # Object-key order is a grammar property. Keep it in an immutable string.
        schema_json = json.dumps(schema, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        # The public binding ledger and frontier are part of the identity. A
        # changed observed value cannot inherit the previous action support.
        context_digest = hashlib.sha256(json.dumps(
            context, sort_keys=True, ensure_ascii=False, allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        key = (schema_json, context_digest, mode)
        if key in self.contracts:
            self.contracts.move_to_end(key)
            return deepcopy(self.contracts[key])
        number = next(self.contract_ids)
        contract = {
            "contract_id": f"contract-{number}", "context_id": f"context-{number}",
            "decoder_version": DECODER_VERSION, "xgrammar_version": self.xgrammar_version,
            "vocab_size": self.vocab_size,
            "eos_token_ids": list(self.eos_token_ids), "mode": mode,
            "schema_json": schema_json,
            "public_context_sha256": context_digest,
            "compile_options": {"any_whitespace": False, "strict_mode": True,
                                "any_order": False, "separators": [",", ":"]},
        }
        if key not in self.contracts:
            self.contracts[key] = contract
            if len(self.contracts) > 256:
                self.contracts.popitem(last=False)
        self.contracts.move_to_end(key)
        return deepcopy(self.contracts[key])

    def _compiled(self, contract: Mapping[str, Any]) -> Any:
        key = (contract["schema_json"], json.dumps(contract["compile_options"], sort_keys=True))
        if key not in self.compiled:
            options = dict(contract["compile_options"])
            options["separators"] = tuple(options["separators"])
            self.compiled[key] = self.compiler.compile_json_schema(
                contract["schema_json"], **options)
            if len(self.compiled) > 256:
                self.compiled.popitem(last=False)
        self.compiled.move_to_end(key)
        return self.compiled[key]

    def matcher(self, contract: Mapping[str, Any]) -> Any:
        return self.xgr.GrammarMatcher(self._compiled(contract),
                                       override_stop_tokens=list(self.eos_token_ids),
                                       terminate_without_stop_token=False)

    def processor(self, contract: Mapping[str, Any], prompt_length: int) -> "ProgramLogitsProcessor":
        return ProgramLogitsProcessor(self, contract, prompt_length)

    def mask_logits(self, logits: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
        """Torch's differentiable mask preserves gradients on exactly the allowed support."""
        if logits.shape[-1] != self.vocab_size:
            raise ValueError("Program mask and model vocabulary differ.")
        words = packed.to(device=logits.device)
        columns = torch.arange(self.vocab_size, device=logits.device)
        allowed = ((words[:, columns // 32] >> (columns % 32)) & 1).bool()
        return logits.masked_fill(~allowed, -torch.inf)

    def scoring_mask(self, contract: Mapping[str, Any], response_ids: Sequence[int], *,
                     counterfactual: bool = False) -> torch.Tensor:
        mask_key = (
            contract["schema_json"],
            json.dumps(contract["compile_options"], sort_keys=True),
            tuple(int(token) for token in response_ids),
        )
        if mask_key in self.masks:
            self.masks.move_to_end(mask_key)
            return self.masks[mask_key]
        matcher = self.matcher(contract)
        packed = self.xgr.allocate_token_bitmask(len(response_ids), self.vocab_size)
        packed.fill_(-1)
        for index, token in enumerate(response_ids):
            matcher.fill_next_token_bitmask(packed, index)
            if not matcher.accept_token(int(token)):
                if counterfactual:
                    raise ProgramOutOfSupport(index, int(token), contract)
                raise ValueError("Recorded response token lies outside its sampling-time program grammar.")
        self.masks[mask_key] = packed
        if len(self.masks) > 256:
            self.masks.popitem(last=False)
        return packed


class ProgramLogitsProcessor:
    def __init__(self, decoder: ProgramDecoder, contract: Mapping[str, Any], prompt_length: int) -> None:
        self.decoder = decoder
        self.contract = deepcopy(dict(contract))
        self.prompt_length = int(prompt_length)
        self.matcher = decoder.matcher(contract)
        self.packed = decoder.xgr.allocate_token_bitmask(1, decoder.vocab_size)
        self.consumed = 0

    def _accept(self, tokens: Sequence[int]) -> None:
        for token in tokens:
            if not self.matcher.accept_token(int(token)):
                raise RuntimeError("The sampler emitted a token excluded by its program mask.")
            self.consumed += 1

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[0] != 1:
            raise ValueError("Each native resident actor decodes one program at a time.")
        self._accept(input_ids[0, self.prompt_length + self.consumed:].tolist())
        self.packed.fill_(-1)
        self.matcher.fill_next_token_bitmask(self.packed)
        return self.decoder.mask_logits(scores, self.packed)

    def completed(self, response_ids: Sequence[int]) -> bool:
        """Recognize the real closing token without synthesizing an EOS or suffix."""
        self._accept(response_ids[self.consumed:])
        return bool(self.matcher.is_terminated() or self.matcher.is_completed())

    def finish(self, response_ids: Sequence[int]) -> dict[str, bool]:
        complete = self.completed(response_ids)
        return {"program_complete": complete,
                "program_eos_accepted": bool(self.matcher.is_terminated())}
