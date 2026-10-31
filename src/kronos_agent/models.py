"""Kronos model specifications (verified against the Hugging Face API on 2026-10-29).

Nothing in the weights enforces the tokenizer pairing or the context length (both tokenizers emit
10+10-bit tokens and ``max_context`` is not stored in ``config.json``), so this table is the
source of truth. Element counts are ``state_dict()`` totals (persistent buffers included), i.e.
the totals Hugging Face reports for ``model.safetensors``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo_id: str
    tokenizer_repo_id: str
    max_context: int
    d_model: int
    state_dict_numel: int
    tokenizer_state_dict_numel: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "small": ModelSpec(
        name="small",
        repo_id="NeoQuasar/Kronos-small",
        tokenizer_repo_id="NeoQuasar/Kronos-Tokenizer-base",
        max_context=512,
        d_model=512,
        state_dict_numel=24_741_696,
        tokenizer_state_dict_numel=3_958_258,
    ),
    "base": ModelSpec(
        name="base",
        repo_id="NeoQuasar/Kronos-base",
        tokenizer_repo_id="NeoQuasar/Kronos-Tokenizer-base",
        max_context=512,
        d_model=832,
        state_dict_numel=102_311_008,
        tokenizer_state_dict_numel=3_958_258,
    ),
    "mini": ModelSpec(
        name="mini",
        repo_id="NeoQuasar/Kronos-mini",
        tokenizer_repo_id="NeoQuasar/Kronos-Tokenizer-2k",
        max_context=2048,
        d_model=256,
        state_dict_numel=4_108_192,
        tokenizer_state_dict_numel=3_958_259,
    ),
}


def spec_for(name: str) -> ModelSpec:
    try:
        return MODEL_SPECS[name]
    except KeyError:
        raise KeyError(
            f"unknown Kronos model {name!r}; expected one of {sorted(MODEL_SPECS)}"
        ) from None
