"""Compatibility launcher used by BACH Studio.

This deliberately keeps the upstream infer.py intact. It adds two narrow compatibility
fixes before executing the original script:

1. AutoModelForCausalLM.from_pretrained falls back from FlashAttention 2 when the
   local environment does not have a working flash-attn installation.
2. BACH Studio appends a private sentinel lyric section so the original script's
   segment slicing includes the user's actual final section. The sentinel itself is
   never passed to generation because of the upstream slice boundary.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import transformers


_ORIGINAL_FROM_PRETRAINED = transformers.AutoModelForCausalLM.from_pretrained


def _from_pretrained_with_attention_fallback(*args, **kwargs):
    requested = kwargs.get("attn_implementation")
    if requested != "flash_attention_2":
        return _ORIGINAL_FROM_PRETRAINED(*args, **kwargs)

    try:
        return _ORIGINAL_FROM_PRETRAINED(*args, **kwargs)
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"[BACH Studio] FlashAttention 2 unavailable ({exc}). Falling back to SDPA/eager attention.")
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop("attn_implementation", None)
        try:
            fallback_kwargs["attn_implementation"] = "sdpa"
            return _ORIGINAL_FROM_PRETRAINED(*args, **fallback_kwargs)
        except Exception:
            fallback_kwargs.pop("attn_implementation", None)
            return _ORIGINAL_FROM_PRETRAINED(*args, **fallback_kwargs)


transformers.AutoModelForCausalLM.from_pretrained = _from_pretrained_with_attention_fallback

_original_script = Path(__file__).with_name("infer.py")
runpy.run_path(str(_original_script), run_name="__main__")
