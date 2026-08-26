"""Compatibility launcher used by BACH Studio.

The upstream infer.py is kept intact on disk. This launcher applies narrow runtime
compatibility fixes before executing it:

1. On Windows, the Transformers FlashAttention-2 request is replaced with PyTorch
   SDPA, avoiding the fragile native flash-attn Windows build requirement. On other
   platforms, FlashAttention-2 is attempted first and falls back to SDPA/eager.
2. torch.compile is disabled on Windows for broader native-Windows compatibility.
3. The upstream section-count boundary is corrected at runtime so the final lyric
   section is generated instead of being silently omitted.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import transformers


_ORIGINAL_FROM_PRETRAINED = transformers.AutoModelForCausalLM.from_pretrained


def _load_with_sdpa_fallback(args, kwargs):
    sdpa_kwargs = dict(kwargs)
    sdpa_kwargs["attn_implementation"] = "sdpa"
    try:
        return _ORIGINAL_FROM_PRETRAINED(*args, **sdpa_kwargs)
    except Exception as sdpa_exc:
        print(f"[BACH Studio] SDPA unavailable ({sdpa_exc}). Falling back to eager attention.")
        eager_kwargs = dict(kwargs)
        eager_kwargs.pop("attn_implementation", None)
        return _ORIGINAL_FROM_PRETRAINED(*args, **eager_kwargs)


def _from_pretrained_with_attention_fallback(*args, **kwargs):
    requested = kwargs.get("attn_implementation")
    if requested != "flash_attention_2":
        return _ORIGINAL_FROM_PRETRAINED(*args, **kwargs)

    if os.name == "nt":
        print("[BACH Studio] Windows detected: using PyTorch SDPA instead of external FlashAttention-2.")
        return _load_with_sdpa_fallback(args, kwargs)

    try:
        return _ORIGINAL_FROM_PRETRAINED(*args, **kwargs)
    except Exception as exc:
        message = str(exc).lower()
        if not isinstance(exc, (ImportError, ModuleNotFoundError)) and "flash" not in message:
            raise
        print(f"[BACH Studio] FlashAttention-2 unavailable ({exc}). Falling back to SDPA/eager attention.")
        return _load_with_sdpa_fallback(args, kwargs)


transformers.AutoModelForCausalLM.from_pretrained = _from_pretrained_with_attention_fallback

if os.name == "nt" and hasattr(torch, "compile"):
    def _windows_no_compile(model, *args, **kwargs):
        print("[BACH Studio] Windows detected: skipping torch.compile for compatibility.")
        return model

    torch.compile = _windows_no_compile


_original_script = Path(__file__).with_name("infer.py")
source = _original_script.read_text(encoding="utf-8")
old_boundary = "run_n_segments = min(args.run_n_segments+1, len(lyrics))"
new_boundary = "run_n_segments = min(args.run_n_segments+1, len(lyrics)+1)"
if old_boundary not in source:
    raise RuntimeError(
        "BACH Studio could not locate the expected upstream section-count line in infer.py. "
        "The upstream script may have changed; update the compatibility launcher before running it."
    )
source = source.replace(old_boundary, new_boundary, 1)

compiled = compile(source, str(_original_script), "exec")
globals_for_infer = {
    "__name__": "__main__",
    "__file__": str(_original_script),
    "__package__": None,
    "__builtins__": __builtins__,
}
exec(compiled, globals_for_infer, globals_for_infer)
