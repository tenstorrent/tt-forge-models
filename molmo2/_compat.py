# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Compatibility shims for ``allenai/Molmo2-8B`` under transformers >= 5.5.

The Molmo2 checkpoint ships custom (``trust_remote_code``) modeling code that was
written against an older transformers. Three things break on transformers 5.5.x;
all three are fixed here from the loader side so we never have to downgrade
transformers (which would desync the torch / torch-xla stack):

1. ``ROPE_INIT_FUNCTIONS['default']`` was removed in transformers >= 5.5, so the
   text decoder's ``Molmo2RotaryEmbedding`` (rope_type "default") raises
   ``KeyError: 'default'`` at construction. :func:`register_default_rope`
   re-registers a faithful default RoPE init.

2. ``inv_freq`` is a non-persistent buffer. ``from_pretrained`` materializes the
   model through a meta device and never restores non-persistent buffers from the
   checkpoint, so ``inv_freq`` ends up zeroed -> NaN cos/sin -> NaN forward.
   :func:`fix_rotary_inv_freq` recomputes and overwrites the buffer post-load
   (the "default" RoPE is non-dynamic, so the value survives ``forward``).

3. ``transformers.masking_utils.find_packed_sequence_indices`` does an int64
   ``cumsum`` that is traced unconditionally into the compiled graph; the TT
   backend's integer accumulation only supports Int32/UInt32/UInt16, so the
   compile aborts. :func:`patch_packed_sequence_indices` makes it return ``None``
   (no packed sequences) for single-sequence inference.
"""

import torch


def _default_inv_freq(config) -> torch.Tensor:
    """Standard (non-scaled) RoPE inverse frequencies for ``config``."""
    base = getattr(config, "rope_theta", None)
    if base is None:
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        base = rope_scaling.get("rope_theta", 10000.0)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads
    partial = getattr(config, "partial_rotary_factor", 1.0) or 1.0
    dim = int(head_dim * partial)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
    )
    return inv_freq


def register_default_rope() -> None:
    """Re-register the 'default' RoPE init removed in transformers >= 5.5.

    Idempotent. Must run before the model is constructed.
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" in ROPE_INIT_FUNCTIONS:
        return

    def _default(config, device=None, seq_len=None, **kwargs):
        inv_freq = _default_inv_freq(config)
        if device is not None:
            inv_freq = inv_freq.to(device)
        return inv_freq, 1.0  # (inv_freq, attention_scaling)

    ROPE_INIT_FUNCTIONS["default"] = _default


def fix_rotary_inv_freq(module: torch.nn.Module) -> int:
    """Recompute corrupted default-RoPE ``inv_freq`` buffers under ``module``.

    Returns the number of rotary-embedding modules fixed.
    """
    fixed = 0
    for m in module.modules():
        if (
            getattr(m, "rope_type", None) == "default"
            and hasattr(m, "inv_freq")
            and hasattr(m, "config")
        ):
            inv_freq = _default_inv_freq(m.config).float()
            device = m.inv_freq.device if isinstance(m.inv_freq, torch.Tensor) else None
            if device is not None and device.type != "meta":
                inv_freq = inv_freq.to(device)
            m.register_buffer("inv_freq", inv_freq, persistent=False)
            m.original_inv_freq = inv_freq
            fixed += 1
    return fixed


def patch_packed_sequence_indices() -> None:
    """Disable int64-cumsum packed-sequence detection (single-sequence inference).

    Idempotent.
    """
    import transformers.masking_utils as masking_utils

    current = getattr(masking_utils, "find_packed_sequence_indices", None)
    if current is not None and getattr(current, "_molmo2_patched", False):
        return

    def _no_packed_sequences(*args, **kwargs):
        return None

    _no_packed_sequences._molmo2_patched = True
    masking_utils.find_packed_sequence_indices = _no_packed_sequences
