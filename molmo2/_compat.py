# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Compatibility shims for bringing up allenai/Molmo2-8B on transformers >= 5.5.

Molmo2 is a custom ``trust_remote_code`` VLM (Qwen3-8B text decoder + SigLIP-style
ViT). Three things changed in transformers 5.5 that break it; do NOT downgrade
transformers (that desyncs the torch / torch-xla stack). Instead apply these
loader-side fixes:

1. ``ROPE_INIT_FUNCTIONS['default']`` was removed (only 'linear', 'dynamic',
   'yarn', 'longrope', 'llama3', 'proportional' remain). The custom modeling code
   looks it up by name and dies with ``KeyError: 'default'`` before the model can
   even be constructed. ``register_default_rope`` re-registers it.

2. Even after re-registering, the rotary embedding's ``inv_freq`` buffer is left
   corrupted at init (mostly zeros) -> NaN cos/sin -> NaN forward.
   ``fix_rotary_inv_freq`` recomputes every ``inv_freq`` buffer post-load. The
   'default' RoPE is non-dynamic, so the recomputed buffer survives ``forward``.

3. ``transformers.masking_utils.find_packed_sequence_indices`` builds an int64
   ``cumsum`` that is traced unconditionally into the compiled graph; the TT
   backend's integer reductions only support Int32/UInt32/UInt16, so the compile
   aborts. ``patch_packed_sequence_indices`` makes it return ``None`` (no packed
   sequences) for single-sequence inference.
"""

import torch


def _compute_default_rope_parameters(
    config=None, device=None, seq_len=None, layer_type=None, **rope_kwargs
):
    """Standard (unscaled) RoPE inverse frequencies.

    Mirrors transformers' linear-scaling initializer but without the ``factor``
    division, matching the 'default' function that existed before transformers 5.5.
    """
    config.standardize_rope_params()
    rope_parameters_dict = (
        config.rope_parameters[layer_type]
        if layer_type is not None
        else config.rope_parameters
    )
    base = rope_parameters_dict["rope_theta"]
    partial_rotary_factor = rope_parameters_dict.get("partial_rotary_factor", 1.0)
    head_dim = (
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads
    )
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float
            )
            / dim
        )
    )
    return inv_freq, attention_factor


def register_default_rope():
    """Re-register the 'default' RoPE initializer removed in transformers >= 5.5."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    ROPE_INIT_FUNCTIONS.setdefault("default", _compute_default_rope_parameters)


def fix_rotary_inv_freq(model):
    """Recompute every rotary ``inv_freq`` buffer in ``model`` in-place.

    The buffer is corrupted (mostly zeros) at init under transformers 5.5 even
    after the 'default' RoPE function is re-registered, which yields NaN cos/sin.
    We recompute ``inv_freq = 1 / theta ** (arange(0, dim, 2) / dim)`` from the
    buffer's own length (``dim = 2 * len(inv_freq)``) and the rope base resolved
    from the owning module / model config.
    """
    # Resolve a fallback rope base from the model config tree.
    cfg = getattr(model, "config", None)

    def _resolve_theta(module):
        for attr in ("rope_theta", "base", "theta"):
            v = getattr(module, attr, None)
            if v is not None:
                return float(v)
        mcfg = getattr(module, "config", None)
        for c in (mcfg, getattr(cfg, "text_config", None), cfg):
            if c is None:
                continue
            v = getattr(c, "rope_theta", None)
            if v is not None:
                return float(v)
        return 1.0e6  # Qwen3 / Molmo2 text-decoder default

    for module in model.modules():
        inv_freq = getattr(module, "inv_freq", None)
        if not isinstance(inv_freq, torch.Tensor) or inv_freq.ndim != 1:
            continue
        dim = 2 * inv_freq.numel()
        theta = _resolve_theta(module)
        new_inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim
            )
        )
        with torch.no_grad():
            inv_freq.copy_(new_inv_freq.to(inv_freq.dtype).to(inv_freq.device))
    return model


def patch_packed_sequence_indices():
    """Force single-sequence packing: make the int64-cumsum helper return None.

    The TT backend cannot build the unconditional int64 ``cumsum`` that
    ``find_packed_sequence_indices`` traces into the compiled graph. For
    single-sequence inference there are no packed sequences, so returning ``None``
    is numerically exact and removes the unsupported op.
    """
    import transformers.masking_utils as masking_utils

    def _no_packed_sequences(position_ids):
        return None

    masking_utils.find_packed_sequence_indices = _no_packed_sequences


def apply_molmo2_compat():
    """Apply the import-time global patches (rope registration + packed-seq)."""
    register_default_rope()
    patch_packed_sequence_indices()
