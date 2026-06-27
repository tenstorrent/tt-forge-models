# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Compatibility shims for running allenai/Molmo2-8B under transformers >=5.5.

Molmo2 ships custom (`trust_remote_code`) modeling code that assumes an older
transformers RoPE/masking surface. Three in-loader fixes are required:

1. ``ROPE_INIT_FUNCTIONS['default']`` was removed in transformers >=5.5, but the
   Molmo2 text config requests ``rope_type='default'`` -> the model fails to
   construct (``KeyError: 'default'``). Re-register a standard implementation.
2. Even after re-registering, the default-RoPE ``inv_freq`` buffer is left
   corrupted (mostly zeros) after ``from_pretrained`` -> NaN ``cos``/``sin`` ->
   NaN forward. Recompute it from ``theta``/``head_dim`` and overwrite the buffer
   (the ``'default'`` RoPE is non-dynamic, so the fix survives ``forward``).
3. The int64 ``cumsum`` in ``transformers.masking_utils.find_packed_sequence_indices``
   is traced unconditionally into the compiled graph; the TT backend cannot build
   int64 accumulation -> compile abort. For single-sequence inference there are no
   packed sequences, so monkeypatch it to return ``None``.

Do NOT downgrade transformers to work around these — it desyncs the torch /
torch-xla stack that the TT device path depends on.
"""

import torch


def _rotary_dim(config):
    """Rotary dimension for the Molmo2 text decoder (head_dim, full rotary)."""
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads
    return head_dim


def register_default_rope():
    """Re-register ``ROPE_INIT_FUNCTIONS['default']`` (fix #1)."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" in ROPE_INIT_FUNCTIONS:
        return

    def _compute_default_rope(config, device=None, seq_len=None, **kwargs):
        base = config.rope_theta
        dim = _rotary_dim(config)
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(device=device).float()
                / dim
            )
        )
        return inv_freq, 1.0  # (inv_freq, attention_scaling)

    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope


def fix_rotary_inv_freq(text_model):
    """Recompute and overwrite the corrupted default-RoPE ``inv_freq`` (fix #2)."""
    config = text_model.config
    base = config.rope_theta
    dim = _rotary_dim(config)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
    )

    def _overwrite(rotary):
        good = inv_freq.to(device=rotary.inv_freq.device, dtype=rotary.inv_freq.dtype)
        rotary.inv_freq = good
        rotary.original_inv_freq = good

    if hasattr(text_model, "rotary_emb"):
        _overwrite(text_model.rotary_emb)
    if hasattr(text_model, "rotary_embs"):
        for rotary in text_model.rotary_embs.values():
            _overwrite(rotary)


def patch_packed_sequence_indices():
    """Disable int64 packed-sequence indexing for single-sequence inference (fix #3)."""
    import transformers.masking_utils as masking_utils

    masking_utils.find_packed_sequence_indices = lambda *args, **kwargs: None


def apply_all(text_model=None):
    """Apply the construction-time and compile-time patches.

    ``register_default_rope`` / ``patch_packed_sequence_indices`` are global and
    must be applied before constructing the model and before device compile,
    respectively. ``fix_rotary_inv_freq`` needs the loaded text model.
    """
    register_default_rope()
    patch_packed_sequence_indices()
    if text_model is not None:
        fix_rotary_inv_freq(text_model)
