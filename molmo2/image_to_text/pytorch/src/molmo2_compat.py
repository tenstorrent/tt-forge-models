# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Compatibility shims for running allenai/Molmo2-8B under transformers 5.x.

Molmo2-8B ships its architecture as Hub ``custom_code`` (``modeling_molmo2.py`` /
``processing_molmo2.py``) written against transformers ~4.57. The tt-xla venv pins
``transformers==5.5.1`` (required by vllm/surya/etc.), which cannot be downgraded.
Three small, self-contained API gaps break the model under 5.x; each is patched here
without modifying the vendored Hub code:

1. ``ROPE_INIT_FUNCTIONS["default"]`` was removed in 5.x (standard RoPE moved to
   ``RotaryEmbeddingConfigMixin``); the Molmo2 rotary embedding still looks it up by
   name. We re-register a "default" entry computing the standard inverse frequencies.
2. The top-level ``Molmo2Config`` no longer auto-exposes ``use_cache`` /
   ``output_attentions`` / ``output_hidden_states`` (they live on ``text_config``),
   but ``Molmo2ForConditionalGeneration.forward`` reads them off the top-level config.
   We propagate them from ``text_config`` onto the top-level config in the loader.
3. ``Molmo2Processor.__init__`` forwards its Molmo2-specific optional attributes
   (``image_use_col_tokens`` etc.) as kwargs to ``ProcessorMixin.__init__``, which
   5.x rejects (``Unexpected keyword argument``). We wrap ``__init__`` to set those as
   plain instance attributes and call the base init with only the sub-processors.

All shims are idempotent and safe to call repeatedly.
"""

import importlib

import torch

# Pinned revision of allenai/Molmo2-8B this loader was validated against.
MOLMO2_REVISION = "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"

# Molmo2-specific optional processor attributes that 5.x's ProcessorMixin rejects.
_PROCESSOR_OPTIONAL_ATTRS = (
    "image_use_col_tokens",
    "use_single_crop_col_tokens",
    "use_single_crop_start_token",
    "video_use_col_tokens",
    "use_frame_special_tokens",
    "time_mode",
)

_ROPE_PATCHED = False
_PROCESSOR_PATCHED = False


def _default_rope(config, device=None, seq_len=None, **kwargs):
    """Standard (non-scaled) RoPE inverse frequencies — the 5.x replacement for the
    removed ``ROPE_INIT_FUNCTIONS['default']`` entry the Molmo2 code expects."""
    base = getattr(config, "rope_theta", None)
    if base is None:
        base = getattr(config, "rope_parameters", {}).get("rope_theta", 10000.0)
    base = float(base)
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    dim = int(head_dim)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
    )
    return inv_freq, 1.0


def _patch_rope():
    global _ROPE_PATCHED
    if _ROPE_PATCHED:
        return
    import transformers.modeling_rope_utils as rope_utils

    if "default" not in rope_utils.ROPE_INIT_FUNCTIONS:
        rope_utils.ROPE_INIT_FUNCTIONS["default"] = _default_rope
    _ROPE_PATCHED = True


def _patch_processor(revision=MOLMO2_REVISION):
    """Wrap Molmo2Processor.__init__ so it instantiates under transformers 5.x."""
    global _PROCESSOR_PATCHED
    if _PROCESSOR_PATCHED:
        return
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.processing_utils import ProcessorMixin

    proc_cls = get_class_from_dynamic_module(
        "processing_molmo2.Molmo2Processor", "allenai/Molmo2-8B", revision=revision
    )
    mod = importlib.import_module(proc_cls.__module__)

    def _patched_init(
        self,
        image_processor=None,
        video_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        for attr in _PROCESSOR_OPTIONAL_ATTRS:
            setattr(self, attr, kwargs.pop(attr, None))
        ProcessorMixin.__init__(
            self,
            image_processor,
            video_processor,
            tokenizer,
            chat_template=chat_template,
        )
        self.image_placeholder_token = mod.IMAGE_PROMPT
        self.video_placeholder_token = mod.VIDEO_PROMPT
        self.image_token_ids = [
            tokenizer.convert_tokens_to_ids(token) for token in mod.IMAGE_TOKENS
        ]

    proc_cls.__init__ = _patched_init
    _PROCESSOR_PATCHED = True


def apply_model_shims():
    """Shims required before building/loading the Molmo2 model."""
    _patch_rope()


def apply_processor_shims(revision=MOLMO2_REVISION):
    """Shims required before building the Molmo2 processor."""
    _patch_rope()
    _patch_processor(revision=revision)


def propagate_top_level_config_attrs(config):
    """Copy attrs the Molmo2 forward reads off the top-level config but which now live
    only on ``text_config`` under transformers 5.x. Forces single-forward inference
    (``use_cache=False``). Returns the mutated config for chaining."""
    config.use_cache = False
    for attr in ("output_attentions", "output_hidden_states"):
        if not hasattr(config, attr):
            setattr(config, attr, getattr(config.text_config, attr, False))
    return config
