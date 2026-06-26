# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Molmo2-8B text-decoder loader.

Molmo2 (`Molmo2ForConditionalGeneration`, allenai/Molmo2-8B) pairs a
SigLIP-style ViT image encoder (brought up under ``molmo2/vision``) with a
Qwen3-8B-style text decoder (`Molmo2TextModel`, i.e. `model.transformer`):
36 decoder layers, hidden 4096, GQA (32 q : 8 kv heads, head_dim 128),
SwiGLU FFN (intermediate 12288), RMSNorm, QK-norm (Qwen3 style), RoPE
theta 1e6. The model ships as custom code (`trust_remote_code=True`).

This loader brings up the **text decoder** as a single prefill forward pass
(``use_cache=False``), returning the last hidden state.
"""

import torch
from typing import Optional

from transformers import AutoModelForImageTextToText

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


def _patch_default_rope():
    """Restore ``ROPE_INIT_FUNCTIONS['default']`` for Molmo2's custom modeling code.

    allenai/Molmo2-8B (custom code, transformers_version 5.5.1) constructs its
    rotary embedding via ``ROPE_INIT_FUNCTIONS['default']``, but transformers
    >=5.5 dropped the ``'default'`` key (standard RoPE is now expressed as
    ``'proportional'`` with ``factor=1.0`` / ``partial_rotary_factor=1.0`` — the
    inverse-frequency formula is identical, ``1/base**(arange(0,d,2)/d)``).
    Registering the proportional implementation under ``'default'`` lets the
    model build without forcing a transformers downgrade (which would risk
    pulling the torch / torch-xla stack out of sync).
    """
    from transformers.modeling_rope_utils import (
        ROPE_INIT_FUNCTIONS,
        _compute_proportional_rope_parameters,
    )

    ROPE_INIT_FUNCTIONS.setdefault("default", _compute_proportional_rope_parameters)


def _patch_packed_sequence_indices():
    """Disable transformers' packed-sequence detection for single-sequence inference.

    ``transformers.masking_utils.find_packed_sequence_indices`` runs an integer
    ``cumsum`` over ``position_ids`` to detect several sequences packed into one
    batch row. During ``torch.compile`` tracing this cumsum is *always* captured
    into the graph (its dynamic short-circuit is skipped while tracing), and the
    TT device backend cannot build the resulting int64 accumulation kernel
    (``accumulation_compute``: ``fill_tile_int`` / ``add_int`` only support
    Int32/UInt32/UInt16), which aborts compilation.

    We never pack multiple sequences into a batch row here, so the correct result
    is ``None`` ("no packed sequences") — the standard unpacked-sequence causal
    mask. Returning ``None`` removes the unsupported int64 cumsum from the graph
    without changing single-sequence semantics.
    """
    import transformers.masking_utils as _mu

    if getattr(_mu.find_packed_sequence_indices, "_tt_patched", False):
        return

    def _no_packing(position_ids):  # noqa: ANN001
        return None

    _no_packing._tt_patched = True
    _mu.find_packed_sequence_indices = _no_packing


def _fix_rotary_inv_freq(module: torch.nn.Module, config) -> int:
    """Overwrite corrupted rotary ``inv_freq`` buffers with the standard values.

    Beyond merely missing the ``'default'`` key, transformers 5.5.1 mis-computes
    the inverse frequencies for Molmo2's ``rope_type='default'`` at construction
    time (the proportional/standardize path yields a corrupted, mostly-zero
    buffer), which makes the rotary ``cos``/``sin`` — and therefore the whole
    decoder forward — NaN. Since ``'default'`` RoPE is not dynamic, the buffer is
    never recomputed in ``forward``, so recomputing the canonical
    ``inv_freq = 1 / theta**(arange(0, head_dim, 2) / head_dim)`` once after load
    and writing it back permanently fixes the model.

    Returns the number of rotary modules patched.
    """
    theta = float(getattr(config, "rope_theta", None) or 1000000.0)
    head_dim = int(
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads
    )
    correct = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim)
    )
    n_fixed = 0
    for submodule in module.modules():
        inv_freq = getattr(submodule, "inv_freq", None)
        if isinstance(inv_freq, torch.Tensor) and inv_freq.numel() == correct.numel():
            fixed = correct.to(device=inv_freq.device, dtype=inv_freq.dtype).clone()
            submodule.register_buffer("inv_freq", fixed, persistent=False)
            if hasattr(submodule, "original_inv_freq"):
                submodule.original_inv_freq = fixed.clone()
            n_fixed += 1
    return n_fixed


class _TextDecoderWrapper(torch.nn.Module):
    """Thin wrapper exposing the text decoder as a single-tensor forward.

    ``Molmo2TextModel`` returns a ``BaseModelOutputWithPast``; we run a prefill
    forward (``use_cache=False``) and return the last hidden state so the device
    comparison sees one clean tensor and no KV-cache pytree.
    """

    def __init__(self, transformer: torch.nn.Module):
        super().__init__()
        self.transformer = transformer

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, use_cache=False)
        return outputs.last_hidden_state


class ModelVariant(StrEnum):
    """Available Molmo2 text-decoder variants."""

    MOLMO2_8B = "8b"


class ModelLoader(ForgeModel):
    """Loader for the Molmo2-8B text decoder (Qwen3-8B-style decoder-only stack)."""

    _VARIANTS = {
        ModelVariant.MOLMO2_8B: LLMModelConfig(
            pretrained_model_name="allenai/Molmo2-8B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOLMO2_8B

    # Pin the custom-code / weight revision so reruns are reproducible.
    _REVISION = "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"

    # Prefill sequence length used for the bringup forward pass.
    _SEQ_LEN = 128

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="molmo2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load the Molmo2 text decoder wrapped as a single-tensor module.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The text-decoder wrapper.
        """
        _patch_default_rope()
        _patch_packed_sequence_indices()
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        full_model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            revision=self._REVISION,
            **model_kwargs,
        )

        transformer = full_model.model.transformer
        _fix_rotary_inv_freq(transformer, transformer.config)
        wrapper = _TextDecoderWrapper(transformer).eval()
        if dtype_override is not None:
            wrapper = wrapper.to(dtype_override)
        return wrapper

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Return sample ``input_ids`` for the text decoder.

        Args:
            dtype_override: Unused for integer token ids (kept for interface parity).
            batch_size: Batch size for the inputs.

        Returns:
            dict: ``{"input_ids": LongTensor[batch, seq_len]}``.
        """
        # Vocab is 151936; keep ids comfortably inside the base vocab range.
        generator = torch.Generator().manual_seed(0)
        input_ids = torch.randint(
            0,
            150000,
            (batch_size, self._SEQ_LEN),
            generator=generator,
            dtype=torch.long,
        )
        return {"input_ids": input_ids}
