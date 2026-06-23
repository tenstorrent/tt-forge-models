# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VibeVoice model loader implementation.

VibeVoice (microsoft/VibeVoice-1.5B) is a long-form, generation-based
text-to-speech model. The HuggingFace repo ships weights only; the model
code lives in the standalone https://github.com/microsoft/VibeVoice repo
and is vendored (port mode) under ``src/`` here. See
``src/SRC_VENDORED_FROM.txt`` for provenance and the local compat edits.

The entry class ``VibeVoiceForConditionalGeneration`` is built from the real
1.5B ``config.json`` (vendored alongside this loader) with random weights —
no multi-GB safetensors download. The bringup forward path uses
``speech_tensors=None`` so the model reduces to: embed(input_ids) -> Qwen2.5
decoder -> lm_head -> logits, which is a clean tensor-in / tensor-out forward.

Note: the full model is ~2.7B params (the "1.5B" refers to the Qwen LLM
backbone only; the acoustic/semantic VAE tokenizers, diffusion head and
connectors add the rest).
"""

import os
import sys
from typing import Optional

import torch

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

# --- vendored source on sys.path -------------------------------------------
# The ported code uses absolute imports (``from vibevoice.schedule...``), so we
# put the vendored ``src/`` dir on sys.path and import the package by its
# original name rather than rewriting every internal import.
_SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def _patch_idempotent_register():
    """Make AutoModel(.ForCausalLM).register idempotent.

    transformers>=5 pre-registers the VibeVoice sub-configs, so the explicit
    ``AutoModel.register(...)`` calls at module import time would raise
    "already used by a Transformers model". The registrations are identical
    config->model pairs, so forcing ``exist_ok=True`` is safe. Upstream pins
    transformers<5.0.0; this lets the ported code import under transformers 5.
    """
    import transformers

    for cls_name in ("AutoModel", "AutoModelForCausalLM"):
        cls = getattr(transformers, cls_name)
        if getattr(cls.register, "_tt_idempotent", False):
            continue
        orig_bound = cls.register

        def _make(orig):
            def register(config_class, model_class=None, exist_ok=False):
                return orig(config_class, model_class, exist_ok=True)

            register._tt_idempotent = True
            return staticmethod(register)

        setattr(cls, "register", _make(orig_bound))


class _VibeVoiceLogitsWrapper(torch.nn.Module):
    """Wrap VibeVoice so forward() returns only the logits tensor.

    The inference test harness compares the *raw* forward output between CPU
    and TT with a pytree comparator (it does not call unpack_forward_output —
    that is training-only). VibeVoice's native output
    ``VibeVoiceCausalLMOutputWithPast`` contains a non-tensor leaf
    (``speech_token_num``, a Python int), which makes the comparator's
    ``torch.equal()`` fail. Returning a single tensor keeps the device
    computation identical while giving the comparator a clean tensor on both
    sides.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        kwargs.setdefault("return_dict", True)
        out = self.model(*args, **kwargs)
        return out.logits


class ModelVariant(StrEnum):
    """Available VibeVoice model variants."""

    VIBEVOICE_1_5B = "1.5B"


class ModelLoader(ForgeModel):
    """VibeVoice model loader implementation."""

    _VARIANTS = {
        ModelVariant.VIBEVOICE_1_5B: ModelConfig(
            pretrained_model_name="microsoft/VibeVoice-1.5B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VIBEVOICE_1_5B

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="VibeVoice",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_TTS,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def _load_config(self):
        """Build the real 1.5B VibeVoiceConfig from the vendored config.json."""
        _patch_idempotent_register()
        from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig

        self.config = VibeVoiceConfig.from_pretrained(os.path.dirname(__file__))
        return self.config

    def load_model(self, dtype_override=torch.bfloat16, **kwargs):
        """Load and return the VibeVoice model instance (random weights).

        Args:
            dtype_override: dtype to cast the whole model to. The upstream
                model is internally mixed-precision (tokenizers/connectors/head
                follow ``config.torch_dtype`` while the Qwen LM stays fp32);
                casting the whole module to one dtype is required for a clean
                forward. Defaults to bfloat16.

        Returns:
            torch.nn.Module: The VibeVoice model instance.
        """
        _patch_idempotent_register()
        from vibevoice.modular.modeling_vibevoice import (
            VibeVoiceForConditionalGeneration,
        )

        config = self._load_config()
        model = VibeVoiceForConditionalGeneration(config)
        if dtype_override is not None:
            model = model.to(dtype_override)
        # Wrap so forward() returns only the logits tensor (the inference
        # harness compares raw forward outputs and cannot handle the
        # non-tensor leaves in VibeVoiceCausalLMOutputWithPast).
        return _VibeVoiceLogitsWrapper(model.eval()).eval()

    def load_inputs(self, batch_size=1, seq_len=32, dtype_override=torch.bfloat16):
        """Load and return sample inputs for the VibeVoice model.

        The bringup forward path keeps ``speech_tensors=None`` so the model
        behaves as a Qwen2.5 causal LM with a single (unused) semantic
        connector call. ``speech_semantic_tensors`` is therefore required but
        its result is not consumed downstream.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.config is None:
            self._load_config()

        vocab_size = self.config.decoder_config.vocab_size
        semantic_vae_dim = self.config.semantic_vae_dim

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        speech_semantic_tensors = torch.randn(
            batch_size, 1, semantic_vae_dim, dtype=dtype_override
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "speech_semantic_tensors": speech_semantic_tensors,
            "return_dict": True,
        }

    def unpack_forward_output(self, output):
        """Return the logits tensor.

        load_model() wraps the model to already return a bare logits tensor,
        so pass tensors through unchanged; fall back to ``.logits`` if a raw
        dataclass output is ever handed in.
        """
        if torch.is_tensor(output):
            return output
        return output.logits
