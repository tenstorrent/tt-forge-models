# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Voxtral-Small-24B-2507 text-only (causal LM) model loader.

``mistralai/Voxtral-Small-24B-2507`` is an audio+text model. This loader brings
up the text tower only (``input_ids`` + ``attention_mask``); audio tower weights
stay unreplicated / out of the shard map.
"""

from typing import Optional

import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration

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


def _patch_chat_template_guard():
    """Work around a transformers bug with Voxtral's mistral_common backend.

    ``VoxtralProcessor.apply_chat_template`` calls ``_get_template_variables(
    chat_template)`` unconditionally. Voxtral uses the mistral_common / tekken
    tokenizer and has no Jinja ``chat_template`` (it is ``None``), which makes
    jinja raise ``Can't compile non template nodes``.

    This guard returns an empty frozenset when ``chat_template is None`` so
    processor setup stays safe. The patch is idempotent (``_tt_patched``) so
    repeated loader calls do not wrap the helper multiple times.
    """
    import transformers.models.voxtral.processing_voxtral as _vox

    if getattr(_vox._get_template_variables, "_tt_patched", False):
        return

    _orig = _vox._get_template_variables

    def _safe_get_template_variables(chat_template):
        if chat_template is None:
            return frozenset()
        return _orig(chat_template)

    _safe_get_template_variables._tt_patched = True
    _vox._get_template_variables = _safe_get_template_variables


class ModelVariant(StrEnum):
    """Available Voxtral text-only model variants."""

    VOXTRAL_SMALL_24B = "Voxtral-Small-24B-2507"


class ModelLoader(ForgeModel):
    """Voxtral-Small-24B-2507 text-only causal LM loader."""

    _VARIANTS = {
        ModelVariant.VOXTRAL_SMALL_24B: LLMModelConfig(
            pretrained_model_name="mistralai/Voxtral-Small-24B-2507",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VOXTRAL_SMALL_24B

    sample_text = "Who are you?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._model_name = self._variant_config.pretrained_model_name
        self.processor = None
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Voxtral",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _text_config(self):
        return getattr(self.config, "text_config", self.config)

    def _load_processor(self):
        """Load processor for the current variant."""
        if self.processor is None:
            _patch_chat_template_guard()
            self.processor = AutoProcessor.from_pretrained(self._model_name)
            # Voxtral ships without a pad token; reuse EOS so padding works.
            tokenizer = self.processor.tokenizer
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Voxtral model for text-only bring-up.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            VoxtralForConditionalGeneration: The loaded model instance.
        """
        _patch_chat_template_guard()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = VoxtralForConditionalGeneration.from_pretrained(
            self._model_name, **model_kwargs
        )
        model.eval()
        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs):
        """Load sample text-only inputs (no audio).

        Voxtral's tokenizer has no Jinja ``chat_template`` (mistral_common /
        tekken). ``VoxtralProcessor.__call__`` is the supported text-only
        path; ``apply_chat_template`` is for audio(+text) conversations.

        Args:
            dtype_override: Unused for tokenized integer inputs.
            batch_size: Batch size for the inputs.

        Returns:
            dict: ``input_ids`` and ``attention_mask`` tensors.
        """
        processor = self._load_processor()
        max_length = self._variant_config.max_length
        inputs = processor(
            text=self.sample_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size or 1, dim=0)

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

    def unpack_forward_output(self, fwd_output):
        """Extract logits from the causal LM output."""
        if hasattr(fwd_output, "logits"):
            return fwd_output.logits
        return super().unpack_forward_output(fwd_output)

    def get_mesh_config(self, num_devices: int):
        """Return mesh shape and axis names for tensor parallel."""
        text_cfg = self._text_config()
        attn_heads = text_cfg.num_attention_heads
        mesh_shape = (1, num_devices)
        if attn_heads % mesh_shape[1] != 0:
            raise ValueError(
                f"Cannot evenly distribute {attn_heads} attention heads "
                f"across model axis size {mesh_shape[1]}"
            )
        return mesh_shape, ("batch", "model")

    @staticmethod
    def _get_language_model(model):
        """Get the language_model sub-module, handling nested model wrapping."""
        if hasattr(model, "language_model"):
            return model.language_model
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            return model.model.language_model
        raise AttributeError("Cannot find language_model on the model")

    def load_shard_spec(self, model):
        """Megatron-style TP map for the Voxtral language model.

        Column-parallel on q/k/v/gate/up, row-parallel on o/down. Audio tower
        weights stay out of the map on the text-only path.
        """
        shard_specs = {}
        language_model = self._get_language_model(model)
        for layer in language_model.model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

        return shard_specs
