# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Apriel-1.6-15b-Thinker model loader implementation.

Apriel-1.6-15b-Thinker is a vision-language model built as a
``LlavaForConditionalGeneration`` architecture: a Mistral text decoder
(hidden_size=5120, 48 layers) paired with a Pixtral vision encoder. It is the
same architecture family as ``mistral/pixtral`` already supported in this repo.

Note on the GGUF source repo:
    The bringup target ``mradermacher/Apriel-1.6-15b-Thinker-heretic-i1-GGUF``
    only ships imatrix (i1) GGUF k-quant weights. Those quantized blocks are not
    dequantizable by HuggingFace ``transformers`` for the Llava/Pixtral
    architecture (transformers GGUF support is limited to a fixed set of
    text-only architectures), and TT-XLA has no path to execute GGUF k-quant
    kernels. This loader therefore targets the upstream full-precision
    safetensors checkpoint that those GGUFs quantize:
    ``MRockatansky/Apriel-1.6-15b-Thinker-heretic``.
"""

from typing import Optional

import torch
from transformers import AutoTokenizer, LlavaForConditionalGeneration

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Apriel model variants."""

    THINKER_15B_HERETIC = "1.6_15b_thinker_heretic"


class ModelLoader(ForgeModel):
    """Apriel-1.6-15b-Thinker model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.THINKER_15B_HERETIC: LLMModelConfig(
            pretrained_model_name="MRockatansky/Apriel-1.6-15b-Thinker-heretic",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.THINKER_15B_HERETIC

    # Sample prompt for text-only forward pass
    sample_text = "Hey how are you doing today?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
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
            model="Apriel",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load and cache the tokenizer for the current variant."""
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Apriel model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model uses its checkpoint dtype.

        Returns:
            torch.nn.Module: The Apriel LlavaForConditionalGeneration model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is available for load_inputs.
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = LlavaForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text-only inputs for the Apriel model.

        A text-only forward pass exercises the Mistral language model path of the
        VLM (no ``pixel_values``), matching the ``mistral/pixtral`` reference.

        Args:
            dtype_override: Unused for integer token inputs; accepted for API parity.
            batch_size: Batch size to expand the sample inputs to (default 1).

        Returns:
            dict: Input tensors (input_ids, attention_mask) for the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length or 128
        encoded = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        )

        inputs = {
            "input_ids": encoded["input_ids"].repeat_interleave(batch_size, dim=0),
            "attention_mask": encoded["attention_mask"].repeat_interleave(
                batch_size, dim=0
            ),
        }
        return inputs

    def get_mesh_config(self, num_devices: int):
        """Mesh configuration for tensor-parallel execution."""
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    @staticmethod
    def _get_language_model(model):
        """Get the language_model sub-module, handling nested model wrapping."""
        if hasattr(model, "language_model"):
            return model.language_model
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            return model.model.language_model
        raise AttributeError("Cannot find language_model on the model")

    @staticmethod
    def _get_vision_tower(model):
        """Get the vision_tower sub-module, handling nested model wrapping."""
        if hasattr(model, "vision_tower"):
            return model.vision_tower
        if hasattr(model, "model") and hasattr(model.model, "vision_tower"):
            return model.model.vision_tower
        raise AttributeError("Cannot find vision_tower on the model")

    def load_shard_spec(self, model):
        """Tensor-parallel shard specs for the language and vision sub-modules."""
        shard_specs = {}
        language_model = self._get_language_model(model)
        for layer in language_model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

        vision_tower = self._get_vision_tower(model)
        for layer in vision_tower.transformer.layers:
            shard_specs[layer.feed_forward.up_proj.weight] = ("model", "batch")
            shard_specs[layer.feed_forward.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.feed_forward.down_proj.weight] = ("batch", "model")

            shard_specs[layer.attention.q_proj.weight] = ("model", "batch")
            shard_specs[layer.attention.k_proj.weight] = ("model", "batch")
            shard_specs[layer.attention.v_proj.weight] = ("model", "batch")
            shard_specs[layer.attention.o_proj.weight] = ("batch", "model")

        return shard_specs
