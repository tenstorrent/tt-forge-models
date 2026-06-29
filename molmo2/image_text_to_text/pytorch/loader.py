# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Molmo2 image-text-to-text model loader implementation.

Molmo2-8B (allenai/Molmo2-8B) is a multimodal vision-language model:
a SigLIP-style vision transformer + pooling adapter feeding a
Qwen3-8B-based decoder-only language model (``molmo2_text``).

This loader drives the *key component* — the text decoder — by running the
full ``Molmo2ForConditionalGeneration`` model with text-only inputs (no
``pixel_values``). With no image, ``merge_visual_inputs`` returns ``None`` and
the forward pass exercises the token embedding, the 36-layer decoder, and the
LM head only (a standard causal-LM forward). The vision tower is brought up
separately as an op pre-check (see ``molmo2/vision/pytorch/loader.py``); the
end-to-end multimodal forward is blocked on device by data-dependent vision
backbone ops (boolean-mask select / gather-scatter pooling).

The model is custom-code on the Hub (``trust_remote_code=True``).

.. note::
    The published remote modeling code requires ``transformers==4.57.1`` (per the
    model card) and is **incompatible with transformers 5.x**: it indexes the
    removed ``ROPE_INIT_FUNCTIONS['default']`` key and reads ``config.use_cache``
    on the top-level config. The tt-forge device stack (``tt_torch.moe_backend``)
    requires ``transformers>=5.x`` (``transformers.integrations.moe``), so the two
    cannot currently coexist. This loader + forward are verified correct under
    ``transformers==4.57.1`` on CPU; on-device bringup is blocked on this
    transformers-version conflict (a human stack decision).
"""

import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
from typing import Optional

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


class ModelVariant(StrEnum):
    """Available Molmo2 image-text-to-text variants."""

    MOLMO2_8B = "8b"


class ModelLoader(ForgeModel):
    """Molmo2 model loader implementation for image-text-to-text tasks.

    Device validation runs the text-decoder path (text-only inputs); see the
    module docstring.
    """

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.MOLMO2_8B: LLMModelConfig(
            pretrained_model_name="allenai/Molmo2-8B",
            max_length=64,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.MOLMO2_8B

    # Shared configuration parameters
    sample_text = "The capital of France is"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="molmo2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Molmo2 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default
                            dtype. If not provided, the model uses its default
                            dtype (float32).

        Returns:
            torch.nn.Module: The Molmo2 model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return text-only sample inputs for the Molmo2 text decoder.

        Args:
            dtype_override: Unused for integer token inputs; kept for interface
                            consistency.
            batch_size: Batch size for the inputs.

        Returns:
            dict: input_ids / attention_mask tensors (text-only).
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        prompts = [self.sample_text]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self._variant_config.max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
