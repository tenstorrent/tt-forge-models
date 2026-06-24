# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.6 multimodal model loader implementation for image-to-text.

Qwen3.6-27B (model_type ``qwen3_5``, class ``Qwen3_5ForConditionalGeneration``)
is a vision-language model: a SigLIP-style vision encoder (depth 27, hidden
1152, patch 16, spatial-merge 2) projecting into a hybrid Qwen3.5 text decoder.
The text stack interleaves Gated DeltaNet linear-attention layers with a full
attention layer every 4th block (layout 3x linear_attention + 1x full_attention
repeated over 64 layers), uses GQA (24 query : 4 kv heads, head_dim 256) on the
full-attention layers, mRoPE interleaved positional encoding (partial rotary
factor 0.25), and carries an MTP head. Branded "Qwen3.6" but shares the
``qwen3_5`` architecture family, hence this lives under ``qwen_3_5``.
"""
import torch
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
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
    """Available Qwen 3.6 multimodal variants for image-to-text."""

    QWEN_3_6_27B = "27b"


class ModelLoader(ForgeModel):
    """Qwen 3.6 multimodal model loader for image-to-text tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.QWEN_3_6_27B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.6-27B",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.QWEN_3_6_27B

    # Shared configuration parameters
    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )
    sample_prompt = "Describe this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None

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
            model="Qwen 3.6",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant.

        Returns:
            The loaded processor instance
        """
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen 3.6 multimodal model instance for this variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model uses its checkpoint dtype.

        Returns:
            torch.nn.Module: The Qwen 3.6 model instance for image-to-text.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {"low_cpu_mem_usage": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        # Force use_cache=False on the text sub-config so the forward output does
        # not include a Qwen3_5DynamicCache, which the runner's pytree comparator
        # can't diff leaf-wise against the CPU golden. Same rationale as the
        # qwen_3_5 causal_lm and qwen_2_5_vl loaders.
        model.config.text_config.use_cache = False
        model.eval()

        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Qwen 3.6 model.

        Args:
            dtype_override: Optional torch.dtype to convert pixel_values to.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.sample_image},
                    {"type": "text", "text": self.sample_prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Convert pixel_values to the requested dtype if provided.
        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs
