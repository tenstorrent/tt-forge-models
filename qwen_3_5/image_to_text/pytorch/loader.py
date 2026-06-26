# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.5 MoE (multimodal) model loader implementation for image to text.

Qwen3.6-35B-A3B is a vision-language model built on the qwen3_5_moe
architecture: a SigLIP-style vision tower (27 blocks, hidden 1152, patch 16)
feeding a hybrid linear/full-attention Mixture-of-Experts text decoder
(40 layers, 256 experts / 8 active, Gated DeltaNet linear-attention layers
interleaved with full-attention every 4th layer, mRoPE). 35B total params,
~3B active per token. Class: Qwen3_5MoeForConditionalGeneration.
"""
from transformers import AutoModelForImageTextToText, AutoProcessor
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
    """Available Qwen 3.5 MoE VLM variants for image to text."""

    QWEN_3_6_35B_A3B = "35b_a3b"


class ModelLoader(ForgeModel):
    """Qwen 3.5 MoE VLM loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_6_35B_A3B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.6-35B-A3B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_6_35B_A3B

    # Shared configuration parameters
    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

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
            model="Qwen 3.5 MoE VLM",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self._variant_config.pretrained_model_name
            )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen 3.5 MoE VLM instance for this variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Qwen 3.5 MoE VLM instance for image to text.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        # Force use_cache=False so the forward output does not include a
        # DynamicCache, which the runner's pytree comparator cannot diff
        # leaf-wise against the CPU golden (same pattern as the qwen_3_5
        # causal_lm and qwen_2_5_vl loaders).
        model.config.use_cache = False
        if getattr(model.config, "text_config", None) is not None:
            model.config.text_config.use_cache = False

        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample multimodal inputs for the Qwen 3.5 MoE VLM.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
            batch_size: Batch size for the inputs (only 1 is supported here).

        Returns:
            dict: Input tensors (input_ids, attention_mask, pixel_values,
                  image_grid_thw, ...) that can be fed to the model.
        """
        self._load_processor()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.sample_image},
                    {"type": "text", "text": "Describe this image."},
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

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs
