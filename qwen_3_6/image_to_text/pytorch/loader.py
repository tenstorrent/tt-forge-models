# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.6 (Qwen3.5-MoE) model loader implementation for image-to-text.

Qwen3.6-35B-A3B is a multimodal Mixture-of-Experts vision-language model
(`Qwen3_5MoeForConditionalGeneration`). It couples a SigLIP-style vision
tower (depth 27, hidden 1152, patch 16, spatial-merge 2) with a sparse MoE
text decoder (40 layers, hidden 2048, 256 experts, 8 experts/token, a shared
expert, GQA 16 q : 2 kv, head_dim 256, mRoPE with partial rotary factor 0.25).
Total ~35B params, ~3B active per token.
"""
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig
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
    """Available Qwen 3.6 MoE VL model variants for image-to-text."""

    QWEN_3_6_35B_A3B = "35b_a3b"


class ModelLoader(ForgeModel):
    """Qwen 3.6 MoE VL model loader implementation for image-to-text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_6_35B_A3B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.6-35B-A3B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_6_35B_A3B

    # Shared configuration parameters. A deterministic, locally-generated image
    # is used instead of a remote URL so the loader is reproducible offline (the
    # CI runner cannot reliably reach external image hosts).
    sample_text = "Describe this image."

    @staticmethod
    def _sample_image():
        """Build a deterministic synthetic RGB image (no network dependency)."""
        from PIL import Image
        import numpy as np

        h = w = 448
        yy, xx = np.mgrid[0:h, 0:w]
        r = (xx * 255 // w).astype(np.uint8)
        g = (yy * 255 // h).astype(np.uint8)
        b = ((xx + yy) * 255 // (h + w)).astype(np.uint8)
        arr = np.stack([r, g, b], axis=-1)
        return Image.fromarray(arr, mode="RGB")

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None
        self.config = None

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
            model="Qwen 3.6 MoE VL",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen 3.6 MoE VL model instance for this variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Qwen 3.6 MoE VL model instance for image-to-text.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {"low_cpu_mem_usage": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        # Force use_cache=False so the forward output does not include a
        # DynamicCache, which the runner's pytree comparator can't diff
        # leaf-wise against the CPU golden (same pattern as the qwen_2_5_vl
        # and qwen_3_5 loaders).
        model.config.use_cache = False
        if getattr(model.config, "text_config", None) is not None:
            model.config.text_config.use_cache = False

        model.eval()
        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Qwen 3.6 MoE VL model.

        Args:
            dtype_override: Optional torch.dtype for the pixel_values.
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
                    {"type": "image", "image": self._sample_image()},
                    {"type": "text", "text": self.sample_text},
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

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
