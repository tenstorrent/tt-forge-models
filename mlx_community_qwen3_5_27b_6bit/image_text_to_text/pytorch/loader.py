# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mlx-community/Qwen3.5-27B-6bit model loader for image-text-to-text generation.
"""

import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....tools.utils import get_file


class ModelVariant(StrEnum):
    """Available mlx-community Qwen3.5-27B-6bit model variants."""

    QWEN_3_5_27B_6BIT = "27b_6bit"


class ModelLoader(ForgeModel):
    """mlx-community Qwen3.5-27B-6bit model loader for image-text-to-text generation."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_27B_6BIT: ModelConfig(
            pretrained_model_name="mlx-community/Qwen3.5-27B-6bit",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_27B_6BIT

    sample_text = "Describe this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="mlx-community Qwen3.5-27B-6bit",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the mlx-community Qwen3.5-27B-6bit model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        # MLX-community models store weights in MLX's packed-integer format (uint32 +
        # bfloat16 scales/biases), which is incompatible with standard safetensors
        # loading. For compile-only targets we only need the correct architecture, so
        # we load the config (stripping the MLX quantization_config which lacks the
        # `quant_method` key required by transformers 5.2+) and initialise fresh
        # float weights via from_config.
        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        if hasattr(config, "quantization_config") and isinstance(
            config.quantization_config, dict
        ):
            del config.quantization_config

        model = AutoModelForImageTextToText.from_config(config)
        if dtype_override is not None:
            model = model.to(dtype_override)
        model.eval()

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the mlx-community Qwen3.5-27B-6bit model."""
        if self.processor is None:
            self._load_processor()

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file).convert("RGB")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.sample_text},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].is_floating_point():
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
