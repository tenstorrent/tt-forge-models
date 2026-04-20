# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VLM-3R LLaVA-Video Qwen2 LoRA model loader implementation for multimodal conditional generation.
"""

from typing import Optional

import numpy as np
from peft import PeftModel
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

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
from ...tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available VLM-3R LLaVA-Video Qwen2 LoRA model variants."""

    VLM_3R_LLAVA_QWEN2_LORA = "vlm-3r-llava-qwen2-lora"


class ModelLoader(ForgeModel):
    """VLM-3R LLaVA-Video Qwen2 LoRA model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.VLM_3R_LLAVA_QWEN2_LORA: ModelConfig(
            pretrained_model_name="Journey9ni/vlm-3r-llava-qwen2-lora",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VLM_3R_LLAVA_QWEN2_LORA

    BASE_MODEL_NAME = "lmms-lab/LLaVA-Video-7B-Qwen2"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize VLM-3R LLaVA-Video Qwen2 LoRA model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="VLM-3R-LLaVA-Qwen2-LoRA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(self.BASE_MODEL_NAME)
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the VLM-3R LoRA-merged model instance."""
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        base_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.BASE_MODEL_NAME, **model_kwargs
        )

        adapter_name = self._variant_config.pretrained_model_name
        model = PeftModel.from_pretrained(base_model, adapter_name)
        model = model.merge_and_unload()

        for param in model.parameters():
            param.requires_grad = False

        model.eval()

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for VLM-3R."""
        if self.processor is None:
            self._load_processor()

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": "Describe this video in detail."},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        # Create a small synthetic video (8 frames of 32x32 RGB)
        video = np.random.randint(0, 255, (8, 32, 32, 3), dtype=np.uint8)

        inputs = self.processor(text=text_prompt, videos=[video], return_tensors="pt")

        if dtype_override:
            inputs = {
                k: cast_input_to_type(v, dtype_override) for k, v in inputs.items()
            }

        return dict(inputs)
