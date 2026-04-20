# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TinyLLaVA-Qwen2-0.5B-SigLIP model loader implementation for image-text-to-text tasks.
"""

from typing import Optional

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor

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
from ....tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available TinyLLaVA-Qwen2-0.5B-SigLIP model variants."""

    TINY_LLAVA_QWEN2_0_5B_SIGLIP = "tiny_llava_qwen2_0_5b_siglip"


class ModelLoader(ForgeModel):
    """TinyLLaVA-Qwen2-0.5B-SigLIP model loader for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.TINY_LLAVA_QWEN2_0_5B_SIGLIP: ModelConfig(
            pretrained_model_name="Zhang199/TinyLLaVA-Qwen2-0.5B-SigLIP",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_LLAVA_QWEN2_0_5B_SIGLIP

    sample_text = "What is shown in this image?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize TinyLLaVA-Qwen2-0.5B-SigLIP model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="TinyLLaVA-Qwen2-0.5B-SigLIP",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the TinyLLaVA-Qwen2-0.5B-SigLIP model instance."""
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.eval()

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for TinyLLaVA-Qwen2-0.5B-SigLIP."""
        if self.processor is None:
            self._load_processor()

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.sample_text},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            conversation, padding=True, add_generation_prompt=True
        )

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")

        if dtype_override is not None:
            inputs = {
                k: cast_input_to_type(v, dtype_override) for k, v in inputs.items()
            }

        if batch_size > 1:
            inputs = {
                k: v.repeat_interleave(batch_size, dim=0) for k, v in inputs.items()
            }

        return inputs
