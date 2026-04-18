# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image-Edit Object Adder LoRA model loader implementation.

Loads the Qwen-Image-Edit-2511 base diffusion pipeline and applies the
prithivMLmods/Qwen-Image-Edit-2511-Object-Adder LoRA weights for
object addition in images while preserving background and lighting.

Available variants:
- OBJECT_ADDER_2511: Object Adder LoRA on Qwen-Image-Edit 2511
"""

from typing import Any, Optional

import torch
from diffusers import DiffusionPipeline

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)

BASE_MODEL = "Qwen/Qwen-Image-Edit-2511"
LORA_REPO = "prithivMLmods/Qwen-Image-Edit-2511-Object-Adder"


class ModelVariant(StrEnum):
    """Available Qwen-Image-Edit Object Adder variants."""

    OBJECT_ADDER_2511 = "ObjectAdder_2511"


class ModelLoader(ForgeModel):
    """Qwen-Image-Edit Object Adder LoRA model loader."""

    _VARIANTS = {
        ModelVariant.OBJECT_ADDER_2511: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.OBJECT_ADDER_2511

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[DiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_IMAGE_EDIT_OBJECT_ADDER",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype: torch.dtype = torch.float32):
        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )
        self.pipeline.load_lora_weights(LORA_REPO)
        return self.pipeline

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        dtype = dtype_override if dtype_override is not None else torch.float32

        if self.pipeline is None:
            self._load_pipeline(dtype)
        elif dtype_override is not None:
            self.pipeline.transformer = self.pipeline.transformer.to(dtype_override)

        return self.pipeline.transformer

    def load_inputs(
        self, dtype_override: Optional[torch.dtype] = None, **kwargs
    ) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.float32

        if self.pipeline is None:
            self._load_pipeline(dtype)

        prompt = (
            "Add the specified objects to the image while preserving "
            "the background lighting and surrounding elements"
        )

        text_inputs = self.pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_embeds = self.pipeline.text_encoder(
            text_inputs.input_ids,
            output_hidden_states=True,
        ).hidden_states[-1]
        prompt_embeds = prompt_embeds.to(dtype=dtype)

        in_channels = self.pipeline.transformer.config.in_channels
        hidden_states = torch.randn(1, in_channels, 8, 8, dtype=dtype)

        timestep = torch.tensor([1.0], dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
        }
