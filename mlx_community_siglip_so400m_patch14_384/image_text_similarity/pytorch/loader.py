# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mlx-community/siglip-so400m-patch14-384 model loader for image-text similarity.
"""

import torch
from transformers import AutoProcessor, AutoModel
from typing import Optional
from datasets import load_dataset

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


class ModelVariant(StrEnum):
    """Available mlx-community SigLIP So400m Patch14 384 model variants."""

    SO400M_PATCH14_384 = "So400m_Patch14_384"


class ModelLoader(ForgeModel):
    """mlx-community/siglip-so400m-patch14-384 model loader for image-text similarity."""

    _VARIANTS = {
        ModelVariant.SO400M_PATCH14_384: ModelConfig(
            pretrained_model_name="mlx-community/siglip-so400m-patch14-384",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SO400M_PATCH14_384

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.text_prompts = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="mlx-community SigLIP So400m Patch14 384",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        self.text_prompts = ["a photo of 2 cats", "a photo of 2 dogs"]

        inputs = self.processor(
            text=self.text_prompts,
            images=image,
            return_tensors="pt",
            padding="max_length",
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].dtype == torch.float32:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
