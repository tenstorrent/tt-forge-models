# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Prompt Depth Anything ViT-Small model loader implementation for
prompt-conditioned metric depth estimation.
"""
import requests
import torch
from PIL import Image
from transformers import (
    PromptDepthAnythingForDepthEstimation,
    PromptDepthAnythingImageProcessor,
)
from typing import Optional

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available Prompt Depth Anything ViT-Small model variants."""

    VITS = "vits"


class ModelLoader(ForgeModel):
    """Prompt Depth Anything ViT-Small model loader implementation."""

    _VARIANTS = {
        ModelVariant.VITS: ModelConfig(
            pretrained_model_name="depth-anything/prompt-depth-anything-vits-hf",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VITS

    _IMAGE_URL = (
        "https://github.com/DepthAnything/PromptDA/blob/main/"
        "assets/example_images/image.jpg?raw=true"
    )
    _PROMPT_DEPTH_URL = (
        "https://github.com/DepthAnything/PromptDA/blob/main/"
        "assets/example_images/arkit_depth.png?raw=true"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="PromptDepthAnythingViTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_DEPTH_EST,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.processor = PromptDepthAnythingImageProcessor.from_pretrained(
            pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = PromptDepthAnythingForDepthEstimation.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        image = Image.open(requests.get(self._IMAGE_URL, stream=True).raw)
        prompt_depth = Image.open(requests.get(self._PROMPT_DEPTH_URL, stream=True).raw)

        inputs = self.processor(
            images=image, return_tensors="pt", prompt_depth=prompt_depth
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].dtype.is_floating_point:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
