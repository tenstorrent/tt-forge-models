# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen Image Edit 2509 Multi-Angle Lighting LoRA model loader implementation.

Loads the dx8152/Qwen-Edit-2509-Multi-Angle-Lighting LoRA adapter on top of the
Qwen/Qwen-Image-Edit-2509 base diffusion pipeline and extracts the transformer
component for compilation.
"""

from typing import Any, Optional

import torch
from PIL import Image

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
from .src.model_utils import (
    load_qwen_image_edit_plus_pipeline,
    qwen_image_edit_plus_preprocessing,
)


class ModelVariant(StrEnum):
    """Available Qwen Image Edit Multi-Angle Lighting model variants."""

    V251121 = "v251121"


class ModelLoader(ForgeModel):
    """Qwen Image Edit 2509 Multi-Angle Lighting LoRA model loader."""

    _VARIANTS = {
        ModelVariant.V251121: ModelConfig(
            pretrained_model_name="dx8152/Qwen-Edit-2509-Multi-Angle-Lighting",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V251121

    _BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"

    _LORA_WEIGHT_NAMES = {
        ModelVariant.V251121: "多角度灯光-251121.safetensors",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="Qwen Image Edit 2509 Multi-Angle Lighting",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override or torch.bfloat16
        self.pipeline = load_qwen_image_edit_plus_pipeline(
            self._BASE_MODEL,
            self._variant_config.pretrained_model_name,
            self._LORA_WEIGHT_NAMES[self._variant],
            dtype=dtype,
        )
        return self.pipeline.transformer

    def load_inputs(self, **kwargs) -> Any:
        if self.pipeline is None:
            self.load_model()

        image = Image.new("RGB", (512, 512), color=(128, 128, 128))
        luminance_map = Image.new("RGB", (512, 512), color=(255, 255, 255))
        prompt = "使用图2的亮度贴图对图1重新照明(光源来自前方)"

        (
            latent_model_input,
            timestep,
            prompt_embeds,
            prompt_embeds_mask,
            img_shapes,
            guidance,
        ) = qwen_image_edit_plus_preprocessing(
            self.pipeline, prompt, [image, luminance_map]
        )

        inputs = {
            "hidden_states": latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "encoder_hidden_states_mask": prompt_embeds_mask,
            "img_shapes": img_shapes,
            "guidance": guidance,
            "return_dict": False,
        }

        return inputs
