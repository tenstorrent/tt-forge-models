# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VanillaMotionBotModels Wan 2.2 I2V model loader implementation for video generation.

Loads the Wan 2.2 I2V A14B transformer from
Wan-AI/Wan2.2-I2V-A14B-Diffusers.
"""

from typing import Any, Optional

import torch

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

BASE_REPO = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

TRANSFORMER_NUM_FRAMES = 2
TRANSFORMER_HEIGHT = 4
TRANSFORMER_WIDTH = 4
TRANSFORMER_TEXT_SEQ_LEN = 8


class ModelVariant(StrEnum):
    """Available VanillaMotionBotModels variants."""

    I2V_14B_HIGHNOISE_Q8_0 = "I2V_14B_HighNoise_Q8_0"
    I2V_14B_LOWNOISE_Q8_0 = "I2V_14B_LowNoise_Q8_0"


class ModelLoader(ForgeModel):
    """VanillaMotionBotModels Wan 2.2 I2V model loader for video generation tasks."""

    _VARIANTS = {
        ModelVariant.I2V_14B_HIGHNOISE_Q8_0: ModelConfig(
            pretrained_model_name=BASE_REPO,
        ),
        ModelVariant.I2V_14B_LOWNOISE_Q8_0: ModelConfig(
            pretrained_model_name=BASE_REPO,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.I2V_14B_HIGHNOISE_Q8_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="VanillaMotionBotModels",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from diffusers import WanTransformer3DModel

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        self._transformer = WanTransformer3DModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder="transformer",
            torch_dtype=compute_dtype,
        )

        return self._transformer

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        if self._transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self._transformer.config

        return {
            "hidden_states": torch.randn(
                1,
                config.in_channels,
                TRANSFORMER_NUM_FRAMES,
                TRANSFORMER_HEIGHT,
                TRANSFORMER_WIDTH,
                dtype=dtype,
            ),
            "encoder_hidden_states": torch.randn(
                1,
                TRANSFORMER_TEXT_SEQ_LEN,
                config.text_dim,
                dtype=dtype,
            ),
            "timestep": torch.tensor([500], dtype=torch.long),
            "return_dict": False,
        }
