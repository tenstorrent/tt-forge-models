# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SkyReels-V2 Diffusion Forcing model loader for tt_forge_models.

SkyReels-V2 is an autoregressive Diffusion Forcing Transformer for long-form
video generation. It supports text-to-video and image-to-video synthesis with
per-token noise levels, enabling synchronous or asynchronous denoising
schedules across frames.

Repository:
- https://huggingface.co/Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers
"""

from typing import Any, Optional

import torch
from diffusers import SkyReelsV2DiffusionForcingPipeline  # type: ignore[import]

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


class ModelVariant(StrEnum):
    """Available SkyReels-V2 Diffusion Forcing variants."""

    SKYREELS_V2_DF_1_3B_540P = "1.3B_540P"


class ModelLoader(ForgeModel):
    """SkyReels-V2 Diffusion Forcing model loader for text-to-video generation."""

    _VARIANTS = {
        ModelVariant.SKYREELS_V2_DF_1_3B_540P: ModelConfig(
            pretrained_model_name="Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.SKYREELS_V2_DF_1_3B_540P

    DEFAULT_PROMPT = (
        "A graceful white swan swimming in a serene lake at dawn, "
        "soft golden light reflecting on the water, cinematic composition"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[SkyReelsV2DiffusionForcingPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SkyReelsV2_DF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the SkyReels-V2 Diffusion Forcing pipeline.

        Returns:
            SkyReelsV2DiffusionForcingPipeline ready for inference.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        self.pipeline = SkyReelsV2DiffusionForcingPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for the SkyReels-V2 Diffusion Forcing pipeline.

        Returns:
            Dict with prompt for the pipeline.
        """
        prompt_value = prompt if prompt is not None else self.DEFAULT_PROMPT
        return {"prompt": prompt_value}
