# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TurboWan2.1-T2V-1.3B-Diffusers text-to-video model loader implementation.

TurboWan2.1-T2V-1.3B-Diffusers is a distilled/turbo variant of the
Wan-AI/Wan2.1-T2V-1.3B text-to-video diffusion model, finetuned from
TurboDiffusion/TurboWan2.1-T2V-1.3B-480P for fast few-step inference.
"""

from typing import Any, Optional

import torch
from diffusers import DiffusionPipeline  # type: ignore[import]

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


class ModelVariant(StrEnum):
    """Available TurboWan2.1-T2V-1.3B model variants."""

    TURBOWAN_2_1_T2V_1_3B = "2.1_T2V_1.3B"


class ModelLoader(ForgeModel):
    """TurboWan2.1-T2V-1.3B-Diffusers text-to-video model loader implementation."""

    _VARIANTS = {
        ModelVariant.TURBOWAN_2_1_T2V_1_3B: ModelConfig(
            pretrained_model_name="IPostYellow/TurboWan2.1-T2V-1.3B-Diffusers",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.TURBOWAN_2_1_T2V_1_3B

    DEFAULT_PROMPT = (
        "A stylish woman walks down a Tokyo street filled with warm glowing neon"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[DiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="TurboWan2.1-T2V-1.3B",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(
        self, dtype_override: Optional[torch.dtype] = None
    ) -> DiffusionPipeline:
        pipe_kwargs = {
            "torch_dtype": (
                dtype_override if dtype_override is not None else torch.bfloat16
            ),
        }

        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            **pipe_kwargs,
        )

        return self.pipeline

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the TurboWan2.1-T2V-1.3B pipeline.

        Args:
            dtype_override: Optional torch dtype to instantiate the pipeline with.

        Returns:
            DiffusionPipeline: The TurboWan2.1-T2V-1.3B text-to-video pipeline.
        """
        if self.pipeline is None:
            self._load_pipeline(dtype_override=dtype_override)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype=dtype_override)

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for the TurboWan2.1-T2V-1.3B pipeline.

        Args:
            prompt: Optional text prompt for video generation.

        Returns:
            dict: Input dictionary with prompt for the pipeline.
        """
        prompt_value = prompt if prompt is not None else self.DEFAULT_PROMPT
        return {"prompt": prompt_value}
