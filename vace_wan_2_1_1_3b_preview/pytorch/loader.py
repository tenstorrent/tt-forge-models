#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VACE-Wan2.1-1.3B-Preview model loader implementation.

Loads the ali-vilab/VACE-Wan2.1-1.3B-Preview all-in-one video creation and
editing model and constructs a WanVACEPipeline for reference-to-video
generation. The preview repo packages the original Wan2.1-T2V-1.3B weights
fine-tuned with the VACE adapter.
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


class ModelVariant(StrEnum):
    """Available VACE-Wan2.1-1.3B-Preview variants."""

    VACE_WAN_2_1_1_3B_PREVIEW = "VACE-Wan2.1-1.3B-Preview"


class ModelLoader(ForgeModel):
    """VACE-Wan2.1-1.3B-Preview model loader for reference-to-video generation."""

    _VARIANTS = {
        ModelVariant.VACE_WAN_2_1_1_3B_PREVIEW: ModelConfig(
            pretrained_model_name="ali-vilab/VACE-Wan2.1-1.3B-Preview",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VACE_WAN_2_1_1_3B_PREVIEW

    DEFAULT_PROMPT = (
        "A character walking gracefully across a sunlit garden, "
        "smooth animation, detailed motion, cinematic lighting"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="VACE-Wan2.1-1.3B-Preview",
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
        """Load the VACE-Wan2.1-1.3B-Preview pipeline."""
        from diffusers import AutoencoderKLWan, WanVACEPipeline

        dtype = dtype_override if dtype_override is not None else torch.float32
        pretrained_model_name = self._variant_config.pretrained_model_name

        vae = AutoencoderKLWan.from_pretrained(
            pretrained_model_name,
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        self.pipeline = WanVACEPipeline.from_pretrained(
            pretrained_model_name,
            vae=vae,
            torch_dtype=dtype,
        )
        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for VACE reference-to-video generation."""
        prompt_value = prompt if prompt is not None else self.DEFAULT_PROMPT

        ref_image = Image.new("RGB", (832, 480), color=(128, 128, 200))

        return {
            "prompt": prompt_value,
            "reference_images": [ref_image],
            "height": 480,
            "width": 832,
            "num_frames": 9,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
        }
