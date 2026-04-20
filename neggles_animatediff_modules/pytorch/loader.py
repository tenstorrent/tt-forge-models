# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
neggles/animatediff-modules model loader implementation.

The neggles/animatediff-modules repository re-hosts the original AnimateDiff
motion module weights (mm_sd_v14, mm_sd_v15, mm_sd_v15_v2) as raw safetensors.
This loader downloads one of the motion module checkpoints, uses
``MotionAdapter.from_single_file`` to build the diffusers MotionAdapter, and
wraps it with a Stable Diffusion 1.5 base to return a UNetMotionModel for
text-to-video generation.

Reference: https://huggingface.co/neggles/animatediff-modules
"""

from typing import Any, Optional

import torch
from diffusers import AnimateDiffPipeline, MotionAdapter
from huggingface_hub import hf_hub_download

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

BASE_MODEL = "emilianJR/epiCRealism"


class ModelVariant(StrEnum):
    """Available neggles/animatediff-modules variants."""

    MM_SD_V14 = "mm_sd_v14"
    MM_SD_V15 = "mm_sd_v15"
    MM_SD_V15_V2 = "mm_sd_v15_v2"


class ModelLoader(ForgeModel):
    """neggles/animatediff-modules motion module loader."""

    _VARIANTS = {
        ModelVariant.MM_SD_V14: ModelConfig(
            pretrained_model_name="neggles/animatediff-modules",
        ),
        ModelVariant.MM_SD_V15: ModelConfig(
            pretrained_model_name="neggles/animatediff-modules",
        ),
        ModelVariant.MM_SD_V15_V2: ModelConfig(
            pretrained_model_name="neggles/animatediff-modules",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MM_SD_V15_V2

    _MODULE_FILES = {
        ModelVariant.MM_SD_V14: "mm_sd_v14.fp16.safetensors",
        ModelVariant.MM_SD_V15: "mm_sd_v15.fp16.safetensors",
        ModelVariant.MM_SD_V15_V2: "mm_sd_v15_v2.fp16.safetensors",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[AnimateDiffPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="neggles_animatediff_modules",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the UNetMotionModel backed by the neggles motion module weights."""
        dtype = dtype_override if dtype_override is not None else torch.float32

        ckpt_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename=self._MODULE_FILES[self._variant],
        )

        adapter = MotionAdapter.from_single_file(ckpt_path, torch_dtype=dtype)

        self.pipeline = AnimateDiffPipeline.from_pretrained(
            BASE_MODEL,
            motion_adapter=adapter,
            torch_dtype=dtype,
        )

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        """Build dummy latents for the UNetMotionModel forward pass."""
        dtype = dtype_override if dtype_override is not None else torch.float32

        batch_size = 1
        num_frames = 16
        height = 64
        width = 64
        in_channels = 4
        cross_attention_dim = 768

        sample = torch.randn(
            (batch_size, in_channels, num_frames, height // 8, width // 8),
            dtype=dtype,
        )
        timestep = torch.randint(0, 1000, (1,))
        encoder_hidden_states = torch.randn(
            (batch_size, 77, cross_attention_dim),
            dtype=dtype,
        )

        return {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if hasattr(output, "sample"):
            return output.sample
        if isinstance(output, tuple):
            return output[0]
        return output
