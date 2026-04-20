# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
neggles/animatediff-modules model loader implementation.

The neggles/animatediff-modules repository re-hosts the original AnimateDiff
motion module weights (mm_sd_v14, mm_sd_v15, mm_sd_v15_v2) as raw safetensors.
This loader downloads one of the motion module checkpoints, converts its key
names to the diffusers MotionAdapter layout, and wraps it with a Stable
Diffusion 1.5 base UNet to form a UNetMotionModel for text-to-video generation.

Reference: https://huggingface.co/neggles/animatediff-modules
"""

from typing import Any, Optional

import torch
from diffusers import AnimateDiffPipeline, MotionAdapter
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

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

    # Motion module sequence length: v1 uses 24 frames, v2 uses 32.
    _MOTION_MAX_SEQ_LENGTH = {
        ModelVariant.MM_SD_V14: 24,
        ModelVariant.MM_SD_V15: 24,
        ModelVariant.MM_SD_V15_V2: 32,
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

    @staticmethod
    def _convert_motion_module_state_dict(raw_state_dict: dict) -> dict:
        """Rename raw AnimateDiff motion module keys to diffusers layout."""
        converted = {}
        for key, value in raw_state_dict.items():
            if "pos_encoder" in key:
                continue
            new_key = (
                key.replace(".norms.0", ".norm1")
                .replace(".norms.1", ".norm2")
                .replace(".ff_norm", ".norm3")
                .replace(".attention_blocks.0", ".attn1")
                .replace(".attention_blocks.1", ".attn2")
            )
            converted[new_key] = value
        return converted

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the UNetMotionModel backed by the neggles motion module weights."""
        dtype = dtype_override if dtype_override is not None else torch.float32

        ckpt_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename=self._MODULE_FILES[self._variant],
        )
        raw_state_dict = load_file(ckpt_path)
        converted_state_dict = self._convert_motion_module_state_dict(raw_state_dict)

        adapter = MotionAdapter(
            block_out_channels=(320, 640, 1280, 1280),
            motion_num_attention_heads=8,
            motion_max_seq_length=self._MOTION_MAX_SEQ_LENGTH[self._variant],
        )
        # strict=False: raw checkpoints omit positional encoding buffers that
        # MotionAdapter reconstructs from motion_max_seq_length.
        adapter.load_state_dict(converted_state_dict, strict=False)

        if dtype_override is not None:
            adapter = adapter.to(dtype=dtype)

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
