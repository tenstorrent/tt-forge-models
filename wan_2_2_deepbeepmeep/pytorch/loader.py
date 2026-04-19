#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepBeepMeep/Wan2.2 single-file safetensors model loader implementation.

Loads Wan 2.2 text-to-video diffusion models from single-file safetensors
checkpoints hosted at DeepBeepMeep/Wan2.2.

The Wan 2.2 T2V models use a Mixture-of-Experts (MoE) diffusion transformer
architecture with separate high-noise and low-noise expert checkpoints:
- High-noise expert: handles early denoising steps (overall layout)
- Low-noise expert: handles later denoising steps (detail refinement)

Available variants:
- WAN22_T2V_14B_HIGH_BF16: Text-to-Video 14B high-noise expert, bf16 precision
- WAN22_T2V_5B_BF16: Text-to-Video 5B, bf16 precision
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

SINGLE_FILE_REPO = "DeepBeepMeep/Wan2.2"
BASE_PIPELINE_14B = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
BASE_PIPELINE_5B = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

TRANSFORMER_NUM_FRAMES = 2
TRANSFORMER_HEIGHT = 4
TRANSFORMER_WIDTH = 4
TRANSFORMER_TEXT_SEQ_LEN = 8


class ModelVariant(StrEnum):
    """Available DeepBeepMeep/Wan2.2 variants."""

    WAN22_T2V_14B_HIGH_BF16 = "2.2_T2V_14B_HighNoise_bf16"
    WAN22_T2V_5B_BF16 = "2.2_T2V_5B_bf16"


_SINGLE_FILES = {
    ModelVariant.WAN22_T2V_14B_HIGH_BF16: {
        "file": "wan2.2_text2video_14B_high_mbf16.safetensors",
        "base_pipeline": BASE_PIPELINE_14B,
    },
    ModelVariant.WAN22_T2V_5B_BF16: {
        "file": "wan2.2_text2video_5B_mbf16.safetensors",
        "base_pipeline": BASE_PIPELINE_5B,
    },
}


class ModelLoader(ForgeModel):
    """DeepBeepMeep/Wan2.2 single-file safetensors model loader."""

    _VARIANTS = {
        ModelVariant.WAN22_T2V_14B_HIGH_BF16: ModelConfig(
            pretrained_model_name=SINGLE_FILE_REPO,
        ),
        ModelVariant.WAN22_T2V_5B_BF16: ModelConfig(
            pretrained_model_name=SINGLE_FILE_REPO,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_T2V_5B_BF16

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_2_2_DEEPBEEPMEEP",
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
        """Load the Wan 2.2 T2V transformer from a single-file checkpoint.

        Returns the transformer nn.Module directly for compilation testing.
        """
        from diffusers import WanTransformer3DModel
        from huggingface_hub import hf_hub_download

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        variant_info = _SINGLE_FILES[self._variant]
        local_path = hf_hub_download(
            repo_id=SINGLE_FILE_REPO,
            filename=variant_info["file"],
        )

        self._transformer = WanTransformer3DModel.from_single_file(
            local_path,
            config=variant_info["base_pipeline"],
            subfolder="transformer",
            torch_dtype=compute_dtype,
        )

        return self._transformer

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare tensor inputs for the WanTransformer3DModel forward pass."""
        if self._transformer is None:
            self.load_model()

        dtype = torch.bfloat16
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
