#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 Distill Models loader implementation.

Loads the Wan 2.2 I2V base pipeline and applies distilled transformer weights
from lightx2v/Wan2.2-Distill-Models for fast 4-step image-to-video generation.

These distilled models replace the full DIT (transformer) weights rather than
using LoRA adapters, enabling 4-step inference instead of the typical 50+ steps.

Available variants:
- WAN22_I2V_HIGH_NOISE: Creative, diverse outputs (high noise distillation)
- WAN22_I2V_LOW_NOISE: Faithful, stable outputs (low noise distillation)
"""

from typing import Any, Dict, Optional

import torch
from diffusers import WanImageToVideoPipeline  # type: ignore[import]
from huggingface_hub import hf_hub_download  # type: ignore[import]
from safetensors.torch import load_file  # type: ignore[import]

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

BASE_MODEL = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
DISTILL_REPO = "lightx2v/Wan2.2-Distill-Models"

# Distilled weight filenames (BF16 variants)
WEIGHT_HIGH_NOISE = "wan2.2_i2v_A14b_high_noise_lightx2v_4step.safetensors"
WEIGHT_LOW_NOISE = "wan2.2_i2v_A14b_low_noise_lightx2v_4step.safetensors"

# Small test dimensions for transformer inputs
# Must be divisible by patch_size (1, 2, 2)
TRANSFORMER_NUM_FRAMES = 2
TRANSFORMER_HEIGHT = 4
TRANSFORMER_WIDTH = 4
TRANSFORMER_TEXT_SEQ_LEN = 8


class ModelVariant(StrEnum):
    """Available Wan 2.2 Distill Model variants."""

    WAN22_I2V_HIGH_NOISE = "2.2_I2V_HighNoise"
    WAN22_I2V_LOW_NOISE = "2.2_I2V_LowNoise"


_WEIGHT_FILES = {
    ModelVariant.WAN22_I2V_HIGH_NOISE: WEIGHT_HIGH_NOISE,
    ModelVariant.WAN22_I2V_LOW_NOISE: WEIGHT_LOW_NOISE,
}


class ModelLoader(ForgeModel):
    """Wan 2.2 Distill Models loader."""

    _VARIANTS = {
        ModelVariant.WAN22_I2V_HIGH_NOISE: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.WAN22_I2V_LOW_NOISE: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_I2V_HIGH_NOISE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[WanImageToVideoPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_DISTILL_MODELS",
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
        """Load the Wan 2.2 I2V transformer with distilled weights.

        Downloads the distilled safetensor weights from the HuggingFace repo
        and loads them into the pipeline's transformer component.

        Returns:
            WanTransformer3DModel with distilled weights.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        self.pipeline = WanImageToVideoPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

        weight_file = _WEIGHT_FILES[self._variant]
        weight_path = hf_hub_download(
            repo_id=DISTILL_REPO,
            filename=weight_file,
        )

        state_dict = load_file(weight_path)
        self.pipeline.transformer.load_state_dict(state_dict, strict=False)

        return self.pipeline.transformer

    def _load_transformer_inputs(self, dtype: torch.dtype) -> Dict[str, Any]:
        """Prepare synthetic inputs for the WanTransformer3DModel forward pass."""
        config = self.pipeline.transformer.config
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

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for the transformer forward pass.

        Returns:
            dict with tensor inputs for WanTransformer3DModel.
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        return self._load_transformer_inputs(dtype)
