#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 I2V A14B GGUF model loader implementation.

Loads GGUF-quantized Wan 2.2 Image-to-Video diffusion transformers from
bullerwins/Wan2.2-I2V-A14B-GGUF. Uses the upstream Wan-AI/Wan2.2-I2V-A14B-Diffusers
config for model construction.

Available variants:
- WAN22_I2V_HIGH_NOISE_Q4_K_M: High-noise expert, Q4_K_M quantization
- WAN22_I2V_LOW_NOISE_Q4_K_M: Low-noise expert, Q4_K_M quantization
"""

from typing import Any, Optional

import torch
from diffusers import GGUFQuantizationConfig, WanTransformer3DModel
from huggingface_hub import hf_hub_download

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

GGUF_REPO = "bullerwins/Wan2.2-I2V-A14B-GGUF"
CONFIG_REPO = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"


class ModelVariant(StrEnum):
    """Available Wan 2.2 I2V A14B GGUF variants."""

    WAN22_I2V_HIGH_NOISE_Q4_K_M = "2.2_I2V_HighNoise_Q4_K_M"
    WAN22_I2V_LOW_NOISE_Q4_K_M = "2.2_I2V_LowNoise_Q4_K_M"


_GGUF_FILES = {
    ModelVariant.WAN22_I2V_HIGH_NOISE_Q4_K_M: "wan2.2_i2v_high_noise_14B_Q4_K_M.gguf",
    ModelVariant.WAN22_I2V_LOW_NOISE_Q4_K_M: "wan2.2_i2v_low_noise_14B_Q4_K_M.gguf",
}


class ModelLoader(ForgeModel):
    """Wan 2.2 I2V A14B GGUF model loader for the diffusion transformer."""

    _VARIANTS = {
        ModelVariant.WAN22_I2V_HIGH_NOISE_Q4_K_M: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
        ModelVariant.WAN22_I2V_LOW_NOISE_Q4_K_M: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_I2V_HIGH_NOISE_Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_2_2_I2V_GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_transformer(
        self, dtype: torch.dtype = torch.float32
    ) -> WanTransformer3DModel:
        gguf_file = _GGUF_FILES[self._variant]
        model_path = hf_hub_download(repo_id=GGUF_REPO, filename=gguf_file)

        quantization_config = GGUFQuantizationConfig(compute_dtype=dtype)
        self._transformer = WanTransformer3DModel.from_single_file(
            model_path,
            quantization_config=quantization_config,
            config=CONFIG_REPO,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        self._transformer.eval()
        return self._transformer

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._transformer is None:
            return self._load_transformer(dtype)
        if dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)
        return self._transformer

    def load_inputs(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        batch_size: int = 1,
        **kwargs,
    ) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.float32

        in_channels = 36
        text_dim = 4096
        txt_seq_len = 32
        num_frames, height, width = 5, 8, 8

        hidden_states = torch.randn(
            batch_size, in_channels, num_frames, height, width, dtype=dtype
        )
        encoder_hidden_states = torch.randn(
            batch_size, txt_seq_len, text_dim, dtype=dtype
        )
        timestep = torch.tensor([500.0] * batch_size, dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
        }
