# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SD3.5 Medium GGUF (calcuis/sd3.5-medium-gguf) model loader implementation.

Stable Diffusion 3.5 Medium is a text-to-image generation model in GGUF quantized format,
based on the SD3 MMDiT transformer architecture with 2B parameters.

Available variants:
- SD3_5_MEDIUM_Q4_K_M: Q4_K_M quantized variant
"""

from pathlib import Path
from typing import Optional

import torch
from diffusers import SD3Transformer2DModel, GGUFQuantizationConfig
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

REPO_ID = "calcuis/sd3.5-medium-gguf"


class ModelVariant(StrEnum):
    """Available SD3.5 Medium GGUF model variants."""

    SD3_5_MEDIUM_Q4_K_M = "sd3.5_medium_Q4_K_M"


class ModelLoader(ForgeModel):
    """SD3.5 Medium GGUF model loader implementation."""

    _VARIANTS = {
        ModelVariant.SD3_5_MEDIUM_Q4_K_M: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SD3_5_MEDIUM_Q4_K_M

    GGUF_FILE = "sd3.5_medium-q4_k_m.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SD3.5 Medium GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.transformer is None:
            model_path = hf_hub_download(repo_id=REPO_ID, filename=self.GGUF_FILE)
            config_dir = str(Path(__file__).parent / "config" / "transformer")
            compute_dtype = (
                dtype_override if dtype_override is not None else torch.float32
            )
            quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)
            self.transformer = SD3Transformer2DModel.from_single_file(
                model_path,
                config=config_dir,
                quantization_config=quantization_config,
                torch_dtype=compute_dtype,
            )

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.float32
        config = self.transformer.config

        latent_height = 128
        latent_width = 128

        hidden_states = torch.randn(
            batch_size,
            config.in_channels,
            latent_height,
            latent_width,
            dtype=dtype,
        )

        encoder_hidden_states = torch.randn(
            batch_size, 256, config.joint_attention_dim, dtype=dtype
        )

        pooled_projections = torch.randn(
            batch_size, config.pooled_projection_dim, dtype=dtype
        )

        timestep = torch.tensor([1.0], dtype=dtype).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_projections,
            "timestep": timestep,
        }
