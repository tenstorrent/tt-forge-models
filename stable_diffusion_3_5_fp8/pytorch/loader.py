# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion 3.5 FP8 model loader implementation.

Loads FP8-quantized single-file checkpoints from the Comfy-Org/stable-diffusion-3.5-fp8 repository.
Avoids accessing gated stabilityai/* repos by using a local transformer config
and generating synthetic inputs.

Available variants:
- LARGE_FP8: sd3.5_large_fp8_scaled.safetensors
- MEDIUM_FP8: sd3.5_medium_incl_clips_t5xxlfp8scaled.safetensors
"""

from typing import Optional

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
from .src.model_utils import load_transformer, TRANSFORMER_CONFIGS


class ModelVariant(StrEnum):
    """Available Stable Diffusion 3.5 FP8 model variants."""

    LARGE_FP8 = "Large_FP8"
    MEDIUM_FP8 = "Medium_FP8"


class ModelLoader(ForgeModel):
    """Stable Diffusion 3.5 FP8 model loader implementation."""

    _VARIANTS = {
        ModelVariant.LARGE_FP8: ModelConfig(
            pretrained_model_name="sd3.5_large_fp8_scaled.safetensors",
        ),
        ModelVariant.MEDIUM_FP8: ModelConfig(
            pretrained_model_name="sd3.5_medium_incl_clips_t5xxlfp8scaled.safetensors",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_FP8

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Stable Diffusion 3.5 FP8",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Stable Diffusion 3.5 FP8 transformer."""
        filename = self._variant_config.pretrained_model_name

        self._transformer = load_transformer(filename)

        if dtype_override is not None:
            self._transformer = self._transformer.to(dtype_override)

        return self._transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Generate synthetic inputs for the SD3.5 FP8 transformer."""
        if self._transformer is None:
            self.load_model(dtype_override=dtype_override)

        filename = self._variant_config.pretrained_model_name
        config = TRANSFORMER_CONFIGS[filename]
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        in_channels = config["in_channels"]
        joint_attention_dim = config["joint_attention_dim"]
        pooled_projection_dim = config["pooled_projection_dim"]
        patch_size = config["patch_size"]
        sample_size = config["sample_size"]

        height = sample_size
        width = sample_size

        hidden_states = torch.randn(
            batch_size,
            in_channels,
            height // patch_size,
            width // patch_size,
            dtype=dtype,
        )
        encoder_hidden_states = torch.randn(
            batch_size, 256, joint_attention_dim, dtype=dtype
        )
        pooled_projections = torch.randn(batch_size, pooled_projection_dim, dtype=dtype)
        timestep = torch.tensor([1.0] * batch_size, dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_projections,
            "timestep": timestep,
        }
