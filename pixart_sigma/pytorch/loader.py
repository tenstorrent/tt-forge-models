# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PixArt-Sigma model loader implementation
"""

import torch
from typing import Optional

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from diffusers import PixArtTransformer2DModel


class ModelVariant(StrEnum):
    """Available PixArt-Sigma model variants."""

    XL_2_1024_MS = "XL-2-1024-MS"
    SDXLVAE_T5_DIFFUSERS = "sdxlvae-T5-diffusers"


class ModelLoader(ForgeModel):
    """PixArt-Sigma model loader implementation."""

    _VARIANTS = {
        ModelVariant.XL_2_1024_MS: ModelConfig(
            pretrained_model_name="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        ),
        ModelVariant.SDXLVAE_T5_DIFFUSERS: ModelConfig(
            # The sdxlvae_T5_diffusers repo is a partial pipeline (VAE/tokenizer/scheduler
            # only, no transformer). The transformer is sourced from PixArt-Sigma-XL-2-1024-MS.
            pretrained_model_name="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.XL_2_1024_MS

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="PixArt-Sigma",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the PixArt-Sigma transformer.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            PixArtTransformer2DModel: The pre-trained PixArt-Sigma transformer.
        """
        load_kwargs = {"use_safetensors": True}
        if dtype_override is not None:
            load_kwargs["torch_dtype"] = dtype_override
        load_kwargs |= kwargs

        self.transformer = PixArtTransformer2DModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder="transformer",
            **load_kwargs,
        )

        if dtype_override is not None:
            self.transformer = self.transformer.to(dtype_override)

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the PixArt-Sigma transformer.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size for the inputs.

        Returns:
            dict: Dictionary of tensor inputs for the transformer.
        """
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.float32
        config = self.transformer.config

        sample_size = config.sample_size
        in_channels = config.in_channels
        cross_attention_dim = config.cross_attention_dim

        hidden_states = torch.randn(
            batch_size, in_channels, sample_size, sample_size, dtype=dtype
        )

        max_sequence_length = 300
        encoder_hidden_states = torch.randn(
            batch_size, max_sequence_length, cross_attention_dim, dtype=dtype
        )

        timestep = torch.tensor([1], dtype=torch.long).expand(batch_size)

        encoder_attention_mask = torch.ones(
            batch_size, max_sequence_length, dtype=dtype
        )

        inputs = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "encoder_attention_mask": encoder_attention_mask,
        }

        if self.transformer.use_additional_conditions:
            image_resolution = float(sample_size * 8)
            resolution = torch.tensor([image_resolution], dtype=dtype).expand(
                batch_size
            )
            aspect_ratio = torch.tensor([1.0], dtype=dtype).expand(batch_size)
            inputs["added_cond_kwargs"] = {
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
            }

        return inputs
