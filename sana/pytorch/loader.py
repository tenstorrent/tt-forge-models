# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Sana model loader implementation for text-to-image generation.

Sana is a fast text-to-image diffusion transformer from NVIDIA / Efficient-Large-Model.
Repository: https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers

This loader returns only the SanaTransformer2DModel and synthesizes inputs
matching the diffusers pipeline contract (latents from a DC-AE-style VAE
with 32x downscale, encoder hidden states from a Gemma-2 text encoder).
The text encoder and VAE are not materialized to keep the bringup focused
on the diffusion transformer.
"""
from typing import Optional

import torch
from diffusers import SanaTransformer2DModel

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
    """Available Sana model variants."""

    SANA_1600M_1024PX = "1600M_1024px"


class ModelLoader(ForgeModel):
    """Sana transformer loader for text-to-image generation."""

    _VARIANTS = {
        ModelVariant.SANA_1600M_1024PX: ModelConfig(
            pretrained_model_name="Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SANA_1600M_1024PX

    # Pipeline-side constants used by load_inputs(). These match the
    # diffusers Sana defaults at 1024px and the transformer config
    # exposed at the HF repo above.
    _MAX_SEQUENCE_LENGTH = 300
    _CAPTION_CHANNELS = 2304  # Gemma-2 hidden size
    _LATENT_CHANNELS = 32  # SanaTransformer2DModel.in_channels
    _LATENT_HW = 32  # sample_size for 1024px / 32x VAE downscale

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Sana",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Return the SanaTransformer2DModel for the configured variant."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        model = SanaTransformer2DModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Synthesize transformer inputs matching the SanaPipeline call site.

        SanaPipeline.__call__ invokes the transformer as::

            self.transformer(
                latent_model_input,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                timestep=timestep,
                return_dict=False,
                attention_kwargs=...,
            )
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        hidden_states = torch.randn(
            batch_size,
            self._LATENT_CHANNELS,
            self._LATENT_HW,
            self._LATENT_HW,
            dtype=dtype,
        )
        encoder_hidden_states = torch.randn(
            batch_size,
            self._MAX_SEQUENCE_LENGTH,
            self._CAPTION_CHANNELS,
            dtype=dtype,
        )
        timestep = torch.tensor([1.0] * batch_size, dtype=dtype)

        # encoder_attention_mask intentionally omitted: TT-MLIR's
        # ttir.scaled_dot_product_attention requires mask dim 2 == query
        # sequence length, but Sana cross-attention has Q=1024 (32x32 latents)
        # vs K=300 (text). The op accepts encoder_attention_mask=None.
        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "return_dict": False,
        }

    def unpack_forward_output(self, output):
        """Unpack the transformer output to a single tensor."""
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
