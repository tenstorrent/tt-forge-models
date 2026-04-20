# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kandinsky 2.2 Decoder UNet model loader implementation.

Extracts the UNet2DConditionModel decoder from the Kandinsky 2.2 pipeline for
direct inference testing with synthetic tensor inputs. The decoder is image
embedding conditioned (via CLIP-ViT-G) rather than text conditioned.
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
from diffusers import UNet2DConditionModel


class ModelVariant(StrEnum):
    """Available Kandinsky 2.2 Decoder model variants."""

    KANDINSKY_2_2_DECODER = "kandinsky-2-2-decoder"


class ModelLoader(ForgeModel):
    """Kandinsky 2.2 Decoder UNet model loader implementation."""

    _VARIANTS = {
        ModelVariant.KANDINSKY_2_2_DECODER: ModelConfig(
            pretrained_model_name="kandinsky-community/kandinsky-2-2-decoder",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KANDINSKY_2_2_DECODER

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Kandinsky 2.2 Decoder",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Kandinsky 2.2 decoder UNet.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            UNet2DConditionModel: The pre-trained Kandinsky 2.2 decoder UNet.
        """
        dtype = dtype_override or torch.float32
        unet = UNet2DConditionModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder="unet",
            torch_dtype=dtype,
            **kwargs,
        )
        return unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return synthetic tensor inputs for the Kandinsky 2.2 UNet.

        The Kandinsky 2.2 decoder UNet is image-embedding conditioned via the
        encoder_hid_proj / addition_embed image projection heads. It expects:
        - sample: noised latent (batch, in_channels=4, height=64, width=64)
        - timestep: diffusion timestep
        - added_cond_kwargs: image_embeds (batch, encoder_hid_dim=1280)

        Args:
            dtype_override: Optional torch.dtype for input tensors.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Dictionary of input tensors for the UNet forward pass.
        """
        dtype = dtype_override or torch.float32
        return {
            "sample": torch.randn(batch_size, 4, 64, 64, dtype=dtype),
            "timestep": torch.tensor([0]),
            "encoder_hidden_states": None,
            "added_cond_kwargs": {
                "image_embeds": torch.randn(batch_size, 1280, dtype=dtype),
            },
        }
