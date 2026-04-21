# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kandinsky 3 UNet model loader implementation.

Extracts the Kandinsky3UNet from the Kandinsky 3 text-to-image pipeline for
direct inference testing with synthetic tensor inputs.
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
from diffusers import Kandinsky3UNet


class ModelVariant(StrEnum):
    """Available Kandinsky 3 model variants."""

    KANDINSKY_3 = "kandinsky-3"


class ModelLoader(ForgeModel):
    """Kandinsky 3 UNet model loader implementation."""

    _VARIANTS = {
        ModelVariant.KANDINSKY_3: ModelConfig(
            pretrained_model_name="kandinsky-community/kandinsky-3",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KANDINSKY_3

    # Text encoder max sequence length used by the Kandinsky 3 pipeline.
    max_sequence_length = 128

    # Default latent spatial size (1024px image with 8x VAE downsample).
    sample_size = 128

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.unet = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Kandinsky 3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Kandinsky 3 UNet.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            Kandinsky3UNet: The pre-trained Kandinsky 3 UNet.
        """
        load_kwargs = {}
        if dtype_override is not None:
            load_kwargs["torch_dtype"] = dtype_override
        load_kwargs |= kwargs

        self.unet = Kandinsky3UNet.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder="unet",
            **load_kwargs,
        )
        return self.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return synthetic tensor inputs for the Kandinsky 3 UNet.

        The UNet expects:
        - sample: noised latent (batch, in_channels=4, sample_size, sample_size)
        - timestep: diffusion timestep
        - encoder_hidden_states: text encoding (batch, 128, encoder_hid_dim=4096)
        - encoder_attention_mask: text attention mask (batch, 128)

        Args:
            dtype_override: Optional torch.dtype for input tensors.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Dictionary of input tensors for the UNet forward pass.
        """
        if self.unet is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override or torch.float32
        config = self.unet.config

        sample = torch.randn(
            batch_size,
            config.in_channels,
            self.sample_size,
            self.sample_size,
            dtype=dtype,
        )
        timestep = torch.tensor([1], dtype=torch.long).expand(batch_size)
        encoder_hidden_states = torch.randn(
            batch_size, self.max_sequence_length, config.encoder_hid_dim, dtype=dtype
        )
        encoder_attention_mask = torch.ones(
            batch_size, self.max_sequence_length, dtype=torch.long
        )

        return {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
        }
