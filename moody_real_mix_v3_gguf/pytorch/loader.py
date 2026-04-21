# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Moody Real Mix v3 GGUF (Gthalmie1/moody-real-mix-v3-gguf) model loader implementation.

Moody Real Mix v3 is a text-to-image diffusion model in GGUF quantized format,
based on the Lumina2 architecture. It is a GGUF quantization of the
catlover1937/moody-real-mix CivitAI model.
"""

from typing import Optional

import torch
from diffusers import GGUFQuantizationConfig, Lumina2Transformer2DModel

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

REPO_ID = "Gthalmie1/moody-real-mix-v3-gguf"

# Lumina-Image-2.0 architecture constants (Gemma-2-2b text encoder)
IN_CHANNELS = 16
CAP_FEAT_DIM = 2304


class ModelVariant(StrEnum):
    """Available Moody Real Mix v3 GGUF model variants."""

    MOODY_REAL_MIX_V3_Q4_K_M = "moodyRealMix_zitV3_q4_k_m"


_GGUF_FILES = {
    ModelVariant.MOODY_REAL_MIX_V3_Q4_K_M: "moodyRealMix_zitV3_q4_k_m.gguf",
}


class ModelLoader(ForgeModel):
    """Moody Real Mix v3 GGUF model loader for text-to-image generation."""

    _VARIANTS = {
        ModelVariant.MOODY_REAL_MIX_V3_Q4_K_M: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOODY_REAL_MIX_V3_Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Moody Real Mix v3 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        gguf_filename = _GGUF_FILES[self._variant]
        self.transformer = Lumina2Transformer2DModel.from_single_file(
            f"https://huggingface.co/{REPO_ID}/resolve/main/{gguf_filename}",
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
        )
        self.transformer.eval()
        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # Latent image: (B, in_channels, H, W)
        height = 128
        width = 128
        hidden_states = torch.randn(batch_size, IN_CHANNELS, height, width, dtype=dtype)

        # Timestep: (B,)
        timestep = torch.tensor([1.0 / 1000], dtype=dtype).expand(batch_size)

        # Text encoder hidden states from Gemma-2-2b: (B, seq_len, cap_feat_dim)
        max_sequence_length = 128
        encoder_hidden_states = torch.randn(
            batch_size, max_sequence_length, CAP_FEAT_DIM, dtype=dtype
        )

        # Encoder attention mask: (B, seq_len)
        encoder_attention_mask = torch.ones(
            batch_size, max_sequence_length, dtype=torch.bool
        )

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
        }
