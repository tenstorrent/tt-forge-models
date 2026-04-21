# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mochi GGUF model loader implementation.

Mochi is a ~10B text-to-video diffusion model by Genmo. This loader uses the
GGUF-quantized transformer repackaged by calcuis for ComfyUI / gguf-node.
The GGUF transformer is loaded via diffusers'
MochiTransformer3DModel.from_single_file.

Repository:
- https://huggingface.co/calcuis/mochi
"""

from typing import Optional

import torch
from diffusers import MochiTransformer3DModel

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

GGUF_BASE_URL = "https://huggingface.co/calcuis/mochi/blob/main"


class ModelVariant(StrEnum):
    """Available Mochi GGUF model variants."""

    Q3_K_M = "Q3_K_M"


class ModelLoader(ForgeModel):
    """Mochi GGUF model loader for text-to-video generation."""

    _VARIANTS = {
        ModelVariant.Q3_K_M: ModelConfig(
            pretrained_model_name="calcuis/mochi",
        ),
    }

    _GGUF_FILES = {
        ModelVariant.Q3_K_M: "mochi-q3_k_m.gguf",
    }

    DEFAULT_VARIANT = ModelVariant.Q3_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Mochi GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GGUF-quantized Mochi transformer."""
        gguf_file = self._GGUF_FILES[self._variant]
        gguf_url = f"{GGUF_BASE_URL}/{gguf_file}"

        load_kwargs = {}
        if dtype_override is not None:
            load_kwargs["torch_dtype"] = dtype_override

        self.transformer = MochiTransformer3DModel.from_single_file(
            gguf_url,
            **load_kwargs,
        )

        if dtype_override is not None:
            self.transformer = self.transformer.to(dtype_override)

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample inputs for the Mochi transformer."""
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.transformer.config

        # Mochi latent dimensions (12-channel latents, 6x temporal / 8x spatial
        # compression). Using small test dimensions.
        num_channels = config.in_channels
        num_frames = 2
        height = 12
        width = 12
        seq_len = 128  # matches MochiPipeline's max_sequence_length
        text_embed_dim = config.text_embed_dim

        hidden_states = torch.randn(
            batch_size, num_channels, num_frames, height, width, dtype=dtype
        )

        timestep = torch.tensor([500], dtype=torch.long).expand(batch_size)

        encoder_hidden_states = torch.randn(
            batch_size, seq_len, text_embed_dim, dtype=dtype
        )

        encoder_attention_mask = torch.ones(batch_size, seq_len, dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "encoder_attention_mask": encoder_attention_mask,
        }
