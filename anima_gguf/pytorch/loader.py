# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Anima GGUF model loader implementation.

Loads GGUF-quantized variants of the ANIMA text-to-image diffusion transformer from
Bedovyy/Anima-GGUF. ANIMA is a 2B parameter diffusion model built on NVIDIA's
Cosmos-Predict2-2B-Text2Image architecture and is fine-tuned for anime-style
illustration generation. The upstream nvidia/Cosmos-Predict2-2B-Text2Image
repository provides the transformer config used during loading.

Available variants:
- PREVIEW3_BASE_Q5_K_M: anima-preview3-base 5-bit (medium) quantization
"""

import os
from typing import Any, Optional

import torch
from diffusers import CosmosTransformer3DModel, GGUFQuantizationConfig
from huggingface_hub import hf_hub_download

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

REPO_ID = "Bedovyy/Anima-GGUF"
# Local config avoids downloading the gated nvidia/Cosmos-Predict2-2B-Text2Image repo.
# Architecture inferred from GGUF tensor shapes: 16 heads, 128 head_dim, 28 layers.
_LOCAL_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_config")


class ModelVariant(StrEnum):
    """Available Anima GGUF quantization variants."""

    PREVIEW3_BASE_Q5_K_M = "preview3_base_Q5_K_M"


_GGUF_FILES = {
    ModelVariant.PREVIEW3_BASE_Q5_K_M: "anima-preview3-base-Q5_K_M.gguf",
}


class ModelLoader(ForgeModel):
    """Anima GGUF model loader for the text-to-image diffusion transformer."""

    _VARIANTS = {
        ModelVariant.PREVIEW3_BASE_Q5_K_M: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PREVIEW3_BASE_Q5_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer: Optional[CosmosTransformer3DModel] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ANIMA_GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_transformer(
        self, dtype: torch.dtype = torch.float32
    ) -> CosmosTransformer3DModel:
        """Load the Cosmos diffusion transformer from the selected GGUF file."""
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=_GGUF_FILES[self._variant],
        )

        self._transformer = CosmosTransformer3DModel.from_single_file(
            model_path,
            config=_LOCAL_CONFIG,
            subfolder="transformer",
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            torch_dtype=dtype,
        )
        self._transformer.eval()
        return self._transformer

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Anima GGUF diffusion transformer."""
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._transformer is None:
            return self._load_transformer(dtype)
        if dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)
        return self._transformer

    def load_inputs(
        self, *, dtype_override: Optional[torch.dtype] = None, **kwargs
    ) -> Any:
        """Prepare synthetic inputs for the Anima text-to-image transformer."""
        dtype = dtype_override if dtype_override is not None else torch.float32
        batch_size = kwargs.get("batch_size", 1)

        if self._transformer is None:
            self._load_transformer(dtype)

        config = self._transformer.config

        # Text-to-image runs the video transformer with a single latent frame.
        latent_num_frames = 1
        latent_height = 2
        latent_width = 2

        hidden_states = torch.randn(
            batch_size,
            config.in_channels,
            latent_num_frames,
            latent_height,
            latent_width,
            dtype=dtype,
        )
        encoder_hidden_states = torch.randn(
            batch_size, 8, config.text_embed_dim, dtype=dtype
        )
        timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)
        # padding_mask required when concat_padding_mask=True; all-ones = no padding
        padding_mask = torch.ones(
            batch_size, 1, latent_height, latent_width, dtype=dtype
        )

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "padding_mask": padding_mask,
            "return_dict": False,
        }
