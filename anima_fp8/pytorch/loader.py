# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Anima FP8 model loader implementation.

Loads FP8-quantized variants of the Anima text-to-image diffusion
transformer from Bedovyy/Anima-FP8. Anima is a 2B parameter model
derived from NVIDIA Cosmos-Predict2-2B-Text2Image, using a Qwen3
0.6B text encoder and the Qwen-Image VAE.
"""

from typing import Any, Optional

import torch
from diffusers import CosmosTransformer3DModel
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

FP8_REPO_ID = "Bedovyy/Anima-FP8"


class ModelVariant(StrEnum):
    """Available Anima FP8 model variants."""

    PREVIEW3_BASE_FP8 = "preview3-base-fp8"
    PREVIEW3_BASE_MXFP8 = "preview3-base-mxfp8"
    PREVIEW3_BASE_NVFP4MIXED = "preview3-base-nvfp4mixed"


class ModelLoader(ForgeModel):
    """Anima FP8 model loader."""

    _VARIANTS = {
        ModelVariant.PREVIEW3_BASE_FP8: ModelConfig(
            pretrained_model_name=FP8_REPO_ID,
        ),
        ModelVariant.PREVIEW3_BASE_MXFP8: ModelConfig(
            pretrained_model_name=FP8_REPO_ID,
        ),
        ModelVariant.PREVIEW3_BASE_NVFP4MIXED: ModelConfig(
            pretrained_model_name=FP8_REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.PREVIEW3_BASE_FP8

    _FP8_FILES = {
        ModelVariant.PREVIEW3_BASE_FP8: "anima-preview3-base-fp8.safetensors",
        ModelVariant.PREVIEW3_BASE_MXFP8: "anima-preview3-base-mxfp8.safetensors",
        ModelVariant.PREVIEW3_BASE_NVFP4MIXED: "anima-preview3-base-nvfp4mixed.safetensors",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer: Optional[CosmosTransformer3DModel] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ANIMA_FP8",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_transformer(self, dtype: torch.dtype) -> CosmosTransformer3DModel:
        """Load the FP8-quantized transformer from a single safetensors file."""
        fp8_file = self._FP8_FILES[self._variant]
        fp8_path = hf_hub_download(repo_id=FP8_REPO_ID, filename=fp8_file)
        self.transformer = CosmosTransformer3DModel.from_single_file(
            fp8_path,
            config="nvidia/Cosmos-Predict2-2B-Text2Image",
            torch_dtype=dtype,
        )
        return self.transformer

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the FP8-quantized Anima transformer."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self.transformer is None:
            self._load_transformer(dtype)
        return self.transformer

    def load_inputs(
        self, dtype_override: Optional[torch.dtype] = None, **kwargs
    ) -> Any:
        """Prepare synthetic inputs for the Anima transformer forward pass."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self.transformer is None:
            self._load_transformer(dtype)

        batch_size = 1
        config = self.transformer.config

        # Use small latent dimensions for testing
        latent_num_frames = 1
        latent_height = 2
        latent_width = 2

        in_channels = config.in_channels
        hidden_states = torch.randn(
            batch_size,
            in_channels,
            latent_num_frames,
            latent_height,
            latent_width,
            dtype=dtype,
        )

        # Text encoder hidden states (Qwen3 0.6B embedding dimension)
        text_embed_dim = config.text_embed_dim
        encoder_hidden_states = torch.randn(batch_size, 8, text_embed_dim, dtype=dtype)

        timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "return_dict": False,
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
