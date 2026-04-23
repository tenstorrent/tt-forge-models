# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Argmax MLX Stable Diffusion 3.5 Large 4-bit Quantized model loader implementation.

Loads the argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized single-file
safetensors checkpoint. This is a 4-bit quantized MLX conversion of the SD3
MMDiT transformer from stabilityai/stable-diffusion-3.5-large.

Available variants:
- 4BIT: sd3.5_large_4bit_quantized.safetensors
"""

import os
from typing import Any, Optional

import torch
from diffusers import SD3Transformer2DModel
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

REPO_ID = "argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized"
CHECKPOINT_FILENAME = "sd3.5_large_4bit_quantized.safetensors"

# SD3.5 Large transformer config source
TRANSFORMER_CONFIG = "stabilityai/stable-diffusion-3.5-large"
TRANSFORMER_SUBFOLDER = "transformer"

# SD3.5 Large transformer input dimensions
LATENT_CHANNELS = 16
LATENT_HEIGHT = 64
LATENT_WIDTH = 64
JOINT_ATTENTION_DIM = 4096
POOLED_PROJECTION_DIM = 2048
MAX_SEQ_LEN = 154

# SD3.5 Large transformer architecture config (38 heads × 64 dim = 2432 hidden size).
# caption_projection_dim must equal inner_dim (num_attention_heads * attention_head_dim).
# Used when TT_RANDOM_WEIGHTS=1 to avoid downloading the gated stabilityai repo.
_SD35_LARGE_CONFIG = {
    "attention_head_dim": 64,
    "caption_projection_dim": 2432,
    "in_channels": 16,
    "joint_attention_dim": 4096,
    "num_attention_heads": 38,
    "num_layers": 38,
    "out_channels": 16,
    "patch_size": 2,
    "pooled_projection_dim": 2048,
    "pos_embed_max_size": 96,
    "sample_size": 128,
}


class ModelVariant(StrEnum):
    """Available Argmax MLX SD3.5 Large 4-bit quantized model variants."""

    MLX_4BIT = "mlx_4bit"


class ModelLoader(ForgeModel):
    """Argmax MLX SD3.5 Large 4-bit quantized transformer loader."""

    _VARIANTS = {
        ModelVariant.MLX_4BIT: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.MLX_4BIT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Argmax_MLX_SD3.5_Large_4bit",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the SD3.5 Large transformer from the MLX checkpoint.

        Returns:
            SD3Transformer2DModel instance loaded from the 4-bit MLX safetensors.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._transformer is None:
            if os.environ.get("TT_RANDOM_WEIGHTS") == "1":
                # stabilityai/stable-diffusion-3.5-large is gated; use hardcoded
                # architecture config so compilation can proceed without weights.
                self._transformer = SD3Transformer2DModel(**_SD35_LARGE_CONFIG)
                self._transformer = self._transformer.to(dtype=dtype)
            else:
                checkpoint_path = hf_hub_download(REPO_ID, CHECKPOINT_FILENAME)
                self._transformer = SD3Transformer2DModel.from_single_file(
                    checkpoint_path,
                    config=TRANSFORMER_CONFIG,
                    subfolder=TRANSFORMER_SUBFOLDER,
                    torch_dtype=dtype,
                )
            self._transformer.eval()
        elif dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)
        return self._transformer

    def load_inputs(
        self, *, dtype_override: Optional[torch.dtype] = None, **kwargs
    ) -> Any:
        """Prepare synthetic inputs for the SD3.5 transformer.

        Returns:
            dict: Input tensors matching the SD3Transformer2DModel forward signature:
                - hidden_states: Latent tensor [batch, channels, height, width]
                - timestep: Scalar timestep tensor
                - encoder_hidden_states: Text encoder outputs [batch, seq_len, dim]
                - pooled_projections: Pooled text embeddings [batch, pooled_dim]
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        return {
            "hidden_states": torch.randn(
                1, LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH, dtype=dtype
            ),
            "timestep": torch.tensor([1.0], dtype=dtype),
            "encoder_hidden_states": torch.randn(
                1, MAX_SEQ_LEN, JOINT_ATTENTION_DIM, dtype=dtype
            ),
            "pooled_projections": torch.randn(1, POOLED_PROJECTION_DIM, dtype=dtype),
        }
