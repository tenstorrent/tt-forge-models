# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Argmax MLX Stable Diffusion 3.5 Large 4-bit Quantized model loader implementation.

The argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized checkpoint stores weights
in MLX's proprietary 4-bit quantized format (separate q/k/v projections with packed
int4 weights, scales, and biases). This format is incompatible with diffusers'
SD3 converter, which expects standard joint_blocks.* keys.

We instantiate SD3Transformer2DModel directly with the known SD3.5-Large architecture
config (derived from the checkpoint structure) rather than loading the actual weights.

Available variants:
- 4BIT: sd3.5_large_4bit_quantized.safetensors
"""

from typing import Any, Optional

import torch
from diffusers import SD3Transformer2DModel

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

# SD3.5-Large transformer architecture config.
# Derived from the checkpoint: pos_embed has 36864=192^2 positions, attention
# projection dim is 2432=38*64, context_embedder maps 4096->2432, etc.
SD3_5_LARGE_CONFIG = {
    "sample_size": 128,
    "patch_size": 2,
    "in_channels": 16,
    "num_layers": 38,
    "attention_head_dim": 64,
    "num_attention_heads": 38,
    "joint_attention_dim": 4096,
    "caption_projection_dim": 2432,
    "pooled_projection_dim": 2048,
    "out_channels": 16,
    "pos_embed_max_size": 192,
    "qk_norm": "rms_norm",
}

# SD3.5 Large transformer input dimensions
LATENT_CHANNELS = 16
LATENT_HEIGHT = 64
LATENT_WIDTH = 64
JOINT_ATTENTION_DIM = 4096
POOLED_PROJECTION_DIM = 2048
MAX_SEQ_LEN = 154


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
        """Load and return the SD3.5 Large transformer with the known architecture config.

        The MLX 4-bit checkpoint uses a proprietary weight format incompatible with
        diffusers' from_single_file converter, so we initialize with random weights.

        Returns:
            SD3Transformer2DModel instance with SD3.5-Large architecture.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._transformer is None:
            self._transformer = SD3Transformer2DModel(**SD3_5_LARGE_CONFIG)
            self._transformer = self._transformer.to(dtype=dtype)
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
