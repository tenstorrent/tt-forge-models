# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ovis-Image-7B diffusion transformer model loader implementation.

Loads the diffusion transformer component from the upstream
AIDC-AI/Ovis-Image-7B text-to-image generation model.

Available variants:
- OVIS_IMAGE_7B: Ovis-Image-7B diffusion transformer
"""

from typing import Any, Optional

import torch
from diffusers import OvisImageTransformer2DModel

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

REPO_ID = "AIDC-AI/Ovis-Image-7B"

# From transformer config: in_channels=64, joint_attention_dim=2048
IMG_DIM = 64
TEXT_DIM = 2048
TXT_SEQ_LEN = 32
# img_seq_len = frame * height * width for positional encoding
FRAME, HEIGHT, WIDTH = 1, 8, 8
IMG_SEQ_LEN = FRAME * HEIGHT * WIDTH


class ModelVariant(StrEnum):
    """Available Ovis-Image-7B model variants."""

    OVIS_IMAGE_7B = "Ovis_Image_7B"


class ModelLoader(ForgeModel):
    """Ovis-Image-7B model loader for the diffusion transformer."""

    _VARIANTS = {
        ModelVariant.OVIS_IMAGE_7B: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.OVIS_IMAGE_7B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="OVIS_IMAGE_7B",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_transformer(
        self, dtype: torch.dtype = torch.bfloat16
    ) -> OvisImageTransformer2DModel:
        """Load the diffusion transformer from Ovis-Image-7B."""
        self._transformer = OvisImageTransformer2DModel.from_pretrained(
            REPO_ID,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        self._transformer.eval()
        return self._transformer

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Ovis-Image-7B diffusion transformer.

        Returns:
            OvisImageTransformer2DModel instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._transformer is None:
            return self._load_transformer(dtype)
        if dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the diffusion transformer.

        Returns a dict matching OvisImageTransformer2DModel.forward() signature.
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        batch_size = kwargs.get("batch_size", 1)

        hidden_states = torch.randn(batch_size, IMG_SEQ_LEN, IMG_DIM, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, TXT_SEQ_LEN, TEXT_DIM, dtype=dtype
        )
        timestep = torch.tensor([500.0] * batch_size, dtype=dtype)

        # Position IDs for image and text tokens (2D: [seq_len, 3])
        img_ids = torch.zeros(IMG_SEQ_LEN, 3, dtype=dtype)
        for i in range(IMG_SEQ_LEN):
            img_ids[i, 0] = 0  # frame
            img_ids[i, 1] = i // WIDTH  # height
            img_ids[i, 2] = i % WIDTH  # width

        txt_ids = torch.zeros(TXT_SEQ_LEN, 3, dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
        }
