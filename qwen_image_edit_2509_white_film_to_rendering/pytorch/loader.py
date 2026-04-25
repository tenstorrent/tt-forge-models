# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen Image Edit 2509 White Film to Rendering LoRA model loader implementation.

Loads the dx8152/Qwen-Image-Edit-2509-White_film_to_rendering LoRA adapter on
top of the Qwen/Qwen-Image-Edit-2509 base diffusion pipeline for converting
white-model (white film) renders into textured/material renderings.
"""

from typing import Any, Optional

import torch
from diffusers import QwenImageEditPlusPipeline

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

BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"
LORA_REPO = "dx8152/Qwen-Image-Edit-2509-White_film_to_rendering"
LORA_WEIGHT_NAME = "白膜转材质.safetensors"


class ModelVariant(StrEnum):
    """Available Qwen Image Edit 2509 White Film to Rendering model variants."""

    WHITE_FILM_TO_RENDERING = "White_Film_To_Rendering"


class ModelLoader(ForgeModel):
    """Qwen Image Edit 2509 White Film to Rendering LoRA model loader."""

    _VARIANTS = {
        ModelVariant.WHITE_FILM_TO_RENDERING: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.WHITE_FILM_TO_RENDERING

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="Qwen Image Edit 2509 White Film to Rendering",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load the Qwen-Image-Edit-2509 transformer with white film to rendering LoRA applied.

        Returns:
            QwenImageTransformer2DModel instance with LoRA weights merged.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=dtype,
        )
        pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=LORA_WEIGHT_NAME,
        )
        pipeline.fuse_lora(safe_fusing=True)

        self._transformer = pipeline.transformer
        self._transformer.eval()
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the diffusion transformer.

        Returns a dict matching QwenImageTransformer2DModel.forward() signature.
        """
        dtype = kwargs.get("dtype_override", torch.float32)
        batch_size = kwargs.get("batch_size", 1)

        img_dim = 64
        text_dim = 3584
        txt_seq_len = 32

        frame, height, width = 1, 8, 8
        img_seq_len = frame * height * width

        hidden_states = torch.randn(batch_size, img_seq_len, img_dim, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, txt_seq_len, text_dim, dtype=dtype
        )
        encoder_hidden_states_mask = torch.ones(batch_size, txt_seq_len, dtype=dtype)
        timestep = torch.tensor([500.0] * batch_size, dtype=dtype)
        img_shapes = [(frame, height, width)] * batch_size

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
        }
