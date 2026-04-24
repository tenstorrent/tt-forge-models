# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image-Edit-2511 Anime LoRA model loader.

Loads the Qwen/Qwen-Image-Edit-2511 base diffusion pipeline and applies
the prithivMLmods/Qwen-Image-Edit-2511-Anime LoRA adapter for flat
cel-shaded anime-style image editing, then extracts the diffusion transformer.

Available variants:
- QWEN_IMAGE_EDIT_2511_ANIME: Anime LoRA (bf16)
"""

from typing import Any, Optional

import torch
from diffusers import DiffusionPipeline

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

BASE_REPO_ID = "Qwen/Qwen-Image-Edit-2511"
LORA_REPO_ID = "prithivMLmods/Qwen-Image-Edit-2511-Anime"


class ModelVariant(StrEnum):
    """Available Qwen-Image-Edit-2511 Anime model variants."""

    QWEN_IMAGE_EDIT_2511_ANIME = "Anime"


class ModelLoader(ForgeModel):
    """Qwen-Image-Edit-2511 Anime LoRA model loader."""

    _VARIANTS = {
        ModelVariant.QWEN_IMAGE_EDIT_2511_ANIME: ModelConfig(
            pretrained_model_name=BASE_REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.QWEN_IMAGE_EDIT_2511_ANIME

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen_Image_Edit_2511_Anime",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Qwen-Image-Edit-2511 transformer with Anime LoRA weights applied.

        Returns:
            QwenImageTransformer2DModel: The diffusion transformer with LoRA applied.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        pipeline.load_lora_weights(LORA_REPO_ID)
        self._transformer = pipeline.transformer
        del pipeline
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Load sample inputs for the diffusion transformer.

        Returns:
            dict: Inputs matching QwenImageTransformer2DModel.forward() signature.
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
