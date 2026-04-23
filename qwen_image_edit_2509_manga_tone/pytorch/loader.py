# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image-Edit-2509 Manga Tone LoRA model loader.

Loads the Qwen/Qwen-Image-Edit-2509 diffusion transformer and applies
the nappa114514/Qwen-Image-Edit-2509-Manga-Tone LoRA adapter for
manga tone style image editing.

Available variants:
- QWEN_IMAGE_EDIT_2509_MANGA_TONE: Manga Tone LoRA (float32)
"""

from typing import Any, Optional

import torch
from diffusers import DiffusionPipeline, QwenImageTransformer2DModel

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

BASE_REPO_ID = "Qwen/Qwen-Image-Edit-2509"
LORA_REPO_ID = "nappa114514/Qwen-Image-Edit-2509-Manga-Tone"
LORA_WEIGHT_NAME = "tone001.safetensors"


class ModelVariant(StrEnum):
    """Available Qwen-Image-Edit-2509 Manga Tone model variants."""

    QWEN_IMAGE_EDIT_2509_MANGA_TONE = "Manga_Tone"


class ModelLoader(ForgeModel):
    """Qwen-Image-Edit-2509 Manga Tone LoRA model loader."""

    _VARIANTS = {
        ModelVariant.QWEN_IMAGE_EDIT_2509_MANGA_TONE: ModelConfig(
            pretrained_model_name=BASE_REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.QWEN_IMAGE_EDIT_2509_MANGA_TONE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen_Image_Edit_2509_Manga_Tone",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Qwen-Image-Edit-2509 transformer with Manga Tone LoRA weights fused.

        Returns:
            QwenImageTransformer2DModel with LoRA adapter merged.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        pipe = DiffusionPipeline.from_pretrained(
            BASE_REPO_ID,
            torch_dtype=dtype,
        )
        pipe.load_lora_weights(LORA_REPO_ID, weight_name=LORA_WEIGHT_NAME)
        pipe.fuse_lora()
        pipe.unload_lora_weights()
        self._transformer = pipe.transformer.to(dtype=dtype)
        self._transformer.eval()
        del pipe
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
