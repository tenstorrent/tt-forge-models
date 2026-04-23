# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen Image Edit 2509 White-to-Scene LoRA model loader implementation.

Loads the dx8152/Qwen-Image-Edit-2509-White_to_Scene LoRA adapter on top of the
Qwen/Qwen-Image-Edit-2509 base diffusion pipeline for converting white
background product images into scenic backgrounds.
"""

from typing import Any, Dict, Optional

import torch
from diffusers import QwenImageEditPlusPipeline

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


class ModelVariant(StrEnum):
    """Available Qwen Image Edit White-to-Scene model variants."""

    WHITE_TO_SCENE = "white_to_scene"


class ModelLoader(ForgeModel):
    """Qwen Image Edit 2509 White-to-Scene LoRA model loader."""

    _VARIANTS = {
        ModelVariant.WHITE_TO_SCENE: ModelConfig(
            pretrained_model_name="dx8152/Qwen-Image-Edit-2509-White_to_Scene",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.WHITE_TO_SCENE

    _BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"

    _LORA_WEIGHT_NAMES = {
        ModelVariant.WHITE_TO_SCENE: "白底图转场景.safetensors",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._pipe: Optional[QwenImageEditPlusPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="Qwen Image Edit 2509 White-to-Scene",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override or torch.bfloat16
        if self._pipe is None:
            self._pipe = QwenImageEditPlusPipeline.from_pretrained(
                self._BASE_MODEL, torch_dtype=dtype, **kwargs
            )
            self._pipe.load_lora_weights(
                self._variant_config.pretrained_model_name,
                weight_name=self._LORA_WEIGHT_NAMES[self._variant],
            )
        return self._pipe.transformer

    def load_inputs(self, dtype_override=None, batch_size=1) -> Dict[str, Any]:
        dtype = dtype_override or torch.bfloat16

        # From Qwen-Image-Edit-2509 transformer config: in_channels=64
        img_dim = 64
        # joint_attention_dim from config = 3584
        text_dim = 3584
        txt_seq_len = 32

        # img_seq_len must equal frame * height * width for positional encoding
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
