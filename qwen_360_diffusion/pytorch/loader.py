#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 360 Diffusion LoRA model loader implementation.

Loads the Qwen-Image base pipeline and applies the 360-degree equirectangular
panorama LoRA weights from ProGamerGov/qwen-360-diffusion for text-to-image
generation of 360-degree panoramic images.

Available variants:
- INT8_V1: int8-bf16 v1 LoRA on Qwen/Qwen-Image (default)
- INT4_V1: int4-bf16 v1 LoRA on Qwen/Qwen-Image
- INT4_V1B: int4-bf16 v1-b LoRA on Qwen/Qwen-Image
- INT8_V2_2512: int8-bf16 v2 LoRA on Qwen/Qwen-Image-2512
"""

from typing import Any, Dict, Optional

import torch
from diffusers import DiffusionPipeline  # type: ignore[import]

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

BASE_MODEL_QWEN_IMAGE = "Qwen/Qwen-Image"
BASE_MODEL_QWEN_IMAGE_2512 = "Qwen/Qwen-Image-2512"
LORA_REPO = "ProGamerGov/qwen-360-diffusion"

LORA_INT8_V1 = "qwen-360-diffusion-int8-bf16-v1.safetensors"
LORA_INT4_V1 = "qwen-360-diffusion-int4-bf16-v1.safetensors"
LORA_INT4_V1B = "qwen-360-diffusion-int4-bf16-v1-b.safetensors"
LORA_INT8_V2_2512 = "qwen-360-diffusion-2512-int8-bf16-v2.safetensors"


class ModelVariant(StrEnum):
    """Available Qwen 360 Diffusion LoRA variants."""

    INT8_V1 = "int8_v1"
    INT4_V1 = "int4_v1"
    INT4_V1B = "int4_v1b"
    INT8_V2_2512 = "int8_v2_2512"


_LORA_FILES = {
    ModelVariant.INT8_V1: LORA_INT8_V1,
    ModelVariant.INT4_V1: LORA_INT4_V1,
    ModelVariant.INT4_V1B: LORA_INT4_V1B,
    ModelVariant.INT8_V2_2512: LORA_INT8_V2_2512,
}


class ModelLoader(ForgeModel):
    """Qwen 360 Diffusion LoRA model loader."""

    _VARIANTS = {
        ModelVariant.INT8_V1: ModelConfig(
            pretrained_model_name=BASE_MODEL_QWEN_IMAGE,
        ),
        ModelVariant.INT4_V1: ModelConfig(
            pretrained_model_name=BASE_MODEL_QWEN_IMAGE,
        ),
        ModelVariant.INT4_V1B: ModelConfig(
            pretrained_model_name=BASE_MODEL_QWEN_IMAGE,
        ),
        ModelVariant.INT8_V2_2512: ModelConfig(
            pretrained_model_name=BASE_MODEL_QWEN_IMAGE_2512,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.INT8_V1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_360_DIFFUSION",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the Qwen-Image transformer with 360 diffusion LoRA weights fused.

        Returns:
            QwenImageTransformer2DModel with LoRA weights fused.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self._transformer is None:
            pipe = DiffusionPipeline.from_pretrained(
                self._variant_config.pretrained_model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
            )
            lora_file = _LORA_FILES[self._variant]
            pipe.load_lora_weights(LORA_REPO, weight_name=lora_file)
            pipe.fuse_lora()
            self._transformer = pipe.transformer
            self._transformer.eval()
            del pipe
        elif dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)

        return self._transformer

    def load_inputs(self, **kwargs) -> Dict[str, Any]:
        """Prepare sample inputs for the diffusion transformer.

        Returns a dict matching QwenImageTransformer2DModel.forward() signature.
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        batch_size = kwargs.get("batch_size", 1)

        # From Qwen-Image transformer config: in_channels=64
        img_dim = 64
        # joint_attention_dim from config = 4096
        text_dim = 4096
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
