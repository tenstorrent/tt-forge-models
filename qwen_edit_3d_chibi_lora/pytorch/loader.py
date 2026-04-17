#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Edit-3DChibi-LoRA model loader implementation.

Loads the Qwen/Qwen-Image-Edit-2509 base transformer and applies LoRA weights
from rsshekhawat/Qwen-Edit-3DChibi-LoRA for 3D Chibi style image editing.

Available variants:
- V1: Default LoRA weights (qwen_3d_chibi_lora_v1_000000820.safetensors)
"""

from typing import Any, Optional

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

BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"
LORA_REPO = "rsshekhawat/Qwen-Edit-3DChibi-LoRA"
LORA_WEIGHT_NAME = "qwen_3d_chibi_lora_v1_000000820.safetensors"


class ModelVariant(StrEnum):
    """Available Qwen-Edit-3DChibi-LoRA variants."""

    V1 = "v1"


class ModelLoader(ForgeModel):
    """Qwen-Edit-3DChibi-LoRA model loader."""

    _VARIANTS = {
        ModelVariant.V1: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.V1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[QwenImageEditPlusPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_EDIT_3D_CHIBI_LORA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the Qwen Image Edit transformer with 3DChibi LoRA weights fused.

        Returns:
            QwenImageTransformer2DModel with LoRA weights merged.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=LORA_WEIGHT_NAME,
        )
        self.pipeline.fuse_lora()

        transformer = self.pipeline.transformer
        transformer.eval()
        return transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the diffusion transformer."""
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
