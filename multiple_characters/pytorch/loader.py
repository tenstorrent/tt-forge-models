# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Multiple Characters LoRA image-to-image model loader implementation.

Loads the Qwen-Image-Edit-2509 base pipeline and applies the
YaoJiefu/multiple-characters LoRA adapter for multi-character scene generation.

Reference: https://huggingface.co/YaoJiefu/multiple-characters
"""

from typing import Optional

import torch

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
    """Available Multiple Characters model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Multiple Characters LoRA image-to-image model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="YaoJiefu/multiple-characters",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    base_model = "Qwen/Qwen-Image-Edit-2509"
    prompt = "Generate two people watching TV on the sofa in the image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Multiple Characters",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from diffusers import DiffusionPipeline

        dtype = dtype_override or torch.bfloat16
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.base_model, torch_dtype=dtype, **kwargs
        )
        self.pipeline.load_lora_weights(
            self._variant_config.pretrained_model_name,
        )
        return self.pipeline.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        dtype = dtype_override or torch.bfloat16

        if self.pipeline is None:
            self.load_model(dtype_override=dtype)

        config = self.pipeline.transformer.config
        frame, height, width = 1, 2, 2
        img_seq_len = frame * height * width
        txt_seq_len = 4
        hidden_states = torch.randn(
            batch_size,
            img_seq_len,
            config.in_channels,
            dtype=dtype,
        )
        encoder_hidden_states = torch.randn(
            batch_size,
            txt_seq_len,
            config.joint_attention_dim,
            dtype=dtype,
        )
        timestep = torch.tensor([1000], dtype=torch.long).expand(batch_size)
        img_shapes = [[(frame, height, width)]] * batch_size

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "img_shapes": img_shapes,
            "return_dict": False,
        }
