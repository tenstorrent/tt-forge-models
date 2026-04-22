#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image-Edit-2509-Caricature-LoRA model loader implementation.

Loads the Qwen/Qwen-Image-Edit-2509 base pipeline and applies LoRA weights
from drbaph/Qwen-Image-Edit-2509-Caricature-LoRA to transform input images
into sketched caricature art with exaggerated features.

Available variants:
- V1_1: Default LoRA weights (qwen-edit-2509-caricature_v1.1.safetensors)
"""

from typing import Any, Optional

import torch

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
LORA_REPO = "drbaph/Qwen-Image-Edit-2509-Caricature-LoRA"
LORA_WEIGHT_NAME = "qwen-edit-2509-caricature_v1.1.safetensors"


class ModelVariant(StrEnum):
    """Available Qwen-Image-Edit-2509-Caricature-LoRA variants."""

    V1_1 = "v1.1"


class ModelLoader(ForgeModel):
    """Qwen-Image-Edit-2509-Caricature-LoRA model loader."""

    _VARIANTS = {
        ModelVariant.V1_1: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.V1_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_IMAGE_EDIT_2509_CARICATURE_LORA",
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
        """Load the Qwen Image Edit 2509 transformer with Caricature LoRA weights fused.

        Loads the full pipeline, applies the LoRA weights, fuses them into the
        transformer, and returns the transformer component.

        Returns:
            QwenImageTransformer2DModel with LoRA weights fused.
        """
        from diffusers import QwenImageEditPlusPipeline  # type: ignore[import]

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        pipe = QwenImageEditPlusPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )
        pipe.load_lora_weights(
            LORA_REPO,
            weight_name=LORA_WEIGHT_NAME,
        )
        pipe.fuse_lora()

        self._transformer = pipe.transformer
        self._transformer.eval()
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the diffusion transformer.

        Returns a dict matching QwenImageTransformer2DModel.forward() signature.
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        batch_size = kwargs.get("batch_size", 1)

        # From model config: in_channels=64 (img_in linear input dimension)
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
