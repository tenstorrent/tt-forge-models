# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
QIE-2511-Zoom-Master LoRA model loader.

Loads the Qwen/Qwen-Image-Edit-2511 base diffusion pipeline and applies
the prithivMLmods/QIE-2511-Zoom-Master LoRA adapter for precise,
high-quality zoom-in transformations on marked image regions.

Available variants:
- QIE_2511_ZOOM_MASTER: Zoom Master LoRA (bf16)
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
LORA_REPO_ID = "prithivMLmods/QIE-2511-Zoom-Master"


class ModelVariant(StrEnum):
    """Available QIE-2511-Zoom-Master model variants."""

    QIE_2511_ZOOM_MASTER = "Zoom_Master"


class ModelLoader(ForgeModel):
    """QIE-2511-Zoom-Master LoRA model loader."""

    _VARIANTS = {
        ModelVariant.QIE_2511_ZOOM_MASTER: ModelConfig(
            pretrained_model_name=BASE_REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.QIE_2511_ZOOM_MASTER

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QIE_2511_Zoom_Master",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Qwen-Image-Edit-2511 pipeline with Zoom Master LoRA and return transformer.

        Returns:
            torch.nn.Module: The transformer with LoRA adapter applied.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        self.pipeline.load_lora_weights(LORA_REPO_ID)
        if dtype_override is not None:
            self.pipeline.transformer = self.pipeline.transformer.to(dtype_override)
        return self.pipeline.transformer

    def load_inputs(self, *, dtype_override=None, **kwargs) -> Any:
        """Load sample inputs for the QwenImageTransformer2DModel.

        Returns:
            dict: Keyword arguments matching QwenImageTransformer2DModel.forward().
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        batch_size = kwargs.get("batch_size", 1)

        # Model config: in_channels=64 (packed latent token dimension)
        img_dim = 64
        # joint_attention_dim from config = 3584
        text_dim = 3584
        txt_seq_len = 32

        # img_seq_len = frame * height * width for positional encoding
        frame, height, width = 1, 8, 8
        img_seq_len = frame * height * width

        hidden_states = torch.randn(batch_size, img_seq_len, img_dim, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, txt_seq_len, text_dim, dtype=dtype
        )
        encoder_hidden_states_mask = torch.ones(batch_size, txt_seq_len, dtype=dtype)
        timestep = torch.tensor([0.5] * batch_size, dtype=dtype)
        img_shapes = [(frame, height, width)] * batch_size

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
        }
