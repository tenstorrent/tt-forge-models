# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image-Edit FlatLogColor LoRA model loader implementation.

Loads the Qwen-Image-Edit-2509 diffusion pipeline, applies the
tlennon-ie/QwenEdit2509-FlatLogColor LoRA weights, and returns
the transformer for compile-only testing.

Available variants:
- FLAT_LOG_COLOR_2509: FlatLogColor LoRA on Qwen-Image-Edit 2509
"""

from typing import Any, Optional

import torch
from diffusers import QwenImageEditPipeline

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
LORA_REPO = "tlennon-ie/QwenEdit2509-FlatLogColor"

IN_CHANNELS = 64
JOINT_ATTENTION_DIM = 3584


class ModelVariant(StrEnum):
    """Available Qwen-Image-Edit FlatLogColor variants."""

    FLAT_LOG_COLOR_2509 = "FlatLogColor_2509"


class ModelLoader(ForgeModel):
    """Qwen-Image-Edit FlatLogColor LoRA model loader."""

    _VARIANTS = {
        ModelVariant.FLAT_LOG_COLOR_2509: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.FLAT_LOG_COLOR_2509

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_IMAGE_EDIT_FLAT_LOG_COLOR",
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
        """Load the Qwen-Image-Edit transformer with FlatLogColor LoRA weights.

        Returns:
            QwenImageTransformer2DModel with LoRA weights fused.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        self._pipe = QwenImageEditPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

        self._pipe.load_lora_weights(LORA_REPO)
        self._pipe.fuse_lora()
        self._pipe.unload_lora_weights()

        transformer = self._pipe.transformer
        transformer.eval()

        return transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample tensor inputs for the diffusion transformer.

        Returns:
            dict matching QwenImageTransformer2DModel.forward() signature.
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        batch_size = kwargs.get("batch_size", 1)

        txt_seq_len = 32
        frame, height, width = 1, 8, 8
        img_seq_len = frame * height * width

        hidden_states = torch.randn(batch_size, img_seq_len, IN_CHANNELS, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, txt_seq_len, JOINT_ATTENTION_DIM, dtype=dtype
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
