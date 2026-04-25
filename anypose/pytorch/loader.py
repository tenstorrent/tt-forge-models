# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AnyPose model loader implementation
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
from .src.model_utils import load_anypose_pipe


class ModelVariant(StrEnum):
    """Available AnyPose model variants."""

    ANYPOSE_QWEN_2511 = "AnyPose_Qwen_2511"


class ModelLoader(ForgeModel):
    """AnyPose model loader implementation.

    AnyPose is a LoRA adapter for Qwen/Qwen-Image-Edit-2511 that transfers
    poses from one image to another while preserving the character's appearance
    and background.
    """

    _VARIANTS = {
        ModelVariant.ANYPOSE_QWEN_2511: ModelConfig(
            pretrained_model_name="lilylilith/AnyPose",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ANYPOSE_QWEN_2511

    base_model = "Qwen/Qwen-Image-Edit-2511"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="AnyPose",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the AnyPose diffusion transformer.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            QwenImageTransformer2DModel: The transformer with LoRA weights fused.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_anypose_pipe(self.base_model, pretrained_model_name)
        self.pipeline.fuse_lora()

        transformer = self.pipeline.transformer
        if dtype_override is not None:
            transformer = transformer.to(dtype_override)

        transformer.eval()
        return transformer

    def load_inputs(self, dtype_override=None, **kwargs):
        """Load and return sample inputs for the AnyPose transformer.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Keyword arguments for QwenImageTransformer2DModel.forward().
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        batch_size = 1

        # Transformer config: in_channels=64, joint_attention_dim=3584
        img_dim = 64
        text_dim = 3584
        txt_seq_len = 32

        frame, height, width = 1, 8, 8
        img_seq_len = frame * height * width

        return {
            "hidden_states": torch.randn(batch_size, img_seq_len, img_dim, dtype=dtype),
            "encoder_hidden_states": torch.randn(
                batch_size, txt_seq_len, text_dim, dtype=dtype
            ),
            "encoder_hidden_states_mask": torch.ones(
                batch_size, txt_seq_len, dtype=dtype
            ),
            "timestep": torch.tensor([500.0] * batch_size, dtype=dtype),
            "img_shapes": [(frame, height, width)] * batch_size,
        }
