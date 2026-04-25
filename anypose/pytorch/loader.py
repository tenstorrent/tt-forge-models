# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AnyPose model loader implementation
"""

from typing import Any, Dict, Optional

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

    # Small spatial dims for synthetic transformer inputs (H_latent x W_latent per image)
    _DUMMY_LATENT_H = 4
    _DUMMY_LATENT_W = 4
    _DUMMY_TEXT_SEQ_LEN = 16
    # AnyPose takes 1 output image + 2 condition images
    _NUM_IMAGES = 3

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
        """Load the AnyPose pipeline and return its transformer nn.Module.

        Returns:
            QwenImageTransformer2DModel: The transformer component with LoRA adapters.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.pipeline = load_anypose_pipe(self.base_model, pretrained_model_name)

        if dtype_override is not None:
            self.pipeline.transformer = self.pipeline.transformer.to(dtype_override)

        return self.pipeline.transformer

    def load_inputs(self, dtype_override=None) -> Dict[str, Any]:
        """Return synthetic inputs for the QwenImageTransformer2DModel forward pass.

        hidden_states has shape (batch, total_seq, in_channels) where total_seq is the
        sum of T*H*W over all img_shapes tuples (output image + condition images).
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        config = self.pipeline.transformer.config
        in_channels = config.in_channels
        joint_attention_dim = config.joint_attention_dim
        dtype = dtype_override if dtype_override is not None else torch.float32

        # Each packed image contributes (H//2)*(W//2) tokens to the sequence.
        h = self._DUMMY_LATENT_H // 2
        w = self._DUMMY_LATENT_W // 2
        img_shape = (1, h, w)
        img_shapes = [[img_shape] * self._NUM_IMAGES]
        total_seq = self._NUM_IMAGES * h * w

        return {
            "hidden_states": torch.randn(1, total_seq, in_channels, dtype=dtype),
            "encoder_hidden_states": torch.randn(
                1, self._DUMMY_TEXT_SEQ_LEN, joint_attention_dim, dtype=dtype
            ),
            "encoder_hidden_states_mask": torch.ones(
                1, self._DUMMY_TEXT_SEQ_LEN, dtype=torch.bool
            ),
            "timestep": torch.tensor([0.5], dtype=dtype),
            "img_shapes": img_shapes,
            "return_dict": False,
        }
