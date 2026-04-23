# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
InternVideoNext feature extraction model loader implementation for PyTorch.

InternVideoNext is a video foundation model for extracting video features/embeddings.
It processes video frames using a Vision Transformer architecture with patch size 14
and produces dense feature representations.
"""

from typing import Optional

import torch
from transformers import AutoConfig, AutoModel
from transformers import PreTrainedModel

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available InternVideoNext feature extraction model variants."""

    LARGE_P14_RES224_F16 = "Large_P14_Res224_F16"
    BASE_P14_RES224_F16 = "Base_P14_Res224_F16"


class ModelLoader(ForgeModel):
    """InternVideoNext feature extraction model loader implementation for PyTorch."""

    _VARIANTS = {
        ModelVariant.LARGE_P14_RES224_F16: ModelConfig(
            pretrained_model_name="revliter/internvideo_next_large_p14_res224_f16",
        ),
        ModelVariant.BASE_P14_RES224_F16: ModelConfig(
            pretrained_model_name="revliter/internvideo_next_base_p14_res224_f16",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_P14_RES224_F16

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="InternVideoNext",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_VIDEO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the InternVideoNext feature extraction model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            model: The loaded InternVideoNext model instance
        """
        config = AutoConfig.from_pretrained(self._model_name, trust_remote_code=True)
        # Disable flash_attn/fused ops which require CUDA-only packages
        config.model_config["use_flash_attn"] = False
        config.model_config["use_fused_rmsnorm"] = False
        config.model_config["use_fused_mlp"] = False

        # Patch get_init_context to avoid meta device: model __init__ calls
        # .item() on tensors which fails under torch.device("meta") context
        _orig_get_init_context = PreTrainedModel.get_init_context.__func__

        @classmethod
        def _patched_get_init_context(cls, dtype, is_quantized, _is_ds_init_called):
            return [
                ctx
                for ctx in _orig_get_init_context(
                    cls, dtype, is_quantized, _is_ds_init_called
                )
                if not isinstance(ctx, torch.device)
            ]

        PreTrainedModel.get_init_context = _patched_get_init_context
        try:
            model = AutoModel.from_pretrained(
                self._model_name,
                config=config,
                trust_remote_code=True,
                **kwargs,
            )
        finally:
            PreTrainedModel.get_init_context = classmethod(
                lambda cls, dtype, is_quantized, _is_ds_init_called: _orig_get_init_context(
                    cls, dtype, is_quantized, _is_ds_init_called
                )
            )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the model.

        Generates synthetic video input matching the expected format:
        (batch, channels, frames, height, width) with 16 frames at 224x224 resolution.

        Args:
            dtype_override: Optional torch.dtype override.
            batch_size: Batch size (default: 1).

        Returns:
            dict: Input tensors with pixel_values for video frames.
        """
        num_frames = 16
        height = 224
        width = 224
        channels = 3

        # Model expects input shape: (B, C, T, H, W)
        pixel_values = torch.randn(batch_size, channels, num_frames, height, width)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return {"pixel_values": pixel_values}
