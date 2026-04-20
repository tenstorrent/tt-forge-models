# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image-2512-Fun-Controlnet-Union model loader implementation.

Loads the ControlNet Union model for Qwen-Image-2512 from alibaba-pai.
This ControlNet supports multiple control conditions including Canny,
Depth, Pose, MLSD, HED, Scribble, and Gray.

Available variants:
- CONTROLNET_UNION: Original ControlNet Union weights
- CONTROLNET_UNION_2602: Updated ControlNet Union weights (2602 revision)
"""

import torch
from typing import Optional

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
from .src.model_utils import (
    load_controlnet_model,
    HIDDEN_SIZE,
    NUM_HEADS,
    HEAD_DIM,
    IN_CHANNELS,
    PATCH_SIZE,
)  # noqa: F401


class ModelVariant(StrEnum):
    """Available Qwen-Image-2512-Fun-Controlnet-Union model variants."""

    CONTROLNET_UNION = "ControlNet_Union"
    CONTROLNET_UNION_2602 = "ControlNet_Union_2602"


_VARIANT_FILENAMES = {
    ModelVariant.CONTROLNET_UNION: "Qwen-Image-2512-Fun-Controlnet-Union.safetensors",
    ModelVariant.CONTROLNET_UNION_2602: "Qwen-Image-2512-Fun-Controlnet-Union-2602.safetensors",
}


class ModelLoader(ForgeModel):
    """Qwen-Image-2512-Fun-Controlnet-Union model loader implementation."""

    _VARIANTS = {
        ModelVariant.CONTROLNET_UNION: ModelConfig(
            pretrained_model_name="alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union",
        ),
        ModelVariant.CONTROLNET_UNION_2602: ModelConfig(
            pretrained_model_name="alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CONTROLNET_UNION

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen-Image-2512-Fun-Controlnet-Union",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the ControlNet Union model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The ControlNet Union model with loaded weights.
        """
        filename = _VARIANT_FILENAMES[self._variant]
        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        self._model = load_controlnet_model(filename, dtype=compute_dtype)
        return self._model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the ControlNet Union model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size (default: 1).

        Returns:
            dict: Input tensors for the ControlNet model.
        """
        if self._model is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        height = 128
        width = 128
        vae_scale_factor = 8
        patch_size = PATCH_SIZE

        h_latent = height // vae_scale_factor
        w_latent = width // vae_scale_factor
        h_patched = h_latent // patch_size
        w_patched = w_latent // patch_size
        seq_len = h_patched * w_patched

        hidden_states = torch.randn(batch_size, seq_len, HIDDEN_SIZE, dtype=dtype)

        text_seq_len = 128
        joint_attention_dim = NUM_HEADS * HEAD_DIM
        encoder_hidden_states = torch.randn(
            batch_size, text_seq_len, joint_attention_dim, dtype=dtype
        )

        temb = torch.randn(batch_size, HIDDEN_SIZE, dtype=dtype)

        controlnet_cond = torch.randn(
            batch_size, seq_len, IN_CHANNELS * PATCH_SIZE + 4, dtype=dtype
        )

        return {
            "hidden_states": hidden_states,
            "temb": temb,
            "encoder_hidden_states": encoder_hidden_states,
            "controlnet_cond": controlnet_cond,
            "conditioning_scale": 1.0,
        }
