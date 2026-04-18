# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image InstantX ControlNets model loader implementation.

Loads FluxControlNetModel variants from the InstantX FLUX ControlNet repositories.
Supports Union ControlNet variant.

Available variants:
- UNION: ControlNet supporting multiple control modes (canny, tile, depth, blur, pose, gray, low-quality)
"""

from typing import Any, Optional

import torch
from diffusers import FluxControlNetModel

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

_VARIANT_REPO_IDS = {
    "union": "InstantX/FLUX.1-dev-Controlnet-Union",
}


class ModelVariant(StrEnum):
    """Available Qwen-Image InstantX ControlNet model variants."""

    UNION = "Union"


class ModelLoader(ForgeModel):
    """Qwen-Image InstantX ControlNets model loader."""

    _VARIANTS = {
        ModelVariant.UNION: ModelConfig(
            pretrained_model_name=_VARIANT_REPO_IDS["union"],
        ),
    }
    DEFAULT_VARIANT = ModelVariant.UNION

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._controlnet = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_IMAGE_INSTANTX_CONTROLNETS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_controlnet(
        self, dtype: torch.dtype = torch.float32
    ) -> FluxControlNetModel:
        """Load ControlNet using from_pretrained with proper diffusers config."""
        repo_id = _VARIANT_REPO_IDS[self._variant.value.lower()]
        self._controlnet = FluxControlNetModel.from_pretrained(
            repo_id, torch_dtype=dtype
        )
        self._controlnet.eval()
        return self._controlnet

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the ControlNet model.

        Returns:
            FluxControlNetModel instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._controlnet is None:
            return self._load_controlnet(dtype)
        if dtype_override is not None:
            self._controlnet = self._controlnet.to(dtype=dtype_override)
        return self._controlnet

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the ControlNet.

        Returns a dict matching FluxControlNetModel.forward() signature.
        """
        dtype = kwargs.get("dtype_override", torch.float32)
        batch_size = kwargs.get("batch_size", 1)

        # FluxControlNetModel default config values
        in_channels = 64
        joint_attention_dim = 4096
        pooled_projection_dim = 768

        # Sequence lengths for image and text tokens
        img_seq_len = 64  # e.g. 8x8 patch grid
        txt_seq_len = 32

        hidden_states = torch.randn(batch_size, img_seq_len, in_channels, dtype=dtype)
        controlnet_cond = torch.randn(batch_size, img_seq_len, in_channels, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, txt_seq_len, joint_attention_dim, dtype=dtype
        )
        pooled_projections = torch.randn(batch_size, pooled_projection_dim, dtype=dtype)
        timestep = torch.tensor([500.0] * batch_size, dtype=dtype)
        img_ids = torch.zeros(img_seq_len, 3, dtype=dtype)
        txt_ids = torch.zeros(txt_seq_len, 3, dtype=dtype)

        inputs = {
            "hidden_states": hidden_states,
            "controlnet_cond": controlnet_cond,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_projections,
            "timestep": timestep,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
        }

        # Union variant requires controlnet_mode
        if self._variant == ModelVariant.UNION:
            # Use mode 0 (canny) as default
            inputs["controlnet_mode"] = torch.zeros(batch_size, dtype=torch.long)

        return inputs
