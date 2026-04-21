# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
XLabs FLUX ControlNet Depth v3 model loader implementation.

Loads the XLabs-AI flux-controlnet-depth-v3 ControlNet from a single-file
safetensors distribution for depth-guided FLUX.1-dev image generation.

Repository: https://huggingface.co/XLabs-AI/flux-controlnet-depth-v3
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

REPO_ID = "XLabs-AI/flux-controlnet-depth-v3"
SAFETENSORS_FILE = "flux-depth-controlnet-v3.safetensors"


class ModelVariant(StrEnum):
    """Available XLabs FLUX ControlNet Depth v3 model variants."""

    DEPTH_V3 = "depth-v3"


class ModelLoader(ForgeModel):
    """XLabs FLUX ControlNet Depth v3 model loader for depth-guided image generation."""

    _VARIANTS = {
        ModelVariant.DEPTH_V3: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEPTH_V3

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.controlnet = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="XLabs FLUX ControlNet Depth v3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the XLabs FLUX ControlNet Depth v3 model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            FluxControlNetModel: The ControlNet model instance.
        """
        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        repo_id = self._variant_config.pretrained_model_name
        self.controlnet = FluxControlNetModel.from_single_file(
            f"https://huggingface.co/{repo_id}/resolve/main/{SAFETENSORS_FILE}",
            torch_dtype=compute_dtype,
        )
        self.controlnet.eval()
        return self.controlnet

    def load_inputs(
        self, dtype_override: Optional[torch.dtype] = None, batch_size: int = 1
    ) -> Any:
        """Prepare sample inputs for the ControlNet.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size (default: 1).

        Returns:
            dict: Input tensors matching FluxControlNetModel.forward() signature.
        """
        if self.controlnet is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.controlnet.config

        in_channels = config.in_channels
        joint_attention_dim = config.joint_attention_dim
        pooled_projection_dim = config.pooled_projection_dim

        img_seq_len = 64
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

        return {
            "hidden_states": hidden_states,
            "controlnet_cond": controlnet_cond,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_projections,
            "timestep": timestep,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
        }
