# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX ControlNet Canny v3 model loader implementation.

Loads the XLabs-AI/flux-controlnet-canny-v3 ControlNet from the single-file
safetensors checkpoint shipped in the repo (no config.json is provided).

Available variants:
- XLABS_V3: XLabs-AI canny edge ControlNet v3 for FLUX.1-dev
"""

from typing import Any, Optional

import torch
from diffusers import FluxControlNetModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

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

REPO_ID = "XLabs-AI/flux-controlnet-canny-v3"
CHECKPOINT_FILE = "flux-canny-controlnet-v3.safetensors"

# XLabs ControlNets use a much smaller architecture than the InstantX union
# ControlNet: 2 double-stream transformer blocks and no single-stream blocks.
_NUM_LAYERS = 2
_NUM_SINGLE_LAYERS = 0


class ModelVariant(StrEnum):
    """Available FLUX ControlNet Canny v3 variants."""

    XLABS_V3 = "XLabs-v3"


class ModelLoader(ForgeModel):
    """FLUX ControlNet Canny v3 model loader."""

    _VARIANTS = {
        ModelVariant.XLABS_V3: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.XLABS_V3

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._controlnet = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX_CONTROLNET_CANNY_V3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_controlnet(
        self, dtype: torch.dtype = torch.float32
    ) -> FluxControlNetModel:
        """Load the ControlNet from the single-file safetensors checkpoint."""
        model_path = hf_hub_download(repo_id=REPO_ID, filename=CHECKPOINT_FILE)

        self._controlnet = FluxControlNetModel(
            num_layers=_NUM_LAYERS,
            num_single_layers=_NUM_SINGLE_LAYERS,
        )

        state_dict = load_file(model_path)
        self._controlnet.load_state_dict(state_dict)
        self._controlnet.to(dtype=dtype)
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

    def load_inputs(
        self,
        dtype_override: Optional[torch.dtype] = None,
        batch_size: int = 1,
        **kwargs,
    ) -> Any:
        """Prepare sample inputs for the ControlNet.

        Returns a dict matching FluxControlNetModel.forward() signature.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

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

        return {
            "hidden_states": hidden_states,
            "controlnet_cond": controlnet_cond,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_projections,
            "timestep": timestep,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
        }
