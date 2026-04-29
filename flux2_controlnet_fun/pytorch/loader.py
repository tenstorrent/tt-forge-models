# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.2 ControlNet Fun Union model loader implementation for conditional image generation.

Loads the FLUX.2 ControlNet Fun Union model from alibaba-pai, which supports
multiple control conditions (Canny, HED, Depth, Pose, MLSD, Scribble, Gray)
in a single unified model.

Repository: https://huggingface.co/alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union
"""

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
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
from .src.model import FluxFunControlNetModel

REPO_ID = "alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union"
SAFETENSORS_FILE = "FLUX.2-dev-Fun-Controlnet-Union.safetensors"

# Architectural constants derived from the checkpoint
_INNER_DIM = 6144
_NUM_HEADS = 48
_HEAD_DIM = 128
_NUM_CONTROL_LAYERS = 4
_IN_CHANNELS = 260


class ModelVariant(StrEnum):
    """Available FLUX.2 ControlNet Fun Union model variants."""

    CONTROLNET_UNION = "ControlNet_Union"


class ModelLoader(ForgeModel):
    """FLUX.2 ControlNet Fun Union model loader for conditional image generation."""

    _VARIANTS = {
        ModelVariant.CONTROLNET_UNION: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CONTROLNET_UNION

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="FLUX.2 ControlNet Fun",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the FLUX.2 ControlNet Fun Union model.

        Downloads the single-file safetensors checkpoint and loads it into a
        FluxFunControlNetModel nn.Module.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The ControlNet model instance.
        """
        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        local_path = hf_hub_download(repo_id=REPO_ID, filename=SAFETENSORS_FILE)
        state_dict = load_file(local_path)

        self.model = FluxFunControlNetModel(
            inner_dim=_INNER_DIM,
            num_heads=_NUM_HEADS,
            head_dim=_HEAD_DIM,
            num_control_layers=_NUM_CONTROL_LAYERS,
            in_channels=_IN_CHANNELS,
        )
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(compute_dtype)

        return self.model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the FLUX.2 ControlNet Fun Union model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size (default: 1).

        Returns:
            dict: Input tensors for the ControlNet model forward pass.
        """
        if self.model is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # Spatial dimensions: 128×128 image, 8× VAE + 2× patch packing
        height, width = 128, 128
        vae_scale_factor = 8
        patch_size = 2
        h_packed = height // (vae_scale_factor * patch_size)
        w_packed = width // (vae_scale_factor * patch_size)
        img_seq_len = h_packed * w_packed

        # Text encoder sequence length
        txt_seq_len = 256

        hidden_states = torch.randn(batch_size, img_seq_len, _INNER_DIM, dtype=dtype)
        encoder_hidden_states = torch.randn(batch_size, txt_seq_len, _INNER_DIM, dtype=dtype)
        controlnet_cond = torch.randn(batch_size, img_seq_len, _IN_CHANNELS, dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "controlnet_cond": controlnet_cond,
        }
