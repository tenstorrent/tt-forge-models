#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VACE-Wan2.1-1.3B-Preview model loader implementation.

Loads the ali-vilab/VACE-Wan2.1-1.3B-Preview diffusion transformer using
from_single_file with config from Wan-AI/Wan2.1-VACE-1.3B-Diffusers, since
the Preview repo ships raw .pth weights rather than a diffusers-format pipeline.
"""

from typing import Any, Optional

import torch
from huggingface_hub import hf_hub_download

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

DIFFUSERS_CONFIG_REPO = "Wan-AI/Wan2.1-VACE-1.3B-Diffusers"


class ModelVariant(StrEnum):
    """Available VACE-Wan2.1-1.3B-Preview variants."""

    VACE_WAN_2_1_1_3B_PREVIEW = "VACE-Wan2.1-1.3B-Preview"


class ModelLoader(ForgeModel):
    """VACE-Wan2.1-1.3B-Preview transformer loader for video generation."""

    _VARIANTS = {
        ModelVariant.VACE_WAN_2_1_1_3B_PREVIEW: ModelConfig(
            pretrained_model_name="ali-vilab/VACE-Wan2.1-1.3B-Preview",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VACE_WAN_2_1_1_3B_PREVIEW

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="VACE-Wan2.1-1.3B-Preview",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the VACE-Wan2.1-1.3B-Preview transformer."""
        from diffusers import WanVACETransformer3DModel

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        pretrained_model_name = self._variant_config.pretrained_model_name

        self._transformer = WanVACETransformer3DModel.from_single_file(
            hf_hub_download(
                pretrained_model_name, "diffusion_pytorch_model.safetensors"
            ),
            config=DIFFUSERS_CONFIG_REPO,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        self._transformer.eval()
        return self._transformer

    def load_inputs(
        self, dtype_override: Optional[torch.dtype] = None, **kwargs
    ) -> Any:
        """Prepare synthetic inputs for the VACE transformer forward pass."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        batch_size = 1
        in_channels = 16
        vace_in_channels = 96
        num_frames = 1
        height = 8
        width = 8
        text_seq_len = 32
        text_dim = 4096

        hidden_states = torch.randn(
            batch_size, in_channels, num_frames, height, width, dtype=dtype
        )
        control_hidden_states = torch.randn(
            batch_size, vace_in_channels, num_frames, height, width, dtype=dtype
        )
        encoder_hidden_states = torch.randn(
            batch_size, text_seq_len, text_dim, dtype=dtype
        )
        timestep = torch.tensor([500] * batch_size, dtype=torch.long)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "control_hidden_states": control_hidden_states,
            "return_dict": False,
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, (tuple, list)):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
