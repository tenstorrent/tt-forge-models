# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
smoothMixWan22 T2V FP16 model loader implementation.

Loads the FP16 safetensors Wan 2.2 text-to-video diffusion transformer
variants from BigDannyPt/WAN-2.2-SmoothMix-FP16. Uses the upstream
Wan-AI/Wan2.2-T2V-A14B-Diffusers config for model construction.

Available variants:
- HIGH_NOISE_FP16: HighNoise FP16 safetensors expert
- LOW_NOISE_FP16: LowNoise FP16 safetensors expert
"""

from typing import Any, Optional

import torch
from diffusers import WanTransformer3DModel
from huggingface_hub import hf_hub_download

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

REPO_ID = "BigDannyPt/WAN-2.2-SmoothMix-FP16"
CONFIG_REPO = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"


class ModelVariant(StrEnum):
    """Available smoothMixWan22 T2V FP16 variants."""

    HIGH_NOISE_FP16 = "HighNoise_FP16"
    LOW_NOISE_FP16 = "LowNoise_FP16"


_SAFETENSORS_FILES = {
    ModelVariant.HIGH_NOISE_FP16: "smoothmixTxt2vidHigh-FP16.safetensors",
    ModelVariant.LOW_NOISE_FP16: "smoothmixTxt2vidLow-FP16.safetensors",
}


class ModelLoader(ForgeModel):
    """smoothMixWan22 T2V FP16 model loader for the diffusion transformer."""

    _VARIANTS = {
        ModelVariant.HIGH_NOISE_FP16: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.LOW_NOISE_FP16: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.HIGH_NOISE_FP16

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SMOOTH_MIX_WAN22_T2V_FP16",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_transformer(
        self, dtype: torch.dtype = torch.float16
    ) -> WanTransformer3DModel:
        """Load diffusion transformer from FP16 safetensors file."""
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=_SAFETENSORS_FILES[self._variant],
        )

        self._transformer = WanTransformer3DModel.from_single_file(
            model_path,
            config=CONFIG_REPO,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        self._transformer.eval()
        return self._transformer

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the smoothMixWan22 T2V FP16 diffusion transformer."""
        dtype = dtype_override if dtype_override is not None else torch.float16
        if self._transformer is None:
            return self._load_transformer(dtype)
        if dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the Wan T2V diffusion transformer."""
        dtype = kwargs.get("dtype_override", torch.float16)
        batch_size = kwargs.get("batch_size", 1)

        # Wan T2V 14B transformer config dimensions
        in_channels = 16  # 16 latent channels for T2V
        text_dim = 4096  # text_dim from Wan config
        txt_seq_len = 32

        # Spatial/temporal latent dimensions
        frame, height, width = 2, 8, 8
        seq_len = frame * height * width

        hidden_states = torch.randn(batch_size, seq_len, in_channels, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, txt_seq_len, text_dim, dtype=dtype
        )
        timestep = torch.tensor([500.0] * batch_size, dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
        }
