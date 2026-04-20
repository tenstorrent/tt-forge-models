# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Isi99999/Wan2.1-I2V-14B-480P model loader implementation.

Loads the Wan 2.1 Image-to-Video 14B 480P diffusion transformer from the
Isi99999/Wan2.1-I2V-14B-480P rehost, which ships the weights as a single
``diffusion_pytorch_model.safetensors`` file in the original Wan format.
Uses the upstream Wan-AI/Wan2.1-I2V-14B-480P-Diffusers repo for the
transformer config.

Available variants:
- WAN21_I2V_14B_480P: Wan 2.1 I2V 14B 480P single-file safetensors
"""

from typing import Any, Optional

import torch
from diffusers import WanTransformer3DModel
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

REPO_ID = "Isi99999/Wan2.1-I2V-14B-480P"
CONFIG_REPO = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
TRANSFORMER_FILENAME = "diffusion_pytorch_model.safetensors"


class ModelVariant(StrEnum):
    """Available Isi99999/Wan2.1-I2V-14B-480P model variants."""

    WAN21_I2V_14B_480P = "2.1_I2V_14B_480P"


class ModelLoader(ForgeModel):
    """Isi99999/Wan2.1-I2V-14B-480P loader for the Wan diffusion transformer."""

    _VARIANTS = {
        ModelVariant.WAN21_I2V_14B_480P: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN21_I2V_14B_480P

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ISI99999_WAN_2_1_I2V_14B_480P",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_transformer(
        self, dtype: torch.dtype = torch.bfloat16
    ) -> WanTransformer3DModel:
        """Load the Wan 2.1 I2V transformer from a single safetensors file."""
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=TRANSFORMER_FILENAME,
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
        """Load and return the Wan 2.1 I2V 14B 480P diffusion transformer."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._transformer is None:
            return self._load_transformer(dtype)
        if dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare synthetic inputs for the Wan 2.1 I2V transformer forward pass.

        The I2V 14B 480P variant has 36 input channels (16 latent + 16 latent
        reference + 4 mask channels) and a text embedding dim of 4096.
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        batch_size = kwargs.get("batch_size", 1)

        in_channels = 36
        text_dim = 4096
        txt_seq_len = 32

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

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
