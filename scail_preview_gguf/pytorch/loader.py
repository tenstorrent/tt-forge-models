# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SCAIL-Preview GGUF model loader implementation for image-to-video generation.

SCAIL (Studio-Grade Character Animation via In-Context Learning) animates
characters from a single image using 3D-consistent pose representations.
Based on the Wan 2.1 14B architecture in GGUF format for ComfyUI.

Repository:
- https://huggingface.co/vantagewithai/SCAIL-Preview-GGUF
"""
from typing import Optional

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

REPO_ID = "vantagewithai/SCAIL-Preview-GGUF"
CONFIG_REPO = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"


class ModelVariant(StrEnum):
    """Available SCAIL-Preview GGUF model variants."""

    Q4_K_M = "Q4_K_M"
    Q8_0 = "Q8_0"


class ModelLoader(ForgeModel):
    """SCAIL-Preview GGUF model loader for image-to-video character animation."""

    _VARIANTS = {
        ModelVariant.Q4_K_M: ModelConfig(
            pretrained_model_name="vantagewithai/SCAIL-Preview-GGUF",
        ),
        ModelVariant.Q8_0: ModelConfig(
            pretrained_model_name="vantagewithai/SCAIL-Preview-GGUF",
        ),
    }

    _GGUF_FILES = {
        ModelVariant.Q4_K_M: "Wan21-14B-SCAIL-preview_comfy-Q4_K_M.gguf",
        ModelVariant.Q8_0: "Wan21-14B-SCAIL-preview_comfy-Q8_0.gguf",
    }

    DEFAULT_VARIANT = ModelVariant.Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="SCAIL-Preview GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        gguf_file = self._GGUF_FILES[self._variant]
        dtype = dtype_override if dtype_override is not None else torch.float32

        model_path = hf_hub_download(repo_id=REPO_ID, filename=gguf_file)

        self.transformer = WanTransformer3DModel.from_single_file(
            model_path,
            config=CONFIG_REPO,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        self.transformer.eval()

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.float32
        config = self.transformer.config

        in_channels = config.in_channels
        text_dim = config.text_dim
        frame, height, width = 2, 8, 8
        seq_len = frame * height * width

        hidden_states = torch.randn(batch_size, seq_len, in_channels, dtype=dtype)
        encoder_hidden_states = torch.randn(batch_size, 32, text_dim, dtype=dtype)
        timestep = torch.tensor([500.0] * batch_size, dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
        }
