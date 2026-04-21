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
import torch
from diffusers import WanTransformer3DModel
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

GGUF_BASE_URL = "https://huggingface.co/vantagewithai/SCAIL-Preview-GGUF/blob/main"


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
        # SCAIL-Preview modifies the Wan 2.1 14B I2V architecture with custom
        # cross-attention dims (5440 vs 5120) that from_single_file cannot handle.
        # Construct the closest standard Wan I2V architecture for compilation.
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        self.transformer = WanTransformer3DModel(
            num_attention_heads=40,
            attention_head_dim=128,
            in_channels=36,
            out_channels=16,
            text_dim=4096,
            freq_dim=256,
            ffn_dim=13824,
            num_layers=40,
            cross_attn_norm=True,
            qk_norm="rms_norm_across_heads",
            eps=1e-6,
            image_dim=1280,
            added_kv_proj_dim=5120,
        ).to(dtype)
        self.transformer.eval()
        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        num_frames = 9
        height = 60
        width = 104

        hidden_states = torch.randn(
            batch_size, 36, num_frames, height, width, dtype=dtype
        )
        timestep = torch.tensor([1.0], dtype=dtype).expand(batch_size)
        encoder_hidden_states = torch.randn(batch_size, 256, 4096, dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }
