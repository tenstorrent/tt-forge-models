# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FireRed-Image-Edit 1.1 ComfyUI Quants model loader implementation.

Loads FP8/NVFP4 single-file safetensors diffusion transformer variants from
drbaph/FireRed-Image-Edit-1.1_ComfyUI_Quants. Uses the upstream
FireRedTeam/FireRed-Image-Edit-1.1 diffusers config for model construction.

Available variants:
- FP8_E4M3FN: FP8 E4M3FN quantized transformer
- FP8_SCALED_E4M3FN: FP8 E4M3FN scaled quantized transformer
- NVFP4: NVFP4 quantized transformer
"""

from typing import Any, Optional

import torch
from diffusers import QwenImageTransformer2DModel
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

REPO_ID = "drbaph/FireRed-Image-Edit-1.1_ComfyUI_Quants"

# Upstream diffusers config source
_CONFIG_REPO = "FireRedTeam/FireRed-Image-Edit-1.1"


class ModelVariant(StrEnum):
    """Available FireRed-Image-Edit 1.1 ComfyUI Quants model variants."""

    FP8_E4M3FN = "fp8_e4m3fn"
    FP8_SCALED_E4M3FN = "fp8_scaled_e4m3fn"
    NVFP4 = "nvfp4"


_DIFFUSION_FILES = {
    ModelVariant.FP8_E4M3FN: "firered_image_edit_1.1_fp8_e4m3fn.safetensors",
    ModelVariant.FP8_SCALED_E4M3FN: "firered_image_edit_1.1_fp8_scaled_e4m3fn.safetensors",
    ModelVariant.NVFP4: "firered_image_edit_1.1_nvfp4.safetensors.safetensors",
}


class ModelLoader(ForgeModel):
    """FireRed-Image-Edit 1.1 ComfyUI Quants model loader for the diffusion transformer."""

    _VARIANTS = {
        ModelVariant.FP8_E4M3FN: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.FP8_SCALED_E4M3FN: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.NVFP4: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.FP8_E4M3FN

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FIRERED_IMAGE_EDIT_1_1_QUANTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_transformer(
        self, dtype: torch.dtype = torch.bfloat16
    ) -> QwenImageTransformer2DModel:
        """Load diffusion transformer from single-file safetensors."""
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=_DIFFUSION_FILES[self._variant],
        )

        self._transformer = QwenImageTransformer2DModel.from_single_file(
            model_path,
            config=_CONFIG_REPO,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        self._transformer.eval()
        return self._transformer

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the FireRed-Image-Edit 1.1 quantized diffusion transformer.

        Returns:
            QwenImageTransformer2DModel instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._transformer is None:
            return self._load_transformer(dtype)
        if dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the diffusion transformer.

        Returns a dict matching QwenImageTransformer2DModel.forward() signature.
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        batch_size = kwargs.get("batch_size", 1)

        # From model config: in_channels=64 (img_in linear input dimension)
        img_dim = 64
        # joint_attention_dim from config = 3584
        text_dim = 3584
        txt_seq_len = 32

        # img_seq_len must equal frame * height * width for positional encoding
        frame, height, width = 1, 8, 8
        img_seq_len = frame * height * width

        hidden_states = torch.randn(batch_size, img_seq_len, img_dim, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, txt_seq_len, text_dim, dtype=dtype
        )
        encoder_hidden_states_mask = torch.ones(batch_size, txt_seq_len, dtype=dtype)
        timestep = torch.tensor([500.0] * batch_size, dtype=dtype)
        # img_shapes: list of (frame, height, width) tuples per batch item
        img_shapes = [(frame, height, width)] * batch_size

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
        }
