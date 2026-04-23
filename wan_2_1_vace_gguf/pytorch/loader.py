#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.1 VACE 14B GGUF model loader implementation.

Loads GGUF-quantized Wan 2.1 VACE transformers from the QuantStack
VACE GGUF repositories and builds a WanVACEPipeline.

The Wan 2.1 VACE (Video All-in-one Creation Engine) model supports
versatile video creation and editing tasks including reference-to-video
generation. This loader uses GGUF-quantized weights for reduced memory
usage.

Available variants:
- WAN21_VACE_Q4_K_M: Q4_K_M quantization
- WAN21_VACE_Q8_0: Q8_0 quantization
- WAN21_FUSIONX_VACE_Q4_K_M: FusionX fine-tune, Q4_K_M quantization
- WAN21_FUSIONX_VACE_Q8_0: FusionX fine-tune, Q8_0 quantization
"""

from typing import Any, Dict, Optional

import torch

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

VACE_GGUF_REPO = "QuantStack/Wan2.1_14B_VACE-GGUF"
FUSIONX_VACE_GGUF_REPO = "QuantStack/Wan2.1_T2V_14B_FusionX_VACE-GGUF"
BASE_PIPELINE = "Wan-AI/Wan2.1-VACE-14B-diffusers"

# Small synthetic dimensions for transformer forward pass
# patch_size is [1, 2, 2]; spatial dims must be divisible by patch_size
TRANSFORMER_NUM_FRAMES = 1
TRANSFORMER_HEIGHT = 4
TRANSFORMER_WIDTH = 4
TRANSFORMER_TEXT_SEQ_LEN = 8


class ModelVariant(StrEnum):
    """Available Wan 2.1 VACE 14B GGUF variants."""

    WAN21_VACE_Q4_K_M = "2.1_VACE_Q4_K_M"
    WAN21_VACE_Q8_0 = "2.1_VACE_Q8_0"
    WAN21_FUSIONX_VACE_Q4_K_M = "2.1_FusionX_VACE_Q4_K_M"
    WAN21_FUSIONX_VACE_Q8_0 = "2.1_FusionX_VACE_Q8_0"


_GGUF_FILES = {
    ModelVariant.WAN21_VACE_Q4_K_M: "Wan2.1_14B_VACE-Q4_K_M.gguf",
    ModelVariant.WAN21_VACE_Q8_0: "Wan2.1_14B_VACE-Q8_0.gguf",
    ModelVariant.WAN21_FUSIONX_VACE_Q4_K_M: "Wan2.1_T2V_14B_FusionX_VACE-Q4_K_M.gguf",
    ModelVariant.WAN21_FUSIONX_VACE_Q8_0: "Wan2.1_T2V_14B_FusionX_VACE-Q8_0.gguf",
}

_GGUF_REPOS = {
    ModelVariant.WAN21_VACE_Q4_K_M: VACE_GGUF_REPO,
    ModelVariant.WAN21_VACE_Q8_0: VACE_GGUF_REPO,
    ModelVariant.WAN21_FUSIONX_VACE_Q4_K_M: FUSIONX_VACE_GGUF_REPO,
    ModelVariant.WAN21_FUSIONX_VACE_Q8_0: FUSIONX_VACE_GGUF_REPO,
}


class ModelLoader(ForgeModel):
    """Wan 2.1 VACE 14B GGUF model loader."""

    _VARIANTS = {
        ModelVariant.WAN21_VACE_Q4_K_M: ModelConfig(
            pretrained_model_name=VACE_GGUF_REPO,
        ),
        ModelVariant.WAN21_VACE_Q8_0: ModelConfig(
            pretrained_model_name=VACE_GGUF_REPO,
        ),
        ModelVariant.WAN21_FUSIONX_VACE_Q4_K_M: ModelConfig(
            pretrained_model_name=FUSIONX_VACE_GGUF_REPO,
        ),
        ModelVariant.WAN21_FUSIONX_VACE_Q8_0: ModelConfig(
            pretrained_model_name=FUSIONX_VACE_GGUF_REPO,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN21_VACE_Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_2_1_VACE_GGUF",
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
        """Load the GGUF-quantized Wan 2.1 VACE transformer and build the pipeline.

        Uses diffusers GGUFQuantizationConfig to load the quantized transformer,
        then constructs the full WanVACEPipeline with the base model's VAE in
        float32 for numerical stability.

        Returns:
            WanVACETransformer3DModel extracted from the pipeline.
        """
        from diffusers import AutoencoderKLWan, GGUFQuantizationConfig, WanVACEPipeline
        from diffusers.models.transformers.transformer_wan_vace import (
            WanVACETransformer3DModel,
        )

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_file = _GGUF_FILES[self._variant]
        gguf_repo = _GGUF_REPOS[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        transformer = WanVACETransformer3DModel.from_single_file(
            f"https://huggingface.co/{gguf_repo}/{gguf_file}",
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
        )

        vae = AutoencoderKLWan.from_pretrained(
            BASE_PIPELINE,
            subfolder="vae",
            torch_dtype=torch.float32,
        )

        self.pipeline = WanVACEPipeline.from_pretrained(
            BASE_PIPELINE,
            transformer=transformer,
            vae=vae,
            torch_dtype=compute_dtype,
        )

        return self.pipeline.transformer

    def _load_transformer_inputs(self, dtype: torch.dtype) -> Dict[str, Any]:
        """Prepare synthetic tensor inputs for the WanVACETransformer3DModel forward pass."""
        config = self.pipeline.transformer.config
        return {
            "hidden_states": torch.randn(
                1,
                config.in_channels,
                TRANSFORMER_NUM_FRAMES,
                TRANSFORMER_HEIGHT,
                TRANSFORMER_WIDTH,
                dtype=dtype,
            ),
            "encoder_hidden_states": torch.randn(
                1,
                TRANSFORMER_TEXT_SEQ_LEN,
                config.text_dim,
                dtype=dtype,
            ),
            "timestep": torch.tensor([500], dtype=torch.long),
            "return_dict": False,
        }

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for the VACE transformer forward pass."""
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        return self._load_transformer_inputs(dtype)
