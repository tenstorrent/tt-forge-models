#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 NSFW Enhanced FASTMOVE V2 GGUF model loader implementation.

Loads GGUF-quantized Wan 2.2 Enhanced NSFW Camera Prompt FASTMOVE V2
transformer experts from Cassanovason69/Wan22nsfwenhanced. These are
community Q8 quantizations of a Wan 2.2 14B Mixture-of-Experts
text-to-video fine-tune, packaged as separate high-noise and low-noise
expert files.

Available variants:
- WAN22_NSFW_ENHANCED_FASTMOVE_V2_Q8_HIGH: high-noise expert, Q8
- WAN22_NSFW_ENHANCED_FASTMOVE_V2_Q8_LOW: low-noise expert, Q8
"""

from typing import Any, Optional

import torch

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

GGUF_REPO = "Cassanovason69/Wan22nsfwenhanced"

# Small spatial dimensions for compile-only testing
TRANSFORMER_NUM_FRAMES = 2
TRANSFORMER_HEIGHT = 4
TRANSFORMER_WIDTH = 4
TRANSFORMER_TEXT_SEQ_LEN = 8


class ModelVariant(StrEnum):
    """Available Wan 2.2 NSFW Enhanced FASTMOVE V2 GGUF variants."""

    WAN22_NSFW_ENHANCED_FASTMOVE_V2_Q8_HIGH = "2.2_NSFW_Enhanced_FASTMOVE_V2_Q8_High"
    WAN22_NSFW_ENHANCED_FASTMOVE_V2_Q8_LOW = "2.2_NSFW_Enhanced_FASTMOVE_V2_Q8_Low"


_GGUF_FILES = {
    ModelVariant.WAN22_NSFW_ENHANCED_FASTMOVE_V2_Q8_HIGH: "wan22EnhancedNSFWCameraPrompt_nsfwFASTMOVEV2Q8H.gguf",
    ModelVariant.WAN22_NSFW_ENHANCED_FASTMOVE_V2_Q8_LOW: "wan22EnhancedNSFWCameraPrompt_nsfwFASTMOVEV2Q8L.gguf",
}


class ModelLoader(ForgeModel):
    """Wan 2.2 NSFW Enhanced FASTMOVE V2 GGUF model loader."""

    _VARIANTS = {
        ModelVariant.WAN22_NSFW_ENHANCED_FASTMOVE_V2_Q8_HIGH: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
        ModelVariant.WAN22_NSFW_ENHANCED_FASTMOVE_V2_Q8_LOW: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_NSFW_ENHANCED_FASTMOVE_V2_Q8_HIGH

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_22_NSFW_ENHANCED_GGUF",
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
        """Load the GGUF-quantized Wan 2.2 NSFW Enhanced FASTMOVE V2 transformer.

        Uses diffusers GGUFQuantizationConfig to load the quantized transformer.
        Returns the transformer nn.Module directly for compilation testing.
        """
        import diffusers.utils.import_utils as _diffusers_import_utils

        if not _diffusers_import_utils._gguf_available:
            import importlib.util

            if importlib.util.find_spec("gguf") is not None:
                _diffusers_import_utils._gguf_available = True

        from diffusers import (
            GGUFQuantizationConfig,
            WanTransformer3DModel,
        )

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_file = _GGUF_FILES[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        self._transformer = WanTransformer3DModel.from_single_file(
            f"https://huggingface.co/{GGUF_REPO}/resolve/main/{gguf_file}",
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
        )

        return self._transformer

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare tensor inputs for the WanTransformer3DModel forward pass."""
        if self._transformer is None:
            self.load_model()

        dtype = torch.bfloat16
        config = self._transformer.config

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
