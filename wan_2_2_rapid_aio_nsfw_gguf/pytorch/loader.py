#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 14B Rapid AllInOne NSFW v10 I2V GGUF model loader implementation.

Loads GGUF-quantized Wan 2.2 Image-to-Video Rapid AllInOne transformers from
DoorZekor/WAN2.2-14B-Rapid-AllInOne-GGUF-NSFW-v10. These are community
quantizations of Phr00t/WAN2.2-14B-Rapid-AllInOne combining both
high-noise and low-noise experts into a single transformer.

Available variants:
- WAN22_RAPID_AIO_NSFW_V10_Q4_K_M: Q4_K_M quantization
- WAN22_RAPID_AIO_NSFW_V10_Q8_0: Q8_0 quantization
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

GGUF_REPO = "DoorZekor/WAN2.2-14B-Rapid-AllInOne-GGUF-NSFW-v10"

# Small spatial dimensions for compile-only testing
TRANSFORMER_NUM_FRAMES = 2
TRANSFORMER_HEIGHT = 4
TRANSFORMER_WIDTH = 4
TRANSFORMER_TEXT_SEQ_LEN = 8


class ModelVariant(StrEnum):
    """Available Wan 2.2 Rapid AllInOne NSFW v10 I2V GGUF variants."""

    WAN22_RAPID_AIO_NSFW_V10_Q4_K_M = "2.2_Rapid_AIO_NSFW_v10_Q4_K_M"
    WAN22_RAPID_AIO_NSFW_V10_Q8_0 = "2.2_Rapid_AIO_NSFW_v10_Q8_0"


_GGUF_FILES = {
    ModelVariant.WAN22_RAPID_AIO_NSFW_V10_Q4_K_M: "wan2.2-i2v-rapid-aio-v10-nsfw-Q4_K_M.gguf",
    ModelVariant.WAN22_RAPID_AIO_NSFW_V10_Q8_0: "wan2.2-i2v-rapid-aio-v10-nsfw-Q8_0.gguf",
}


class ModelLoader(ForgeModel):
    """Wan 2.2 14B Rapid AllInOne NSFW v10 I2V GGUF model loader."""

    _VARIANTS = {
        ModelVariant.WAN22_RAPID_AIO_NSFW_V10_Q4_K_M: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
        ModelVariant.WAN22_RAPID_AIO_NSFW_V10_Q8_0: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_RAPID_AIO_NSFW_V10_Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_2_2_RAPID_AIO_NSFW_GGUF",
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
        """Load the GGUF-quantized Wan 2.2 Rapid AllInOne I2V transformer.

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
