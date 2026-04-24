#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.1 VACE 14B GGUF model loader implementation.

Loads GGUF-quantized Wan 2.1 VACE transformers from the QuantStack
VACE GGUF repositories. Returns the WanVACETransformer3DModel directly
for compilation testing.

Available variants:
- WAN21_VACE_Q4_K_M: Q4_K_M quantization
- WAN21_VACE_Q8_0: Q8_0 quantization
- WAN21_FUSIONX_VACE_Q4_K_M: FusionX fine-tune, Q4_K_M quantization
- WAN21_FUSIONX_VACE_Q8_0: FusionX fine-tune, Q8_0 quantization
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

VACE_GGUF_REPO = "QuantStack/Wan2.1_14B_VACE-GGUF"
FUSIONX_VACE_GGUF_REPO = "QuantStack/Wan2.1_T2V_14B_FusionX_VACE-GGUF"

TRANSFORMER_NUM_FRAMES = 2
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
        self._transformer = None

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
        """Load the GGUF-quantized Wan 2.1 VACE transformer.

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
            WanVACETransformer3DModel,
        )

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_file = _GGUF_FILES[self._variant]
        gguf_repo = _GGUF_REPOS[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        self._transformer = WanVACETransformer3DModel.from_single_file(
            f"https://huggingface.co/{gguf_repo}/resolve/main/{gguf_file}",
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
        )

        return self._transformer

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare tensor inputs for the WanVACETransformer3DModel forward pass."""
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
