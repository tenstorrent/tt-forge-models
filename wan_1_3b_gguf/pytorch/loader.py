#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.1 T2V 1.3B GGUF model loader implementation.

Loads GGUF-quantized Wan 2.1 Text-to-Video 1.3B transformers from
calcuis/wan-1.3b-gguf and returns the transformer nn.Module directly
for compilation testing.

Available variants:
- WAN21_T2V_1_3B_Q4_0: Q4_0 quantization (~917 MB)
- WAN21_T2V_1_3B_F16: F16 (~2.89 GB)
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

GGUF_REPO_ID = "calcuis/wan-1.3b-gguf"

# Small spatial dimensions for compile-only testing
TRANSFORMER_NUM_FRAMES = 2
TRANSFORMER_HEIGHT = 4
TRANSFORMER_WIDTH = 4
TRANSFORMER_TEXT_SEQ_LEN = 8


class ModelVariant(StrEnum):
    """Available Wan 2.1 T2V 1.3B GGUF variants."""

    WAN21_T2V_1_3B_Q4_0 = "2.1_T2V_1.3B_Q4_0"
    WAN21_T2V_1_3B_F16 = "2.1_T2V_1.3B_F16"


_GGUF_FILES = {
    ModelVariant.WAN21_T2V_1_3B_Q4_0: "wan2.1_t2v_1.3b-q4_0.gguf",
    ModelVariant.WAN21_T2V_1_3B_F16: "wan2.1_t2v_1.3b-f16.gguf",
}


class ModelLoader(ForgeModel):
    """Wan 2.1 T2V 1.3B GGUF model loader."""

    _VARIANTS = {
        ModelVariant.WAN21_T2V_1_3B_Q4_0: ModelConfig(
            pretrained_model_name=GGUF_REPO_ID,
        ),
        ModelVariant.WAN21_T2V_1_3B_F16: ModelConfig(
            pretrained_model_name=GGUF_REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN21_T2V_1_3B_Q4_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_1_3B_GGUF",
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
        """Load the GGUF-quantized Wan 2.1 T2V 1.3B transformer.

        Returns the transformer nn.Module directly for compilation testing.
        """
        import diffusers.utils.import_utils as _diffusers_import_utils

        if not _diffusers_import_utils._gguf_available:
            import importlib.util

            if importlib.util.find_spec("gguf") is not None:
                _diffusers_import_utils._gguf_available = True

        from diffusers import GGUFQuantizationConfig, WanTransformer3DModel

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_filename = _GGUF_FILES[self._variant]
        gguf_path = hf_hub_download(
            repo_id=GGUF_REPO_ID,
            filename=gguf_filename,
        )

        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)
        self._transformer = WanTransformer3DModel.from_single_file(
            gguf_path,
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
