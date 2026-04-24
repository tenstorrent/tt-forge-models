#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SkyReels-V3 14B GGUF model loader implementation.

Loads GGUF-quantized SkyReels-V3 transformers from
vantagewithai/SkyReels-V3-14B-GGUF.

SkyReels-V3 is a multimodal video generation model built on the Wan 2.1
framework. It supports three generative tasks:
- Reference-to-Video (R2V): generate video from reference images + prompt
- Video-to-Video (V2V): extend existing video segments
- Audio-to-Video (A2V): talking avatar from image + audio

Available variants:
- SKYREELS_V3_R2V_Q8_0: Reference-to-Video Q8_0 quantization
- SKYREELS_V3_V2V_Q8_0: Video-to-Video Q8_0 quantization
- SKYREELS_V3_A2V_Q8_0: Audio-to-Video Q8_0 quantization
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

GGUF_REPO = "vantagewithai/SkyReels-V3-14B-GGUF"

# Small spatial dimensions for compile-only testing
TRANSFORMER_NUM_FRAMES = 2
TRANSFORMER_HEIGHT = 4
TRANSFORMER_WIDTH = 4
TRANSFORMER_TEXT_SEQ_LEN = 8


class ModelVariant(StrEnum):
    """Available SkyReels-V3 14B GGUF variants."""

    SKYREELS_V3_R2V_Q8_0 = "R2V_Q8_0"
    SKYREELS_V3_V2V_Q8_0 = "V2V_Q8_0"
    SKYREELS_V3_A2V_Q8_0 = "A2V_Q8_0"


_GGUF_FILES = {
    ModelVariant.SKYREELS_V3_R2V_Q8_0: "r2v/SkyReels-v3-r2v-Q8_0.gguf",
    ModelVariant.SKYREELS_V3_V2V_Q8_0: "v2v/SkyReels-v3-v2v-Q8_0.gguf",
    ModelVariant.SKYREELS_V3_A2V_Q8_0: "a2v/SkyReels-v3-a2v-Q8_0.gguf",
}


class ModelLoader(ForgeModel):
    """SkyReels-V3 14B GGUF model loader."""

    _VARIANTS = {
        ModelVariant.SKYREELS_V3_R2V_Q8_0: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
        ModelVariant.SKYREELS_V3_V2V_Q8_0: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
        ModelVariant.SKYREELS_V3_A2V_Q8_0: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.SKYREELS_V3_R2V_Q8_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SKYREELS_V3_14B_GGUF",
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
        """Load the GGUF-quantized SkyReels-V3 transformer.

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
        import diffusers.loaders.single_file_model as _sfm

        _orig_load = _sfm.load_model_dict_into_meta

        def _patched_load(model, state_dict, **kwargs):
            # GGUF stores scale_shift_table as [6, dim] but model expects [1, 6, dim]
            for key in list(state_dict.keys()):
                if "scale_shift_table" in key and state_dict[key].ndim == 2:
                    state_dict[key] = state_dict[key].unsqueeze(0)
            return _orig_load(model, state_dict, **kwargs)

        _sfm.load_model_dict_into_meta = _patched_load
        try:
            compute_dtype = (
                dtype_override if dtype_override is not None else torch.bfloat16
            )

            gguf_file = _GGUF_FILES[self._variant]
            quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

            self._transformer = WanTransformer3DModel.from_single_file(
                f"https://huggingface.co/{GGUF_REPO}/{gguf_file}",
                quantization_config=quantization_config,
                torch_dtype=compute_dtype,
            )
        finally:
            _sfm.load_model_dict_into_meta = _orig_load

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
