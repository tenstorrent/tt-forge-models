# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 VACE Fun A14B GGUF model loader implementation for video generation
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

GGUF_REPO = "QuantStack/Wan2.2-VACE-Fun-A14B-GGUF"
GGUF_BASE_URL = f"https://huggingface.co/{GGUF_REPO}/blob/main"

# Small spatial dimensions for compile-only testing
TRANSFORMER_NUM_FRAMES = 2
TRANSFORMER_HEIGHT = 4
TRANSFORMER_WIDTH = 4
TRANSFORMER_TEXT_SEQ_LEN = 8


class ModelVariant(StrEnum):
    """Available Wan 2.2 VACE Fun A14B GGUF model variants."""

    A14B_HIGHNOISE_Q4_K_M = "A14B_HighNoise_Q4_K_M"


_GGUF_FILES = {
    ModelVariant.A14B_HIGHNOISE_Q4_K_M: "HighNoise/Wan2.2-VACE-Fun-A14B-high-noise-Q4_K_M.gguf",
}


class ModelLoader(ForgeModel):
    """Wan 2.2 VACE Fun A14B GGUF model loader for video generation tasks."""

    _VARIANTS = {
        ModelVariant.A14B_HIGHNOISE_Q4_K_M: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.A14B_HIGHNOISE_Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Wan 2.2 VACE Fun A14B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        import diffusers.utils.import_utils as _diffusers_import_utils

        if not _diffusers_import_utils._gguf_available:
            import importlib.util

            if importlib.util.find_spec("gguf") is not None:
                _diffusers_import_utils._gguf_available = True

        from diffusers import GGUFQuantizationConfig, WanTransformer3DModel
        from diffusers.quantizers.gguf.utils import _dequantize_gguf_and_restore_linear

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        gguf_file = _GGUF_FILES[self._variant]
        gguf_url = f"{GGUF_BASE_URL}/{gguf_file}"

        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)
        self._transformer = WanTransformer3DModel.from_single_file(
            gguf_url,
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
        )

        # Dequantize GGUFLinear layers to regular Linear with float weights so
        # that TorchFunctionMode does not see quantized byte tensors at runtime.
        _dequantize_gguf_and_restore_linear(self._transformer)
        # Clear quantization markers so .to(dtype) is not blocked.
        self._transformer.quantization_method = None
        self._transformer.is_quantized = False
        self._transformer = self._transformer.to(compute_dtype)

        return self._transformer

    def load_inputs(
        self, dtype_override: Optional[torch.dtype] = None, **kwargs
    ) -> Any:
        if self._transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
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
