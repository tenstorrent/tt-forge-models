# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 Fun InP GGUF model loader implementation for video generation
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

GGUF_REPO = "QuantStack/Wan2.2-Fun-A14B-InP-GGUF"
GGUF_FILE = "HighNoise/Wan2.2-Fun-A14B-InP_HighNoise-Q4_K_M.gguf"
CONFIG_REPO = "alibaba-pai/Wan2.2-Fun-A14B-InP"
CONFIG_SUBFOLDER = "high_noise_model"

TRANSFORMER_NUM_FRAMES = 2
TRANSFORMER_HEIGHT = 4
TRANSFORMER_WIDTH = 4
TRANSFORMER_TEXT_SEQ_LEN = 8


class ModelVariant(StrEnum):
    """Available Wan 2.2 Fun InP GGUF model variants."""

    A14B_HIGHNOISE_Q4_K_M = "A14B_HighNoise_Q4_K_M"


class ModelLoader(ForgeModel):
    """Wan 2.2 Fun InP GGUF model loader for video generation tasks."""

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
            model="Wan 2.2 Fun InP GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        import diffusers.utils.import_utils as _diffusers_import_utils

        if not _diffusers_import_utils._gguf_available:
            import importlib.metadata
            import importlib.util

            if importlib.util.find_spec("gguf") is not None:
                _diffusers_import_utils._gguf_available = True
                try:
                    _diffusers_import_utils._gguf_version = importlib.metadata.version(
                        "gguf"
                    )
                except importlib.metadata.PackageNotFoundError:
                    pass

                # gguf_quantizer.py was imported at collection time before gguf
                # was installed, so its module-level conditional imports were
                # skipped. Inject the missing symbols now.
                import diffusers.quantizers.gguf.gguf_quantizer as _gguf_q

                if not hasattr(_gguf_q, "_replace_with_gguf_linear"):
                    from diffusers.quantizers.gguf.utils import (
                        GGML_QUANT_SIZES,
                        GGUFParameter,
                        _dequantize_gguf_and_restore_linear,
                        _quant_shape_from_byte_shape,
                        _replace_with_gguf_linear,
                    )

                    _gguf_q.GGML_QUANT_SIZES = GGML_QUANT_SIZES
                    _gguf_q.GGUFParameter = GGUFParameter
                    _gguf_q._dequantize_gguf_and_restore_linear = (
                        _dequantize_gguf_and_restore_linear
                    )
                    _gguf_q._quant_shape_from_byte_shape = _quant_shape_from_byte_shape
                    _gguf_q._replace_with_gguf_linear = _replace_with_gguf_linear

        from diffusers import GGUFQuantizationConfig, WanTransformer3DModel

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        self._transformer = WanTransformer3DModel.from_single_file(
            f"https://huggingface.co/{GGUF_REPO}/{GGUF_FILE}",
            config=CONFIG_REPO,
            subfolder=CONFIG_SUBFOLDER,
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
        )

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
