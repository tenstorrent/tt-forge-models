#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.1 I2V 14B FusionX GGUF model loader implementation.

Loads GGUF-quantized Wan 2.1 Image-to-Video FusionX transformers from
QuantStack/Wan2.1_I2V_14B_FusionX-GGUF.

FusionX is a fine-tune of Wan 2.1 adapted for image-to-video generation
built on the Wan2.1 14B architecture (~14B parameters).

Available variants:
- WAN21_I2V_14B_FUSIONX_Q4_K_S: Q4_K_S quantization (~9.95 GB)
- WAN21_I2V_14B_FUSIONX_Q8_0: Q8_0 quantization (~17.7 GB)
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

GGUF_REPO = "QuantStack/Wan2.1_I2V_14B_FusionX-GGUF"

# Small spatial dimensions for compile-only testing
TRANSFORMER_NUM_FRAMES = 2
TRANSFORMER_HEIGHT = 4
TRANSFORMER_WIDTH = 4
TRANSFORMER_TEXT_SEQ_LEN = 8


class ModelVariant(StrEnum):
    """Available Wan 2.1 I2V 14B FusionX GGUF variants."""

    WAN21_I2V_14B_FUSIONX_Q4_K_S = "2.1_I2V_14B_FusionX_Q4_K_S"
    WAN21_I2V_14B_FUSIONX_Q8_0 = "2.1_I2V_14B_FusionX_Q8_0"


_GGUF_FILES = {
    ModelVariant.WAN21_I2V_14B_FUSIONX_Q4_K_S: "Wan2.1_I2V_14B_FusionX-Q4_K_S.gguf",
    ModelVariant.WAN21_I2V_14B_FUSIONX_Q8_0: "Wan2.1_I2V_14B_FusionX-Q8_0.gguf",
}


class ModelLoader(ForgeModel):
    """Wan 2.1 I2V 14B FusionX GGUF model loader."""

    _VARIANTS = {
        ModelVariant.WAN21_I2V_14B_FUSIONX_Q4_K_S: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
        ModelVariant.WAN21_I2V_14B_FUSIONX_Q8_0: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN21_I2V_14B_FUSIONX_Q4_K_S

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_2_1_I2V_14B_FUSIONX_GGUF",
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
        """Load the GGUF-quantized Wan 2.1 I2V 14B FusionX transformer.

        Uses diffusers GGUFQuantizationConfig to load the quantized transformer.
        Returns the transformer nn.Module directly for compilation testing.
        """
        import sys

        import diffusers.utils.import_utils as _diffusers_import_utils

        if not _diffusers_import_utils._gguf_available:
            import importlib.util

            if importlib.util.find_spec("gguf") is not None:
                import importlib.metadata

                _diffusers_import_utils._gguf_available = True
                try:
                    _diffusers_import_utils._gguf_version = importlib.metadata.version(
                        "gguf"
                    )
                except importlib.metadata.PackageNotFoundError:
                    pass

        # Patch missing gguf utilities into the quantizer module when diffusers was
        # imported before gguf was installed (lazy requirements.txt install timing).
        _gguf_q_mod = sys.modules.get("diffusers.quantizers.gguf.gguf_quantizer")
        if _gguf_q_mod is not None and not hasattr(
            _gguf_q_mod, "_replace_with_gguf_linear"
        ):
            from diffusers.quantizers.gguf.utils import (
                GGML_QUANT_SIZES,
                GGUFParameter,
                _dequantize_gguf_and_restore_linear,
                _quant_shape_from_byte_shape,
                _replace_with_gguf_linear,
            )

            _gguf_q_mod.GGML_QUANT_SIZES = GGML_QUANT_SIZES
            _gguf_q_mod.GGUFParameter = GGUFParameter
            _gguf_q_mod._dequantize_gguf_and_restore_linear = (
                _dequantize_gguf_and_restore_linear
            )
            _gguf_q_mod._quant_shape_from_byte_shape = _quant_shape_from_byte_shape
            _gguf_q_mod._replace_with_gguf_linear = _replace_with_gguf_linear

        from diffusers import (
            GGUFQuantizationConfig,
            WanTransformer3DModel,
        )

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_file = _GGUF_FILES[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        self._transformer = WanTransformer3DModel.from_single_file(
            f"https://huggingface.co/{GGUF_REPO}/{gguf_file}",
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
