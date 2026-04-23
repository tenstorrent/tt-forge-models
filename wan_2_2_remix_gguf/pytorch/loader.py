#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 Remix GGUF model loader implementation.

Loads GGUF-quantized Wan 2.2 Remix transformers from community
re-uploads (BigDannyPt/Wan-2.2-Remix-GGUF and
huchukato/Wan2.2-Remix-I2V-v2.1-GGUF).

The Wan 2.2 Remix is a community fine-tune of the Wan 2.2 14B model
supporting both text-to-video (T2V) and image-to-video (I2V) generation.

Available variants:
- WAN22_REMIX_T2V_HIGH_V2_Q4_K_M: T2V high-noise expert v2.0, Q4_K_M
- WAN22_REMIX_I2V_HIGH_V3_Q4_K_M: I2V high-noise expert v3.0, Q4_K_M
- WAN22_REMIX_I2V_HIGH_V2_1_Q4_K_M: I2V high-noise expert v2.1, Q4_K_M
- WAN22_REMIX_I2V_LOW_V2_1_Q4_K_M: I2V low-noise expert v2.1, Q4_K_M
"""

from typing import Any, Optional

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

BIGDANNYPT_GGUF_REPO = "BigDannyPt/Wan-2.2-Remix-GGUF"
HUCHUKATO_I2V_V2_1_GGUF_REPO = "huchukato/Wan2.2-Remix-I2V-v2.1-GGUF"

# Small spatial dimensions for compile-only testing
TRANSFORMER_NUM_FRAMES = 2
TRANSFORMER_HEIGHT = 4
TRANSFORMER_WIDTH = 4
TRANSFORMER_TEXT_SEQ_LEN = 8


class ModelVariant(StrEnum):
    """Available Wan 2.2 Remix GGUF variants."""

    WAN22_REMIX_T2V_HIGH_V2_Q4_K_M = "2.2_Remix_T2V_High_v2.0_Q4_K_M"
    WAN22_REMIX_I2V_HIGH_V3_Q4_K_M = "2.2_Remix_I2V_High_v3.0_Q4_K_M"
    WAN22_REMIX_I2V_HIGH_V2_1_Q4_K_M = "2.2_Remix_I2V_High_v2.1_Q4_K_M"
    WAN22_REMIX_I2V_LOW_V2_1_Q4_K_M = "2.2_Remix_I2V_Low_v2.1_Q4_K_M"


_GGUF_REPOS = {
    ModelVariant.WAN22_REMIX_T2V_HIGH_V2_Q4_K_M: BIGDANNYPT_GGUF_REPO,
    ModelVariant.WAN22_REMIX_I2V_HIGH_V3_Q4_K_M: BIGDANNYPT_GGUF_REPO,
    ModelVariant.WAN22_REMIX_I2V_HIGH_V2_1_Q4_K_M: HUCHUKATO_I2V_V2_1_GGUF_REPO,
    ModelVariant.WAN22_REMIX_I2V_LOW_V2_1_Q4_K_M: HUCHUKATO_I2V_V2_1_GGUF_REPO,
}

_GGUF_FILES = {
    ModelVariant.WAN22_REMIX_T2V_HIGH_V2_Q4_K_M: "T2V/v2.0/High/wan22RemixT2VI2V_t2vHighV20-Q4_K_M.gguf",
    ModelVariant.WAN22_REMIX_I2V_HIGH_V3_Q4_K_M: "I2V/v3.0/High/wan22RemixT2VI2V_i2vHighV30-Q4_K_M.gguf",
    ModelVariant.WAN22_REMIX_I2V_HIGH_V2_1_Q4_K_M: "High/wan22RemixT2VI2V_i2vHighV21-Q4_K_M.gguf",
    ModelVariant.WAN22_REMIX_I2V_LOW_V2_1_Q4_K_M: "Low/wan22RemixT2VI2V_i2vLowV21-Q4_K_M.gguf",
}


class ModelLoader(ForgeModel):
    """Wan 2.2 Remix GGUF model loader."""

    _VARIANTS = {
        ModelVariant.WAN22_REMIX_T2V_HIGH_V2_Q4_K_M: ModelConfig(
            pretrained_model_name=BIGDANNYPT_GGUF_REPO,
        ),
        ModelVariant.WAN22_REMIX_I2V_HIGH_V3_Q4_K_M: ModelConfig(
            pretrained_model_name=BIGDANNYPT_GGUF_REPO,
        ),
        ModelVariant.WAN22_REMIX_I2V_HIGH_V2_1_Q4_K_M: ModelConfig(
            pretrained_model_name=HUCHUKATO_I2V_V2_1_GGUF_REPO,
        ),
        ModelVariant.WAN22_REMIX_I2V_LOW_V2_1_Q4_K_M: ModelConfig(
            pretrained_model_name=HUCHUKATO_I2V_V2_1_GGUF_REPO,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_REMIX_T2V_HIGH_V2_Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_2_2_REMIX_GGUF",
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
        """Load the GGUF-quantized Wan 2.2 Remix transformer.

        Uses diffusers GGUFQuantizationConfig to load the quantized transformer.
        Returns the transformer nn.Module directly for compilation testing.

        The GGUF files may omit certain I2V-specific weights (image cross-attention
        projections). A patched dispatch_model zero-initialises those missing
        parameters so the model can be loaded and compiled without errors.
        """
        import importlib.metadata

        import diffusers.quantizers.gguf.gguf_quantizer as _gguf_quantizer_mod
        import diffusers.utils.import_utils as _diffusers_import_utils

        if not _diffusers_import_utils._gguf_available:
            try:
                _gguf_ver = importlib.metadata.version("gguf")
                _diffusers_import_utils._gguf_available = True
                _diffusers_import_utils._gguf_version = _gguf_ver
                from diffusers.quantizers.gguf.utils import (
                    GGML_QUANT_SIZES,
                    GGUFParameter,
                    _dequantize_gguf_and_restore_linear,
                    _quant_shape_from_byte_shape,
                    _replace_with_gguf_linear,
                )

                _gguf_quantizer_mod.GGML_QUANT_SIZES = GGML_QUANT_SIZES
                _gguf_quantizer_mod.GGUFParameter = GGUFParameter
                _gguf_quantizer_mod._dequantize_gguf_and_restore_linear = (
                    _dequantize_gguf_and_restore_linear
                )
                _gguf_quantizer_mod._quant_shape_from_byte_shape = (
                    _quant_shape_from_byte_shape
                )
                _gguf_quantizer_mod._replace_with_gguf_linear = (
                    _replace_with_gguf_linear
                )
            except importlib.metadata.PackageNotFoundError:
                pass

        import accelerate.big_modeling as _accel_bm
        import diffusers.loaders.single_file_model as _sfm
        import torch.nn as nn
        from diffusers import GGUFQuantizationConfig, WanTransformer3DModel
        from huggingface_hub import hf_hub_download

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_repo = _GGUF_REPOS[self._variant]
        gguf_file = _GGUF_FILES[self._variant]
        gguf_path = hf_hub_download(repo_id=gguf_repo, filename=gguf_file)
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        orig_dispatch = _accel_bm.dispatch_model

        def _patched_dispatch(model, device_map=None, **kwargs):
            if device_map:
                for name, param in list(model.named_parameters()):
                    if param.device.type != "meta":
                        continue
                    parts = name.split(".")
                    module = model
                    for part in parts[:-1]:
                        module = getattr(module, part)
                    module._parameters[parts[-1]] = nn.Parameter(
                        torch.zeros(param.shape, dtype=compute_dtype, device="cpu"),
                        requires_grad=param.requires_grad,
                    )
            return orig_dispatch(model, device_map=device_map, **kwargs)

        _accel_bm.dispatch_model = _patched_dispatch
        _sfm.dispatch_model = _patched_dispatch
        try:
            self._transformer = WanTransformer3DModel.from_single_file(
                gguf_path,
                quantization_config=quantization_config,
                torch_dtype=compute_dtype,
            )
        finally:
            _accel_bm.dispatch_model = orig_dispatch
            _sfm.dispatch_model = orig_dispatch

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
