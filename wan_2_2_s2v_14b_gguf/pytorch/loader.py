#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
QuantStack Wan 2.2 Sound-to-Video 14B GGUF model loader implementation.

Loads the GGUF-quantized Wan 2.2 S2V 14B denoising transformer from
QuantStack/Wan2.2-S2V-14B-GGUF. This model generates video from audio input.

Base model: Wan-AI/Wan2.2-S2V-14B
Format: GGUF (quantized for ComfyUI inference via ComfyUI-GGUF)

Available variants:
- WAN22_S2V_14B_Q4_0: Q4_0 quantization (12.8 GB)
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

GGUF_REPO = "QuantStack/Wan2.2-S2V-14B-GGUF"

# Small spatial dimensions for compile-only testing
TRANSFORMER_NUM_FRAMES = 2
TRANSFORMER_HEIGHT = 4
TRANSFORMER_WIDTH = 4
TRANSFORMER_TEXT_SEQ_LEN = 8

_GGUF_FILES = {
    "Q4_0": "Wan2.2-S2V-14B-Q4_0.gguf",
}


class ModelVariant(StrEnum):
    """Available Wan 2.2 S2V 14B GGUF model variants."""

    WAN22_S2V_14B_Q4_0 = "S2V_14B_Q4_0"


class ModelLoader(ForgeModel):
    """Wan 2.2 Sound-to-Video 14B GGUF model loader."""

    _VARIANTS = {
        ModelVariant.WAN22_S2V_14B_Q4_0: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_S2V_14B_Q4_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_2_2_S2V_14B_GGUF",
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
        """Load the GGUF-quantized Wan 2.2 S2V 14B transformer.

        Uses diffusers GGUFQuantizationConfig to load the quantized transformer,
        then dequantizes all GGUF layers to plain nn.Linear with float weights
        so that TT-XLA's __torch_function__ override does not encounter the
        byte-packed GGUF weight tensors during compilation.
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

        gguf_file = _GGUF_FILES["Q4_0"]
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        from huggingface_hub import hf_hub_download

        gguf_path = hf_hub_download(
            repo_id=GGUF_REPO,
            filename=gguf_file,
        )

        self._transformer = WanTransformer3DModel.from_single_file(
            gguf_path,
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
        )

        # GGUFLinear stores weights as packed byte tensors and relies on
        # GGUFParameter.__torch_function__ to dequantize on-the-fly. The TT-XLA
        # __torch_function__ override intercepts F.linear before GGUF can
        # dequantize, causing a dtype mismatch. Convert all GGUFLinear modules
        # back to plain nn.Linear with float weights before compilation.
        from diffusers.quantizers.gguf.utils import _dequantize_gguf_and_restore_linear

        _dequantize_gguf_and_restore_linear(self._transformer)
        # Remove quantizer markers so diffusers' .to() guard doesn't block dtype cast.
        self._transformer.hf_quantizer = None
        self._transformer.is_quantized = False
        self._transformer.to(compute_dtype)

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
