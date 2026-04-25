#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
calcuis/wan2-gguf model loader implementation.

Loads GGUF-quantized Wan 2.2 diffusion transformers from the
calcuis/wan2-gguf repository. The repo packages Wan 2.2 Animate and
Wan 2.2 I2V (high-noise and low-noise experts) variants for ComfyUI use.

Available variants:
- WAN22_ANIMATE_Q4_K_M: Wan 2.2 Animate 14B, Q4_K_M quantization
- WAN22_I2V_HIGH_NOISE_Q4_K_M: Wan 2.2 I2V 14B high-noise expert, Q4_K_M
- WAN22_I2V_LOW_NOISE_Q4_K_M: Wan 2.2 I2V 14B low-noise expert, Q4_K_M
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

GGUF_REPO = "calcuis/wan2-gguf"


def _from_single_file_safe(model_cls, url, torch_dtype=None, **kwargs):
    """Load a diffusers model from a GGUF single file, materializing any meta tensors left over from
    shape mismatches between the GGUF checkpoint and the inferred model config before dispatch."""
    import diffusers.loaders.single_file_model as _sfm

    _orig_dispatch = _sfm.dispatch_model
    fallback_dtype = torch_dtype or torch.float32

    def _dispatch_materializing_meta(model, **dispatch_kwargs):
        device_map = dispatch_kwargs.get("device_map", {})
        target_device = next(iter(device_map.values())) if device_map else "cpu"
        for name, param in list(model.named_parameters()):
            if param.is_meta:
                parts = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(
                    parent,
                    parts[-1],
                    torch.nn.Parameter(
                        torch.empty(
                            param.shape, dtype=fallback_dtype, device=target_device
                        ),
                        requires_grad=param.requires_grad,
                    ),
                )
        for name, buf in list(model.named_buffers()):
            if buf is not None and buf.is_meta:
                parts = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                parent.register_buffer(
                    parts[-1],
                    torch.empty(buf.shape, dtype=fallback_dtype, device=target_device),
                )
        return _orig_dispatch(model, **dispatch_kwargs)

    _sfm.dispatch_model = _dispatch_materializing_meta
    try:
        return model_cls.from_single_file(url, torch_dtype=torch_dtype, **kwargs)
    finally:
        _sfm.dispatch_model = _orig_dispatch


# Small spatial dimensions for compile-only testing
TRANSFORMER_NUM_FRAMES = 2
TRANSFORMER_HEIGHT = 4
TRANSFORMER_WIDTH = 4
TRANSFORMER_TEXT_SEQ_LEN = 8


class ModelVariant(StrEnum):
    """Available calcuis/wan2-gguf variants."""

    WAN22_ANIMATE_Q4_K_M = "2.2_Animate_Q4_K_M"
    WAN22_I2V_HIGH_NOISE_Q4_K_M = "2.2_I2V_HighNoise_Q4_K_M"
    WAN22_I2V_LOW_NOISE_Q4_K_M = "2.2_I2V_LowNoise_Q4_K_M"


_GGUF_FILES = {
    ModelVariant.WAN22_ANIMATE_Q4_K_M: "wan2.2-animate-14b-q4_k_m.gguf",
    ModelVariant.WAN22_I2V_HIGH_NOISE_Q4_K_M: "wan2.2-i2v-14b-high-noise-q4_k_m.gguf",
    ModelVariant.WAN22_I2V_LOW_NOISE_Q4_K_M: "wan2.2-i2v-14b-low-noise-q4_k_m.gguf",
}


class ModelLoader(ForgeModel):
    """calcuis/wan2-gguf model loader."""

    _VARIANTS = {
        ModelVariant.WAN22_ANIMATE_Q4_K_M: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
        ModelVariant.WAN22_I2V_HIGH_NOISE_Q4_K_M: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
        ModelVariant.WAN22_I2V_LOW_NOISE_Q4_K_M: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_ANIMATE_Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="CALCUIS_WAN2_GGUF",
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
        """Load the GGUF-quantized Wan 2.2 transformer.

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

        self._transformer = _from_single_file_safe(
            WanTransformer3DModel,
            f"https://huggingface.co/{GGUF_REPO}/{gguf_file}",
            torch_dtype=compute_dtype,
            quantization_config=quantization_config,
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
