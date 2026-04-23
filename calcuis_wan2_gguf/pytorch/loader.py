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
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

GGUF_REPO = "calcuis/wan2-gguf"

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


def _make_meta_param_initializer(original_dispatch):
    """Return a patched dispatch_model that zero-initializes any remaining meta parameters.

    The calcuis/wan2-gguf GGUF files omit certain I2V-specific weights (image
    cross-attention projections, image embedder) that the diffusers
    WanTransformer3DModel creates as meta tensors.  accelerate's dispatch_model
    then calls model.to(device) which raises NotImplementedError on meta tensors.
    This wrapper materialises those meta parameters as zeros before dispatching.
    """
    import torch.nn as nn

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
                    torch.zeros(param.shape, dtype=param.dtype, device="cpu"),
                    requires_grad=param.requires_grad,
                )
        return original_dispatch(model, device_map=device_map, **kwargs)

    return _patched_dispatch


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

        The calcuis GGUF files detect as I2V (36-channel patch embedding) but
        omit I2V-specific cross-attention weights.  A patched dispatch_model
        zero-initialises those missing parameters so the model can be loaded
        and compiled without errors.
        """
        import diffusers.utils.import_utils as _diffusers_import_utils

        if not _diffusers_import_utils._gguf_available:
            import importlib.util

            if importlib.util.find_spec("gguf") is not None:
                _diffusers_import_utils._gguf_available = True

        import accelerate.big_modeling as _accel_bm
        import diffusers.loaders.single_file_model as _sfm
        from diffusers import GGUFQuantizationConfig, WanTransformer3DModel
        from huggingface_hub import hf_hub_download

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_file = _GGUF_FILES[self._variant]
        gguf_path = hf_hub_download(repo_id=GGUF_REPO, filename=gguf_file)

        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        orig_dispatch = _accel_bm.dispatch_model
        patched = _make_meta_param_initializer(orig_dispatch)
        _accel_bm.dispatch_model = patched
        _sfm.dispatch_model = patched
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
