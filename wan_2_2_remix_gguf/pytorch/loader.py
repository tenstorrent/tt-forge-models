#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 Remix GGUF model loader implementation.

Loads GGUF-quantized Wan 2.2 Remix transformers from community
re-uploads (BigDannyPt/Wan-2.2-Remix-GGUF and
huchukato/Wan2.2-Remix-I2V-v2.1-GGUF) and returns the transformer
nn.Module directly for compilation testing.

The Wan 2.2 Remix is a community fine-tune of the Wan 2.2 14B model
supporting both text-to-video (T2V) and image-to-video (I2V) generation.
Each mode has high-noise and low-noise expert variants following the
Mixture-of-Experts (MoE) architecture.

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
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)

BIGDANNYPT_GGUF_REPO = "BigDannyPt/Wan-2.2-Remix-GGUF"
HUCHUKATO_I2V_V2_1_GGUF_REPO = "huchukato/Wan2.2-Remix-I2V-v2.1-GGUF"

# Small spatial dimensions for compile-only testing
TRANSFORMER_NUM_FRAMES = 2
TRANSFORMER_HEIGHT = 4
TRANSFORMER_WIDTH = 4
TRANSFORMER_TEXT_SEQ_LEN = 8
TRANSFORMER_IMAGE_SEQ_LEN = 4


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

_IS_I2V = {
    ModelVariant.WAN22_REMIX_T2V_HIGH_V2_Q4_K_M: False,
    ModelVariant.WAN22_REMIX_I2V_HIGH_V3_Q4_K_M: True,
    ModelVariant.WAN22_REMIX_I2V_HIGH_V2_1_Q4_K_M: True,
    ModelVariant.WAN22_REMIX_I2V_LOW_V2_1_Q4_K_M: True,
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

        Returns the transformer nn.Module directly for compilation testing.
        Uses diffusers GGUFQuantizationConfig to load the quantized weights.
        """
        from diffusers import (
            GGUFQuantizationConfig,
            WanTransformer3DModel,
        )
        from huggingface_hub import hf_hub_download

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_repo = _GGUF_REPOS[self._variant]
        gguf_file = _GGUF_FILES[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        gguf_path = hf_hub_download(repo_id=gguf_repo, filename=gguf_file)

        # Patch dispatch_model to handle meta tensors left by GGUF loading.
        # Some non-quantized params may not appear in the GGUF state dict and
        # remain as meta tensors after load_model_dict_into_meta; dispatch_model
        # then fails when calling model.to(device) on them.
        import accelerate.big_modeling as _accel
        import diffusers.loaders.single_file_model as _sfm

        _orig_dispatch = _accel.dispatch_model

        def _patched_dispatch(model, device_map=None, **kwargs):
            if device_map:
                device = list(device_map.values())[0]
                for module in model.modules():
                    for pname, param in list(module._parameters.items()):
                        if param is not None and param.is_meta:
                            module._parameters[pname] = torch.nn.Parameter(
                                torch.empty(
                                    param.shape,
                                    dtype=compute_dtype,
                                    device=device,
                                ),
                                requires_grad=param.requires_grad,
                            )
                    for bname, buf in list(module._buffers.items()):
                        if buf is not None and buf.is_meta:
                            module._buffers[bname] = torch.empty(
                                buf.shape, dtype=compute_dtype, device=device
                            )
            return _orig_dispatch(model, device_map=device_map, **kwargs)

        _accel.dispatch_model = _patched_dispatch
        _sfm.dispatch_model = _patched_dispatch
        try:
            self._transformer = WanTransformer3DModel.from_single_file(
                gguf_path,
                quantization_config=quantization_config,
                torch_dtype=compute_dtype,
            )
        finally:
            _accel.dispatch_model = _orig_dispatch
            _sfm.dispatch_model = _orig_dispatch

        return self._transformer

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare tensor inputs for the WanTransformer3DModel forward pass."""
        if self._transformer is None:
            self.load_model()

        dtype = torch.bfloat16
        config = self._transformer.config
        is_i2v = _IS_I2V[self._variant]

        inputs = {
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

        if is_i2v and getattr(config, "image_dim", None) is not None:
            inputs["encoder_hidden_states_image"] = torch.randn(
                1,
                TRANSFORMER_IMAGE_SEQ_LEN,
                config.image_dim,
                dtype=dtype,
            )

        return inputs
