#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 Remix GGUF model loader implementation.

Loads GGUF-quantized Wan 2.2 Remix transformers from community
re-uploads (BigDannyPt/Wan-2.2-Remix-GGUF and
huchukato/Wan2.2-Remix-I2V-v2.1-GGUF) and builds text-to-video or
image-to-video pipelines.

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
from PIL import Image

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
T2V_BASE_PIPELINE = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
I2V_BASE_PIPELINE = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"


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
        self.pipeline = None

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
        """Load the GGUF-quantized Wan 2.2 Remix transformer and build the pipeline.

        Uses diffusers GGUFQuantizationConfig to load the quantized transformer,
        then constructs the appropriate pipeline (T2V or I2V) with the base model's
        VAE in float32 for numerical stability.
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
        from diffusers import (
            AutoencoderKLWan,
            GGUFQuantizationConfig,
            WanTransformer3DModel,
        )
        from huggingface_hub import hf_hub_download

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_repo = _GGUF_REPOS[self._variant]
        gguf_file = _GGUF_FILES[self._variant]
        gguf_path = hf_hub_download(repo_id=gguf_repo, filename=gguf_file)
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

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
            return _orig_dispatch(model, device_map=device_map, **kwargs)

        _orig_dispatch = _accel_bm.dispatch_model
        _accel_bm.dispatch_model = _patched_dispatch
        _sfm.dispatch_model = _patched_dispatch
        try:
            transformer = WanTransformer3DModel.from_single_file(
                gguf_path,
                quantization_config=quantization_config,
                torch_dtype=compute_dtype,
            )
        finally:
            _accel_bm.dispatch_model = _orig_dispatch
            _sfm.dispatch_model = _orig_dispatch

        is_i2v = _IS_I2V[self._variant]
        base_pipeline = I2V_BASE_PIPELINE if is_i2v else T2V_BASE_PIPELINE

        vae = AutoencoderKLWan.from_pretrained(
            base_pipeline,
            subfolder="vae",
            torch_dtype=torch.float32,
        )

        if is_i2v:
            from diffusers import WanImageToVideoPipeline

            self.pipeline = WanImageToVideoPipeline.from_pretrained(
                base_pipeline,
                transformer=transformer,
                vae=vae,
                torch_dtype=compute_dtype,
            )
        else:
            from diffusers import WanPipeline

            self.pipeline = WanPipeline.from_pretrained(
                base_pipeline,
                transformer=transformer,
                vae=vae,
                torch_dtype=compute_dtype,
            )

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for video generation."""
        if prompt is None:
            prompt = (
                "A cat walking gracefully across a sunlit garden, "
                "detailed fur texture, cinematic lighting"
            )

        is_i2v = _IS_I2V[self._variant]

        if is_i2v:
            image = Image.new("RGB", (832, 480), color=(128, 128, 200))
            return {
                "prompt": prompt,
                "image": image,
                "height": 480,
                "width": 832,
                "num_frames": 9,
                "num_inference_steps": 2,
                "guidance_scale": 5.0,
            }

        return {
            "prompt": prompt,
            "height": 480,
            "width": 832,
            "num_frames": 9,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
        }
