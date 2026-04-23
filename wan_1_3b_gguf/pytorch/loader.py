#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.1 T2V 1.3B GGUF model loader implementation.

Loads GGUF-quantized Wan 2.1 Text-to-Video 1.3B transformers from
calcuis/wan-1.3b-gguf and builds a WanPipeline using the VAE and text
encoder from the base Wan-AI/Wan2.1-T2V-1.3B-Diffusers repo.

Available variants:
- WAN21_T2V_1_3B_Q4_0: Q4_0 quantization (~917 MB)
- WAN21_T2V_1_3B_F16: F16 (~2.89 GB)
"""

from typing import Any, Optional

import torch
from diffusers import (
    AutoencoderKLWan,
    GGUFQuantizationConfig,
    WanPipeline,
    WanTransformer3DModel,
)
from huggingface_hub import hf_hub_download

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

GGUF_REPO_ID = "calcuis/wan-1.3b-gguf"
BASE_PIPELINE = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


class ModelVariant(StrEnum):
    """Available Wan 2.1 T2V 1.3B GGUF variants."""

    WAN21_T2V_1_3B_Q4_0 = "2.1_T2V_1.3B_Q4_0"
    WAN21_T2V_1_3B_F16 = "2.1_T2V_1.3B_F16"


_GGUF_FILES = {
    ModelVariant.WAN21_T2V_1_3B_Q4_0: "wan2.1_t2v_1.3b-q4_0.gguf",
    ModelVariant.WAN21_T2V_1_3B_F16: "wan2.1_t2v_1.3b-f16.gguf",
}


class ModelLoader(ForgeModel):
    """Wan 2.1 T2V 1.3B GGUF model loader."""

    _VARIANTS = {
        ModelVariant.WAN21_T2V_1_3B_Q4_0: ModelConfig(
            pretrained_model_name=GGUF_REPO_ID,
        ),
        ModelVariant.WAN21_T2V_1_3B_F16: ModelConfig(
            pretrained_model_name=GGUF_REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN21_T2V_1_3B_Q4_0

    DEFAULT_PROMPT = (
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_1_3B_GGUF",
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
        """Load the GGUF-quantized Wan 2.1 T2V 1.3B transformer and build the pipeline.

        Uses diffusers GGUFQuantizationConfig to load the quantized transformer,
        then constructs the full WanPipeline with the base model's VAE in
        float32 for numerical stability.
        """
        import diffusers.utils.import_utils as _diffusers_import_utils

        if not _diffusers_import_utils._gguf_available:
            import importlib
            import importlib.metadata
            import importlib.util

            if importlib.util.find_spec("gguf") is not None:
                _diffusers_import_utils._gguf_available = True
                _diffusers_import_utils._gguf_version = importlib.metadata.version(
                    "gguf"
                )
                import diffusers.quantizers.gguf.gguf_quantizer as _gguf_quantizer

                importlib.reload(_gguf_quantizer)

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_filename = _GGUF_FILES[self._variant]
        gguf_path = hf_hub_download(
            repo_id=GGUF_REPO_ID,
            filename=gguf_filename,
        )

        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)
        transformer = WanTransformer3DModel.from_single_file(
            gguf_path,
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
        )

        vae = AutoencoderKLWan.from_pretrained(
            BASE_PIPELINE,
            subfolder="vae",
            torch_dtype=torch.float32,
        )

        self.pipeline = WanPipeline.from_pretrained(
            BASE_PIPELINE,
            transformer=transformer,
            vae=vae,
            torch_dtype=compute_dtype,
        )

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for text-to-video generation."""
        prompt_value = prompt if prompt is not None else self.DEFAULT_PROMPT
        return {
            "prompt": prompt_value,
            "height": 480,
            "width": 832,
            "num_frames": 9,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
        }
