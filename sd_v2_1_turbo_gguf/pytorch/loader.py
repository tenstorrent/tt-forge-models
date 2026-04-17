# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion v2.1 Turbo GGUF model loader implementation.

Loads GGUF-quantized UNet from gpustack/stable-diffusion-v2-1-turbo-GGUF
and builds a text-to-image pipeline using the base stabilityai/sd-turbo model.

SD-Turbo is a distilled version of Stable Diffusion 2.1, trained using
Adversarial Diffusion Distillation (ADD) for high-quality single-step
image generation at 512x512 resolution.

Available variants:
- Q4_0: 4-bit quantization (~2.19 GB)
- Q4_1: 4-bit quantization (~2.2 GB)
- Q8_0: 8-bit quantization (~2.32 GB)
"""

import os
from typing import Optional

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

GGUF_REPO = "gpustack/stable-diffusion-v2-1-turbo-GGUF"
BASE_PIPELINE = "stabilityai/sd-turbo"


class ModelVariant(StrEnum):
    """Available Stable Diffusion v2.1 Turbo GGUF variants."""

    Q4_0 = "Q4_0"
    Q4_1 = "Q4_1"
    Q8_0 = "Q8_0"


_GGUF_FILES = {
    ModelVariant.Q4_0: "stable-diffusion-v2-1-turbo-Q4_0.gguf",
    ModelVariant.Q4_1: "stable-diffusion-v2-1-turbo-Q4_1.gguf",
    ModelVariant.Q8_0: "stable-diffusion-v2-1-turbo-Q8_0.gguf",
}


class ModelLoader(ForgeModel):
    """Stable Diffusion v2.1 Turbo GGUF model loader."""

    _VARIANTS = {
        ModelVariant.Q4_0: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
        ModelVariant.Q4_1: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
        ModelVariant.Q8_0: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.Q8_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SD_V2_1_TURBO_GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _ensure_gguf_available(self):
        import importlib
        import importlib.metadata
        import sys

        import diffusers.utils.import_utils as _diu

        if not _diu._gguf_available:
            _diu._gguf_available = True
            _diu._gguf_version = importlib.metadata.version("gguf")
            mod = "diffusers.quantizers.gguf.gguf_quantizer"
            if mod in sys.modules:
                importlib.reload(sys.modules[mod])

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        from diffusers import UNet2DConditionModel

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if os.environ.get("TT_RANDOM_WEIGHTS") == "1":
            config = UNet2DConditionModel.load_config(BASE_PIPELINE, subfolder="unet")
            unet = UNet2DConditionModel.from_config(config).to(compute_dtype)
        else:
            self._ensure_gguf_available()
            from diffusers import GGUFQuantizationConfig
            from huggingface_hub import hf_hub_download

            gguf_file = _GGUF_FILES[self._variant]
            quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)
            gguf_path = hf_hub_download(repo_id=GGUF_REPO, filename=gguf_file)
            unet = UNet2DConditionModel.from_single_file(
                gguf_path,
                quantization_config=quantization_config,
                torch_dtype=compute_dtype,
            )

        self._unet = unet
        return unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        unet = self._unet
        in_channels = unet.config.in_channels
        cross_attention_dim = unet.config.cross_attention_dim

        latents = torch.randn(batch_size, in_channels, 64, 64, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, 77, cross_attention_dim, dtype=dtype
        )

        return {
            "sample": latents,
            "timestep": 0,
            "encoder_hidden_states": encoder_hidden_states,
        }
