# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion v1.5 GGUF model loader implementation.

Loads a GGUF-quantized UNet from gpustack/stable-diffusion-v1-5-GGUF and
builds a text-to-image pipeline using the base sd-legacy/stable-diffusion-v1-5
model for the remaining components (text encoder, tokenizer, VAE, scheduler).

Available variants:
- FP16: Full precision (~2.13 GB)
- Q4_0: 4-bit quantization (~1.75 GB)
- Q4_1: 4-bit quantization (~1.76 GB)
- Q8_0: 8-bit quantization (~1.88 GB)
"""

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

GGUF_REPO = "gpustack/stable-diffusion-v1-5-GGUF"
BASE_PIPELINE = "sd-legacy/stable-diffusion-v1-5"


class ModelVariant(StrEnum):
    """Available Stable Diffusion v1.5 GGUF variants."""

    FP16 = "FP16"
    Q4_0 = "Q4_0"
    Q4_1 = "Q4_1"
    Q8_0 = "Q8_0"


_GGUF_FILES = {
    ModelVariant.FP16: "stable-diffusion-v1-5-FP16.gguf",
    ModelVariant.Q4_0: "stable-diffusion-v1-5-Q4_0.gguf",
    ModelVariant.Q4_1: "stable-diffusion-v1-5-Q4_1.gguf",
    ModelVariant.Q8_0: "stable-diffusion-v1-5-Q8_0.gguf",
}


class ModelLoader(ForgeModel):
    """Stable Diffusion v1.5 GGUF model loader."""

    _VARIANTS = {
        ModelVariant.FP16: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
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
            model="SD_V1_5_GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _dequantize_non_linear_gguf_params(module, compute_dtype):
        """Dequantize GGUFParameter weights in non-linear modules (e.g., GroupNorm).

        Diffusers places raw GGUFParameter (quantized bytes) into ALL module
        parameters, but only GGUFLinear knows how to dequantize on the fly.
        Non-linear layers like GroupNorm need plain float tensors.
        """
        from diffusers.quantizers.gguf.utils import (
            GGUFLinear,
            GGUFParameter,
            dequantize_gguf_tensor,
        )

        for child in module.modules():
            if isinstance(child, GGUFLinear):
                continue
            for name, param in list(child.named_parameters(recurse=False)):
                if isinstance(param, GGUFParameter):
                    dequantized = dequantize_gguf_tensor(param).to(compute_dtype)
                    child._parameters[name] = torch.nn.Parameter(dequantized)

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the GGUF-quantized UNet and build the SD v1.5 pipeline."""
        from diffusers import (
            GGUFQuantizationConfig,
            StableDiffusionPipeline,
            UNet2DConditionModel,
        )
        from huggingface_hub import hf_hub_download

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_file = _GGUF_FILES[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        gguf_path = hf_hub_download(repo_id=GGUF_REPO, filename=gguf_file)
        unet = UNet2DConditionModel.from_single_file(
            gguf_path,
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
        )

        self._dequantize_non_linear_gguf_params(unet, compute_dtype)

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            BASE_PIPELINE,
            unet=unet,
            torch_dtype=compute_dtype,
        )

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs):
        """Load and return sample inputs for the SD v1.5 UNet.

        Returns:
            dict: Keyword arguments for the UNet forward method.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        in_channels = 4
        height, width = 64, 64
        cross_attention_dim = 768

        sample = torch.randn((batch_size, in_channels, height, width), dtype=dtype)
        timestep = torch.randint(0, 1000, (1,))
        encoder_hidden_states = torch.randn(
            (batch_size, 77, cross_attention_dim), dtype=dtype
        )

        return {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }
