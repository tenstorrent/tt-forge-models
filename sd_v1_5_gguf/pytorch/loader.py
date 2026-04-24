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

# SD v1.5 UNet input dimensions
LATENT_CHANNELS = 4
LATENT_HEIGHT = 64
LATENT_WIDTH = 64
CROSS_ATTENTION_DIM = 768
MAX_SEQ_LEN = 77


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
        self._unet = None

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

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load and return the GGUF-quantized SD v1.5 UNet.

        Returns:
            UNet2DConditionModel instance loaded from GGUF checkpoint.
        """
        from diffusers import GGUFQuantizationConfig, UNet2DConditionModel
        from huggingface_hub import hf_hub_download

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self._unet is None:
            gguf_file = _GGUF_FILES[self._variant]
            gguf_path = hf_hub_download(repo_id=GGUF_REPO, filename=gguf_file)
            quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)
            self._unet = UNet2DConditionModel.from_single_file(
                gguf_path,
                quantization_config=quantization_config,
                torch_dtype=compute_dtype,
            )
            self._unet.eval()
        elif dtype_override is not None:
            self._unet = self._unet.to(dtype=dtype_override)

        return self._unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Return synthetic UNet inputs for SD v1.5.

        Returns:
            dict: Input tensors matching UNet2DConditionModel.forward():
                - sample: Noisy latents [batch, 4, 64, 64]
                - timestep: Diffusion timestep tensor
                - encoder_hidden_states: CLIP text embeddings [batch, 77, 768]
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        return {
            "sample": torch.randn(
                batch_size, LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH, dtype=dtype
            ),
            "timestep": torch.tensor([1.0], dtype=dtype),
            "encoder_hidden_states": torch.randn(
                batch_size, MAX_SEQ_LEN, CROSS_ATTENTION_DIM, dtype=dtype
            ),
        }
