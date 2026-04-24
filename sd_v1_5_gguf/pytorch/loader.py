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

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the GGUF-quantized UNet and build the SD v1.5 pipeline.

        Returns the UNet (a torch.nn.Module) while storing the full pipeline
        in self.pipeline so load_inputs can access tokenizer and text encoder.
        """
        from diffusers import (
            GGUFQuantizationConfig,
            StableDiffusionPipeline,
            UNet2DConditionModel,
        )

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_file = _GGUF_FILES[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        unet = UNet2DConditionModel.from_single_file(
            f"https://huggingface.co/{GGUF_REPO}/blob/main/{gguf_file}",
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
        )

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            BASE_PIPELINE,
            unet=unet,
            torch_dtype=compute_dtype,
        )

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare preprocessed tensor inputs for the UNet.

        Returns:
            list: [latent_sample, timestep, encoder_hidden_states]
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        dtype = self.pipeline.unet.dtype

        prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

        text_inputs = self.pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            encoder_hidden_states = self.pipeline.text_encoder(text_inputs.input_ids)[
                0
            ].to(dtype)

        in_channels = self.pipeline.unet.config.in_channels
        sample_size = self.pipeline.unet.config.sample_size
        latent_sample = torch.randn(
            batch_size, in_channels, sample_size, sample_size, dtype=dtype
        )
        timestep = torch.tensor([1.0], dtype=dtype)

        return [latent_sample, timestep, encoder_hidden_states]

    def unpack_forward_output(self, fwd_output: Any) -> torch.Tensor:
        """Unpack UNet output to the sample tensor."""
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        if hasattr(fwd_output, "sample"):
            return fwd_output.sample
        return fwd_output
