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

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the GGUF-quantized UNet and build the SD-Turbo pipeline.

        Uses diffusers GGUFQuantizationConfig to load the quantized UNet,
        then constructs the StableDiffusionPipeline with the base model's
        other components.
        """
        from diffusers import (
            GGUFQuantizationConfig,
            StableDiffusionPipeline,
            UNet2DConditionModel,
        )

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        from huggingface_hub import hf_hub_download

        gguf_file = _GGUF_FILES[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        gguf_path = hf_hub_download(repo_id=GGUF_REPO, filename=gguf_file)
        unet = UNet2DConditionModel.from_single_file(
            gguf_path,
            config=BASE_PIPELINE,
            subfolder="unet",
            use_linear_projection=False,
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
        )

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            BASE_PIPELINE,
            unet=unet,
            torch_dtype=compute_dtype,
            safety_checker=None,
            requires_safety_checker=False,
            ignore_patterns=["unet/*"],
        )

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare tensor inputs for the UNet (latent sample, timestep, text embeddings)."""
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

        if dtype_override:
            latent_sample = latent_sample.to(dtype_override)
            timestep = timestep.to(dtype_override)
            encoder_hidden_states = encoder_hidden_states.to(dtype_override)

        return [latent_sample, timestep, encoder_hidden_states]

    def unpack_forward_output(self, fwd_output):
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        if hasattr(fwd_output, "sample"):
            return fwd_output.sample
        return fwd_output
