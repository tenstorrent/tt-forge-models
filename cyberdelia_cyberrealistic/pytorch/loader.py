# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CyberRealistic (cyberdelia/CyberRealistic) model loader implementation.

CyberRealistic is a photorealistic Stable Diffusion 1.5 text-to-image model
distributed as single-file safetensors checkpoints with a baked-in VAE.

Available variants:
- V8: CyberRealistic v8.0 (CyberRealistic_V8_FP32.safetensors)
"""

from typing import Any, Optional

import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download

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

REPO_ID = "cyberdelia/CyberRealistic"
CHECKPOINT_FILE = "CyberRealistic_V8_FP32.safetensors"

PROMPT = (
    "(masterpiece, best quality), ultra-detailed, realistic photo of a "
    "22-year-old woman, natural lighting, depth of field, candid moment, "
    "color graded, RAW photo, soft cinematic bokeh"
)


class ModelVariant(StrEnum):
    """Available CyberRealistic model variants."""

    V8 = "v8.0"


class ModelLoader(ForgeModel):
    """CyberRealistic model loader implementation."""

    _VARIANTS = {
        ModelVariant.V8: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.V8

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="CyberRealistic",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the CyberRealistic pipeline and return the UNet module.

        Returns:
            torch.nn.Module: The UNet model used for denoising.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        model_path = hf_hub_download(repo_id=REPO_ID, filename=CHECKPOINT_FILE)
        self.pipeline = StableDiffusionPipeline.from_single_file(
            model_path,
            torch_dtype=dtype,
        )
        self.pipeline.unet.eval()
        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load preprocessed tensor inputs for the UNet forward pass.

        Returns:
            list: [latent_sample, timestep, encoder_hidden_states]
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        dtype = self.pipeline.unet.dtype

        text_inputs = self.pipeline.tokenizer(
            PROMPT,
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

        if dtype_override is not None:
            latent_sample = latent_sample.to(dtype_override)
            timestep = timestep.to(dtype_override)
            encoder_hidden_states = encoder_hidden_states.to(dtype_override)

        return [latent_sample, timestep, encoder_hidden_states]

    def unpack_forward_output(self, fwd_output: Any) -> torch.Tensor:
        """Unpack UNet output to the sample tensor."""
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        if hasattr(fwd_output, "sample"):
            return fwd_output.sample
        return fwd_output
