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

from typing import Optional

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
        """Load and return the CyberRealistic UNet from a single-file checkpoint.

        Returns:
            UNet2DConditionModel: The UNet submodule of the loaded pipeline.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        model_path = hf_hub_download(repo_id=REPO_ID, filename=CHECKPOINT_FILE)
        self.pipeline = StableDiffusionPipeline.from_single_file(
            model_path,
            torch_dtype=dtype,
            **kwargs,
        )
        self.in_channels = self.pipeline.unet.in_channels
        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the UNet model.

        Returns:
            dict: Dictionary containing sample, timestep, and encoder_hidden_states.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        prompt = [
            "(masterpiece, best quality), ultra-detailed, realistic photo of a "
            "22-year-old woman, natural lighting, depth of field, candid moment, "
            "color graded, RAW photo, soft cinematic bokeh"
        ] * batch_size

        text_input = self.pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.pipeline.text_encoder(text_input.input_ids)[0]

        height, width = 512, 512
        latents = torch.randn(
            (batch_size, self.in_channels, height // 8, width // 8), dtype=dtype
        )
        self.pipeline.scheduler.set_timesteps(1)
        latents = latents * self.pipeline.scheduler.init_noise_sigma
        latent_model_input = self.pipeline.scheduler.scale_model_input(latents, 0)

        return {
            "sample": latent_model_input.to(dtype),
            "timestep": torch.tensor(0),
            "encoder_hidden_states": text_embeddings.to(dtype),
        }
