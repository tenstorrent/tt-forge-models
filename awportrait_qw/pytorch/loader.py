# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AWPortrait-QW LoRA model loader implementation.

Loads the Qwen/Qwen-Image base diffusion pipeline and applies the
Shakker-Labs/AWPortrait-QW LoRA weights for enhanced portrait generation
with a focus on Chinese facial features and aesthetics.

Available variants:
- AWPORTRAIT_QW: AWPortrait-QW LoRA on Qwen-Image
"""

from typing import Any, Optional

import torch
from diffusers import QwenImagePipeline

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

BASE_MODEL = "Qwen/Qwen-Image"
LORA_REPO = "Shakker-Labs/AWPortrait-QW"
LORA_WEIGHT_NAME = "AWPortrait-QW_1.0.safetensors"


class ModelVariant(StrEnum):
    """Available AWPortrait-QW model variants."""

    AWPORTRAIT_QW = "AWPortrait_QW"


class ModelLoader(ForgeModel):
    """AWPortrait-QW LoRA model loader."""

    _VARIANTS = {
        ModelVariant.AWPORTRAIT_QW: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.AWPORTRAIT_QW

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[QwenImagePipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="AWPORTRAIT_QW",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype):
        self.pipeline = QwenImagePipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )
        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=LORA_WEIGHT_NAME,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        dtype = dtype_override if dtype_override is not None else torch.float32

        if self.pipeline is None:
            self._load_pipeline(dtype)

        if dtype_override is not None:
            self.pipeline.transformer = self.pipeline.transformer.to(dtype_override)

        return self.pipeline.transformer

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs) -> Any:
        if self.pipeline is None:
            self._load_pipeline(
                dtype_override if dtype_override is not None else torch.float32
            )

        dtype = dtype_override if dtype_override is not None else torch.float32
        height = 128
        width = 128
        prompt = "a professional portrait photo of a woman in a studio setting"

        prompt_embeds, prompt_embeds_mask = self.pipeline.encode_prompt(
            prompt=prompt,
            device="cpu",
            num_images_per_prompt=1,
            max_sequence_length=512,
        )

        num_channels_latents = self.pipeline.transformer.config.in_channels // 4
        vae_scale_factor = self.pipeline.vae_scale_factor

        latent_h = height // vae_scale_factor
        latent_w = width // vae_scale_factor
        latent_seq_len = (latent_h // 2) * (latent_w // 2)
        latents = torch.randn(
            batch_size, latent_seq_len, num_channels_latents * 4, dtype=dtype
        )

        img_shapes = [[(1, latent_h // 2, latent_w // 2)]] * batch_size

        timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)

        guidance = None
        if self.pipeline.transformer.config.guidance_embeds:
            guidance = torch.full([batch_size], 3.5, dtype=torch.float32)

        return {
            "hidden_states": latents,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds.to(dtype),
            "encoder_hidden_states_mask": prompt_embeds_mask,
            "img_shapes": img_shapes,
            "guidance": guidance,
            "attention_kwargs": {},
            "return_dict": False,
        }
