# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.1-dev LoRA model loader implementation.

Loads the lzyvegetable/FLUX.1-schnell ungated mirror base pipeline and applies
style LoRA weights from alexrzem/flux-loras for stylized text-to-image generation.

Available variants:
- FLUX_LORA_PIXAR_3D: Pixar-style 3D rendering LoRA
- FLUX_LORA_COMIC_BOOK: Comic book style LoRA
"""

from typing import Any, Optional

import torch
from diffusers import FluxPipeline

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

BASE_MODEL = "lzyvegetable/FLUX.1-schnell"
LORA_REPO = "alexrzem/flux-loras"

# LoRA weight filenames (under dev/ subdirectory in the repo)
LORA_PIXAR_3D = "dev/Anime_-_Pixar_3D_Animation_-_Flux.1_D_-_leimaxiu252537.safetensors"
LORA_COMIC_BOOK = "dev/Comic_Book_v4_-_Adel_AI.safetensors"


class ModelVariant(StrEnum):
    """Available FLUX LoRA style variants."""

    FLUX_LORA_PIXAR_3D = "FLUX_LoRA_Pixar_3D"
    FLUX_LORA_COMIC_BOOK = "FLUX_LoRA_Comic_Book"


_LORA_FILES = {
    ModelVariant.FLUX_LORA_PIXAR_3D: LORA_PIXAR_3D,
    ModelVariant.FLUX_LORA_COMIC_BOOK: LORA_COMIC_BOOK,
}


class ModelLoader(ForgeModel):
    """FLUX.1-dev LoRA model loader."""

    _VARIANTS = {
        ModelVariant.FLUX_LORA_PIXAR_3D: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.FLUX_LORA_COMIC_BOOK: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.FLUX_LORA_PIXAR_3D

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[FluxPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX_LORAS",
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
        """Load the FLUX.1-dev pipeline with style LoRA weights applied.

        Returns:
            FluxPipeline with LoRA weights merged.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = FluxPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

        lora_file = _LORA_FILES[self._variant]
        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=lora_file,
        )

        return self.pipeline.transformer

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs) -> Any:
        """Prepare inputs for the FLUX transformer model.

        Returns:
            dict with tensor inputs for the transformer.
        """
        assert self.pipeline is not None, "load_model must be called before load_inputs"

        prompt = "An astronaut riding a green horse"
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        max_sequence_length = 256
        height = 128
        width = 128
        num_images_per_prompt = 1
        num_channels_latents = self.pipeline.transformer.config.in_channels // 4

        # CLIP text encoding
        text_inputs_clip = self.pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipeline.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        pooled_prompt_embeds = self.pipeline.text_encoder(
            text_inputs_clip.input_ids, output_hidden_states=False
        ).pooler_output.to(dtype=dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(
            batch_size, num_images_per_prompt
        ).view(batch_size * num_images_per_prompt, -1)

        # T5 text encoding
        text_inputs_t5 = self.pipeline.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_embeds = self.pipeline.text_encoder_2(
            text_inputs_t5.input_ids, output_hidden_states=False
        )[0].to(dtype=dtype)
        _, seq_len_t5, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(batch_size, num_images_per_prompt, 1).view(
            batch_size * num_images_per_prompt, seq_len_t5, -1
        )

        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(dtype=dtype)

        # Latents
        height_latent = 2 * (int(height) // (self.pipeline.vae_scale_factor * 2))
        width_latent = 2 * (int(width) // (self.pipeline.vae_scale_factor * 2))
        shape = (
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height_latent,
            width_latent,
        )
        latents = torch.randn(shape, dtype=dtype)
        latents = latents.view(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height_latent // 2,
            2,
            width_latent // 2,
            2,
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5).reshape(
            batch_size * num_images_per_prompt,
            (height_latent // 2) * (width_latent // 2),
            num_channels_latents * 4,
        )

        latent_image_ids = torch.zeros(height_latent // 2, width_latent // 2, 3)
        latent_image_ids[..., 1] += torch.arange(height_latent // 2)[:, None]
        latent_image_ids[..., 2] += torch.arange(width_latent // 2)[None, :]
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        return {
            "hidden_states": latents,
            "timestep": torch.tensor([1.0], dtype=dtype),
            "guidance": None,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }
