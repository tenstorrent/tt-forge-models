# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX-SRPO GGUF model loader implementation for text-to-image generation.

This loader uses GGUF-quantized variants of the FLUX-SRPO model published
in MonsterMMORPG/Wan_GGUF. FLUX-SRPO is a Stepwise Relative Preference
Optimization fine-tune of FLUX.1-dev. The GGUF transformer is loaded via
diffusers' FluxTransformer2DModel.from_single_file and plugged into a
FluxPipeline built from the original black-forest-labs/FLUX.1-dev
repository.
"""

import importlib.metadata
from typing import Optional

import torch
from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig

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

GGUF_REPO = "MonsterMMORPG/Wan_GGUF"
BASE_REPO = "black-forest-labs/FLUX.1-dev"


class ModelVariant(StrEnum):
    """Available FLUX-SRPO GGUF quantization variants."""

    Q4_K = "Q4_K"
    Q5_K = "Q5_K"
    Q6_K = "Q6_K"
    Q8_0 = "Q8_0"


_GGUF_FILES = {
    ModelVariant.Q4_K: "FLUX-SRPO-GGUF_Q4_K.gguf",
    ModelVariant.Q5_K: "FLUX-SRPO-GGUF_Q5_K.gguf",
    ModelVariant.Q6_K: "FLUX-SRPO-GGUF_Q6_K.gguf",
    ModelVariant.Q8_0: "FLUX-SRPO-GGUF_Q8_0.gguf",
}


class ModelLoader(ForgeModel):
    """FLUX-SRPO GGUF model loader for text-to-image generation."""

    _VARIANTS = {
        variant: ModelConfig(pretrained_model_name=GGUF_REPO) for variant in _GGUF_FILES
    }

    DEFAULT_VARIANT = ModelVariant.Q4_K

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX-SRPO GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _fix_gguf_version_detection():
        """Fix gguf availability flag when gguf is installed at runtime by RequirementsManager.

        diffusers sets _gguf_available at import time. When gguf is installed later,
        the cached flag remains False and GGUF loading fails. Patch it directly.
        """
        import diffusers.utils.import_utils as _import_utils

        if not _import_utils.is_gguf_available():
            try:
                importlib.metadata.version("gguf")
                _import_utils._gguf_available = True
            except importlib.metadata.PackageNotFoundError:
                pass

    def _load_pipeline(self, dtype: torch.dtype = torch.bfloat16):
        """Load the FluxPipeline with a GGUF-quantized transformer."""
        self._fix_gguf_version_detection()
        gguf_file = _GGUF_FILES[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=dtype)

        transformer = FluxTransformer2DModel.from_single_file(
            f"https://huggingface.co/{GGUF_REPO}/blob/main/{gguf_file}",
            config=BASE_REPO,
            quantization_config=quantization_config,
            torch_dtype=dtype,
        )

        self.pipe = FluxPipeline.from_pretrained(
            BASE_REPO,
            transformer=transformer,
            torch_dtype=dtype,
            use_safetensors=True,
        )

        return self.pipe

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GGUF-quantized FLUX-SRPO transformer.

        Returns:
            torch.nn.Module: The FLUX-SRPO transformer model instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self.pipe is None:
            self._load_pipeline(dtype)
        elif dtype_override is not None:
            self.pipe.transformer = self.pipe.transformer.to(dtype=dtype_override)
        return self.pipe.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample inputs for the FLUX-SRPO transformer.

        Returns:
            dict: Input tensors for the transformer model.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.pipe is None:
            self._load_pipeline(dtype)

        max_sequence_length = 256
        prompt = "A cat sitting on a windowsill"
        height = 128
        width = 128
        num_images_per_prompt = 1

        # CLIP text encoding
        text_inputs_clip = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        pooled_prompt_embeds = self.pipe.text_encoder(
            text_inputs_clip.input_ids, output_hidden_states=False
        ).pooler_output
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(
            batch_size, num_images_per_prompt
        )
        pooled_prompt_embeds = pooled_prompt_embeds.view(
            batch_size * num_images_per_prompt, -1
        )

        # T5 text encoding
        text_inputs_t5 = self.pipe.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        prompt_embeds = self.pipe.text_encoder_2(
            text_inputs_t5.input_ids, output_hidden_states=False
        )[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype)
        _, seq_len_t5, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(batch_size, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len_t5, -1
        )

        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(dtype=dtype)

        # Latent dimensions
        height_latent = 2 * (int(height) // (self.pipe.vae_scale_factor * 2))
        width_latent = 2 * (int(width) // (self.pipe.vae_scale_factor * 2))
        seq_len = (height_latent // 2) * (width_latent // 2)

        # FLUX packs 2x2 spatial patches, so the channel dim is
        # num_channels_latents * 4, matching the transformer's in_channels.
        num_channels_latents = self.pipe.transformer.config.in_channels // 4
        latents = torch.randn(
            batch_size * num_images_per_prompt,
            seq_len,
            num_channels_latents * 4,
            dtype=dtype,
        )

        latent_image_ids = torch.zeros(height_latent // 2, width_latent // 2, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height_latent // 2)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width_latent // 2)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        # FLUX.1-dev uses classifier-free guidance distillation
        guidance = torch.tensor([3.5], dtype=dtype).expand(
            batch_size * num_images_per_prompt
        )

        return {
            "hidden_states": latents,
            "timestep": torch.tensor([1.0], dtype=dtype).expand(
                batch_size * num_images_per_prompt
            ),
            "guidance": guidance,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }
