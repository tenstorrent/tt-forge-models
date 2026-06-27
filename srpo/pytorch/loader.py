# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SRPO model loader implementation for text-to-image generation.

SRPO (Tencent, arXiv:2509.06942) fine-tunes the FLUX.1-dev MMDiT transformer
("denoiser") with Direct-Align + Semantic Relative Preference Optimization.
The released checkpoint (``tencent/SRPO/diffusion_pytorch_model.safetensors``)
contains *only* the transformer weights; the text encoders (CLIP + T5-XXL),
tokenizers, VAE and scheduler are reused unchanged from FLUX.1-dev.

This loader therefore builds a FLUX.1-dev pipeline whose transformer weights are
overwritten with the SRPO checkpoint, and exposes the transformer (the per-step
denoiser — the heavy compute that must run on device) via ``load_model``.
"""
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from typing import Optional

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


class ModelVariant(StrEnum):
    """Available SRPO model variants."""

    DEV = "Dev"


class ModelLoader(ForgeModel):
    """SRPO model loader implementation for text-to-image generation tasks."""

    # The frozen FLUX.1-dev components (encoders, VAE, tokenizers, scheduler).
    _FLUX_BASE = "black-forest-labs/FLUX.1-dev"
    # The SRPO transformer (denoiser) checkpoint.
    _SRPO_REPO = "tencent/SRPO"
    _SRPO_WEIGHTS = "diffusion_pytorch_model.safetensors"

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.DEV: ModelConfig(
            pretrained_model_name="tencent/SRPO",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DEV

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.pipe = None
        # FLUX.1-dev (and SRPO, which fine-tunes it) is a guidance-distilled
        # model; the SRPO model card uses guidance_scale=3.5.
        self.guidance_scale = 3.5

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="SRPO",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_IMAGE_TTT,  # FIXME: Update task to Text to Image
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override=None):
        """Load the FLUX.1-dev pipeline with the SRPO transformer weights.

        To avoid downloading the FLUX.1-dev transformer (~24 GB) only to throw it
        away, the SRPO transformer is built from the FLUX.1-dev transformer config
        and its weights are loaded directly from the SRPO checkpoint, then injected
        into the pipeline so diffusers skips the base transformer download.

        Args:
            dtype_override: Optional torch.dtype to override the model's default
                            dtype. If not provided, bfloat16 is used (the device
                            default, and the model card's recommended precision).

        Returns:
            The loaded pipeline instance
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # Build the SRPO transformer from the FLUX.1-dev transformer *config only*
        # (no weight download), then load the SRPO checkpoint into it. This skips
        # the ~24 GB FLUX.1-dev transformer download, since SRPO replaces it.
        config = FluxTransformer2DModel.load_config(
            self._FLUX_BASE, subfolder="transformer"
        )
        transformer = FluxTransformer2DModel.from_config(config)
        srpo_path = hf_hub_download(self._SRPO_REPO, self._SRPO_WEIGHTS)
        state_dict = load_file(srpo_path)
        transformer.load_state_dict(state_dict)
        transformer = transformer.to(dtype)

        # Assemble the pipeline from the frozen FLUX.1-dev components, injecting
        # the SRPO transformer (so the base transformer is never re-downloaded).
        self.pipe = FluxPipeline.from_pretrained(
            self._FLUX_BASE,
            transformer=transformer,
            torch_dtype=dtype,
            use_safetensors=True,
        )

        return self.pipe

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SRPO transformer (denoiser) for this variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default
                            dtype (bfloat16).

        Returns:
            torch.nn.Module: The SRPO/FLUX transformer model instance (denoiser).
        """
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        if dtype_override is not None:
            self.pipe.transformer = self.pipe.transformer.to(dtype_override)

        return self.pipe.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Build native-resolution denoiser inputs for the SRPO transformer.

        Inputs are sized for SRPO's native 1024x1024 generation (see the model
        card): 4096 packed latent tokens and a 512-token T5 context, matching the
        composite pipeline's denoising step.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' dtype.
            batch_size: Optional batch size (default 1).

        Returns:
            dict: Input tensors that can be fed to the transformer model.
        """
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        # SRPO native generation settings (from the model card).
        max_sequence_length = 512
        prompt = "An astronaut riding a horse in a futuristic city"
        do_classifier_free_guidance = self.guidance_scale > 1.0
        height = 1024
        width = 1024
        num_images_per_prompt = 1
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        num_channels_latents = self.pipe.transformer.config.in_channels // 4

        # Text encoding for CLIP (pooled projection).
        text_inputs_clip = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_clip = text_inputs_clip.input_ids
        pooled_prompt_embeds = self.pipe.text_encoder(
            text_input_ids_clip, output_hidden_states=False
        ).pooler_output
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(
            batch_size, num_images_per_prompt
        )
        pooled_prompt_embeds = pooled_prompt_embeds.view(
            batch_size * num_images_per_prompt, -1
        )

        # Text encoding for T5.
        text_inputs_t5 = self.pipe.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids_t5 = text_inputs_t5.input_ids
        prompt_embeds = self.pipe.text_encoder_2(
            text_input_ids_t5, output_hidden_states=False
        )[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype)
        _, seq_len_t5, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(batch_size, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len_t5, -1
        )

        # Text position IDs.
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(dtype=dtype)

        # Pack latents into FLUX's [B, (H/2)*(W/2), C*4] token layout.
        height_latent = 2 * (int(height) // (self.pipe.vae_scale_factor * 2))
        width_latent = 2 * (int(width) // (self.pipe.vae_scale_factor * 2))

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
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size * num_images_per_prompt,
            (height_latent // 2) * (width_latent // 2),
            num_channels_latents * 4,
        )

        # Latent position IDs.
        latent_image_ids = torch.zeros(height_latent // 2, width_latent // 2, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height_latent // 2)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width_latent // 2)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        # Guidance embedding (FLUX.1-dev / SRPO are guidance-distilled).
        if do_classifier_free_guidance:
            guidance = torch.full([batch_size], self.guidance_scale, dtype=dtype)
        else:
            guidance = None

        inputs = {
            "hidden_states": latents,
            "timestep": torch.tensor([1.0], dtype=dtype),
            "guidance": guidance,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }

        return inputs
