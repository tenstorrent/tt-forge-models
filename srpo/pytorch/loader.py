# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SRPO model loader implementation for text-to-image generation.

SRPO (Semantic Relative Preference Optimization, Tencent Hunyuan) is a
fine-tune of the FLUX.1-dev diffusion transformer. The HuggingFace repo
``tencent/SRPO`` ships *only* the transformer (denoiser) weights
(``diffusion_pytorch_model.safetensors``); the VAE, the CLIP and T5 text
encoders, the tokenizers and the scheduler are reused unchanged from
``black-forest-labs/FLUX.1-dev``. This loader therefore builds a FLUX.1-dev
pipeline and swaps the SRPO transformer state dict into ``pipe.transformer``,
exposing that transformer as the model under bringup (it is the
compute-dominant denoiser run inside the diffusion loop).
"""
import torch
from diffusers import FluxPipeline, AutoencoderTiny
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

    BASE = "base"


class ModelLoader(ForgeModel):
    """SRPO model loader implementation for text-to-image generation tasks."""

    # Base FLUX pipeline that supplies the VAE, text encoders, tokenizers and
    # scheduler. SRPO only re-trains the transformer.
    _BASE_PIPELINE = "black-forest-labs/FLUX.1-dev"

    # SRPO transformer (denoiser) weights, distributed as a single safetensors.
    _SRPO_REPO = "tencent/SRPO"
    _SRPO_WEIGHTS = "diffusion_pytorch_model.safetensors"

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="tencent/SRPO",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.pipe = None
        # SRPO is a FLUX.1-dev fine-tune -> dev-style guidance.
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
            task=ModelTask.MM_IMAGE_TTT,  # text-to-image
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override=None):
        """Load the FLUX.1-dev pipeline and swap in the SRPO transformer weights.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically bfloat16).

        Returns:
            The loaded pipeline instance
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        pipe_kwargs = {
            "use_safetensors": True,
            "torch_dtype": dtype,
        }

        # Initialize the base FLUX pipeline (VAE / text encoders / scheduler).
        self.pipe = FluxPipeline.from_pretrained(self._BASE_PIPELINE, **pipe_kwargs)

        # Swap in the SRPO fine-tuned transformer weights. The repo ships fp32
        # weights; load them and let load_state_dict cast into the pipeline's
        # transformer dtype.
        weights_path = hf_hub_download(
            repo_id=self._SRPO_REPO, filename=self._SRPO_WEIGHTS
        )
        state_dict = load_file(weights_path)
        self.pipe.transformer.load_state_dict(state_dict)
        del state_dict
        self.pipe.transformer = self.pipe.transformer.to(dtype)

        # Replace VAE with tiny version for efficiency (decoder only; the
        # denoiser is the model under test).
        vae_kwargs = {"torch_dtype": dtype}
        self.pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taef1", **vae_kwargs)

        # Enable optimizations
        self.pipe.enable_attention_slicing()
        self.pipe.enable_vae_tiling()

        return self.pipe

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SRPO transformer (denoiser) for this variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically bfloat16).

        Returns:
            torch.nn.Module: The SRPO/FLUX transformer model instance for text-to-image generation.
        """
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        if dtype_override is not None:
            self.pipe.transformer = self.pipe.transformer.to(dtype_override)

        return self.pipe.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the SRPO transformer.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the transformer model.
        """
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        # Configuration
        max_sequence_length = 256
        prompt = "An astronaut riding a horse in a futuristic city"
        do_classifier_free_guidance = self.guidance_scale > 1.0
        height = 128
        width = 128
        num_images_per_prompt = 1
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        num_channels_latents = self.pipe.transformer.config.in_channels // 4

        # Text encoding for CLIP
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

        # Text encoding for T5
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

        # Create text IDs
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(dtype=dtype)

        # Create latents
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

        # Prepare latent image IDs
        latent_image_ids = torch.zeros(height_latent // 2, width_latent // 2, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height_latent // 2)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width_latent // 2)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        # Prepare guidance
        if do_classifier_free_guidance:
            guidance = torch.full([batch_size], self.guidance_scale, dtype=dtype)
        else:
            guidance = None

        # Prepare inputs
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
