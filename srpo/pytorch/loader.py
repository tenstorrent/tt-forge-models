# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SRPO model loader implementation for text-to-image generation.

SRPO (Semantic Relative Preference Optimization, tencent/SRPO) is a full
fine-tune of the FLUX.1-dev diffusion transformer. The HuggingFace repo ships
only the transformer weights (``diffusion_pytorch_model.safetensors``, fp32);
the CLIP/T5 text encoders, tokenizers, VAE and scheduler are reused verbatim
from ``black-forest-labs/FLUX.1-dev``. This loader therefore builds the FLUX
transformer architecture from the FLUX.1-dev config, loads the SRPO weights into
it, and assembles a ``FluxPipeline`` around it for input pre-processing. The
heavy per-step denoiser (the transformer) is the model returned by
``load_model`` and the on-device bringup target.
"""
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel, AutoencoderTiny
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

# The base FLUX pipeline whose text encoders / tokenizers / VAE / scheduler SRPO reuses.
FLUX_BASE_REPO = "black-forest-labs/FLUX.1-dev"


class ModelVariant(StrEnum):
    """Available SRPO model variants."""

    DEV = "Dev"


class ModelLoader(ForgeModel):
    """SRPO model loader implementation for text-to-image generation tasks."""

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
        # SRPO inherits FLUX.1-dev's guided (distilled-guidance) configuration.
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
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_srpo_transformer(self, dtype_override=None):
        """Build the FLUX transformer architecture and load the SRPO fine-tuned weights.

        SRPO publishes only the transformer safetensors (no config.json), so the
        architecture is taken from the FLUX.1-dev transformer config and the SRPO
        state dict is loaded into it.
        """
        config = FluxTransformer2DModel.load_config(
            FLUX_BASE_REPO, subfolder="transformer"
        )
        transformer = FluxTransformer2DModel.from_config(config)

        weights_path = hf_hub_download(
            self._variant_config.pretrained_model_name,
            "diffusion_pytorch_model.safetensors",
        )
        state_dict = load_file(weights_path)
        transformer.load_state_dict(state_dict)

        if dtype_override is not None:
            transformer = transformer.to(dtype_override)
        return transformer

    def _load_pipeline(self, dtype_override=None):
        """Load pipeline for the current variant.

        Builds the SRPO transformer and assembles a FluxPipeline around the
        FLUX.1-dev encoders / VAE / scheduler. Passing ``transformer=`` avoids
        downloading FLUX.1-dev's own transformer weights.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model uses its default dtype (bfloat16).

        Returns:
            The loaded pipeline instance
        """
        transformer = self._load_srpo_transformer(dtype_override=dtype_override)

        pipe_kwargs = {
            "transformer": transformer,
            "use_safetensors": True,
        }
        if dtype_override is not None:
            pipe_kwargs["torch_dtype"] = dtype_override

        # Initialize pipeline (reuses FLUX.1-dev encoders/tokenizers/VAE/scheduler)
        self.pipe = FluxPipeline.from_pretrained(FLUX_BASE_REPO, **pipe_kwargs)

        vae_kwargs = {}
        if dtype_override is not None:
            vae_kwargs["torch_dtype"] = dtype_override

        # Replace VAE with tiny version for efficiency
        self.pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taef1", **vae_kwargs)

        # Enable optimizations
        self.pipe.enable_attention_slicing()
        self.pipe.enable_vae_tiling()

        return self.pipe

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SRPO transformer (denoiser) for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model uses its default dtype (bfloat16).

        Returns:
            torch.nn.Module: The SRPO transformer model for text-to-image generation.
        """
        # Ensure pipeline is loaded
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        # Apply dtype override if specified
        if dtype_override is not None:
            self.pipe.transformer = self.pipe.transformer.to(dtype_override)

        return self.pipe.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the SRPO transformer.

        Text is encoded on CPU through the FLUX.1-dev CLIP and T5 encoders, and
        the latent grid is built exactly as ``FluxPipeline`` would, so the
        returned dict can be fed straight to the transformer.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the transformer model.
        """
        # Ensure pipeline is initialized
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        # Configuration
        max_sequence_length = 512
        prompt = "A cinematic photo of an astronaut riding a horse in a futuristic city"
        do_classifier_free_guidance = self.guidance_scale > 1.0
        height = 1024
        width = 1024
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
