# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SRPO model loader implementation for text-to-image generation.

SRPO (Semantic Relative Preference Optimization, tencent/SRPO, arXiv:2509.06942)
is a FLUX.1-dev fine-tune: the published checkpoint is *only* the denoiser
(FluxTransformer2DModel) weights. The remaining pipeline components -- CLIP and
T5 text encoders, the AutoencoderKL VAE, tokenizers and the
FlowMatchEulerDiscreteScheduler -- are taken unchanged from
black-forest-labs/FLUX.1-dev. We therefore build a standard FluxPipeline but
inject the SRPO transformer, which also avoids downloading FLUX.1-dev's own 24 GB
transformer subfolder (it would just be overwritten).

The published SRPO checkpoint is fp32 (~47.6 GB on disk); the model card
recommends FP32/BF16 loading, so we run in bf16 on device.
"""
import torch
from typing import Optional

from diffusers import FluxPipeline, FluxTransformer2DModel
from safetensors.torch import load_file
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

# Base pipeline (everything except the denoiser) comes from FLUX.1-dev.
_FLUX_BASE = "black-forest-labs/FLUX.1-dev"


class ModelVariant(StrEnum):
    """Available SRPO model variants."""

    DEV = "Dev"


class ModelLoader(ForgeModel):
    """SRPO (FLUX.1-dev fine-tune) loader for text-to-image generation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.DEV: ModelConfig(
            pretrained_model_name="tencent/SRPO",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DEV

    # Native generation settings (from the SRPO model card quick-start).
    NATIVE_HEIGHT = 1024
    NATIVE_WIDTH = 1024
    MAX_SEQUENCE_LENGTH = 512
    GUIDANCE_SCALE = 3.5
    NUM_INFERENCE_STEPS = 50

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.pipe = None
        self.guidance_scale = self.GUIDANCE_SCALE
        # Denoiser input geometry -- defaults to the model's native resolution so
        # the device gate exercises the real workload. The composite generator
        # reuses the same values.
        self.height = self.NATIVE_HEIGHT
        self.width = self.NATIVE_WIDTH
        self.max_sequence_length = self.MAX_SEQUENCE_LENGTH

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant."""
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

    def _load_pipeline(self, dtype_override=None):
        """Load the FluxPipeline with the SRPO transformer injected.

        Args:
            dtype_override: Optional torch.dtype to override the model's default
                            dtype. If not provided, bfloat16 is used.

        Returns:
            The loaded FluxPipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # Build the denoiser from FLUX.1-dev's transformer config (random init,
        # fp32) and overwrite it with the SRPO fine-tuned weights.
        transformer_config = FluxTransformer2DModel.load_config(
            _FLUX_BASE, subfolder="transformer"
        )
        transformer = FluxTransformer2DModel.from_config(transformer_config)

        srpo_weights = hf_hub_download(
            self._variant_config.pretrained_model_name,
            "diffusion_pytorch_model.safetensors",
        )
        state_dict = load_file(srpo_weights)
        transformer.load_state_dict(state_dict)
        del state_dict
        transformer = transformer.to(dtype)

        # Build the rest of the pipeline from FLUX.1-dev with the SRPO transformer
        # injected (passing transformer= skips downloading FLUX's transformer).
        self.pipe = FluxPipeline.from_pretrained(
            _FLUX_BASE,
            transformer=transformer,
            torch_dtype=dtype,
            use_safetensors=True,
        )

        return self.pipe

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SRPO denoiser (FluxTransformer2DModel).

        The transformer is the SRPO-specific component and the on-device
        compute target; the rest of the pipeline is vanilla FLUX.1-dev.

        Args:
            dtype_override: Optional torch.dtype to override the model's default
                            dtype (bfloat16).

        Returns:
            torch.nn.Module: The SRPO transformer (denoiser).
        """
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        if dtype_override is not None:
            self.pipe.transformer = self.pipe.transformer.to(dtype_override)

        return self.pipe.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Build sample denoiser inputs at the configured resolution.

        Mirrors FluxPipeline's latent / embedding preparation so the returned
        dict can be fed directly to the transformer's forward.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' dtype.
            batch_size: Optional batch size (default 1).

        Returns:
            dict: Input tensors for the transformer model.
        """
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        max_sequence_length = self.max_sequence_length
        prompt = "An astronaut riding a horse in a futuristic city"
        do_classifier_free_guidance = self.guidance_scale > 1.0
        height = self.height
        width = self.width
        num_images_per_prompt = 1
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        num_channels_latents = self.pipe.transformer.config.in_channels // 4

        # CLIP (pooled) text embeddings
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

        # T5 sequence text embeddings
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

        # Pack latents the way FluxPipeline does
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

        latent_image_ids = torch.zeros(height_latent // 2, width_latent // 2, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height_latent // 2)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width_latent // 2)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

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
