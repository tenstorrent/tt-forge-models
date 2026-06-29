# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SRPO model loader implementation for text-to-image generation.

SRPO (tencent/SRPO) is a full fine-tune of the FLUX.1-dev DiT transformer
("Directly Aligning the Full Diffusion Trajectory with Fine-Grained Human
Preference", arXiv:2509.06942). The HuggingFace repo ships only the denoiser
weights (`diffusion_pytorch_model.safetensors`, fp32, ~47.6 GB) as a drop-in
replacement for the FLUX.1-dev transformer; the text encoders (CLIP + T5-XXL),
VAE, tokenizers and scheduler come from black-forest-labs/FLUX.1-dev.

This loader therefore:
  1. builds the FluxTransformer2DModel architecture **from the FLUX.1-dev
     transformer config** (so the 24 GB FLUX transformer weights are never
     downloaded — they would be overwritten anyway), and
  2. injects the SRPO state dict into it.

The denoiser (this transformer) is the key, compute-dominant component and the
device sharding target. `load_model` returns it; `load_inputs` produces a single
denoising-step's inputs at the model's native 1024x1024 / max_sequence_length=512
geometry. The full pipeline (with the real FLUX VAE) is available on `self.pipe`
for the end-to-end composite generation.
"""
import torch
from typing import Optional

from diffusers import FluxPipeline, FluxTransformer2DModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

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

# Base pipeline (encoders / VAE / tokenizers / scheduler + transformer architecture config).
FLUX_BASE = "black-forest-labs/FLUX.1-dev"


class ModelVariant(StrEnum):
    """Available SRPO model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """SRPO model loader implementation for text-to-image generation tasks."""

    # Dictionary of available model variants.
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="tencent/SRPO",
        ),
    }

    # Default variant to use.
    DEFAULT_VARIANT = ModelVariant.BASE

    # Native generation geometry (from the model card's quick-start).
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

    def _load_pipeline(self, dtype_override=None):
        """Build the SRPO FLUX pipeline (FLUX.1-dev components + SRPO denoiser).

        The transformer is built from the FLUX.1-dev transformer config and then
        has the SRPO weights injected, so the FLUX.1-dev transformer shards are
        never downloaded. Passing the prebuilt transformer to
        ``FluxPipeline.from_pretrained`` skips loading that subfolder.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model uses bfloat16.

        Returns:
            The loaded FluxPipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # 1. Build the FLUX.1-dev transformer architecture (config only — no weights download).
        config = FluxTransformer2DModel.load_config(FLUX_BASE, subfolder="transformer")
        transformer = FluxTransformer2DModel.from_config(config)

        # 2. Inject the SRPO denoiser weights (fp32 on disk).
        srpo_path = hf_hub_download(
            self._variant_config.pretrained_model_name,
            "diffusion_pytorch_model.safetensors",
        )
        state_dict = load_file(srpo_path)
        transformer.load_state_dict(state_dict)
        del state_dict
        transformer = transformer.to(dtype)

        # 3. Assemble the rest of the pipeline from FLUX.1-dev (CLIP, T5, VAE, tokenizers, scheduler).
        self.pipe = FluxPipeline.from_pretrained(
            FLUX_BASE,
            transformer=transformer,
            torch_dtype=dtype,
            use_safetensors=True,
        )

        return self.pipe

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SRPO denoiser (FLUX transformer) for this variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model uses bfloat16.

        Returns:
            torch.nn.Module: The SRPO FLUX transformer (denoiser).
        """
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        if dtype_override is not None:
            self.pipe.transformer = self.pipe.transformer.to(dtype_override)

        return self.pipe.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return one denoising-step's inputs at native 1024x1024 geometry.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors for a single FLUX transformer forward pass.
        """
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        max_sequence_length = self.MAX_SEQUENCE_LENGTH
        prompt = (
            "The Death of Ophelia by John Everett Millais, Pre-Raphaelite painting, "
            "Ophelia floating in a river surrounded by flowers, detailed natural "
            "elements, melancholic and tragic atmosphere"
        )
        height = self.NATIVE_HEIGHT
        width = self.NATIVE_WIDTH
        num_images_per_prompt = 1
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        num_channels_latents = self.pipe.transformer.config.in_channels // 4

        # Text encoding for CLIP (pooled projections).
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
        prompt_embeds = self.pipe.text_encoder_2(
            text_inputs_t5.input_ids, output_hidden_states=False
        )[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype)
        _, seq_len_t5, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(batch_size, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len_t5, -1
        )

        # Text position IDs.
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(dtype=dtype)

        # Pack the latents the way FluxPipeline.prepare_latents does.
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

        # FLUX.1-dev is a guidance-distilled model: guidance is always embedded.
        guidance = torch.full([batch_size], self.GUIDANCE_SCALE, dtype=dtype)

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
