# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.2 model loader implementation for text-to-image generation.

FLUX.2 is a flow-matching text-to-image model built from three components:
  * transformer (DiT) - ``Flux2Transformer2DModel`` (~32B params)
  * text_encoder      - ``Mistral3ForConditionalGeneration`` (Mistral-Small-3.1 VLM)
  * vae               - ``AutoencoderKLFlux2``

``load_model``/``load_inputs`` target the DiT (the model exercised by the
single-device test runner). Helper methods (:meth:`load_pipeline`,
:meth:`load_vae`, :meth:`load_text_encoder`, :meth:`generate`) expose the other
components so that the full pipeline can be validated end-to-end.
"""
import torch
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
    """Available FLUX.2 model variants."""

    DEV = "Dev"


class ModelLoader(ForgeModel):
    """FLUX.2 model loader implementation for text-to-image generation tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.DEV: ModelConfig(
            pretrained_model_name="black-forest-labs/FLUX.2-dev",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DEV

    # Default guidance scale used by FLUX.2-dev.
    GUIDANCE_SCALE = 4.0

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.pipe = None
        self._transformer = None

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
            model="FLUX.2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_IMAGE_TTT,  # Text-to-image
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    # ------------------------------------------------------------------ #
    # Component loaders
    # ------------------------------------------------------------------ #
    def load_pipeline(self, dtype_override=None):
        """Load (and cache) the full ``Flux2Pipeline`` for this variant.

        Loading the whole pipeline pulls in the text encoder (~48GB) and VAE in
        addition to the DiT, so prefer :meth:`load_model` when only the
        transformer is needed.

        Args:
            dtype_override: Optional torch.dtype to override the model's default
                dtype. Defaults to bfloat16 when not provided.

        Returns:
            Flux2Pipeline: the loaded pipeline instance.
        """
        from diffusers import Flux2Pipeline

        if self.pipe is not None:
            return self.pipe

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        self.pipe = Flux2Pipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            use_safetensors=True,
        )
        return self.pipe

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the FLUX.2 transformer (DiT) instance for this variant.

        The transformer is loaded directly from its subfolder to avoid pulling in
        the (much larger) text encoder when only the DiT is required.

        Args:
            dtype_override: Optional torch.dtype to override the model's default
                dtype. Defaults to bfloat16 when not provided.

        Returns:
            torch.nn.Module: The FLUX.2 transformer (DiT) model instance.
        """
        from diffusers import Flux2Transformer2DModel

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # Reuse the pipeline's transformer if the full pipeline is already loaded.
        if self.pipe is not None:
            self._transformer = self.pipe.transformer.to(dtype)
            return self._transformer

        if self._transformer is None:
            self._transformer = Flux2Transformer2DModel.from_pretrained(
                self._variant_config.pretrained_model_name,
                subfolder="transformer",
                torch_dtype=dtype,
                use_safetensors=True,
            )
        else:
            self._transformer = self._transformer.to(dtype)

        return self._transformer

    def load_vae(self, dtype_override=None):
        """Load and return the FLUX.2 VAE (``AutoencoderKLFlux2``).

        Args:
            dtype_override: Optional torch.dtype. Defaults to bfloat16.

        Returns:
            torch.nn.Module: the VAE instance.
        """
        from diffusers import AutoencoderKLFlux2

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self.pipe is not None:
            return self.pipe.vae.to(dtype)
        return AutoencoderKLFlux2.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder="vae",
            torch_dtype=dtype,
            use_safetensors=True,
        )

    def load_text_encoder(self, dtype_override=None):
        """Load and return the FLUX.2 text encoder (``Mistral3ForConditionalGeneration``).

        Args:
            dtype_override: Optional torch.dtype. Defaults to bfloat16.

        Returns:
            torch.nn.Module: the text encoder instance.
        """
        from transformers import Mistral3ForConditionalGeneration

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self.pipe is not None:
            return self.pipe.text_encoder.to(dtype)
        return Mistral3ForConditionalGeneration.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder="text_encoder",
            torch_dtype=dtype,
            use_safetensors=True,
        )

    # ------------------------------------------------------------------ #
    # Inputs
    # ------------------------------------------------------------------ #
    def load_inputs(
        self,
        dtype_override=None,
        batch_size=1,
        height=256,
        width=256,
        max_sequence_length=512,
        vae_scale_factor=8,
    ):
        """Build sample inputs for the FLUX.2 transformer (DiT).

        The prompt embeddings are synthesised with the correct shape/dtype rather
        than computed from the text encoder, so the DiT can be exercised without
        loading the ~48GB Mistral text encoder. Position ids are generated exactly
        as the ``Flux2Pipeline`` does.

        Args:
            dtype_override: Optional torch.dtype. Defaults to bfloat16.
            batch_size: Batch size (default 1).
            height/width: Target image size in pixels (default 256x256, kept small
                so the sequence length is modest).
            max_sequence_length: Text sequence length (default 512).
            vae_scale_factor: Spatial compression factor of the VAE (default 8).

        Returns:
            dict: keyword inputs for ``Flux2Transformer2DModel.forward``.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        config = self._transformer.config if self._transformer is not None else None
        in_channels = config.in_channels if config is not None else 128
        joint_attention_dim = (
            config.joint_attention_dim if config is not None else 15360
        )
        num_channels_latents = in_channels // 4

        # Latent grid: mirror Flux2Pipeline.prepare_latents.
        h = 2 * (int(height) // (vae_scale_factor * 2))
        w = 2 * (int(width) // (vae_scale_factor * 2))
        latent_h, latent_w = h // 2, w // 2

        # hidden_states: packed latents (B, H*W, in_channels)
        image_seq_len = latent_h * latent_w
        hidden_states = torch.randn(
            batch_size, image_seq_len, in_channels, dtype=dtype
        )

        # encoder_hidden_states: (B, text_seq_len, joint_attention_dim)
        encoder_hidden_states = torch.randn(
            batch_size, max_sequence_length, joint_attention_dim, dtype=dtype
        )

        # img_ids: (B, H*W, 4) via cartesian_prod(t, h, w, l) — matches pipeline.
        img_ids = torch.cartesian_prod(
            torch.arange(1),
            torch.arange(latent_h),
            torch.arange(latent_w),
            torch.arange(1),
        )
        img_ids = img_ids.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

        # txt_ids: (B, text_seq_len, 4) via cartesian_prod(t, h, w, l).
        txt_ids = torch.cartesian_prod(
            torch.arange(1),
            torch.arange(1),
            torch.arange(1),
            torch.arange(max_sequence_length),
        )
        txt_ids = txt_ids.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

        timestep = torch.full((batch_size,), 1.0, dtype=dtype)
        guidance = torch.full((batch_size,), self.GUIDANCE_SCALE, dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "guidance": guidance,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
        }

    # ------------------------------------------------------------------ #
    # End-to-end generation (exercises all three components)
    # ------------------------------------------------------------------ #
    def generate(
        self,
        prompt="A photorealistic astronaut riding a horse in a futuristic city",
        dtype_override=None,
        height=256,
        width=256,
        num_inference_steps=4,
        max_sequence_length=512,
        seed=0,
    ):
        """Run the full FLUX.2 pipeline (text encoder -> DiT -> VAE) and return a PIL image.

        Args:
            prompt: Text prompt.
            dtype_override: Optional torch.dtype. Defaults to bfloat16.
            height/width: Output image size in pixels.
            num_inference_steps: Number of denoising steps.
            max_sequence_length: Text sequence length.
            seed: RNG seed for reproducibility.

        Returns:
            PIL.Image.Image: the generated image.
        """
        pipe = self.load_pipeline(dtype_override=dtype_override)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        result = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=self.GUIDANCE_SCALE,
            max_sequence_length=max_sequence_length,
            generator=generator,
        )
        return result.images[0]
