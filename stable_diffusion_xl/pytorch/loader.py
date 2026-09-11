# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion XL model loader implementation
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
from .src.model_utils import (
    CLIP_G_SUBFOLDER,
    CLIP_L_SUBFOLDER,
    CLIPGTextEncoderWrapper,
    CLIPTextEncoderWrapper,
    load_clip_text_encoder,
    load_clip_text_encoder_inputs,
    load_pipe,
    stable_diffusion_preprocessing_xl,
)


class ModelVariant(StrEnum):
    """Available Stable Diffusion XL model variants."""

    STABLE_DIFFUSION_XL_BASE_1_0 = "Base_1.0"
    CLIP_TEXT_ENCODER = "ClipTextEncoder"
    CLIP_G_TEXT_ENCODER = "ClipGTextEncoder"


# Subfolder in the SDXL base repo per text-encoder variant. Membership in this
# dict is what marks a variant as a text-encoder component.
CLIP_SUBFOLDERS = {
    ModelVariant.CLIP_TEXT_ENCODER: CLIP_L_SUBFOLDER,
    ModelVariant.CLIP_G_TEXT_ENCODER: CLIP_G_SUBFOLDER,
}

# Wrapper per text-encoder variant. CLIP-L returns one consumed tensor, CLIP-G
# two (see the comment block in ``src/model_utils.py``).
CLIP_WRAPPERS = {
    ModelVariant.CLIP_TEXT_ENCODER: CLIPTextEncoderWrapper,
    ModelVariant.CLIP_G_TEXT_ENCODER: CLIPGTextEncoderWrapper,
}


class ModelLoader(ForgeModel):
    """Stable Diffusion XL model loader implementation.

    Three variants are exposed:
      - ``STABLE_DIFFUSION_XL_BASE_1_0`` (default) -> the UNet, driven with
        preprocessed conditioning from the full pipeline.
      - ``CLIP_TEXT_ENCODER``                      -> the CLIP-L tower
        (``text_encoder``) as an independently compilable TT component.
      - ``CLIP_G_TEXT_ENCODER``                    -> the CLIP-G tower
        (``text_encoder_2``), whose projection head also produces SDXL's
        ``add_text_embeds``.

    With both towers exposed, every text-conditioning input the UNet consumes
    has a TT component; the VAE decoder already runs on device via
    :meth:`decode_vae`.
    """

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.STABLE_DIFFUSION_XL_BASE_1_0: ModelConfig(
            pretrained_model_name="stable-diffusion-xl-base-1.0",
        ),
        ModelVariant.CLIP_TEXT_ENCODER: ModelConfig(
            pretrained_model_name="stable-diffusion-xl-base-1.0",
        ),
        ModelVariant.CLIP_G_TEXT_ENCODER: ModelConfig(
            pretrained_model_name="stable-diffusion-xl-base-1.0",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.STABLE_DIFFUSION_XL_BASE_1_0

    # Shared configuration parameters
    prompt = "An astronaut riding a green horse"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.pipeline = None

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
        task = (
            ModelTask.NLP_EMBED_GEN
            if variant in CLIP_SUBFOLDERS
            else ModelTask.CONDITIONAL_GENERATION
        )
        return ModelInfo(
            model="Stable Diffusion XL",
            variant=variant,
            group=ModelGroup.RED,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Stable Diffusion XL pipeline for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            DiffusionPipeline: the pipeline instance for the default variant, or
            a ``CLIPTextEncoderWrapper`` / ``CLIPGTextEncoderWrapper`` around one
            CLIP tower for the text-encoder variants.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self._variant in CLIP_SUBFOLDERS:
            dtype = dtype_override if dtype_override is not None else torch.float32
            encoder = load_clip_text_encoder(
                pretrained_model_name, dtype, CLIP_SUBFOLDERS[self._variant]
            )
            return CLIP_WRAPPERS[self._variant](encoder).eval()

        # Load the pipeline
        self.pipeline = load_pipe(pretrained_model_name)

        # Apply dtype conversion if specified
        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Stable Diffusion XL model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            List : Input tensors that can be fed to the model:
                - latent_model_input (torch.Tensor): Latent input for the UNet
                - timestep (torch.Tensor): Timestep tensor
                - prompt_embeds (torch.Tensor): Encoded prompt embeddings
                - added_cond_kwargs (dict): Additional conditioning inputs (e.g., text/image embeddings,
                  time IDs, or other auxiliary information required by the pipeline).

            For the text-encoder variants: ``[input_ids]`` of shape ``(1, 77)``
            int64, the tokenized prompt at the CLIP context length.
        """
        if self._variant in CLIP_SUBFOLDERS:
            return load_clip_text_encoder_inputs(
                self._variant_config.pretrained_model_name,
                self.prompt,
                CLIP_SUBFOLDERS[self._variant],
            )

        # Ensure pipeline is initialized
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        # Generate preprocessed inputs
        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            timestep_cond,
            added_cond_kwargs,
            add_time_ids,
        ) = stable_diffusion_preprocessing_xl(self.pipeline, self.prompt)

        # Apply dtype conversion if specified
        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timesteps = timesteps.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return [latent_model_input, timesteps, prompt_embeds, added_cond_kwargs]

    def decode_vae(self, latents, on_tt=False):
        """Decode SDXL VAE latents -> image tensor in [-1, 1].

        With ``on_tt=True`` the AutoencoderKL decoder runs on TT via
        ``torch.compile(backend="tt")`` with ``optimization_level=1`` (the
        composite ttnn.group_norm lowering the VAE group norms need). The full
        SDXL AutoencoderKL decodes correctly on device (verified vs the CPU
        golden), unlike the SD1.5/SD3 AutoencoderKL. ``load_model`` must run first.
        """
        if self.pipeline is None:
            self.load_model()
        vae = self.pipeline.vae
        with torch.no_grad():
            if not on_tt:
                return vae.decode(latents).sample

            import torch_xla
            import torch_xla.core.xla_model as xm

            torch_xla.set_custom_compile_options({"optimization_level": 1})
            dev = xm.xla_device()
            vae_dev = vae.to(dtype=torch.bfloat16).to(dev)
            compiled = torch.compile(lambda z: vae_dev.decode(z).sample, backend="tt")
            out = compiled(latents.to(dtype=torch.bfloat16).to(dev))
            return out.to("cpu").to(torch.float32)
