# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion v3 (SD3 Medium) model loader implementation.

Unlike ``stable_diffusion/pytorch/loader.py`` (which targets SD 3.5 and
returns the full ``StableDiffusion3Pipeline``), this loader returns the
MMDiT transformer (an ``nn.Module``) directly from ``load_model``. This
matches the convention used by other loaders in tt_forge_models (e.g. FLUX)
and lets the tt-xla model tester compile the transformer without first
having to dig into the pipeline.
"""
from typing import Optional

import torch

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
    MESH_NAMES,
    MESH_SHAPES,
    SD3TransformerWrapper,
    T5TextEncoderWrapper,
    VAEDecoderWrapper,
    load_pipe,
    load_sd3_vae,
    load_sd3_vae_inputs,
    load_t5_text_encoder,
    load_t5_text_encoder_inputs,
    shard_t5_text_encoder_specs,
    stable_diffusion_preprocessing_v3,
)


class ModelVariant(StrEnum):
    """Available Stable Diffusion v3 model variants."""

    STABLE_DIFFUSION_3_MEDIUM = "3_Medium"
    TEXT_ENCODER = "TextEncoder"
    VAE = "Vae"


class ModelLoader(ForgeModel):
    """Stable Diffusion v3 (SD3 Medium) model loader.

    Two variants are exposed:
      - ``STABLE_DIFFUSION_3_MEDIUM`` (default) → the MMDiT transformer.
      - ``TEXT_ENCODER``                        → the T5-XXL encoder
        (``text_encoder_3``) as an independently compilable TT component.
    """

    _VARIANTS = {
        ModelVariant.STABLE_DIFFUSION_3_MEDIUM: ModelConfig(
            pretrained_model_name="stable-diffusion-3-medium-diffusers",
        ),
        ModelVariant.TEXT_ENCODER: ModelConfig(
            pretrained_model_name="stable-diffusion-3-medium-diffusers",
        ),
        ModelVariant.VAE: ModelConfig(
            pretrained_model_name="stable-diffusion-3-medium-diffusers",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.STABLE_DIFFUSION_3_MEDIUM

    prompt = "An astronaut riding a green horse"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize the loader for the given SD3 variant.

        Args:
            variant: Optional ``ModelVariant`` — defaults to ``STABLE_DIFFUSION_3_MEDIUM``.
        """
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        task = (
            ModelTask.NLP_EMBED_GEN
            if variant == ModelVariant.TEXT_ENCODER
            else ModelTask.CONDITIONAL_GENERATION
        )
        return ModelInfo(
            model="Stable Diffusion 3",
            variant=variant,
            group=ModelGroup.RED,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override=None):
        """Load (and cache) the underlying ``StableDiffusion3Pipeline``.

        The pipeline is needed by :meth:`load_inputs` for prompt encoding, but
        is intentionally not returned from :meth:`load_model`.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.pipeline = load_pipe(pretrained_model_name)
        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)
        return self.pipeline

    def load_model(self, *, dtype_override=None, **kwargs):
        """Return the TT-compilable ``nn.Module`` for the active variant.

        Args:
            dtype_override: Optional ``torch.dtype`` to cast the model to.

        Returns:
            torch.nn.Module: an ``SD3TransformerWrapper`` around the MMDiT
            transformer for the default variant, a ``T5TextEncoderWrapper``
            around the T5-XXL encoder for ``TEXT_ENCODER``, or a
            ``VAEDecoderWrapper`` around the VAE decoder for ``VAE``.
        """
        if self._variant == ModelVariant.TEXT_ENCODER:
            dtype = dtype_override if dtype_override is not None else torch.float32
            return T5TextEncoderWrapper(load_t5_text_encoder(dtype)).eval()

        if self._variant == ModelVariant.VAE:
            dtype = dtype_override if dtype_override is not None else torch.float32
            return VAEDecoderWrapper(load_sd3_vae(dtype)).eval()

        if self.pipeline is None:
            self._load_pipeline(dtype_override=dtype_override)
        elif dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return SD3TransformerWrapper(self.pipeline.transformer).eval()

    def get_mesh_config(self, num_devices: int):
        """Return ``(mesh_shape, mesh_names)`` for a ("batch", "model") 2D mesh.

        Supported device counts: 1, 2, 4, 8, 32.
        """
        if num_devices not in MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count: {num_devices}. "
                f"Expected one of {sorted(MESH_SHAPES)}."
            )
        return MESH_SHAPES[num_devices], MESH_NAMES

    def load_shard_spec(self, model):
        """Return tensor → partition_spec dict for the active component.

        Only ``TEXT_ENCODER`` currently provides shard specs; the model object
        is the ``T5TextEncoderWrapper`` returned by :meth:`load_model`.
        """
        if self._variant == ModelVariant.TEXT_ENCODER:
            return shard_t5_text_encoder_specs(model.encoder)
        return None

    def load_inputs(self, dtype_override=None):
        """Return sample inputs for the SD3 transformer.

        Args:
            dtype_override: Optional ``torch.dtype`` for the returned tensors.

        Returns:
            list[torch.Tensor]: ``[latent_model_input, timestep, prompt_embeds,
            pooled_prompt_embeds]`` — the positional args expected by the
            wrapper around ``SD3Transformer2DModel.forward``. For
            ``TEXT_ENCODER`` returns ``[input_ids]`` for the T5-XXL encoder.
        """
        if self._variant == ModelVariant.TEXT_ENCODER:
            return load_t5_text_encoder_inputs(dtype_override)

        if self._variant == ModelVariant.VAE:
            return load_sd3_vae_inputs(dtype_override)

        if self.pipeline is None:
            self._load_pipeline(dtype_override=dtype_override)

        (
            latent_model_input,
            timestep,
            prompt_embeds,
            pooled_prompt_embeds,
        ) = stable_diffusion_preprocessing_v3(self.pipeline, self.prompt)

        if dtype_override is not None:
            latent_model_input = latent_model_input.to(dtype_override)
            timestep = timestep.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)
            pooled_prompt_embeds = pooled_prompt_embeds.to(dtype_override)

        return [latent_model_input, timestep, prompt_embeds, pooled_prompt_embeds]
