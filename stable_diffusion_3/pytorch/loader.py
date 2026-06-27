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
    load_pipe,
    shard_transformer_specs,
    stable_diffusion_preprocessing_v3,
)


class ModelVariant(StrEnum):
    """Available Stable Diffusion v3 model variants."""

    STABLE_DIFFUSION_3_MEDIUM = "3_Medium"


class ModelLoader(ForgeModel):
    """Stable Diffusion v3 (SD3 Medium) model loader."""

    _VARIANTS = {
        ModelVariant.STABLE_DIFFUSION_3_MEDIUM: ModelConfig(
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
        return ModelInfo(
            model="Stable Diffusion 3",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override=None):
        """Load (and cache) the underlying ``StableDiffusion3Pipeline``.

        The pipeline is needed by :meth:`load_inputs` for prompt encoding, but
        is intentionally not returned from :meth:`load_model`.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        # Load directly at the requested dtype. Loading fp32 first and casting
        # afterwards materializes the full ~30 GB fp32 pipeline and OOMs a 32 GB
        # host (the T5-XXL text encoder alone is ~19 GB fp32 / ~9.5 GB bf16).
        load_dtype = dtype_override if dtype_override is not None else None
        if load_dtype is not None:
            self.pipeline = load_pipe(pretrained_model_name, dtype=load_dtype)
        else:
            self.pipeline = load_pipe(pretrained_model_name)
        return self.pipeline

    def load_model(self, *, dtype_override=None, **kwargs):
        """Return the SD3 MMDiT transformer (the TT-compilable ``nn.Module``).

        Args:
            dtype_override: Optional ``torch.dtype`` to cast the model to.

        Returns:
            torch.nn.Module: The SD3 ``SD3Transformer2DModel`` instance.
        """
        if self.pipeline is None:
            self._load_pipeline(dtype_override=dtype_override)
        elif dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline.transformer

    def load_inputs(self, dtype_override=None):
        """Return sample inputs for the SD3 transformer.

        Args:
            dtype_override: Optional ``torch.dtype`` for the returned tensors.

        Returns:
            list[torch.Tensor]: ``[latent_model_input, timestep, prompt_embeds,
            pooled_prompt_embeds]`` — the positional args expected by the
            wrapper around ``SD3Transformer2DModel.forward``.
        """
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

    def get_mesh_config(self, num_devices: int):
        """Return ``((batch, model) mesh shape, mesh names)`` for the MMDiT.

        Tensor-parallel baseline for the denoiser: the model axis carries the
        Megatron column→row sharding (24 attention heads divide 2/4/8). The
        batch axis stays data-parallel / replicated.
        """
        if num_devices not in MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count: {num_devices}. "
                f"Expected one of {sorted(MESH_SHAPES)}."
            )
        return MESH_SHAPES[num_devices], MESH_NAMES

    def load_shard_spec(self, model):
        """Return ``{param_tensor: partition_spec}`` for the MMDiT denoiser.

        ``model`` is the ``SD3Transformer2DModel`` (optionally wrapped); the
        wrapper exposes the underlying transformer at ``.model``.
        """
        transformer = getattr(model, "model", model)
        return shard_transformer_specs(transformer)
