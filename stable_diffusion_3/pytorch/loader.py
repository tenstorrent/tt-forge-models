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
import gc
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
from .src.model_utils import load_pipe, stable_diffusion_preprocessing_v3


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
        # Thread the device dtype straight into from_pretrained so the weights
        # materialize directly in (typically) bf16. Loading the full pipeline in
        # fp32 (T5-XXL alone is ~19 GB) then casting OOM-kills the ~32 GB bringup
        # host before the cast ever runs.
        if dtype_override is not None:
            self.pipeline = load_pipe(
                pretrained_model_name, torch_dtype=dtype_override
            )
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

        Returns a dict keyed by ``SD3Transformer2DModel.forward``'s parameter
        names (the FLUX loader convention). The tt-xla model tester feeds a list
        positionally as ``model(*inputs)``; SD3's forward signature is
        ``(hidden_states, encoder_hidden_states, pooled_projections, timestep)``,
        so a positional ``[latent, timestep, prompt_embeds, pooled]`` list
        misaligns ``timestep`` onto ``encoder_hidden_states`` and fails with
        "Timesteps should be a 1d-array". A dict binds each tensor by name.

        Args:
            dtype_override: Optional ``torch.dtype`` for the returned tensors.

        Returns:
            dict[str, torch.Tensor]: keyed by ``hidden_states``,
            ``encoder_hidden_states``, ``pooled_projections``, ``timestep``.
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

        # Prompt encoding is done, so free the heavy text encoders + VAE. They
        # are not needed for the transformer (denoiser) forward, and keeping
        # T5-XXL (~9.4 GB bf16) resident OOM-kills the ~22 GB bringup host during
        # the CPU golden forward that the model tester runs before the device
        # compile. The transformer returned by ``load_model`` is referenced
        # independently, so dropping the other components here is safe.
        if self.pipeline is not None:
            self.pipeline.text_encoder = None
            self.pipeline.text_encoder_2 = None
            self.pipeline.text_encoder_3 = None
            self.pipeline.vae = None
            gc.collect()

        return {
            "hidden_states": latent_model_input,
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_prompt_embeds,
            "timestep": timestep,
        }
