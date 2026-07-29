# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BRIA 2.3 (briaai/BRIA-2.3) model loader implementation.

BRIA 2.3 is an SDXL-class text-to-image model. Rather than living as a
variant inside ``stable_diffusion_xl``, it has its own dedicated loader
package so its dependencies, preprocessing tweaks
(``force_zeros_for_empty_prompt = False``) and bringup state can evolve
independently. This mirrors the layout used by ``stable_diffusion_3``.

``load_model`` returns a wrapped SDXL UNet as an ``nn.Module`` that takes
positional tensor inputs only, so the auto-runner
(``DynamicTorchModelTester``) can compile it without any extra glue.
"""
from typing import Optional

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from .src.model_utils import Bria23UNetWrapper, bria_2_3_preprocessing, load_pipe


class ModelVariant(StrEnum):
    """Available BRIA 2.3 model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """BRIA 2.3 model loader."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="briaai/BRIA-2.3",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    prompt = (
        "A portrait of a Beautiful and playful ethereal singer, "
        "golden designs, highly detailed, blurry background"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize the loader for the given BRIA 2.3 variant.

        Args:
            variant: Optional ``ModelVariant`` — defaults to ``BASE``.
        """
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="BRIA 2.3",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override=None):
        """Load (and cache) the underlying SDXL pipeline for BRIA 2.3."""
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.pipeline = load_pipe(pretrained_model_name)
        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)
        return self.pipeline

    def load_model(self, *, dtype_override=None, **kwargs):
        """Return the wrapped BRIA 2.3 UNet ready for the auto-runner.

        Args:
            dtype_override: Optional ``torch.dtype`` to cast the pipeline to.

        Returns:
            torch.nn.Module: ``Bria23UNetWrapper`` around the SDXL UNet.
        """
        if self.pipeline is None:
            self._load_pipeline(dtype_override=dtype_override)
        elif dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        # Run preprocessing once to capture the added_cond_kwargs dict that
        # the SDXL UNet needs at every step. This avoids the wrapper needing
        # to know anything about prompt encoding.
        _, _, _, added_cond_kwargs = bria_2_3_preprocessing(self.pipeline, self.prompt)
        return Bria23UNetWrapper(self.pipeline.unet, added_cond_kwargs)

    def load_inputs(self, dtype_override=None):
        """Return the positional tensor inputs for ``Bria23UNetWrapper``.

        Args:
            dtype_override: Optional ``torch.dtype`` for the returned tensors.

        Returns:
            list[torch.Tensor]: ``[latent_model_input, timesteps, prompt_embeds]``
            — matches the positional signature of ``Bria23UNetWrapper.forward``.
        """
        if self.pipeline is None:
            self._load_pipeline(dtype_override=dtype_override)

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            _added_cond_kwargs,
        ) = bria_2_3_preprocessing(self.pipeline, self.prompt)

        if dtype_override is not None:
            latent_model_input = latent_model_input.to(dtype_override)
            timesteps = timesteps.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return [latent_model_input, timesteps, prompt_embeds]
