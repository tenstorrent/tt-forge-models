# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FIBO (briaai/FIBO) model loader implementation.

FIBO is BRIA AI's 8B-parameter DiT-based flow-matching text-to-image model.
It uses SmolLM3-3B as the text encoder, Wan 2.2 as the VAE, and a novel
DimFusion conditioning architecture (paper: arXiv 2511.06876).

Rather than treating FIBO as a variant of some existing loader, this
introduces a dedicated ``fibo`` loader package. Layout mirrors
``stable_diffusion_3``, ``bria_2_3``, and ``hidream_i1`` — its preprocessing
tweaks, license-gated weight fetch (the FIBO repo on Hugging Face is gated
under the ``bria-fibo`` license), and bringup state can evolve independently
of any other model.

``load_model`` returns the FIBO transformer wrapped so it accepts the
positional tensor inputs the auto-runner expects. ``load_inputs`` returns
those positional tensors, captured by driving one short ``pipe(prompt=...)``
call and intercepting the first transformer forward — making the loader
robust to schema drift in upstream diffusers (``BriaFiboPipeline`` only lives
on diffusers git-main today).

Reference: https://huggingface.co/briaai/FIBO
"""

from typing import Optional

import torch

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
from .src.model_utils import (
    BRINGUP_PROMPT,
    FiboTransformerWrapper,
    capture_transformer_inputs,
    load_pipe,
    positional_inputs_from_capture,
)


class ModelVariant(StrEnum):
    """Available FIBO model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """FIBO model loader."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="briaai/FIBO",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    # Stub structured-JSON prompt used during input capture. FIBO is trained
    # on structured captions but the pipeline tokenizes via SmolLM3 either way.
    prompt = BRINGUP_PROMPT

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize the loader for the given FIBO variant.

        Args:
            variant: Optional ``ModelVariant`` — defaults to ``BASE``.
        """
        super().__init__(variant)
        self.pipe = None
        self._capture = None
        # FIBO inherits guidance_scale=5.0 from the model card's Generate example.
        self.guidance_scale = 5.0

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FIBO",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override=None):
        """Load (and cache) the FIBO pipeline."""
        if self.pipe is None:
            self.pipe = load_pipe(
                self._variant_config.pretrained_model_name,
                dtype_override=dtype_override,
            )
        return self.pipe

    def _ensure_capture(self, dtype_override=None):
        """Capture transformer inputs once and cache the result."""
        if self._capture is not None:
            return self._capture
        self._load_pipeline(dtype_override=dtype_override)
        self._capture = capture_transformer_inputs(
            self.pipe,
            prompt=self.prompt,
            guidance_scale=self.guidance_scale,
        )
        return self._capture

    def load_model(self, *, dtype_override=None, **kwargs):
        """Return the wrapped FIBO transformer.

        Args:
            dtype_override: Optional ``torch.dtype`` to cast the pipeline to.

        Returns:
            torch.nn.Module: ``FiboTransformerWrapper`` around the FIBO DiT,
            ready to accept the positional tensors returned by ``load_inputs``.
        """
        self._ensure_capture(dtype_override=dtype_override)
        if dtype_override is not None:
            self.pipe.transformer = self.pipe.transformer.to(dtype_override)
        return FiboTransformerWrapper(self.pipe.transformer, self._capture)

    def load_inputs(self, dtype_override=None, batch_size: int = 1):
        """Return positional tensor inputs for the FIBO transformer.

        Args:
            dtype_override: Optional ``torch.dtype`` for tensor inputs.
                Non-tensor inputs (e.g. ``joint_attention_kwargs``) are passed
                through unchanged.
            batch_size: Ignored for now — the pipeline call always produces
                ``batch_size=1`` * CFG (2 effective). Retained for the
                signature the auto-runner expects.

        Returns:
            tuple: Positional inputs matching ``FiboTransformerWrapper.forward``.
        """
        capture = self._ensure_capture(dtype_override=dtype_override)
        inputs = positional_inputs_from_capture(capture)

        if dtype_override is None:
            return inputs

        cast = []
        for value in inputs:
            if torch.is_tensor(value) and value.is_floating_point():
                cast.append(value.to(dtype_override))
            else:
                cast.append(value)
        return tuple(cast)
