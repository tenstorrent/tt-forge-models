# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SRPO (tencent/SRPO) model loader implementation.

SRPO is a FLUX.1-dev fine-tune from Tencent Hunyuan that publishes only the
transformer weights (``diffusion_pytorch_model.safetensors``). Rather than
adding it as a variant of the existing ``flux`` loader, this introduces a
dedicated loader package so its preprocessing tweaks, license-gated weights,
and bringup state can evolve independently. This mirrors the layout used by
``stable_diffusion_3`` and ``bria_2_3``.

``load_model`` returns the FLUX transformer (with SRPO weights overlaid) as
an ``nn.Module``. ``load_inputs`` returns the positional tensors the FLUX
transformer consumes — the same shape contract as ``flux/pytorch/loader.py``.

Reference: https://huggingface.co/tencent/SRPO
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
from .src.model_utils import load_pipe, srpo_preprocessing


class ModelVariant(StrEnum):
    """Available SRPO model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """SRPO model loader."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="tencent/SRPO",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    # Prompt taken from the SRPO Hugging Face model card.
    prompt = (
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize the loader for the given SRPO variant.

        Args:
            variant: Optional ``ModelVariant`` — defaults to ``BASE``.
        """
        super().__init__(variant)
        self.pipe = None
        # SRPO inherits FLUX.1-dev's guidance scale (3.5 per the model card).
        self.guidance_scale = 3.5

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SRPO",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override=None):
        """Load (and cache) the SRPO pipeline (FLUX.1-dev base + SRPO weights)."""
        self.pipe = load_pipe(
            self._variant_config.pretrained_model_name,
            dtype_override=dtype_override,
        )
        return self.pipe

    def load_model(self, *, dtype_override=None, **kwargs):
        """Return the SRPO transformer (FLUX.1-dev architecture, SRPO weights).

        Args:
            dtype_override: Optional ``torch.dtype`` to cast the pipeline to.

        Returns:
            torch.nn.Module: The FLUX transformer with SRPO weights overlaid.
        """
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)
        elif dtype_override is not None:
            self.pipe.transformer = self.pipe.transformer.to(dtype_override)

        return self.pipe.transformer

    def load_inputs(self, dtype_override=None, batch_size: int = 1):
        """Return positional inputs for the FLUX transformer (SRPO weights).

        Args:
            dtype_override: Optional ``torch.dtype`` for the returned tensors.
            batch_size: Batch size for the synthetic input. Defaults to 1.

        Returns:
            dict: Input tensors that can be fed directly to the transformer
            (matches the keyword-argument signature of FLUX's
            ``FluxTransformer2DModel.forward``).
        """
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        (
            hidden_states,
            timestep,
            guidance,
            pooled_projections,
            encoder_hidden_states,
            txt_ids,
            img_ids,
        ) = srpo_preprocessing(
            self.pipe,
            self.prompt,
            dtype=dtype,
            batch_size=batch_size,
            guidance_scale=self.guidance_scale,
        )

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "guidance": guidance,
            "pooled_projections": pooled_projections,
            "encoder_hidden_states": encoder_hidden_states,
            "txt_ids": txt_ids,
            "img_ids": img_ids,
            "joint_attention_kwargs": {},
        }
