# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pyramid Flow SD3 (text-to-video) component loader.

`rain1011/pyramid-flow-sd3` is an autoregressive flow-matching video model with
the Stable-Diffusion-3 component layout. Each variant corresponds to one
independently-compilable component of the pipeline:

  - TextEncoder    -> CLIP-L  (CLIPTextModelWithProjection)   params=0.12B
  - TextEncoder2   -> CLIP-G  (CLIPTextModelWithProjection)   params=0.69B
  - TextEncoder3   -> T5-XXL  (T5EncoderModel)                params=4.76B
  - Transformer    -> PyramidDiffusionMMDiT (SD3 MMDiT)       params=0.84B
  - Vae            -> CausalVideoVAE (decoder)                params=0.39B

The MMDiT denoiser is the heavy compute target and must run on device. Its
model code is vendored under `src/mmdit_modules/`; real pretrained weights are
loaded for every component.
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
from .src.utils import (
    REPO_ID,
    load_text_encoder,
    load_text_encoder_2,
    load_text_encoder_3,
    load_text_encoder_inputs,
    load_text_encoder_2_inputs,
    load_text_encoder_3_inputs,
    load_transformer,
    load_transformer_inputs,
)

# Device default precision for this pipeline.
DTYPE = torch.bfloat16


class ModelVariant(StrEnum):
    """Loadable components of the Pyramid Flow SD3 pipeline."""

    TEXT_ENCODER = "TextEncoder"
    TEXT_ENCODER_2 = "TextEncoder2"
    TEXT_ENCODER_3 = "TextEncoder3"
    TRANSFORMER = "Transformer"
    VAE = "Vae"


class ModelLoader(ForgeModel):
    """Load individual Pyramid Flow SD3 components without the full pipeline."""

    _VARIANTS = {
        ModelVariant.TEXT_ENCODER: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.TEXT_ENCODER_2: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.TEXT_ENCODER_3: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.TRANSFORMER: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.VAE: ModelConfig(pretrained_model_name=REPO_ID),
    }
    DEFAULT_VARIANT = ModelVariant.TRANSFORMER

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        task = (
            ModelTask.NLP_EMBED_GEN
            if variant
            in (
                ModelVariant.TEXT_ENCODER,
                ModelVariant.TEXT_ENCODER_2,
                ModelVariant.TEXT_ENCODER_3,
            )
            else ModelTask.MM_VIDEO_TTT
        )
        return ModelInfo(
            model="PyramidFlowSD3",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the component for this variant as a torch.nn.Module."""
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            return load_text_encoder(dtype)
        if self._variant == ModelVariant.TEXT_ENCODER_2:
            return load_text_encoder_2(dtype)
        if self._variant == ModelVariant.TEXT_ENCODER_3:
            return load_text_encoder_3(dtype)
        if self._variant == ModelVariant.TRANSFORMER:
            return load_transformer(dtype)
        if self._variant == ModelVariant.VAE:
            from .src.vae_utils import load_vae

            return load_vae(dtype)

        raise ValueError(f"Unknown variant: {self._variant}")

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return a dict of synthetic inputs for the active component."""
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            return load_text_encoder_inputs(dtype)
        if self._variant == ModelVariant.TEXT_ENCODER_2:
            return load_text_encoder_2_inputs(dtype)
        if self._variant == ModelVariant.TEXT_ENCODER_3:
            return load_text_encoder_3_inputs(dtype)
        if self._variant == ModelVariant.TRANSFORMER:
            return load_transformer_inputs(dtype)
        if self._variant == ModelVariant.VAE:
            from .src.vae_utils import load_vae_inputs

            return load_vae_inputs(dtype)

        raise ValueError(f"Unknown variant: {self._variant}")

    def unpack_forward_output(self, output, dtype_override=None):
        """Normalize component outputs to a single tensor for comparison."""
        if self._variant == ModelVariant.TRANSFORMER:
            # PyramidDiffusionMMDiT returns a list of per-stage tensors.
            if isinstance(output, list):
                return output[0]
        if hasattr(output, "sample"):
            return output.sample
        if hasattr(output, "text_embeds"):
            return output.text_embeds
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        if isinstance(output, (list, tuple)):
            return output[0]
        return output
