# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ideogram 4 component loader.

Ideogram 4 is a 9.3B single-stream DiT text-to-image model. Published weights
are FP8-only (``ideogram-ai/ideogram-4-fp8``). This loader materializes FP8
linear weights to bfloat16 at load time so tt-xla can compile them with TT
block formats (bfp_bf8 / bfp_bf4) via mixed-precision overrides.

Bringup starts with the conditional transformer at 512x512 packed-sequence
shapes matching the CPU inference smoke test.
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
    DTYPE,
    REPO_ID,
    Ideogram4TransformerWrapper,
    build_synthetic_transformer_inputs,
    load_conditional_transformer,
)


class ModelVariant(StrEnum):
    """Loadable Ideogram 4 components."""

    TRANSFORMER_FP8_512 = "Transformer_FP8_512"


class ModelLoader(ForgeModel):
    """Ideogram 4 conditional DiT loader (FP8 checkpoint → bf16 weights)."""

    _VARIANTS = {
        ModelVariant.TRANSFORMER_FP8_512: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.TRANSFORMER_FP8_512

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Ideogram4",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Return the conditional Ideogram4Transformer with FP8 weights → bf16."""
        dtype = dtype_override if dtype_override is not None else DTYPE
        transformer = load_conditional_transformer(dtype=dtype)
        return Ideogram4TransformerWrapper(transformer).eval()

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs):
        """Synthetic packed-sequence inputs for 512x512 resolution."""
        dtype = dtype_override if dtype_override is not None else DTYPE
        return build_synthetic_transformer_inputs(batch_size=batch_size, dtype=dtype)

    def unpack_forward_output(self, output):
        """Return the velocity prediction tensor."""
        if isinstance(output, tuple):
            return output[0]
        return output
