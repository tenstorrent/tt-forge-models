# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pyramid Flow model loader for tt_forge_models.

Pyramid Flow is an autoregressive video generation model based on flow matching.
Repository: https://github.com/jy0205/Pyramid-Flow
Model: https://huggingface.co/rain1011/pyramid-flow-miniflux

Pyramid Flow has no diffusers integration, so the model code is vendored in
`src/flux_modules/` (verbatim from upstream, with the `trainer_misc` sequence-
parallel imports replaced by a local stub). This loader exposes the
PyramidFluxTransformer DiT for compilation / op-coverage error analysis.
"""

from dataclasses import dataclass
from typing import Any, Optional

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
from .src.utils import load_transformer, load_transformer_inputs


@dataclass
class PyramidFlowConfig(ModelConfig):
    """Configuration for Pyramid Flow variants."""

    source: ModelSource


class ModelVariant(StrEnum):
    """Available Pyramid Flow variants."""

    MINIFLUX_768P = "miniFLUX_768p"


class ModelLoader(ForgeModel):
    """
    Loader for Pyramid Flow miniFLUX DiT (768p).

    Loads only the PyramidFluxTransformer with random weights. The full
    pipeline (text encoder + VAE + scheduler) lives in upstream
    `pyramid_dit.PyramidDiTForVideoGeneration` and is CUDA-only; the DiT
    component is the relevant target for tt-xla compilation tests.
    """

    _VARIANTS = {
        ModelVariant.MINIFLUX_768P: PyramidFlowConfig(
            pretrained_model_name="rain1011/pyramid-flow-miniflux",
            source=ModelSource.HUGGING_FACE,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINIFLUX_768P

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="PyramidFlow",
            variant=variant,
            group=ModelGroup.PRIORITY,
            task=ModelTask.MM_VIDEO_TTT,
            source=cls._VARIANTS[variant].source,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.float32
        return load_transformer(dtype)

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.float32
        return load_transformer_inputs(dtype)

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        # PyramidFluxTransformer returns a list of per-stage tensors; expose
        # the first stage so the comparison harness sees a single tensor.
        if isinstance(output, list):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        if isinstance(output, tuple):
            return output[0]
        return output
