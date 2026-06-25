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
    SD3_384P = "sd3_384p"


class ModelLoader(ForgeModel):
    """
    Loader for the Pyramid Flow video-generation denoiser (DiT).

    Two denoiser architectures are exposed as variants:
      * ``miniFLUX_768p`` — the PyramidFluxTransformer DiT
        (``rain1011/pyramid-flow-miniflux``), random weights.
      * ``sd3_384p`` — the PyramidDiffusionMMDiT (SD3-style joint MMDiT)
        denoiser of ``rain1011/pyramid-flow-sd3``, real pretrained weights so
        the on-device PCC comparison is meaningful.

    Only the DiT denoiser (the heavy per-step compute, and the sharding target)
    is exposed here; the surrounding pipeline (text encoders + causal video VAE
    + scheduler) lives in the sibling component loaders / host Python.
    """

    _VARIANTS = {
        ModelVariant.MINIFLUX_768P: PyramidFlowConfig(
            pretrained_model_name="rain1011/pyramid-flow-miniflux",
            source=ModelSource.HUGGING_FACE,
        ),
        ModelVariant.SD3_384P: PyramidFlowConfig(
            pretrained_model_name="rain1011/pyramid-flow-sd3",
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
        if self._variant == ModelVariant.SD3_384P:
            from .src.mmdit_utils import load_transformer as load_mmdit

            return load_mmdit(dtype)
        return load_transformer(dtype)

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._variant == ModelVariant.SD3_384P:
            from .src.mmdit_utils import load_transformer_inputs as load_mmdit_inputs

            return load_mmdit_inputs(dtype)
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
