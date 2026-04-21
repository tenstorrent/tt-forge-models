# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
kernels-community/triton_kernels loader exposing the SwiGLU reference
implementation as an nn.Module.

The HF repo ships a set of Triton kernels for fast MoE (SwiGLU, routing,
matmul, etc.) along with PyTorch reference implementations (``*_torch``).
The kernel module is loaded at runtime via the ``kernels`` library using
``get_kernel("kernels-community/triton_kernels")``.
"""
from typing import Optional

import torch
import torch.nn as nn

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


class ModelVariant(StrEnum):
    """Available triton_kernels variants."""

    SWIGLU = "swiglu"


class _SwiGLUReference(nn.Module):
    """Wraps ``triton_kernels.swiglu.swiglu_torch`` as a traceable nn.Module."""

    def __init__(self, swiglu_module, alpha: float, limit: float):
        super().__init__()
        self._swiglu_module = swiglu_module
        self._precision_config = swiglu_module.PrecisionConfig(limit=limit)
        self.alpha = alpha

    def forward(self, x):
        return self._swiglu_module.swiglu_torch(x, self.alpha, self._precision_config)


class ModelLoader(ForgeModel):
    """Loader for the kernels-community/triton_kernels SwiGLU reference implementation."""

    _VARIANTS = {
        ModelVariant.SWIGLU: ModelConfig(
            pretrained_model_name="kernels-community/triton_kernels",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SWIGLU

    # Shape and scalar parameters mirror the README quickstart example.
    BATCH = 512
    HIDDEN = 1024
    ALPHA = 0.5
    LIMIT = 1.0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="triton_kernels",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.ATOMIC_ML,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Return the SwiGLU reference implementation wrapped as an nn.Module."""
        from kernels import get_kernel

        triton_kernels = get_kernel(self._variant_config.pretrained_model_name)

        model = _SwiGLUReference(
            triton_kernels.swiglu, alpha=self.ALPHA, limit=self.LIMIT
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=None):
        """Return a random activation tensor sized for the SwiGLU example."""
        dtype = dtype_override or torch.bfloat16
        batch = batch_size or self.BATCH
        return torch.randn(batch, self.HIDDEN, dtype=dtype)
