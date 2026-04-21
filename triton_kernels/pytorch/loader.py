# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Triton Kernels (kernels-community/triton_kernels) model loader implementation.

The repository hosts a collection of optimized Triton kernels for Mixture-of-
Experts inference (SwiGLU activation, token routing, etc.). This loader wraps
the PyTorch reference implementation of the SwiGLU kernel as a torch.nn.Module
so it can be exercised through the standard ForgeModel interface.
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


REPO_ID = "kernels-community/triton_kernels"


class ModelVariant(StrEnum):
    """Available Triton Kernels variants."""

    SWIGLU = "swiglu"


class ModelLoader(ForgeModel):
    """Triton Kernels loader exposing the SwiGLU activation kernel."""

    _VARIANTS = {
        ModelVariant.SWIGLU: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.SWIGLU

    seq_len = 512
    hidden_size = 1024
    alpha = 0.5
    limit = 1.0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._kernel_lib = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="triton_kernels",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.ATOMIC_ML,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_kernel_lib(self):
        if self._kernel_lib is None:
            from kernels import get_kernel

            self._kernel_lib = get_kernel(self._variant_config.pretrained_model_name)
        return self._kernel_lib

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SwiGLU activation wrapped as a torch.nn.Module."""
        kernel_lib = self._load_kernel_lib()
        swiglu_mod = kernel_lib.swiglu

        class SwiGLU(torch.nn.Module):
            def __init__(self, alpha, limit):
                super().__init__()
                self.alpha = alpha
                self.precision_config = swiglu_mod.PrecisionConfig(limit=limit)

            def forward(self, x):
                return swiglu_mod.swiglu_torch(x, self.alpha, self.precision_config)

        model = SwiGLU(self.alpha, self.limit).eval()
        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return a sample activation tensor for the SwiGLU kernel."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        x = torch.randn(self.seq_len, self.hidden_size, dtype=dtype)
        if batch_size != 1:
            x = x.repeat(batch_size, 1)
        return x
