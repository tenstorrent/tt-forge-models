# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek V3.2 model loader implementation.

Uses a locally modified Transformer (model.py) instead of the original
HuggingFace model. The modifications are:
  1. Uses scipy.linalg.hadamard instead of fast_hadamard_transform (no CUDA required).
  2. Stubs out FP8 quantization (act_quant, fp8_gemm, fp8_index) that rely on
     custom tilelang kernels unsupported on TT hardware.
  3. Avoids torch.view_as_complex / view_as_real operations.
"""
from typing import Optional

import torch

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
)
from .modified_model import ModelArgs, Transformer


class ModelLoader(ForgeModel):
    """DeepSeek V3.2 model loader using the locally modified Transformer."""

    def __init__(self, variant=None, n_layers: Optional[int] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional variant string. Unused; kept for API compatibility.
            n_layers: Number of transformer layers to instantiate.
                      Defaults to 1 for a lightweight test model.
        """
        super().__init__(variant)
        self.n_layers = n_layers if n_layers is not None else 1

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Return model metadata for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string.

        Returns:
            ModelInfo: Information about the model and variant.
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="DeepSeek-V3.2",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the modified DeepSeek V3.2 Transformer.

        The model is constructed from ModelArgs defaults, overriding n_layers
        with the value passed at construction time.

        Args:
            dtype_override: Optional torch.dtype to cast the model to after
                            construction (e.g. torch.bfloat16).

        Returns:
            torch.nn.Module: The modified DeepSeek V3.2 Transformer in eval mode.
        """
        args = ModelArgs(n_layers=self.n_layers, **kwargs)
        model = Transformer(args)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model = model.eval()
        self._args = args
        return model

    def load_inputs(self, batch_size: int = 1, seq_len: int = 32):
        """Return sample token inputs for the model.

        Args:
            batch_size: Number of sequences in the batch.
            seq_len: Length of each input sequence.

        Returns:
            torch.Tensor: Integer token tensor of shape (batch_size, seq_len).
        """
        if not hasattr(self, "_args"):
            self.load_model()

        tokens = torch.randint(0, self._args.vocab_size, (batch_size, seq_len))
        return tokens
