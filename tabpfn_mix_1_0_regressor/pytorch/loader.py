# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TabPFNMix regressor model loader implementation for tabular regression.

TabPFNMix is a 12-layer encoder-decoder Transformer foundation model (~37M
parameters) pre-trained on synthetic datasets sampled from a mix of random
regressors. Like TabPFN it performs in-context learning: training samples and
test samples are passed through the transformer together in a single forward
pass.
"""
from typing import Optional

import torch
import torch.nn as nn

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available TabPFNMix regressor model variants."""

    TABPFN_MIX_1_0_REGRESSOR = "autogluon/tabpfn-mix-1.0-regressor"


class TabPFNMixWrapper(nn.Module):
    """Wrapper around the TabPFNMix transformer for XLA-compatible tracing.

    TabPFNMix performs in-context learning: a batch of training samples
    (x_src, y_src) is consumed alongside the test features (x_tgt) in a single
    forward pass. This wrapper bakes in the training context so the forward
    method accepts only test features.
    """

    def __init__(self, model, x_src, y_src):
        super().__init__()
        self.model = model
        self.register_buffer("x_src", x_src, persistent=False)
        self.register_buffer("y_src", y_src, persistent=False)

    def forward(self, x_tgt):
        """Run TabPFNMix regression on test features.

        Args:
            x_tgt: Test features tensor (n_test_samples, n_features).

        Returns:
            Tensor: Regression predictions for each test sample.
        """
        return self.model(self.x_src, self.y_src, x_tgt)


class ModelLoader(ForgeModel):
    """TabPFNMix regressor model loader for tabular regression."""

    _VARIANTS = {
        ModelVariant.TABPFN_MIX_1_0_REGRESSOR: ModelConfig(
            pretrained_model_name="autogluon/tabpfn-mix-1.0-regressor",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TABPFN_MIX_1_0_REGRESSOR

    # Sample in-context dataset sizes for regression
    _N_FEATURES = 100
    _N_TRAIN = 32
    _N_TEST = 8

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="TabPFNMix-Regressor",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.ATOMIC_ML,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, **kwargs):
        """Load TabPFNMix regressor weights from HuggingFace and wrap for tracing.

        Returns:
            torch.nn.Module: The wrapped TabPFNMix regressor model instance.
        """
        from autogluon.tabular.models.tabpfnmix._internal.models.foundation.foundation_transformer import (
            FoundationTransformer,
        )

        repo_id = self._variant_config.pretrained_model_name
        model = FoundationTransformer.from_pretrained(repo_id)
        model.eval()

        # Generate deterministic in-context training data; FoundationTransformer
        # expects a batch dimension: (batch, n_samples, n_features).
        # dtype_override is intentionally not supported: autogluon's
        # FoundationEmbeddingYFloat hard-codes float32 internally.
        torch.manual_seed(42)
        x_src = torch.randn(1, self._N_TRAIN, self._N_FEATURES)
        y_src = torch.randn(1, self._N_TRAIN)

        wrapper = TabPFNMixWrapper(model, x_src, y_src)
        wrapper.eval()
        return wrapper

    def load_inputs(self):
        """Prepare sample test inputs for the TabPFNMix regressor.

        Returns:
            torch.Tensor: Test features tensor (batch, n_test_samples, n_features).
        """
        torch.manual_seed(123)
        return torch.randn(1, self._N_TEST, self._N_FEATURES)
