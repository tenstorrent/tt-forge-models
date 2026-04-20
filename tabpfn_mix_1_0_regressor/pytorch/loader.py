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
import json
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
        return self.model(x_src=self.x_src, y_src=self.y_src, x_tgt=x_tgt)


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

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load TabPFNMix regressor weights from HuggingFace and wrap for tracing.

        Returns:
            torch.nn.Module: The wrapped TabPFNMix regressor model instance.
        """
        from autogluon.tabular.models.tabpfnmix._internal.core.tabpfn_model.tabpfn import (
            TabPFN,
        )
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        repo_id = self._variant_config.pretrained_model_name
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")

        with open(config_path) as f:
            config = json.load(f)

        model = TabPFN(
            n_features=config["n_features"],
            n_classes=config["n_classes"],
            dim=config["dim"],
            n_layers=config["n_layers"],
            n_heads=config["n_heads"],
            attn_dropout=config["attn_dropout"],
            task=config["task"],
            y_as_float_embedding=config["y_as_float_embedding"],
        )
        model.load_state_dict(load_file(weights_path))
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        # Generate deterministic in-context training data
        torch.manual_seed(42)
        dtype = dtype_override if dtype_override is not None else torch.float32
        x_src = torch.randn(self._N_TRAIN, self._N_FEATURES, dtype=dtype)
        y_src = torch.randn(self._N_TRAIN, dtype=dtype)

        wrapper = TabPFNMixWrapper(model, x_src, y_src)
        wrapper.eval()
        return wrapper

    def load_inputs(self, dtype_override=None):
        """Prepare sample test inputs for the TabPFNMix regressor.

        Returns:
            torch.Tensor: Test features tensor (n_test_samples, n_features).
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        torch.manual_seed(123)
        x_tgt = torch.randn(self._N_TEST, self._N_FEATURES, dtype=dtype)

        return x_tgt
