# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TabPFNMix classifier model loader implementation for tabular classification.

TabPFNMix is a tabular foundation model pre-trained on a mix of synthetic
classifiers. It uses an in-context learning paradigm: a support set of labeled
examples and a query set are passed jointly through a 12-layer encoder-decoder
Transformer (37M parameters) to produce class probabilities for the queries.
"""
import torch
from typing import Optional

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
    """Available TabPFNMix classifier model variants."""

    TABPFN_MIX_1_0_CLASSIFIER = "autogluon/tabpfn-mix-1.0-classifier"


class ModelLoader(ForgeModel):
    """TabPFNMix classifier model loader implementation for tabular classification."""

    _VARIANTS = {
        ModelVariant.TABPFN_MIX_1_0_CLASSIFIER: ModelConfig(
            pretrained_model_name="autogluon/tabpfn-mix-1.0-classifier",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TABPFN_MIX_1_0_CLASSIFIER

    # In-context learning parameters
    _N_FEATURES = 4
    _N_SUPPORT = 8
    _N_QUERY = 2
    _N_CLASSES = 3

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="TabPFNMix",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.ATOMIC_ML,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load TabPFNMix classifier model via PyTorchModelHubMixin.from_pretrained.

        Returns:
            torch.nn.Module: The FoundationTransformer model instance.
        """
        from autogluon.tabular.models.tabpfnmix._internal.models.foundation.foundation_transformer import (
            FoundationTransformer,
        )

        model = FoundationTransformer.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )
        if dtype_override is not None:
            model = model.to(dtype_override)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample in-context learning inputs for TabPFNMix classifier.

        Returns:
            list: [x_support, y_support, x_query] tensors.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        batch_size = 1

        # Support set: labeled examples for in-context learning
        x_support = torch.randn(
            batch_size, self._N_SUPPORT, self._N_FEATURES, dtype=dtype
        )
        y_support = torch.randint(
            0, self._N_CLASSES, (batch_size, self._N_SUPPORT), dtype=torch.int64
        )

        # Query set: examples to classify
        x_query = torch.randn(batch_size, self._N_QUERY, self._N_FEATURES, dtype=dtype)

        return [x_support, y_support, x_query]
