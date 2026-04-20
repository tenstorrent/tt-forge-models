# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mitra regressor model loader implementation for tabular regression.

Mitra is a tabular foundation model for regression using an in-context
learning paradigm. It operates on support/query sets of tabular data using
a 12-layer Transformer with 2D attention (across observations and features).
"""
import importlib.util
import os
import sys
import types

import torch
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


def _import_tab2d():
    """Import Tab2D bypassing autogluon's __init__ chain.

    autogluon.tabular.__init__ and autogluon.tabular.models.__init__ pull in
    autogluon.common / autogluon.core which depend on pandas <3.  The base
    test environment pins pandas 3.x, so importing through the normal chain
    crashes with an AttributeError inside pandas._libs.  Tab2D and its
    internal deps only need torch / einops / einx / huggingface_hub /
    safetensors, so we register lightweight stub parent packages to prevent
    the problematic __init__.py files from executing.
    """
    target = "autogluon.tabular.models.mitra._internal.models.tab2d"
    if target in sys.modules:
        return sys.modules[target].Tab2D

    ag_spec = importlib.util.find_spec("autogluon")
    if ag_spec is None:
        raise ImportError("autogluon.tabular is not installed")

    ag_root = ag_spec.submodule_search_locations[0]

    stub_pkgs = {
        "autogluon.tabular": os.path.join(ag_root, "tabular"),
        "autogluon.tabular.models": os.path.join(ag_root, "tabular", "models"),
    }
    for name, path in stub_pkgs.items():
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.__path__ = [path]
            stub.__package__ = name
            sys.modules[name] = stub

    from autogluon.tabular.models.mitra._internal.models.tab2d import Tab2D

    return Tab2D


class ModelVariant(StrEnum):
    """Available Mitra regressor model variants."""

    MITRA_REGRESSOR = "autogluon/mitra-regressor"


class ModelLoader(ForgeModel):
    """Mitra model loader implementation for tabular regression."""

    _VARIANTS = {
        ModelVariant.MITRA_REGRESSOR: ModelConfig(
            pretrained_model_name="autogluon/mitra-regressor",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MITRA_REGRESSOR

    # In-context learning parameters
    _N_FEATURES = 4
    _N_SUPPORT = 8
    _N_QUERY = 2
    _DIM_OUTPUT = 1

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Mitra",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load Mitra model for tabular regression.

        Returns:
            torch.nn.Module: The Mitra Tab2D model instance.
        """
        Tab2D = _import_tab2d()

        model = Tab2D.from_pretrained(
            self._variant_config.pretrained_model_name, device="cpu"
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample in-context learning inputs for the Mitra regressor model.

        Returns:
            list: [x_support, y_support, x_query, padding_features,
                   padding_obs_support, padding_obs_query] tensors.
        """
        dtype = torch.float32
        batch_size = 1

        # Support set: labeled examples for in-context learning
        x_support = torch.randn(
            batch_size, self._N_SUPPORT, self._N_FEATURES, dtype=dtype
        )
        y_support = torch.randn(batch_size, self._N_SUPPORT, dtype=dtype)

        # Query set: examples to predict
        x_query = torch.randn(batch_size, self._N_QUERY, self._N_FEATURES, dtype=dtype)

        # No padding - all features and observations are valid
        padding_features = torch.zeros(batch_size, self._N_FEATURES, dtype=torch.bool)
        padding_obs_support = torch.zeros(batch_size, self._N_SUPPORT, dtype=torch.bool)
        padding_obs_query = torch.zeros(batch_size, self._N_QUERY, dtype=torch.bool)

        return [
            x_support,
            y_support,
            x_query,
            padding_features,
            padding_obs_support,
            padding_obs_query,
        ]
