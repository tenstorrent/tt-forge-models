# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Graphormer base PCQM4Mv2 model loader implementation for graph classification.

Graphormer is a graph Transformer pretrained by Microsoft on the PCQM4M-LSCv2
quantum-chemistry benchmark for molecular property prediction.
"""

from typing import Optional

import torch
from third_party.tt_forge_models.base import ForgeModel
from third_party.tt_forge_models.config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from .src.modeling_graphormer import GraphormerForGraphClassification


class ModelVariant(StrEnum):
    """Available Graphormer base PCQM4Mv2 model variants for graph classification."""

    GRAPHORMER_BASE_PCQM4MV2 = "clefourrier/graphormer-base-pcqm4mv2"


class ModelLoader(ForgeModel):
    """Graphormer base PCQM4Mv2 model loader for graph classification."""

    _VARIANTS = {
        ModelVariant.GRAPHORMER_BASE_PCQM4MV2: ModelConfig(
            pretrained_model_name="clefourrier/graphormer-base-pcqm4mv2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GRAPHORMER_BASE_PCQM4MV2

    # Shape parameters for the sample graph batch.
    _BATCH_SIZE = 1
    _NUM_NODES = 8
    _NUM_NODE_FEATURES = 9
    _NUM_EDGE_FEATURES = 3
    _MULTI_HOP_MAX_DIST = 5

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="graphormer_base_pcqm4mv2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.ATOMIC_ML,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load Graphormer graph classification model from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The model instance.
        """
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = GraphormerForGraphClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare a dummy graph batch for Graphormer graph classification.

        The transformers Graphormer collator emits a dict of pre-padded graph
        tensors; we synthesize an equivalent batch of shape `(batch_size=1,
        num_nodes=8)` so tracing has deterministic inputs.

        Args:
            dtype_override: Optional torch.dtype to override the float tensors'
                default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        float_dtype = dtype_override if dtype_override is not None else torch.float32

        batch = self._BATCH_SIZE
        n = self._NUM_NODES
        n_feat = self._NUM_NODE_FEATURES
        e_feat = self._NUM_EDGE_FEATURES
        k = self._MULTI_HOP_MAX_DIST

        torch.manual_seed(0)
        inputs = {
            "input_nodes": torch.randint(0, 4608, (batch, n, n_feat), dtype=torch.long),
            "input_edges": torch.randint(
                0, 1536, (batch, n, n, k, e_feat), dtype=torch.long
            ),
            "attn_bias": torch.zeros(batch, n + 1, n + 1, dtype=float_dtype),
            "in_degree": torch.randint(1, 8, (batch, n), dtype=torch.long),
            "out_degree": torch.randint(1, 8, (batch, n), dtype=torch.long),
            "spatial_pos": torch.randint(0, 16, (batch, n, n), dtype=torch.long),
            "attn_edge_type": torch.randint(
                0, 1536, (batch, n, n, e_feat), dtype=torch.long
            ),
        }
        return inputs
