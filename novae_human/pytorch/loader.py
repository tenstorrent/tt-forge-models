# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Novae human spatial-transcriptomics foundation model loader implementation.

Novae is a graph-based foundation model for spatial transcriptomics that
performs zero-shot spatial domain assignment over human tissue spatial gene
expression data. Its backbone is a graph attention network that consumes PyG
``Batch`` objects built from spatial neighborhood graphs of cells.
"""
import torch
from typing import Optional

# Capture pandas._libs.lib before any requirements manager purge so that
# our reference survives the sys.modules eviction.  CPython never truly
# unloads a C extension (.so), so this object is the *same* one that any
# subsequent `import pandas._libs.lib` will return – meaning a patch here
# persists into the novae import chain.
try:
    from pandas._libs import lib as _PANDAS_LIB
except ImportError:
    _PANDAS_LIB = None

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
    """Available Novae model variants."""

    NOVAE_HUMAN_0 = "MICS-Lab/novae-human-0"


class ModelLoader(ForgeModel):
    """Novae human foundation model loader for spatial transcriptomics."""

    _VARIANTS = {
        ModelVariant.NOVAE_HUMAN_0: ModelConfig(
            pretrained_model_name="MICS-Lab/novae-human-0",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NOVAE_HUMAN_0

    # Synthetic spatial graph dimensions for sample inputs
    _N_GRAPHS = 2
    _N_NODES_PER_GRAPH = 7
    _N_EDGES_PER_GRAPH = 10
    _N_GENES_USED = 32

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="novae",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.ATOMIC_ML,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Novae model via ``PyTorchModelHubMixin.from_pretrained``.

        Returns:
            torch.nn.Module: The Novae model instance.
        """
        # pandas 3.x removed is_interval from _libs.lib; anndata/scanpy (via novae)
        # still imports it.  Since CPython cannot truly reload a C extension that is
        # already mapped into memory, we patch the existing module object directly.
        if _PANDAS_LIB is not None and not hasattr(_PANDAS_LIB, "is_interval"):
            _PANDAS_LIB.is_interval = lambda x: (
                type(x).__name__ == "Interval"
                and hasattr(type(x), "left")
                and hasattr(type(x), "right")
            )

        import novae

        model = novae.Novae.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        self._model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Build a synthetic batch of spatial-transcriptomics graphs.

        Returns:
            tuple: Single-element tuple containing the batch dict so the test
            infrastructure passes it as the positional ``batch`` argument to
            ``Novae.forward``.
        """
        from torch_geometric.data import Batch, Data

        if self._model is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.float32
        gene_names = self._model.cell_embedder.gene_names
        n_genes_used = min(self._N_GENES_USED, len(gene_names))
        genes_indices = torch.arange(n_genes_used, dtype=torch.long).unsqueeze(0)

        torch.manual_seed(0)

        def _make_graph(slide_id: str):
            return Data(
                x=torch.rand(self._N_NODES_PER_GRAPH, n_genes_used, dtype=dtype),
                edge_index=torch.randint(
                    0,
                    self._N_NODES_PER_GRAPH,
                    (2, self._N_EDGES_PER_GRAPH),
                    dtype=torch.long,
                ),
                edge_attr=torch.rand(self._N_EDGES_PER_GRAPH, 1, dtype=dtype),
                genes_indices=genes_indices,
                slide_id=slide_id,
            )

        main_batch = Batch.from_data_list(
            [_make_graph(f"slide_{i}") for i in range(self._N_GRAPHS)]
        )
        view_batch = Batch.from_data_list(
            [_make_graph(f"slide_{i}") for i in range(self._N_GRAPHS)]
        )

        # Return as a tuple so the test infra passes the dict as a single
        # positional `batch` argument rather than spreading it as **kwargs.
        return ({"main": main_batch, "view": view_batch},)
