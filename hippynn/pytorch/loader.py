# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import os
from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    ModelConfig,
    StrEnum,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available model variants."""

    BASE = "Default"


class ModelLoader(ForgeModel):
    """Forge-compatible loader for the Hippynn model."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(pretrained_model_name="hippynn_model_pretrained")
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    @classmethod
    def _get_model_info(cls, variant: ModelVariant = None):
        """Return model information for Forge dashboards and reporting."""
        return ModelInfo(
            model="HIP-NN",
            variant=variant,
            group=ModelGroup.PRIORITY,
            task=ModelTask.ATOMIC_ML,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _patch_padding_indexer():
        # hippynn's PaddingIndexer.forward uses
        #   torch.nonzero(x, as_tuple=False, out=buf)[:, 0]
        # which is data-dependent and dynamo-hostile: the fake-tensor pass
        # infers the output as 1-D (size=(0,)) and the [:, 0] then errors
        # with "too many indices for tensor of dimension 1". Rewriting it as
        # torch.nonzero(x, as_tuple=True)[0] yields the same eager value and
        # traces cleanly.
        import hippynn.layers.indexers as _indexers
        import torch as _torch

        if getattr(_indexers.PaddingIndexer.forward, "_tt_xla_patched", False):
            return

        def forward(self, features, nonblank):
            dev = features.device
            n_molecules, n_atoms_max = nonblank.shape
            n_fictitious_atoms = nonblank.shape[0] * nonblank.shape[1]
            flat_nonblank = nonblank.reshape(n_fictitious_atoms)
            real_atoms = _torch.nonzero(flat_nonblank, as_tuple=True)[0]
            n_real_atoms = real_atoms.shape[0]
            # Avoid an `inv_real_atoms[real_atoms] = arange(...)` indexed
            # assignment, which lowers to stablehlo.scatter and tt-mlir
            # cannot legalize it for this op. Cumsum gives the correct
            # mapping at every real-atom position; values at blank
            # positions are unread downstream (only positions selected
            # via `pair_presence = nonblank_pair & ...` are indexed into
            # inv_real_atoms — see hippynn/layers/pairs/open.py:47-54).
            inv_real_atoms = flat_nonblank.long().cumsum(0) - 1
            indexed_features = features.reshape(n_molecules * n_atoms_max, -1)[
                real_atoms
            ]
            if indexed_features.ndimension() == 1:
                indexed_features = indexed_features.unsqueeze(1)
            mol_index_shaped = (
                _torch.arange(n_molecules, dtype=_torch.long, device=dev)
                .unsqueeze(1)
                .expand(-1, n_atoms_max)
            )
            atom_index_shaped = (
                _torch.arange(n_atoms_max, dtype=_torch.long, device=dev)
                .unsqueeze(0)
                .expand(n_molecules, -1)
            )
            atom_index = atom_index_shaped.reshape(n_fictitious_atoms)[real_atoms]
            mol_index = mol_index_shaped.reshape(n_fictitious_atoms)[real_atoms]
            return (
                indexed_features,
                real_atoms,
                inv_real_atoms,
                mol_index,
                atom_index,
                n_molecules,
                n_atoms_max,
            )

        forward._tt_xla_patched = True
        _indexers.PaddingIndexer.forward = forward

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Hippynn model wrapped in a Torch module.

        Args:
            dtype_override: Optional torch.dtype to override model precision.
        """

        import hippynn
        from hippynn.graphs import GraphModule, inputs, networks, targets
        from hippynn.custom_kernels import set_custom_kernels

        # Force the pure-PyTorch envsum/sensesum/featsum implementations.
        # The numba CPU kernels call `.numpy()` on the input tensor, which
        # fails for bfloat16 (raised under TT_XLA_ARCH=n150). The env-var
        # equivalent is read at hippynn import time and would be too late
        # here, so we call the runtime selector explicitly.
        set_custom_kernels("pytorch")

        # Disable the low-distance warning code path in SensitivityModule /
        # InverseSensitivityModule (hippynn/layers/hiplayers.py:60-66 and
        # :90-99). That path calls `mu, argmin = self.mu.min(dim=1)`, whose
        # StableHLO lowering produces a 2-output `stablehlo.reduce` with an
        # ArgMin reducer body. tt-mlir's StableHLOToTTIRPatterns.cpp
        # handles ArgMax but not ArgMin, so the SHLO->TTIR conversion
        # fails to legalize the op. The setting is read at forward time,
        # so flipping it here (before the model traces) is sufficient.
        hippynn.settings.WARN_LOW_DISTANCES = False

        self._patch_padding_indexer()

        network_params = {
            "possible_species": [0, 1, 6, 7, 8, 16],
            "n_features": 20,
            "n_sensitivities": 20,
            "dist_soft_min": 1.6,
            "dist_soft_max": 10.0,
            "dist_hard_max": 12.5,
            "n_interaction_layers": 2,
            "n_atom_layers": 3,
        }
        species = inputs.SpeciesNode(db_name="Z")
        positions = inputs.PositionsNode(db_name="R")
        network = networks.Hipnn(
            "hippynn_model", (species, positions), module_kwargs=network_params
        )
        henergy = targets.HEnergyNode("HEnergy", network, db_name="T")

        # Load model
        model = GraphModule([species, positions], [henergy.mol_energy])

        for param in model.parameters():
            param.requires_grad = False

        model.eval()
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def unpack_forward_output(self, fwd_output):
        return fwd_output[0]

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs (species and positions) for the model.

        Args:
            dtype_override: Optional torch.dtype to override input precision.
        """

        import ase.build
        import ase.units

        atoms = ase.build.molecule("H2O")
        positions = (
            torch.as_tensor(atoms.positions / ase.units.Bohr)
            .unsqueeze(0)
            .to(torch.get_default_dtype())
        )

        species = torch.as_tensor(atoms.get_atomic_numbers()).unsqueeze(0)

        if dtype_override is not None:
            positions = positions.to(dtype_override)

        return (species, positions)
