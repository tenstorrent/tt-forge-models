# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
qc_unitary_3qubit model loader for tt_forge_models.

Floki00/qc_unitary_3qubit is a diffusion model for quantum circuit synthesis
that compiles arbitrary 3-qubit unitaries into circuits drawn from the gate
set {h, cx, z, x, ccx, swap} with up to 12 gates. The underlying network is
a QC_Compilation_UNet from the genQC library, conditioned on a text prompt
(via a frozen OpenCLIP encoder) and on the target unitary matrix.

Reference: https://huggingface.co/Floki00/qc_unitary_3qubit
Paper: https://arxiv.org/abs/2311.02041
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


class ModelVariant(StrEnum):
    """Available qc_unitary_3qubit model variants."""

    UNITARY_3QUBIT = "unitary_3qubit"


class ModelLoader(ForgeModel):
    """qc_unitary_3qubit model loader.

    Loads the genQC DiffusionPipeline for Floki00/qc_unitary_3qubit and
    exposes the underlying QC_Compilation_UNet as a torch.nn.Module for
    compilation and inference.
    """

    _VARIANTS = {
        ModelVariant.UNITARY_3QUBIT: ModelConfig(
            pretrained_model_name="Floki00/qc_unitary_3qubit",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.UNITARY_3QUBIT

    # Architectural constants for the 3-qubit unitary compilation model.
    num_qubits = 3
    max_gates = 12
    color_dim = 8
    cond_emb_size = 512
    text_seq_len = 77

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="qc_unitary_3qubit",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from genQC.pipeline.diffusion_pipeline import DiffusionPipeline

        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name, "cpu"
        )

        model = self.pipeline.model
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        dtype = dtype_override if dtype_override is not None else torch.float32

        # x: color-embedded circuit [batch, color_dim, num_qubits, max_gates]
        x = torch.randn(
            batch_size, self.color_dim, self.num_qubits, self.max_gates, dtype=dtype
        )
        # t: diffusion timestep per batch element
        t = torch.randint(0, 1000, (batch_size,))
        # c_emb: frozen OpenCLIP text embeddings [batch, text_seq_len, cond_emb_size]
        c_emb = torch.randn(
            batch_size, self.text_seq_len, self.cond_emb_size, dtype=dtype
        )
        # U: stacked real/imaginary parts of the 8x8 target unitary
        unitary_dim = 2**self.num_qubits
        U = torch.randn(batch_size, 2, unitary_dim, unitary_dim, dtype=dtype)

        return (x, t, c_emb, U)
