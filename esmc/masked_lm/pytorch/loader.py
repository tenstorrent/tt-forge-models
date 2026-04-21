# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ESM Cambrian (ESM C) model loader implementation for protein representation learning.
"""
from typing import Optional

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein

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


class ModelVariant(StrEnum):
    """Available ESM C model variants."""

    ESMC_300M_2024_12 = "EvolutionaryScale/esmc-300m-2024-12"


# Mapping from the HF-id variant to the shortname expected by the ``esm`` package.
_ESM_SHORTNAMES = {
    ModelVariant.ESMC_300M_2024_12: "esmc_300m",
}


class ModelLoader(ForgeModel):
    """ESM C model loader for protein representation learning."""

    _VARIANTS = {
        ModelVariant.ESMC_300M_2024_12: ModelConfig(
            pretrained_model_name="EvolutionaryScale/esmc-300m-2024-12",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ESMC_300M_2024_12

    # Short protein sequence for testing
    sample_sequence = "MGSSHHHHHHSSGLVPRGSHM"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ESMC",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        shortname = _ESM_SHORTNAMES[self._variant]
        model = ESMC.from_pretrained(shortname, **kwargs)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.model is None:
            self.load_model(dtype_override=dtype_override)

        protein = ESMProtein(sequence=self.sample_sequence)
        protein_tensor = self.model.encode(protein)

        return {"sequence_tokens": protein_tensor.sequence.unsqueeze(0)}
