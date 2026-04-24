# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RhoFold+ model loader implementation for RNA 3D structure prediction.

RhoFold+ combines evolutionary information from multiple sequence alignments
with embeddings from the RNA-FM language model to predict RNA tertiary
structures from sequence.

Reference: https://github.com/ml4bio/RhoFold
HuggingFace: https://huggingface.co/cuhkaih/rhofold
"""

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
    """Available RhoFold+ model variants."""

    BASE = "cuhkaih/rhofold"


class ModelLoader(ForgeModel):
    """RhoFold+ model loader for RNA 3D structure prediction."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="cuhkaih/rhofold",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    _WEIGHTS_FILENAME = "rhofold_pretrained_params.pt"

    # Sample RNA sequence taken from the RhoFold+ example input (PDB 3owzA).
    _SAMPLE_SEQUENCE = (
        "GGCUCUGGAGAGAACCGUUUAAUCGGUCGCCGAAGGAGCAAGCUCUGCGG"
        "AAACGCAGAGUGAAACUCUCAGGCAAAAGGACAGAGUC"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="RhoFold+",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.ATOMIC_ML,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the RhoFold+ model with pretrained weights."""
        from .src.model import build_rhofold_model

        # dtype_override is intentionally not applied: the embedded RNA-FM
        # sub-model has hardcoded .float() casts in its attention layers that
        # produce float32 intermediate tensors, which conflict with bfloat16
        # LayerNorm parameters and cause a mixed-dtype RuntimeError on CPU.
        return build_rhofold_model(
            repo_id=self._variant_config.pretrained_model_name,
            weights_filename=self._WEIGHTS_FILENAME,
        )

    def load_inputs(self, dtype_override=None):
        """Prepare RhoFold+ input features for a sample RNA sequence.

        The returned features are integer token tensors and the raw sequence
        string, so ``dtype_override`` is accepted for interface parity but has
        no effect on the inputs.
        """
        from .src.model import build_rhofold_inputs

        return build_rhofold_inputs(self._SAMPLE_SEQUENCE)
