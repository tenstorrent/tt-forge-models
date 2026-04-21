# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Profluent E1 model loader implementation for masked language modeling on protein sequences.

E1 is a Transformer-based protein language model family from Profluent-Bio that
supports single-sequence and retrieval-augmented protein representation, designed
as a drop-in replacement for ESM-family models.

Requires the Profluent-AI/E1 repository to be cloned at /tmp/e1_repo so the
custom ``E1`` package is importable.
"""
import os
import subprocess
import sys

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

E1_REPO_PATH = "/tmp/e1_repo"
E1_REPO_URL = "https://github.com/Profluent-AI/E1.git"


def _ensure_e1_importable():
    """Ensure the Profluent-AI/E1 repo is cloned and importable."""
    if not os.path.isdir(E1_REPO_PATH):
        subprocess.check_call(
            ["git", "clone", "--filter=blob:none", E1_REPO_URL, E1_REPO_PATH]
        )

    src_path = os.path.join(E1_REPO_PATH, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


class ModelVariant(StrEnum):
    """Available Profluent E1 model variants."""

    E1_150M = "Profluent-Bio/E1-150m"


class ModelLoader(ForgeModel):
    """Profluent E1 model loader implementation for masked language modeling on protein sequences."""

    _VARIANTS = {
        ModelVariant.E1_150M: ModelConfig(
            pretrained_model_name="Profluent-Bio/E1-150m",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.E1_150M

    # Short protein sequence with E1's ``?`` mask token for testing
    sample_sequence = "MGSSHHHHHHSSGLVPRGSHM?GSSHHHHHHSSGLVPRGSHM"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.batch_preparer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Profluent E1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        _ensure_e1_importable()
        from E1.modeling import E1ForMaskedLM

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = E1ForMaskedLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        _ensure_e1_importable()
        from E1.batch_preparer import E1BatchPreparer

        if self.batch_preparer is None:
            self.batch_preparer = E1BatchPreparer()

        batch = self.batch_preparer.get_batch_kwargs([self.sample_sequence])

        return {
            "input_ids": batch["input_ids"],
            "within_seq_position_ids": batch["within_seq_position_ids"],
            "global_position_ids": batch["global_position_ids"],
            "sequence_ids": batch["sequence_ids"],
        }
