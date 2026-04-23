# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AIDO.RNA model loader implementation for embedding generation on RNA sequences.
"""
import os
from pathlib import Path
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

_SRC_DIR = Path(__file__).parent / "src"


class ModelVariant(StrEnum):
    """Available AIDO.RNA model variants."""

    AIDO_RNA_1B600M = "genbio-ai/AIDO.RNA-1.6B"
    AIDO_RNA_650M_CDS = "genbio-ai/AIDO.RNA-650M-CDS"


class ModelLoader(ForgeModel):
    """AIDO.RNA model loader for embedding generation on RNA sequences."""

    _VARIANTS = {
        ModelVariant.AIDO_RNA_1B600M: ModelConfig(
            pretrained_model_name="genbio-ai/AIDO.RNA-1.6B",
        ),
        ModelVariant.AIDO_RNA_650M_CDS: ModelConfig(
            pretrained_model_name="genbio-ai/AIDO.RNA-650M-CDS",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.AIDO_RNA_1B600M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="AIDO.RNA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _register_rnabert(self):
        import sys

        sys.path.insert(0, str(_SRC_DIR))
        from transformers import AutoConfig, AutoModelForMaskedLM

        from configuration_rnabert import RNABertConfig
        from modeling_rnabert import RNABertForMaskedLM

        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        if "rnabert" not in CONFIG_MAPPING:
            AutoConfig.register("rnabert", RNABertConfig)
            AutoModelForMaskedLM.register(RNABertConfig, RNABertForMaskedLM)

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModelForMaskedLM

        self._register_rnabert()
        model = AutoModelForMaskedLM.from_pretrained(
            self._variant_config.pretrained_model_name
        ).eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        import sys

        import torch

        sys.path.insert(0, str(_SRC_DIR))
        from tokenization_rnabert import RNABertTokenizer

        vocab_file = str(_SRC_DIR / "vocab.txt")
        tokenizer = RNABertTokenizer(vocab_file, version="v2")

        rna_sequence = "ACGUACGUACGUACGU"
        encoded = tokenizer(
            rna_sequence,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=1024,
        )

        return dict(encoded)
