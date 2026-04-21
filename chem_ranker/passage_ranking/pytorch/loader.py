# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ChemRanker model loader implementation for SMILES pair reranking.

ChemRanker is a Cross Encoder built on the ModChemBERT chemical language model
and fine-tuned for molecular reranking. Given a pair of SMILES strings it emits
a relevance score suitable for reranking candidate molecules against an anchor.

Reference: https://huggingface.co/Derify/ChemRanker-alpha-sim
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
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


class ModelVariant(StrEnum):
    """Available ChemRanker model variants for SMILES pair reranking."""

    ALPHA_SIM = "alpha-sim"


class ModelLoader(ForgeModel):
    """ChemRanker model loader implementation for SMILES pair reranking."""

    _VARIANTS = {
        ModelVariant.ALPHA_SIM: ModelConfig(
            pretrained_model_name="Derify/ChemRanker-alpha-sim",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ALPHA_SIM

    # Sample SMILES pairs (anchor, candidate) for reranking.
    sample_pairs = [
        (
            "c1snnc1C[NH2+]Cc1cc2c(s1)CCC2",
            "c1snnc1CCC[NH2+]Cc1cc2c(s1)CCC2",
        ),
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="ChemRanker",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False, "trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        anchors = [pair[0] for pair in self.sample_pairs]
        candidates = [pair[1] for pair in self.sample_pairs]

        inputs = self.tokenizer(
            anchors,
            candidates,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs
