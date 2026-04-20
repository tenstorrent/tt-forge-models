# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PlantCaduceus model loader implementation for masked language modeling on DNA sequences.
"""
from transformers import AutoTokenizer, AutoModelForMaskedLM
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
    """Available PlantCaduceus model variants."""

    PLANT_CADUCEUS_L32 = "kuleshov-group/PlantCaduceus_l32"


class ModelLoader(ForgeModel):
    """PlantCaduceus model loader implementation for masked language modeling on DNA sequences."""

    _VARIANTS = {
        ModelVariant.PLANT_CADUCEUS_L32: ModelConfig(
            pretrained_model_name="kuleshov-group/PlantCaduceus_l32",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PLANT_CADUCEUS_L32

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="PlantCaduceus",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMaskedLM.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        # DNA sequence for masked language modeling
        sequence = "ATGCGTACGATCGTAG"

        inputs = self.tokenizer.encode_plus(
            sequence,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        return inputs
