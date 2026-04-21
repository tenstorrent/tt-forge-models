# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
THYME2 colon end-to-end model loader implementation for token classification.
"""
from typing import Optional

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available THYME2 colon end-to-end model variants."""

    THYME2_COLON_E2E = "thyme2_colon_e2e"


class ModelLoader(ForgeModel):
    """THYME2 colon end-to-end model loader for clinical temporal token classification."""

    _VARIANTS = {
        ModelVariant.THYME2_COLON_E2E: LLMModelConfig(
            pretrained_model_name="mlml-chip/thyme2_colon_e2e",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.THYME2_COLON_E2E

    sample_text = (
        "The patient underwent a colonoscopy on January 5th, and a follow-up "
        "biopsy was performed two weeks later."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="thyme2_colon_e2e",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        # Importing this module registers the legacy "cnlpt" model type with
        # transformers' AutoConfig/AutoModel, allowing the pretrained weights
        # published under cnlpt_version 0.6.0 to be loaded.
        from cnlpt.legacy.CnlpModelForClassification import (
            CnlpModelForClassification,
        )

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = CnlpModelForClassification.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(
            self.sample_text,
            max_length=self._variant_config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs
